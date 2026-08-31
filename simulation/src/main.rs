//! Chuang et al. (2024) "Simulating Opinion Dynamics with Networks of LLM-based
//! Agents" — 再現実験の CLI エントリポイント．
//!
//! `run`       : 単一設定で dyadic LLM 意見力学を実行する (`--control no-interaction`
//!               で非相互作用統制条件，`--mock` でオフライン scripted 駆動)．
//! `sweep`     : 確証バイアス × フレーミング × トポロジ (× メモリ方式) を走査し，
//!               条件 1 点ごとに子 run を起こして `runs` 本の試行を回す．
//! `reproduce` : 論文 4.3 の見出し的知見 (確証バイアスによる合意→断片化の遷移，
//!               非相互作用統制との対比，トポロジ間の収束比較) を一括再現し，
//!               観測 vs 論文の PASS/off を run スコープの指標とイベントに記録する．
//!
//! 出力の置き場と同一性は runvault が持つ．タイムスタンプ付きディレクトリも
//! `latest` シンボリックリンクもこちらでは作らず，`Run::start` が決めた run
//! ディレクトリへ書く．
//!
//! LLM クライアントは記録を始める **前** に組む．`run.json` の `llm` ブロックは
//! `Run::start` の時点で確定するので，モデル名と endpoint を知っている側 (=
//! クライアントを組んだ側) が先に立たないと，llm ブロックを埋めないまま記録できて
//! しまう．

use std::fs;
use std::path::Path;

use clap::{Parser, Subcommand};
use runvault::{Lineage, Run, RunOptions};
use serde::Serialize;

use chuang_opinion_simulation::config::{
    parse_bias, parse_control, parse_framing, parse_memory, parse_topology, Config,
    ConfirmationBias, Framing, LlmSettings, MemoryMode, Topology,
};
use chuang_opinion_simulation::llm::{build_live_client, OpinionClient};
use chuang_opinion_simulation::record::{self, DOMAIN, EXPERIMENT, REPO_ID};
use chuang_opinion_simulation::reproduce_mock::build_reproduce_client;
use chuang_opinion_simulation::simulation::{run_with_client, save_opinions, SimulationResult};

// ---------------------------------------------------------------------------
// CLI 定義
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "chuang",
    about = "Chuang et al. (2024) Simulating Opinion Dynamics with Networks of LLM-based Agents — 再現実験"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Ollama 接続先 URL（指定時は環境変数 OLLAMA_HOST を上書きする）．
    #[arg(long, global = true)]
    ollama_host: Option<String>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// 単一設定で dyadic LLM 意見力学を実行する．
    Run(RunArgs),
    /// 確証バイアス × フレーミング × トポロジを走査し，最終 B/D を集計する．
    Sweep(SweepArgs),
    /// 論文 4.3 の見出し的知見を一括再現し reproduce_summary.json に集計する．
    Reproduce(ReproduceArgs),
}

#[derive(Parser, Debug)]
struct RunArgs {
    /// エージェント数 N．
    #[arg(long, default_value_t = 10)]
    n_agents: usize,

    /// 議論トピック (ground truth 既知の短い記述; アンダースコア可)．
    #[arg(long, default_value = "flat_earth")]
    topic: String,

    /// フレーミング (true / false)．
    #[arg(long, default_value = "false")]
    framing: String,

    /// 確証バイアス (none / weak / strong)．
    #[arg(long, default_value = "none")]
    bias: String,

    /// メモリ方式 (cumulative / reflective)．
    #[arg(long, default_value = "cumulative")]
    memory: String,

    /// 統制条件 (interaction = 通常 / no-interaction = 近傍を見ない単独進化)．
    /// 非相互作用統制は «網による意見変化» と «LLM 自身の prior によるドリフト»
    /// を分離する鍵となる ablation．
    #[arg(long, default_value = "interaction")]
    control: String,

    /// LLM を呼ばず決定論的 scripted mock で駆動する (オフライン検証用)．
    #[arg(long, default_value_t = false)]
    mock: bool,

    /// トポロジ (full / ws / ba / er)．
    #[arg(long, default_value = "full")]
    topology: String,

    /// WS の各ノードの初期次数 k (偶数)．
    #[arg(long, default_value_t = 4)]
    ws_k: usize,

    /// WS の再配線確率 β．
    #[arg(long, default_value_t = 0.1)]
    ws_beta: f64,

    /// BA の新規ノードあたりの結合数 m．
    #[arg(long, default_value_t = 2)]
    ba_m: usize,

    /// 1 ステップあたりの dyadic interaction 数．
    #[arg(long, default_value_t = 1)]
    events_per_step: usize,

    /// 最大ステップ数 T．
    #[arg(long, default_value_t = 100)]
    max_steps: usize,

    /// 収束判定の意見分散しきい値．
    #[arg(long, default_value_t = 1e-6)]
    tol: f64,

    /// 乱数シード (省略時はランダム; socsim コア層のみ支配)．
    #[arg(long)]
    seed: Option<u64>,

    /// LLM 生成温度 (既定 0.0; 論文は 0.7)．
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// LLM 生成シード (バックエンドへ渡す)．
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,

    /// プロンプト→応答キャッシュの保存先 (既定 .llm_cache/cache.json)．
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,

    /// 結果出力ディレクトリ．
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct SweepArgs {
    /// カンマ区切りの確証バイアスリスト．
    #[arg(long, default_value = "none,weak,strong")]
    bias_values: String,

    /// カンマ区切りのフレーミングリスト．
    #[arg(long, default_value = "true,false")]
    framing_values: String,

    /// カンマ区切りのトポロジリスト．
    #[arg(long, default_value = "full")]
    topology_values: String,

    /// メモリ方式 (cumulative / reflective; sweep では単一固定)．
    #[arg(long, default_value = "cumulative")]
    memory: String,

    /// 議論トピック．
    #[arg(long, default_value = "flat_earth")]
    topic: String,

    /// エージェント数 N．
    #[arg(long, default_value_t = 10)]
    n_agents: usize,

    /// 各条件あたりの独立試行数．
    #[arg(long, default_value_t = 5)]
    runs: usize,

    /// 1 ステップあたりの dyadic interaction 数．
    #[arg(long, default_value_t = 1)]
    events_per_step: usize,

    /// 最大ステップ数 T．
    #[arg(long, default_value_t = 100)]
    max_steps: usize,

    /// 収束判定の意見分散しきい値．
    #[arg(long, default_value_t = 1e-6)]
    tol: f64,

    /// 乱数シード基点 (各試行は derive により独立化する)．
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// LLM 生成温度．
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// LLM 生成シード．
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,

    /// プロンプト→応答キャッシュの保存先 (sweep 全体で共有しヒット率を高める)．
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,

    /// 結果出力ベースディレクトリ．
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct ReproduceArgs {
    /// エージェント数 N．
    #[arg(long, default_value_t = 12)]
    n_agents: usize,

    /// 議論トピック (false framing で flat_earth = «地球平面説は偽»)．
    #[arg(long, default_value = "flat_earth")]
    topic: String,

    /// フレーミング (true / false)．
    #[arg(long, default_value = "false")]
    framing: String,

    /// 各条件あたりの独立試行数．
    #[arg(long, default_value_t = 5)]
    runs: usize,

    /// 1 ステップあたりの dyadic interaction 数．
    #[arg(long, default_value_t = 2)]
    events_per_step: usize,

    /// 最大ステップ数 T．
    #[arg(long, default_value_t = 40)]
    max_steps: usize,

    /// トポロジ比較に用いるトポロジリスト (カンマ区切り)．
    #[arg(long, default_value = "full,er,ws,ba")]
    topology_values: String,

    /// 乱数シード基点 (各条件・試行は derive により独立化する)．
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// LLM を呼ばず決定論的 scripted mock で駆動する (オフライン検証用)．
    /// サンドボックス・CI では `--mock` を付ける (ライブ LLM 不要)．
    #[arg(long, default_value_t = false)]
    mock: bool,

    /// LLM 生成温度 (live 時のみ)．
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// LLM 生成シード (live 時のみ)．
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,

    /// プロンプト→応答キャッシュの保存先 (live 時のみ; 全条件で共有)．
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,

    /// 軽量モード (N と runs と max_steps を縮小; 動作確認用)．
    #[arg(long, default_value_t = false)]
    quick: bool,

    /// 結果出力ベースディレクトリ．
    #[arg(long, default_value = "results")]
    output_dir: String,
}

// ---------------------------------------------------------------------------
// 補助
// ---------------------------------------------------------------------------

/// カンマ区切り文字列を trim 済みの非空リストへ．
fn split_csv(s: &str) -> Vec<String> {
    s.split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

/// LLM クライアントを組み，モデル名と endpoint を先に取り出す．
///
/// `Run::start` は開始時点で `run.json` を書くので，`llm` ブロックを埋めるには記録を
/// 始める前にクライアントを組んでおく必要がある．mock は in-memory キャッシュしか
/// 持たないので，`save()` を呼ばないよう `cache_path` を落とした設定も一緒に返す．
fn build_client(cfg: &Config, mock: bool) -> (Config, OpinionClient, String, String) {
    let (cfg, client) = if mock {
        let mut mock_cfg = cfg.clone();
        mock_cfg.llm.cache_path = None;
        (mock_cfg, build_reproduce_client())
    } else {
        let client = build_live_client(&cfg.llm)
            .unwrap_or_else(|e| panic!("LLM クライアント構築に失敗: {e}"));
        (cfg.clone(), client)
    };
    let model = client.inner().model().to_string();
    let endpoint = client.inner().endpoint().to_string();
    (cfg, client, model, endpoint)
}

/// キャッシュファイルの親ディレクトリを用意する．
fn ensure_cache_dir(cache_path: &str) {
    if let Some(parent) = Path::new(cache_path).parent() {
        let _ = fs::create_dir_all(parent);
    }
}

/// `run` サブコマンドの実験条件．
///
/// 旧 `config.json` の内容から `command` と `output_dir` を落としたもの．どの
/// サブコマンドかは `run.json` が持ち，run ディレクトリが出力先そのものである．
#[derive(Serialize)]
struct RunParameters {
    n_agents: usize,
    topic: String,
    framing: &'static str,
    bias: &'static str,
    memory_mode: &'static str,
    interact: bool,
    topology: &'static str,
    er_p: f64,
    ws_k: usize,
    ws_beta: f64,
    ba_m: usize,
    events_per_step: usize,
    max_steps: usize,
    tol: f64,
    seed: u64,
    llm_temperature: f32,
    llm_seed: u64,
    mock: bool,
}

/// スイープ親 run の実験条件 (グリッド定義そのもの)．
#[derive(Serialize)]
struct SweepParameters {
    bias_values: Vec<String>,
    framing_values: Vec<String>,
    topology_values: Vec<String>,
    memory: &'static str,
    topic: String,
    n_agents: usize,
    runs: usize,
    events_per_step: usize,
    max_steps: usize,
    tol: f64,
    seed: u64,
    llm_temperature: f32,
    llm_seed: u64,
}

/// スイープの子 run (bias × framing × topology の 1 点) の実験条件．
///
/// `run` の条件に `runs` が付いた形で，`run` とは別のサブコマンド名を持つ．同じ
/// `run` を名乗らせると，«1 本のシミュレーション» と «同一条件の `runs` 本» という
/// 中身の違う 2 つが 1 つの名前に同居し，`runvault path --subcommand run` がどちらを
/// 返すか分からなくなる．
#[derive(Serialize)]
struct SweepPointParameters {
    bias: &'static str,
    framing: &'static str,
    topology: &'static str,
    memory: &'static str,
    topic: String,
    n_agents: usize,
    runs: usize,
    events_per_step: usize,
    max_steps: usize,
    tol: f64,
    seed: u64,
    llm_temperature: f32,
    llm_seed: u64,
}

/// `reproduce` の実験条件．
#[derive(Serialize)]
struct ReproduceParameters {
    n_agents: usize,
    topic: String,
    framing: &'static str,
    topology_values: Vec<String>,
    runs: usize,
    events_per_step: usize,
    max_steps: usize,
    tol: f64,
    seed: u64,
    llm_temperature: f32,
    llm_seed: u64,
    mock: bool,
    quick: bool,
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

fn cmd_run(args: RunArgs) {
    let framing = parse_framing(&args.framing).unwrap_or_else(|e| panic!("{}", e));
    let bias = parse_bias(&args.bias).unwrap_or_else(|e| panic!("{}", e));
    let memory_mode = parse_memory(&args.memory).unwrap_or_else(|e| panic!("{}", e));
    let topology = parse_topology(&args.topology).unwrap_or_else(|e| panic!("{}", e));
    let interact = parse_control(&args.control).unwrap_or_else(|e| panic!("{}", e));

    // シードを実体化してから記録する．--seed 省略時にシミュレーション側で
    // rand::random に落とすと，実際に使われたシードがどこにも残らない．
    let seed = args.seed.unwrap_or_else(rand::random::<u64>);

    let cfg = Config {
        n_agents: args.n_agents,
        topic: args.topic.clone(),
        framing,
        bias,
        memory_mode,
        interact,
        topology,
        er_p: 0.3,
        ws_k: args.ws_k,
        ws_beta: args.ws_beta,
        ba_m: args.ba_m,
        events_per_step: args.events_per_step,
        max_steps: args.max_steps,
        tol: args.tol,
        seed: Some(seed),
        llm: LlmSettings {
            temperature: args.temperature,
            seed: args.llm_seed,
            cache_path: Some(args.cache_path.clone()),
        },
    };

    if !args.mock {
        ensure_cache_dir(&args.cache_path);
    }
    let (cfg, client, model, endpoint) = build_client(&cfg, args.mock);

    let parameters = RunParameters {
        n_agents: cfg.n_agents,
        topic: cfg.topic.clone(),
        framing: cfg.framing.label(),
        bias: cfg.bias.label(),
        memory_mode: cfg.memory_mode.label(),
        interact: cfg.interact,
        topology: cfg.topology.label(),
        er_p: cfg.er_p,
        ws_k: cfg.ws_k,
        ws_beta: cfg.ws_beta,
        ba_m: cfg.ba_m,
        events_per_step: cfg.events_per_step,
        max_steps: cfg.max_steps,
        tol: cfg.tol,
        seed,
        llm_temperature: cfg.llm.temperature,
        llm_seed: cfg.llm.seed,
        mock: args.mock,
    };

    let mut rv = Run::start(
        RunOptions::new(EXPERIMENT, "run")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&parameters)
            .expect("runvault: parameters の組み立てに失敗")
            .seed_pointers(["/seed"])
            .master_seed(seed)
            .llm(record::llm_block(&model, &endpoint, cfg.llm.temperature))
            .replication(record::replication()),
    )
    .expect("runvault: run の開始に失敗");

    // run ディレクトリが出力先そのものになる．意見の軌跡は artifacts/ の下へ．
    let artifacts = rv.dir().join("artifacts");
    fs::create_dir_all(&artifacts).expect("artifacts ディレクトリの作成に失敗");

    println!("=== Chuang et al. (2024) LLM 意見力学 再現実験 ===");
    println!(
        "N: {} | topic: {} | framing: {} | bias: {} | memory: {} | topology: {} | control: {}{}",
        cfg.n_agents,
        cfg.topic,
        cfg.framing.label(),
        cfg.bias.label(),
        cfg.memory_mode.label(),
        cfg.topology.label(),
        if cfg.interact {
            "interaction"
        } else {
            "no-interaction"
        },
        if args.mock { " | MOCK" } else { "" },
    );
    println!(
        "events/step: {} | max_steps: {} | tol: {} | seed: {}",
        cfg.events_per_step, cfg.max_steps, cfg.tol, seed
    );
    println!(
        "LLM: model={} temp={} llm_seed={} cache={}",
        model,
        cfg.llm.temperature,
        cfg.llm.seed,
        if args.mock {
            "(in-memory)"
        } else {
            args.cache_path.as_str()
        },
    );
    println!("出力先: {}", rv.dir().display());
    println!("-------------------------------------------------");

    let result = run_with_client(&cfg, client).unwrap_or_else(|e| panic!("実行に失敗: {}", e));

    save_opinions(&result, &artifacts.to_string_lossy());
    record::log_simulation(&mut rv, &result);
    // run は全ステップを metrics.csv に残しているので，観測時刻も全ステップ．
    let observed: Vec<u64> = result.metrics_history.iter().map(|m| m.t as u64).collect();
    record::log_terminal(
        &mut rv,
        "run",
        seed,
        cfg.max_steps,
        cfg.tol,
        observed,
        &result,
    );

    let last = result.metrics_history.last().unwrap();
    println!(
        "収束: {} | ステップ: {}",
        if result.converged { "Yes" } else { "No" },
        result.final_step
    );
    println!(
        "最終 Bias B: {:.4} | Diversity D: {:.4} | クラスタ数: {} | 分極: {:.4}",
        last.bias, last.diversity, last.n_clusters, last.polarization
    );
    println!(
        "LLM 呼び出し: {} 回 | cache-hit: {} ({:.1}%) | model: {}",
        result.metadata.total(),
        result.metadata.cache_hits(),
        result.metadata.cache_hit_rate() * 100.0,
        result.llm_model,
    );

    let dir = rv.finish().expect("runvault: run の完了に失敗");
    println!("意見軌跡   → {}/artifacts/opinions.csv", dir.display());
    println!("メトリクス → {}/metrics.csv", dir.display());
    println!("終端       → {}/events.jsonl", dir.display());
    println!("設定       → {}/config.json", dir.display());
}

// ---------------------------------------------------------------------------
// sweep
// ---------------------------------------------------------------------------

fn cmd_sweep(args: SweepArgs) {
    let biases: Vec<ConfirmationBias> = split_csv(&args.bias_values)
        .iter()
        .map(|s| parse_bias(s).unwrap_or_else(|e| panic!("{}", e)))
        .collect();
    let framings: Vec<Framing> = split_csv(&args.framing_values)
        .iter()
        .map(|s| parse_framing(s).unwrap_or_else(|e| panic!("{}", e)))
        .collect();
    let topologies: Vec<Topology> = split_csv(&args.topology_values)
        .iter()
        .map(|s| parse_topology(s).unwrap_or_else(|e| panic!("{}", e)))
        .collect();
    let memory_mode: MemoryMode = parse_memory(&args.memory).unwrap_or_else(|e| panic!("{}", e));

    ensure_cache_dir(&args.cache_path);

    let n_total = biases.len() * framings.len() * topologies.len() * args.runs;

    let sweep_parameters = SweepParameters {
        bias_values: split_csv(&args.bias_values),
        framing_values: split_csv(&args.framing_values),
        topology_values: split_csv(&args.topology_values),
        memory: memory_mode.label(),
        topic: args.topic.clone(),
        n_agents: args.n_agents,
        runs: args.runs,
        events_per_step: args.events_per_step,
        max_steps: args.max_steps,
        tol: args.tol,
        seed: args.seed,
        llm_temperature: args.temperature,
        llm_seed: args.llm_seed,
    };

    // sweep はライブ LLM のみ (`--mock` を持たない) なので，モデル名は最初に組んだ
    // クライアントが名乗った値を親子で共有する．
    let probe_settings = LlmSettings {
        temperature: args.temperature,
        seed: args.llm_seed,
        cache_path: Some(args.cache_path.clone()),
    };
    let probe = build_live_client(&probe_settings)
        .unwrap_or_else(|e| panic!("LLM クライアント構築に失敗: {e}"));
    let model = probe.inner().model().to_string();
    let endpoint = probe.inner().endpoint().to_string();
    drop(probe);
    let llm = || record::llm_block(&model, &endpoint, args.temperature);

    // 親 run: グリッド定義そのものを parameters に持つ．個別条件の指標は書かない．
    // 親は 1 本のシミュレーションではないので master_seed を名乗らず，base seed は
    // /parameters.seed と seed_pointers 経由で execution_hash に残る．
    let parent = Run::start(
        RunOptions::new(EXPERIMENT, "sweep")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&sweep_parameters)
            .expect("runvault: sweep の parameters の組み立てに失敗")
            .seed_pointers(["/seed"])
            .sweep_parent()
            .llm(llm())
            .replication(record::replication()),
    )
    .expect("runvault: sweep 親 run の開始に失敗");

    let sweep_id = parent
        .sweep_id()
        .expect("runvault: sweep 親に sweep_id がありません")
        .to_string();
    let parent_run_uid = parent.run_uid().to_string();

    println!("=== Chuang et al. (2024) LLM 意見力学 パラメータスイープ ===");
    println!(
        "N: {} | bias: {} 種 | framing: {} 種 | topology: {} 種 | 試行: {} | 合計: {} 実行",
        args.n_agents,
        biases.len(),
        framings.len(),
        topologies.len(),
        args.runs,
        n_total,
    );
    println!("シード (base): {} | model: {}", args.seed, model);
    println!("出力先: {}", parent.dir().display());
    println!("-----------------------------------------------------------");

    let mut done = 0usize;
    // 条件ごとの平均 Diversity D / Bias B (単調増大の確認用; 表示のためだけに持つ)．
    let mut per_bias: Vec<(ConfirmationBias, Vec<f64>, Vec<f64>)> = biases
        .iter()
        .map(|&b| (b, Vec::new(), Vec::new()))
        .collect();

    for &bias in &biases {
        for &framing in &framings {
            for &topology in &topologies {
                let params = SweepPointParameters {
                    bias: bias.label(),
                    framing: framing.label(),
                    topology: topology.label(),
                    memory: memory_mode.label(),
                    topic: args.topic.clone(),
                    n_agents: args.n_agents,
                    runs: args.runs,
                    events_per_step: args.events_per_step,
                    max_steps: args.max_steps,
                    tol: args.tol,
                    seed: args.seed,
                    llm_temperature: args.temperature,
                    llm_seed: args.llm_seed,
                };

                // 子は «その条件の試行群» そのもの．master_seed は親と同じ base で，
                // 条件が違えば config_hash が違うので run としては別物になる．
                // 同じ条件の繰り返しは無いので replicate_index は 0．
                let mut child = Run::start(
                    RunOptions::new(EXPERIMENT, "sweep-point")
                        .repo_id(REPO_ID)
                        .domain(DOMAIN)
                        .results_root(&args.output_dir)
                        .parameters(&params)
                        .expect("runvault: 子 run の parameters の組み立てに失敗")
                        .seed_pointers(["/seed"])
                        .master_seed(args.seed)
                        .replicate_index(0)
                        .llm(llm())
                        .lineage(Lineage {
                            sweep_id: Some(sweep_id.clone()),
                            parent_run_uid: Some(parent_run_uid.clone()),
                            ..Default::default()
                        })
                        .replication(record::replication()),
                )
                .expect("runvault: 子 run の開始に失敗");

                let mut trials: Vec<record::TrialOutcome> = Vec::with_capacity(args.runs);
                for run_idx in 0..args.runs {
                    // 各条件に独立なシードを派生 (explicit identity)．
                    let seed = record::sweep_trial_seed(
                        args.seed,
                        bias.label(),
                        framing.label(),
                        topology.label(),
                        run_idx,
                    );

                    let cfg = Config {
                        n_agents: args.n_agents,
                        topic: args.topic.clone(),
                        framing,
                        bias,
                        memory_mode,
                        interact: true,
                        topology,
                        er_p: 0.3,
                        ws_k: 4,
                        ws_beta: 0.1,
                        ba_m: 2,
                        events_per_step: args.events_per_step,
                        max_steps: args.max_steps,
                        tol: args.tol,
                        seed: Some(seed),
                        llm: LlmSettings {
                            temperature: args.temperature,
                            seed: args.llm_seed,
                            cache_path: Some(args.cache_path.clone()),
                        },
                    };

                    let client = build_live_client(&cfg.llm)
                        .unwrap_or_else(|e| panic!("LLM クライアント構築に失敗: {e}"));
                    let result = run_with_client(&cfg, client)
                        .unwrap_or_else(|e| panic!("実行に失敗: {}", e));

                    // sweep が見るのは各試行の最終ステップだけなので，観測時刻もそこ 1 点．
                    record::log_terminal(
                        &mut child,
                        &format!("trial-{run_idx}"),
                        seed,
                        args.max_steps,
                        args.tol,
                        [result.final_step as u64],
                        &result,
                    );
                    let last = result.metrics_history.last().unwrap();
                    if let Some(slot) = per_bias.iter_mut().find(|(b, _, _)| *b == bias) {
                        slot.1.push(last.diversity);
                        slot.2.push(last.bias);
                    }
                    trials.push(record::TrialOutcome::from_result(&result));

                    done += 1;
                }
                record::log_condition_summary(&mut child, &trials);
                child.finish().expect("runvault: 子 run の完了に失敗");

                println!(
                    "[{}/{}] bias={} framing={} topology={} 完了 ({} 試行)",
                    done,
                    n_total,
                    bias.label(),
                    framing.label(),
                    topology.label(),
                    args.runs,
                );
            }
        }
    }

    let parent_dir = parent
        .finish()
        .expect("runvault: sweep 親 run の完了に失敗");

    // 確証バイアスごとの平均 Diversity D を表示する (単調増大の確認用)．
    println!("===========================================================");
    println!("スイープ完了: {} 実行", n_total);
    println!("-----------------------------------------------------------");
    println!("確証バイアス別の平均 Diversity D (単調増大が論文 Table 1 の知見):");
    for (bias, ds, bs) in &per_bias {
        if ds.is_empty() {
            continue;
        }
        let avg_d = ds.iter().sum::<f64>() / ds.len() as f64;
        let avg_b = bs.iter().sum::<f64>() / bs.len() as f64;
        println!(
            "  {:<7} → D̄ = {:.3} | B̄ = {:.3}",
            bias.label(),
            avg_d,
            avg_b
        );
    }
    println!("-----------------------------------------------------------");
    println!("親 run → {}", parent_dir.display());
    println!("条件ごとの子 run は subcommand=sweep-point (試行は events.jsonl)．");
}

// ---------------------------------------------------------------------------
// reproduce
// ---------------------------------------------------------------------------

/// 1 条件 (bias × control × topology) を `runs` 回回した集計セル．
struct ReproCell {
    /// 条件ラベル (指標名の接頭辞になる slug)．
    label: String,
    /// 試行平均の最終 Bias B (意見平均)．
    mean_final_bias: f64,
    /// 試行平均の最終 Diversity D (意見分布の標準偏差)．
    mean_final_diversity: f64,
    /// 試行平均の最終クラスタ数．
    mean_final_clusters: f64,
    /// 試行平均の «D の縮小幅» (初期 D − 最終 D; 正なら合意方向)．
    mean_diversity_drop: f64,
    /// 試行平均の収束ステップ (収束しなければ max_steps)．
    mean_final_step: f64,
}

/// 1 条件を `runs` 回実行し，集計セルを作って run へ記録する．
///
/// 代表 (run 0) のステップごとの指標は `<label>_<指標名>` として `metrics.csv` に
/// 書く．旧 `metrics_<label>.csv` に相当するが，(step, scope, name) が主キーなので
/// 条件ラベルを名前に畳み込まないと 9 条件が衝突する．
#[allow(clippy::too_many_arguments)]
fn run_repro_cell(
    rv: &mut Run,
    label: &str,
    bias: ConfirmationBias,
    interact: bool,
    topology: Topology,
    base: &Config,
    runs: usize,
    root_seed: u64,
    mock: bool,
) -> ReproCell {
    let mut final_bias = 0.0;
    let mut final_div = 0.0;
    let mut final_clusters = 0.0;
    let mut div_drop = 0.0;
    let mut final_step = 0.0;

    for run_idx in 0..runs {
        let seed =
            record::repro_trial_seed(root_seed, bias.label(), interact, topology.label(), run_idx);
        let cfg = Config {
            bias,
            interact,
            topology,
            seed: Some(seed),
            ..base.clone()
        };
        let (cfg, client, _, _) = build_client(&cfg, mock);
        let result: SimulationResult =
            run_with_client(&cfg, client).unwrap_or_else(|e| panic!("実行に失敗 ({label}): {e}"));

        let first = result.metrics_history.first().unwrap();
        let last = result.metrics_history.last().unwrap();
        final_bias += last.bias;
        final_div += last.diversity;
        final_clusters += last.n_clusters as f64;
        div_drop += first.diversity - last.diversity;
        final_step += result.final_step as f64;

        if run_idx == 0 {
            for m in &result.metrics_history {
                record::log_step(rv, Some(label), m);
            }
        }
    }

    let n = runs.max(1) as f64;
    ReproCell {
        label: label.to_string(),
        mean_final_bias: final_bias / n,
        mean_final_diversity: final_div / n,
        mean_final_clusters: final_clusters / n,
        mean_diversity_drop: div_drop / n,
        mean_final_step: final_step / n,
    }
}

/// セルの集約値を run スコープの指標として書く．
fn log_repro_cell(rv: &mut Run, cell: &ReproCell) {
    let name = |base: &str| format!("{}_{}", cell.label, base);
    rv.log_metrics(
        "run",
        &[
            (name("mean_final_bias").as_str(), cell.mean_final_bias),
            (
                name("mean_final_diversity").as_str(),
                cell.mean_final_diversity,
            ),
            (
                name("mean_final_clusters").as_str(),
                cell.mean_final_clusters,
            ),
            (
                name("mean_diversity_drop").as_str(),
                cell.mean_diversity_drop,
            ),
            (name("mean_final_step").as_str(), cell.mean_final_step),
        ],
    )
    .expect("セル集約の記録に失敗");
}

/// 観測値と論文の定性的知見を突き合わせた 1 アンカー．
struct ReproAnchor {
    /// 指標名になる slug．
    id: String,
    /// 人間向けの説明 (どの差を見ているか)．
    label: String,
    /// 論文側の定性的な主張．
    paper: String,
    observed: f64,
    target_lo: f64,
    /// 上限．`None` は «上限なし»．
    target_hi: Option<f64>,
    pass: bool,
}

fn cmd_reproduce(args: ReproduceArgs) {
    let framing = parse_framing(&args.framing).unwrap_or_else(|e| panic!("{}", e));
    let topologies: Vec<Topology> = split_csv(&args.topology_values)
        .iter()
        .map(|s| parse_topology(s).unwrap_or_else(|e| panic!("{}", e)))
        .collect();

    // quick モードは軽量化 (動作確認用; 論文値検証には使わない)．
    let n_agents = if args.quick { 8 } else { args.n_agents };
    let runs = if args.quick { 2 } else { args.runs };
    let max_steps = if args.quick { 20 } else { args.max_steps };

    if !args.mock {
        ensure_cache_dir(&args.cache_path);
    }

    // 基準設定 (全条件で共通; bias/interact/topology/seed のみ条件ごとに差替)．
    let base = Config {
        n_agents,
        topic: args.topic.clone(),
        framing,
        bias: ConfirmationBias::None,
        memory_mode: MemoryMode::Cumulative,
        interact: true,
        topology: Topology::Full,
        er_p: 0.3,
        ws_k: 4,
        ws_beta: 0.1,
        ba_m: 2,
        events_per_step: args.events_per_step,
        max_steps,
        // 収束で早期停止しないよう厳しめ (各条件を同じ T まで回して比較する)．
        tol: 1e-12,
        seed: Some(args.seed),
        llm: LlmSettings {
            temperature: args.temperature,
            seed: args.llm_seed,
            cache_path: if args.mock {
                None
            } else {
                Some(args.cache_path.clone())
            },
        },
    };

    // llm ブロックのためにモデル名を先に取る (記録は Run::start で確定する)．
    let (_, probe, model, endpoint) = build_client(&base, args.mock);
    drop(probe);

    let parameters = ReproduceParameters {
        n_agents,
        topic: args.topic.clone(),
        framing: framing.label(),
        topology_values: topologies.iter().map(|t| t.label().to_string()).collect(),
        runs,
        events_per_step: args.events_per_step,
        max_steps,
        tol: base.tol,
        seed: args.seed,
        llm_temperature: args.temperature,
        llm_seed: args.llm_seed,
        mock: args.mock,
        quick: args.quick,
    };

    let mut rv = Run::start(
        RunOptions::new(EXPERIMENT, "reproduce")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&parameters)
            .expect("runvault: parameters の組み立てに失敗")
            .seed_pointers(["/seed"])
            .master_seed(args.seed)
            .llm(record::llm_block(&model, &endpoint, args.temperature))
            .replication(record::replication()),
    )
    .expect("runvault: reproduce run の開始に失敗");

    println!("=== Chuang et al. (2024) 見出し的知見 一括再現 ===");
    println!(
        "N: {} | topic: {} | framing: {} | runs: {} | T: {} | mode: {}",
        n_agents,
        args.topic,
        framing.label(),
        runs,
        max_steps,
        if args.mock { "MOCK" } else { "LIVE" },
    );
    println!("出力先: {}", rv.dir().display());
    println!("-------------------------------------------------");

    // --- (1) bias × control 行列 (full topology) ---
    // 論文 4.3: バイアス無し→真値方向の合意 / バイアス強→断片化．non-interaction
    // 統制は «社会的影響» を切り，LLM 自身の prior ドリフトを分離する．
    let biases = [
        ConfirmationBias::None,
        ConfirmationBias::Weak,
        ConfirmationBias::Strong,
    ];
    let mut bias_cells: Vec<ReproCell> = Vec::new();
    for &b in &biases {
        for &interact in &[true, false] {
            let arm = if interact { "interact" } else { "control" };
            let label = format!("bias-{}_{}", b.label(), arm);
            let cell = run_repro_cell(
                &mut rv,
                &label,
                b,
                interact,
                Topology::Full,
                &base,
                runs,
                args.seed,
                args.mock,
            );
            log_repro_cell(&mut rv, &cell);
            bias_cells.push(cell);
        }
    }

    // --- (2) topology 比較 (bias=none, interaction) ---
    let mut topo_cells: Vec<ReproCell> = Vec::new();
    for &topo in &topologies {
        let label = format!("topo-{}", topo.label());
        let cell = run_repro_cell(
            &mut rv,
            &label,
            ConfirmationBias::None,
            true,
            topo,
            &base,
            runs,
            args.seed,
            args.mock,
        );
        log_repro_cell(&mut rv, &cell);
        topo_cells.push(cell);
    }

    // --- アンカー評価 (論文の定性的知見) ---
    fn cell<'a>(cells: &'a [ReproCell], label: &str) -> &'a ReproCell {
        cells
            .iter()
            .find(|c| c.label == label)
            .unwrap_or_else(|| panic!("セル {label} が見つかりません"))
    }
    let none_i = cell(&bias_cells, "bias-none_interact");
    let weak_i = cell(&bias_cells, "bias-weak_interact");
    let strong_i = cell(&bias_cells, "bias-strong_interact");
    let none_c = cell(&bias_cells, "bias-none_control");
    let strong_c = cell(&bias_cells, "bias-strong_control");

    let mut anchors: Vec<ReproAnchor> = Vec::new();
    let mut push = |id: &str, label: &str, paper: &str, obs: f64, lo: f64, hi: Option<f64>| {
        anchors.push(ReproAnchor {
            id: id.to_string(),
            label: label.to_string(),
            paper: paper.to_string(),
            observed: obs,
            target_lo: lo,
            target_hi: hi,
            pass: obs >= lo && hi.map(|h| obs <= h).unwrap_or(true),
        });
    };

    // H1: バイアス無しの相互作用は合意へ向かう (D が縮小; drop>0)．
    push(
        "consensus_drift_no_bias",
        "consensus_drift_no_bias (D drop > 0)",
        "consensus",
        none_i.mean_diversity_drop,
        0.0,
        None,
    );
    // H2: 確証バイアスで Diversity D が単調増大 (none ≤ weak ≤ strong)．
    push(
        "diversity_monotone_weak_ge_none",
        "diversity_monotone_weak>=none",
        "D(weak)>=D(none)",
        weak_i.mean_final_diversity - none_i.mean_final_diversity,
        -1e-9,
        None,
    );
    push(
        "diversity_monotone_strong_ge_weak",
        "diversity_monotone_strong>=weak",
        "D(strong)>=D(weak)",
        strong_i.mean_final_diversity - weak_i.mean_final_diversity,
        -1e-9,
        None,
    );
    // H3 (Wave3): 非相互作用統制では強バイアス下で合意が起きない (D が温存)．
    //   interaction(none) は合意 (低 D)，control(strong) は高 D を保つ → 差 > 0．
    push(
        "interaction_drives_consensus",
        "interaction_drives_consensus (D_control(strong) - D_interact(none) > 0)",
        "social influence matters",
        strong_c.mean_final_diversity - none_i.mean_final_diversity,
        0.0,
        None,
    );
    // H3b: 非相互作用統制の «固有ドリフト» は相互作用より小さい (社会的影響の寄与)．
    //   D drop: interaction(none) ≥ control(none)．
    push(
        "social_amplifies_drift",
        "social_amplifies_drift (drop_interact(none) - drop_control(none) >= 0)",
        "interaction >= isolation",
        none_i.mean_diversity_drop - none_c.mean_diversity_drop,
        -1e-9,
        None,
    );

    let n_pass = anchors.iter().filter(|a| a.pass).count();

    // 観測量は数なので指標に，帯と PASS/off はカテゴリなのでイベントに書く．
    let observations: Vec<(String, f64)> =
        anchors.iter().map(|a| (a.id.clone(), a.observed)).collect();
    record::log_anchor_observations(&mut rv, &observations, n_pass);
    for a in &anchors {
        rv.log_event(
            "x.chuang2024.anchor",
            &record::AnchorEvent {
                id: &a.id,
                label: &a.label,
                paper: &a.paper,
                observed: a.observed,
                target_lo: a.target_lo,
                target_hi: a.target_hi,
                pass: a.pass,
            },
        )
        .expect("アンカーイベントの記録に失敗");
    }

    // --- コンソール出力 ---
    println!("--- bias × control 行列 (full topology) ---");
    println!(
        "{:<24} {:>8} {:>8} {:>8} {:>10}",
        "condition", "B̄", "D̄", "clust", "D-drop"
    );
    for c in &bias_cells {
        println!(
            "{:<24} {:>8.3} {:>8.3} {:>8.2} {:>10.3}",
            c.label,
            c.mean_final_bias,
            c.mean_final_diversity,
            c.mean_final_clusters,
            c.mean_diversity_drop,
        );
    }
    println!("--- topology 比較 (bias=none, interaction) ---");
    for c in &topo_cells {
        println!(
            "{:<24} {:>8.3} {:>8.3} {:>8.2} {:>10.3}",
            c.label,
            c.mean_final_bias,
            c.mean_final_diversity,
            c.mean_final_clusters,
            c.mean_diversity_drop,
        );
    }
    println!("--- 論文知見アンカー ---");
    for a in &anchors {
        let hi = match a.target_hi {
            Some(h) => format!("{:.3}", h),
            None => "∞".to_string(),
        };
        println!(
            "[{}] {:<52} obs={:.4} target=[{:.3},{}]",
            if a.pass { "PASS" } else { "OFF " },
            a.label,
            a.observed,
            a.target_lo,
            hi,
        );
    }
    println!("-------------------------------------------------");
    println!("{}/{} アンカーが in-band", n_pass, anchors.len());

    let dir = rv.finish().expect("runvault: reproduce run の完了に失敗");
    println!("セル集約・アンカー観測量 → {}/metrics.csv", dir.display());
    println!("アンカー判定             → {}/events.jsonl", dir.display());
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    if let Some(host) = cli.ollama_host.as_deref() {
        std::env::set_var("OLLAMA_HOST", host);
    }
    match cli.command {
        Commands::Run(args) => cmd_run(args),
        Commands::Sweep(args) => cmd_sweep(args),
        Commands::Reproduce(args) => cmd_reproduce(args),
    }
}
