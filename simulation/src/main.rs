//! Chuang et al. (2024) "Simulating Opinion Dynamics with Networks of LLM-based
//! Agents" — 再現実験の CLI エントリポイント．
//!
//! `run`   : 単一設定で dyadic LLM 意見力学を実行する．
//! `sweep` : 確証バイアス × フレーミング × トポロジ (× メモリ方式) を走査し，
//!           最終 B / D / 分極などを `sweep_summary.csv` に集計する．
//!
//! Phase 3 の `reproduce` (Table 1 / Fig.4-6 一括再現) は未実装 (拡張点)．

use std::fs;
use std::path::Path;

use clap::{Parser, Subcommand};
use socsim_results::{refresh_latest_symlink, timestamp, write_csv, write_json};

use chuang_opinion_simulation::config::{
    parse_bias, parse_framing, parse_memory, parse_topology, Config, ConfirmationBias, Framing,
    LlmSettings, MemoryMode, Topology,
};
use chuang_opinion_simulation::metrics::convergence_time;
use chuang_opinion_simulation::simulation::{
    ensure_output_dir, run, save_metrics, save_opinions, save_run_metadata,
};

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
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// 単一設定で dyadic LLM 意見力学を実行する．
    Run(RunArgs),
    /// 確証バイアス × フレーミング × トポロジを走査し，最終 B/D を集計する．
    Sweep(SweepArgs),
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

    /// トポロジ (full / ws / ba)．
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

// ---------------------------------------------------------------------------
// 補助
// ---------------------------------------------------------------------------

/// `sweep_summary.csv` の 1 行．
#[derive(serde::Serialize)]
struct SweepRow {
    bias: String,
    framing: String,
    topology: String,
    memory: String,
    run: usize,
    seed: u64,
    converged: bool,
    convergence_time: i64,
    final_step: usize,
    final_bias: f64,
    final_diversity: f64,
    final_variance: f64,
    n_clusters: usize,
    polarization: f64,
    cache_hit_rate: f64,
}

/// `sweep_config.json` の構造体．
#[derive(serde::Serialize)]
struct SweepConfigJson {
    command: &'static str,
    bias_values: Vec<String>,
    framing_values: Vec<String>,
    topology_values: Vec<String>,
    memory: String,
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

/// 派生シードのラベルに使う文字列ハッシュ (explicit identity)．
fn label_hash(label: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in label.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// カンマ区切り文字列を trim 済みの非空リストへ．
fn split_csv(s: &str) -> Vec<String> {
    s.split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

fn cmd_run(args: RunArgs) {
    let framing = parse_framing(&args.framing).unwrap_or_else(|e| panic!("{}", e));
    let bias = parse_bias(&args.bias).unwrap_or_else(|e| panic!("{}", e));
    let memory_mode = parse_memory(&args.memory).unwrap_or_else(|e| panic!("{}", e));
    let topology = parse_topology(&args.topology).unwrap_or_else(|e| panic!("{}", e));

    let timestamp = timestamp();
    let output_dir = format!("{}/{}", args.output_dir, timestamp);

    let cfg = Config {
        n_agents: args.n_agents,
        topic: args.topic.clone(),
        framing,
        bias,
        memory_mode,
        interact: true,
        topology,
        ws_k: args.ws_k,
        ws_beta: args.ws_beta,
        ba_m: args.ba_m,
        events_per_step: args.events_per_step,
        max_steps: args.max_steps,
        tol: args.tol,
        seed: args.seed,
        llm: LlmSettings {
            temperature: args.temperature,
            seed: args.llm_seed,
            cache_path: Some(args.cache_path.clone()),
        },
        output_dir: output_dir.clone(),
    };

    // キャッシュ用ディレクトリを用意する．
    if let Some(parent) = Path::new(&args.cache_path).parent() {
        let _ = fs::create_dir_all(parent);
    }
    ensure_output_dir(&cfg.output_dir);

    println!("=== Chuang et al. (2024) LLM 意見力学 再現実験 ===");
    println!(
        "N: {} | topic: {} | framing: {} | bias: {} | memory: {} | topology: {}",
        cfg.n_agents,
        cfg.topic,
        cfg.framing.label(),
        cfg.bias.label(),
        cfg.memory_mode.label(),
        cfg.topology.label(),
    );
    println!(
        "events/step: {} | max_steps: {} | tol: {} | seed: {:?}",
        cfg.events_per_step, cfg.max_steps, cfg.tol, cfg.seed
    );
    println!(
        "LLM: temp={} llm_seed={} cache={}",
        cfg.llm.temperature, cfg.llm.seed, args.cache_path
    );
    println!("出力先: {}", cfg.output_dir);
    println!("-------------------------------------------------");

    let result = run(&cfg).unwrap_or_else(|e| panic!("実行に失敗: {}", e));

    save_metrics(&result.metrics_history, &cfg.output_dir);
    save_opinions(&result, &cfg.output_dir);
    save_run_metadata(&result, &cfg, &cfg.output_dir);

    // config.json (pretty-print JSON; socsim_results::write_json に委譲)．
    {
        let path = format!("{}/config.json", cfg.output_dir);
        write_json(&cfg.to_run_config_json(), &path).expect("config.json の書き込みに失敗");
    }

    // latest シンボリックリンクを再作成する (best-effort; 従来同様エラーは無視)．
    let _ = refresh_latest_symlink(&args.output_dir, &timestamp);

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
    println!("意見軌跡   → {}/opinions.csv", cfg.output_dir);
    println!("メトリクス → {}/metrics.csv", cfg.output_dir);
    println!("LLM メタ   → {}/run_metadata.json", cfg.output_dir);
    println!("設定       → {}/config.json", cfg.output_dir);
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

    let timestamp = timestamp();
    let sweep_dir = format!("{}/{}_sweep", args.output_dir, timestamp);
    fs::create_dir_all(&sweep_dir).expect("sweep ディレクトリの作成に失敗");
    if let Some(parent) = Path::new(&args.cache_path).parent() {
        let _ = fs::create_dir_all(parent);
    }

    let n_total = biases.len() * framings.len() * topologies.len() * args.runs;

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
    println!("出力先: {}", sweep_dir);
    println!("-----------------------------------------------------------");

    let mut summary_rows: Vec<SweepRow> = Vec::with_capacity(n_total);
    let mut done = 0usize;

    for &bias in &biases {
        for &framing in &framings {
            for &topology in &topologies {
                for run_idx in 0..args.runs {
                    // 各条件に独立なシードを派生 (explicit identity)．
                    let seed = socsim_core::derive_seed(
                        args.seed,
                        &[
                            label_hash(bias.label()),
                            label_hash(framing.label()),
                            label_hash(topology.label()),
                            run_idx as u64,
                        ],
                    );

                    let cfg = Config {
                        n_agents: args.n_agents,
                        topic: args.topic.clone(),
                        framing,
                        bias,
                        memory_mode,
                        interact: true,
                        topology,
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
                        output_dir: sweep_dir.clone(),
                    };

                    let result = run(&cfg).unwrap_or_else(|e| panic!("実行に失敗: {}", e));
                    let last = result.metrics_history.last().unwrap();
                    let variances: Vec<f64> =
                        result.metrics_history.iter().map(|m| m.variance).collect();
                    let conv_t = convergence_time(&variances, cfg.tol)
                        .map(|t| t as i64)
                        .unwrap_or(-1);

                    summary_rows.push(SweepRow {
                        bias: bias.label().to_string(),
                        framing: framing.label().to_string(),
                        topology: topology.label().to_string(),
                        memory: memory_mode.label().to_string(),
                        run: run_idx,
                        seed,
                        converged: result.converged,
                        convergence_time: conv_t,
                        final_step: result.final_step,
                        final_bias: last.bias,
                        final_diversity: last.diversity,
                        final_variance: last.variance,
                        n_clusters: last.n_clusters,
                        polarization: last.polarization,
                        cache_hit_rate: result.metadata.cache_hit_rate(),
                    });

                    done += 1;
                }
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

    // sweep_summary.csv (各行を serialize; socsim_results::write_csv に委譲)．
    {
        let path = format!("{}/sweep_summary.csv", sweep_dir);
        write_csv(&summary_rows, &path).expect("sweep_summary.csv の書き込みに失敗");
    }

    // sweep_config.json
    {
        let config_json = SweepConfigJson {
            command: "sweep",
            bias_values: split_csv(&args.bias_values),
            framing_values: split_csv(&args.framing_values),
            topology_values: split_csv(&args.topology_values),
            memory: memory_mode.label().to_string(),
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
        let path = format!("{}/sweep_config.json", sweep_dir);
        write_json(&config_json, &path).expect("sweep_config.json の書き込みに失敗");
    }

    let _ = refresh_latest_symlink(&args.output_dir, &format!("{}_sweep", timestamp));

    // 確証バイアスごとの平均 Diversity D を表示する (単調増大の確認用)．
    println!("===========================================================");
    println!("スイープ完了: {} 実行", n_total);
    println!("-----------------------------------------------------------");
    println!("確証バイアス別の平均 Diversity D (単調増大が論文 Table 1 の知見):");
    for &bias in &biases {
        let rows: Vec<&SweepRow> = summary_rows
            .iter()
            .filter(|r| r.bias == bias.label())
            .collect();
        if rows.is_empty() {
            continue;
        }
        let avg_d = rows.iter().map(|r| r.final_diversity).sum::<f64>() / rows.len() as f64;
        let avg_b = rows.iter().map(|r| r.final_bias).sum::<f64>() / rows.len() as f64;
        println!(
            "  {:<7} → D̄ = {:.3} | B̄ = {:.3}",
            bias.label(),
            avg_d,
            avg_b
        );
    }
    println!("-----------------------------------------------------------");
    println!("サマリ → {}/sweep_summary.csv", sweep_dir);
    println!("設定   → {}/sweep_config.json", sweep_dir);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run(args) => cmd_run(args),
        Commands::Sweep(args) => cmd_sweep(args),
    }
}
