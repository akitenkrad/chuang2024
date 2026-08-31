//! runvault への記録の共通部分．
//!
//! 論文メタデータ (research) は `run` / `sweep` / `reproduce` のどのサブコマンドでも
//! 同一なので，ここ 1 箇所で組み立てる．ステップごとの指標の落とし方，シミュレーション
//! 1 本の終端行，条件 1 点ぶんの集約もここに集める．

use runvault::{Llm, Replication, Run, Target, Work};
use serde::Serialize;

use crate::metrics::Metrics;
use crate::simulation::SimulationResult;

/// runvault 上の実験名．`runvault path --experiment` に渡す値でもある．
/// バイナリ名 (`chuang`) と揃える．
pub const EXPERIMENT: &str = "chuang";
/// リポジトリの安定 id．git remote の名前とは独立に固定する．
pub const REPO_ID: &str = "chuang2024";
/// 分野．初期意見と話者-聴者ペアの抽選に RNG を使うので `simulation`
/// (= `master_seed` が必須)．
///
/// LLM で駆動されるモデルだが `llm-safety` ではない — 測っているのはモデルの
/// 安全性ではなく，網の上の意見分布の時間発展だからである．LLM 側の同一性は
/// `llm` ブロック ([`llm_block`]) が持つ．
pub const DOMAIN: &str = "simulation";

/// 時間軸の単位．
///
/// 本モデルの刻みは «`events_per_step` 回の dyadic 相互作用» を 1 つにまとめた
/// 離散ステップ $t = 0, 1, 2, \dots$ で，論文 4.3 の表記そのもの．runvault の
/// 語彙では `step`．
const T_UNIT: &str = "step";

/// 指標の粒度．意見分布の特徴量はどれも母集団全体の集約なので `run`．
const SCOPE: &str = "run";

/// この再現実験が対象としている論文．
///
/// 論文は特定の図表ではなく «LLM エージェントは真値方向の合意へ強く引かれ，確証
/// バイアスを入れるとそれが崩れて断片化する» という主張の再現を狙うので，
/// `Target::claim` を使う．
pub fn replication() -> Replication {
    Work::arxiv("2311.09618")
        .title("Simulating Opinion Dynamics with Networks of LLM-based Agents")
        .year(2024)
        .source_version("naacl-findings-2024")
        .target(Target::claim(
            "confirmation-bias-breaks-consensus",
            "LLM agents drift toward truthful consensus, and injecting a confirmation bias fragments opinions instead",
        ))
        .obsidian_note("研究/98_論文レポート/80-再現実験/実装完了/chuang2024/設計書.md")
}

// ---------------------------------------------------------------------------
// LLM ブロック
// ---------------------------------------------------------------------------

/// 実際に応答したバックエンドを `llm` ブロックに落とす．
///
/// `model` / `endpoint` はクライアントが名乗った値をそのまま使う．`provider` は
/// runvault の語彙ではなく自由記述なので，endpoint から «どのゲートウェイが答えたか»
/// を決める (`mock://…` はオフラインの scripted クライアント，それ以外はホスト名で
/// Ollama / OpenAI を分ける)．推測しているのは分類だけで，値そのものは記録から採る．
///
/// `model_snapshot` に入るのは `llama3.2:latest` のような動くエイリアスであることが
/// 多い．socsim-llm はスナップショット id を持たないので，持っていない値を作らずに
/// 名乗られた名前を書く．
pub fn llm_block(model: &str, endpoint: &str, temperature: f32) -> Llm {
    let provider = if endpoint.starts_with("mock://") {
        "mock"
    } else if endpoint.contains("openai") {
        "openai"
    } else {
        "ollama"
    };
    Llm {
        provider: provider.to_string(),
        model_snapshot: model.to_string(),
        temperature: Some(temperature as f64),
        // 話者・聴者プロンプトはエージェントと履歴から毎回組み立てられ，固定の
        // system prompt を持たない．無いものを hash しない．
        system_prompt_hash: None,
    }
}

// ---------------------------------------------------------------------------
// ステップごとの指標
// ---------------------------------------------------------------------------

/// シミュレーション 1 本ぶんの記録 (`run` サブコマンド用)．
///
/// ステップごとの 5 指標 (`t` は時間軸なので値としては書かない) と，run 全体を
/// 1 つの値で表す `converged` / `final_step` / LLM 呼び出しの内訳を書く．
/// 実行時間は `status.json` の `duration_sec` が正本なので指標にはしない．
pub fn log_simulation(run: &mut Run, result: &SimulationResult) {
    for m in &result.metrics_history {
        log_step(run, None, m);
    }
    run.log_metrics(
        SCOPE,
        &[
            ("converged", if result.converged { 1.0 } else { 0.0 }),
            ("final_step", result.final_step as f64),
            ("llm_calls", result.metadata.total() as f64),
            ("llm_cache_hits", result.metadata.cache_hits() as f64),
            ("llm_cache_hit_rate", result.metadata.cache_hit_rate()),
        ],
    )
    .expect("run スコープの指標の記録に失敗");
}

/// [`Metrics`] の数値フィールドを 1 ステップぶんまとめて書く．
///
/// 名前は旧 `metrics.csv` の列名のまま (`variance` / `bias` / `diversity` /
/// `n_clusters` / `polarization`) にしてある．wide から long へ形は変わるが，
/// «移行で数が変わっていないか» を列名の対応表なしに突き合わせられる．
///
/// `prefix` は 1 つの run に複数の条件が同居するとき (`reproduce`) に付ける．
/// `(step, scope, name)` が主キーなので，接頭辞が無いと条件どうしで衝突する．
pub fn log_step(run: &mut Run, prefix: Option<&str>, m: &Metrics) {
    let name = |base: &str| match prefix {
        Some(p) => format!("{p}_{base}"),
        None => base.to_string(),
    };
    run.log_metrics_at(
        m.t as u64,
        T_UNIT,
        SCOPE,
        &[
            (name("variance").as_str(), m.variance),
            (name("bias").as_str(), m.bias),
            (name("diversity").as_str(), m.diversity),
            (name("n_clusters").as_str(), m.n_clusters as f64),
            (name("polarization").as_str(), m.polarization),
        ],
    )
    .unwrap_or_else(|e| panic!("step {} の指標の記録に失敗: {e}", m.t));
}

// ---------------------------------------------------------------------------
// 終端イベント
// ---------------------------------------------------------------------------

/// `events.jsonl` に書く観測行．
///
/// 予約キーだけを持つ．数はここには書かない — ステップごとの値は `metrics.csv`
/// が，試行の最終値は下の [`TerminalEvent`] が正本なので，同じ数を 2 箇所に置くと
/// 食い違う余地ができる．この行が持つのは «その単位をいつ見たか» だけである．
///
/// `terminal` 行だけでも生存時間解析は組めるが (`schema/v1/event.json` の terminal
/// の注記)，`runvault verify --deep` は terminal の `unit_id` が observation にも
/// 現れることを要求するので，観測した時刻を明示的に残す．
#[derive(Serialize)]
struct ObservationEvent<'a> {
    unit_id: &'a str,
    t: u64,
    t_unit: &'static str,
}

/// 観測 1 点を書く．
fn log_observation(run: &mut Run, unit_id: &str, t: u64) {
    run.log_event(
        "observation",
        &ObservationEvent {
            unit_id,
            t,
            t_unit: T_UNIT,
        },
    )
    .unwrap_or_else(|e| panic!("{unit_id} の t={t} の observation の記録に失敗: {e}"));
}

/// `events.jsonl` に書く終端行．
///
/// 先頭 6 フィールドは runvault の予約語 (`terminal` はこれを全部要求する)．
/// 残りは自由欄で，旧 `sweep_summary.csv` の 1 行に対応する．
///
/// 派生シードを `seed` ではなく `trial_seed` と呼ぶのは，`runvault.read` の
/// `sweep_events_table` が «条件の parameters» を同名のイベント列へ上書きするため
/// である．子 run の `parameters` は base seed を `seed` として持つので，イベント側も
/// `seed` にすると試行ごとの派生シードが黙って base seed に化ける．
#[derive(Serialize)]
struct TerminalEvent<'a> {
    unit_id: &'a str,
    t: u64,
    t_unit: &'static str,
    outcome: &'static str,
    censored: bool,
    budget: u64,
    trial_seed: u64,
    /// 意見分散が `tol` を初めて下回ったステップ．到達しなければ **列ごと落とす**
    /// (旧 `sweep_summary.csv` は -1 を書いていたが，欠測を数で埋めない)．
    #[serde(skip_serializing_if = "Option::is_none")]
    convergence_time: Option<u64>,
    final_bias: f64,
    final_diversity: f64,
    final_variance: f64,
    n_clusters: usize,
    polarization: f64,
    cache_hit_rate: f64,
}

/// シミュレーション 1 本を `terminal` イベントとして書く．
///
/// 打ち切り (`censored`) の行は `t == budget` でなければならない．ドライバは意見
/// 分散が `tol` を下回れば停止し，止まらなければ `max_steps` まで回すので，収束
/// しなかった run は必ず上限に達している．この不変条件は runvault が `log_event`
/// の書き込み時に検査するので，ここでは二重に持たない．
///
/// `observed` はこの単位を観測した時刻の列で，終端の `t` を必ず含む．`run` は全
/// ステップを `metrics.csv` に残すので全ステップを，`sweep` は各試行の最終ステップ
/// しか見ないのでその 1 点だけを渡す．
pub fn log_terminal(
    run: &mut Run,
    unit_id: &str,
    trial_seed: u64,
    max_steps: usize,
    tol: f64,
    observed: impl IntoIterator<Item = u64>,
    result: &SimulationResult,
) {
    let last = result
        .metrics_history
        .last()
        .expect("metrics_history は t=0 を含む");

    for t in observed {
        log_observation(run, unit_id, t);
    }

    let variances: Vec<f64> = result.metrics_history.iter().map(|m| m.variance).collect();
    let event = TerminalEvent {
        unit_id,
        t: result.final_step as u64,
        t_unit: T_UNIT,
        outcome: if result.converged {
            "converged"
        } else {
            "unconverged"
        },
        censored: !result.converged,
        budget: max_steps as u64,
        trial_seed,
        convergence_time: crate::metrics::convergence_time(&variances, tol).map(|t| t as u64),
        final_bias: last.bias,
        final_diversity: last.diversity,
        final_variance: last.variance,
        n_clusters: last.n_clusters,
        polarization: last.polarization,
        cache_hit_rate: result.metadata.cache_hit_rate(),
    };
    run.log_event("terminal", &event)
        .unwrap_or_else(|e| panic!("{unit_id} の terminal イベントの記録に失敗: {e}"));
}

// ---------------------------------------------------------------------------
// 条件 1 点ぶんの集約 (sweep の子 run)
// ---------------------------------------------------------------------------

/// 1 条件で回した試行群の最終値．集約の材料になる．
pub struct TrialOutcome {
    /// 収束したか．
    pub converged: bool,
    /// 収束 (または打ち切り) したステップ．
    pub final_step: usize,
    /// 最終ステップの Bias B．
    pub bias: f64,
    /// 最終ステップの Diversity D．
    pub diversity: f64,
    /// 最終ステップの意見の分散．
    pub variance: f64,
    /// 最終ステップのクラスタ数．
    pub n_clusters: usize,
    /// 最終ステップの分極指標．
    pub polarization: f64,
}

impl TrialOutcome {
    /// [`SimulationResult`] の最終ステップから取り出す．
    pub fn from_result(result: &SimulationResult) -> Self {
        let last = result
            .metrics_history
            .last()
            .expect("metrics_history は t=0 を含む");
        TrialOutcome {
            converged: result.converged,
            final_step: result.final_step,
            bias: last.bias,
            diversity: last.diversity,
            variance: last.variance,
            n_clusters: last.n_clusters,
            polarization: last.polarization,
        }
    }
}

/// 1 条件 (bias × framing × topology の 1 点) を 1 つの値で表す指標．
///
/// 試行ごとの値は `events.jsonl` の担当なので，ここには集約しか書かない．試行
/// ごとの `final_diversity` を指標にすると (`run_uid`, `step`, `scope`, `name`) が
/// 重複するので，散らばりが要る図は `events.jsonl` から組み直す．
pub fn log_condition_summary(run: &mut Run, trials: &[TrialOutcome]) {
    let n = trials.len();
    assert!(n > 0, "試行が 1 本もありません");
    let n_f = n as f64;

    let n_converged = trials.iter().filter(|t| t.converged).count();
    let mean = |f: &dyn Fn(&TrialOutcome) -> f64| trials.iter().map(f).sum::<f64>() / n_f;

    run.log_metrics(
        SCOPE,
        &[
            ("n_units", n_f),
            ("n_converged", n_converged as f64),
            ("convergence_rate", n_converged as f64 / n_f),
            ("mean_final_bias", mean(&|t| t.bias)),
            ("mean_final_diversity", mean(&|t| t.diversity)),
            ("mean_final_variance", mean(&|t| t.variance)),
            ("mean_n_clusters", mean(&|t| t.n_clusters as f64)),
            ("mean_polarization", mean(&|t| t.polarization)),
            ("mean_final_step", mean(&|t| t.final_step as f64)),
        ],
    )
    .expect("run スコープの指標の記録に失敗");
}

// ---------------------------------------------------------------------------
// reproduce の帯照合
// ---------------------------------------------------------------------------

/// `events.jsonl` に書くアンカー判定行．
///
/// 照合先の帯はこの再現実装が置いた定性的なアンカーであって論文の報告値ではない
/// ので，出典を要求する `reference.csv` には書かない — 書くと «論文が報告した値»
/// と «こちらが決めた帯» が後から見分けられなくなる．
///
/// 観測量そのものは数なので `metrics.csv` にも `anchor_<id>` として置く
/// ([`log_anchor_observations`])．ここに残すのは比較の向き (帯) と PASS/off という
/// カテゴリで，これは指標にできない．
#[derive(Serialize)]
pub struct AnchorEvent<'a> {
    /// 指標名になる slug．
    pub id: &'a str,
    /// 人間向けの説明 (どの差を見ているか)．
    pub label: &'a str,
    /// 論文側の定性的な主張．
    pub paper: &'a str,
    pub observed: f64,
    pub target_lo: f64,
    /// 上限．«上限なし» は `null` ではなく列ごと落とす (JSON に無限大は無い)．
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_hi: Option<f64>,
    pub pass: bool,
}

/// アンカーの観測量を run スコープの指標として書く．
pub fn log_anchor_observations(run: &mut Run, anchors: &[(String, f64)], n_pass: usize) {
    let named: Vec<(String, f64)> = anchors
        .iter()
        .map(|(id, v)| (format!("anchor_{id}"), *v))
        .collect();
    let values: Vec<(&str, f64)> = named.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    run.log_metrics(SCOPE, &values)
        .expect("アンカー観測量の記録に失敗");
    run.log_metrics(
        SCOPE,
        &[
            ("checks_passed", n_pass as f64),
            ("checks_total", anchors.len() as f64),
        ],
    )
    .expect("アンカー件数の記録に失敗");
}

// ---------------------------------------------------------------------------
// シードの派生
// ---------------------------------------------------------------------------

/// 派生シードのラベルに使う文字列ハッシュ (explicit identity)．
///
/// 移行前の `main.rs` にあった FNV-1a をそのまま持ち込む．ここが変わると全条件の
/// 派生シードが変わり，移行の前後で結果を比較できなくなる．
pub fn label_hash(label: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in label.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// `sweep` の試行 1 本のシードを base seed から決定的に派生させる．
///
/// `master_seed` として記録するのは `base` の方で，実際に各試行が使うシードはこれで
/// 作る．移行前の `cmd_sweep` と同じ引数・同じ順序で `derive_seed` を呼ぶ．
pub fn sweep_trial_seed(base: u64, bias: &str, framing: &str, topology: &str, index: usize) -> u64 {
    socsim_core::derive_seed(
        base,
        &[
            label_hash(bias),
            label_hash(framing),
            label_hash(topology),
            index as u64,
        ],
    )
}

/// `reproduce` のセル 1 試行のシードを派生させる．
///
/// 移行前の `run_repro_cell` と同じ引数・同じ順序で `derive_seed` を呼ぶ
/// (第 2 引数は topology ではなく «相互作用の有無» である点に注意)．
pub fn repro_trial_seed(
    base: u64,
    bias: &str,
    interact: bool,
    topology: &str,
    index: usize,
) -> u64 {
    socsim_core::derive_seed(
        base,
        &[
            label_hash(bias),
            if interact { 1 } else { 0 },
            label_hash(topology),
            index as u64,
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::{label_hash, repro_trial_seed, sweep_trial_seed};

    /// FNV-1a("none")．移行前の `main.rs` の `label_hash` が返していた値．
    const GOLDEN_LABEL_HASH_NONE: u64 = 4_327_140_908_117_611_899;
    /// `sweep_trial_seed(42, "none", "false", "full", 0)`．
    const GOLDEN_SWEEP_SEED: u64 = 17_001_547_236_854_835_892;

    #[test]
    fn same_inputs_give_the_same_seed() {
        assert_eq!(
            sweep_trial_seed(42, "none", "false", "full", 3),
            sweep_trial_seed(42, "none", "false", "full", 3)
        );
        assert_eq!(
            repro_trial_seed(42, "strong", true, "full", 1),
            repro_trial_seed(42, "strong", true, "full", 1)
        );
    }

    #[test]
    fn each_coordinate_changes_the_seed() {
        let base = sweep_trial_seed(42, "none", "false", "full", 0);
        assert_ne!(base, sweep_trial_seed(43, "none", "false", "full", 0));
        assert_ne!(base, sweep_trial_seed(42, "weak", "false", "full", 0));
        assert_ne!(base, sweep_trial_seed(42, "none", "true", "full", 0));
        assert_ne!(base, sweep_trial_seed(42, "none", "false", "ws", 0));
        assert_ne!(base, sweep_trial_seed(42, "none", "false", "full", 1));

        let repro = repro_trial_seed(42, "none", true, "full", 0);
        assert_ne!(repro, repro_trial_seed(42, "none", false, "full", 0));
    }

    /// 具体値を固定する．
    ///
    /// ここが変わるのは socsim の `derive_seed` か [`label_hash`] が変わったときで，
    /// そのときは過去の run と結果を比較できなくなっている．Cargo.lock が socsim の
    /// commit を固定しているので，この値は依存を上げたときにだけ動く．
    #[test]
    fn golden_values_are_pinned() {
        assert_eq!(label_hash("none"), GOLDEN_LABEL_HASH_NONE);
        assert_eq!(
            sweep_trial_seed(42, "none", "false", "full", 0),
            GOLDEN_SWEEP_SEED
        );
    }
}
