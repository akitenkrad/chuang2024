//! 初期化と実行ドライバ (SimulationBuilder 配線 + 二層 LLM レイヤ)．
//!
//! 二層決定論 (socsim-mapping §10) を配線する:
//! - **下層 (決定論的 socsim コア)**: `derive_seed(root, &[0])` で網生成・ペルソナ
//!   /初期意見割当の init RNG を，`derive_seed(root, &[1])` で engine RNG
//!   (= ペア一様サンプリング) を派生する．bit 単位で再現する．
//! - **上層 (非決定的 LLM レイヤ)**: [`crate::llm`] のキャッシュ付き
//!   Ollama→OpenAI フォールバッククライアントに閉じ込め，`temperature=0`/`seed`
//!   固定 + プロンプト→応答キャッシュで擬似決定論化する．モデル・endpoint・
//!   温度・seed・cache-hit を `run_metadata.json` に記録する．

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::rc::Rc;

use csv::Writer;
use rand::Rng;

use socsim_core::{derive_seed, AgentId, SimRng};
use socsim_engine::{SequentialScheduler, SimulationBuilder};
use socsim_llm::{LlmClient, MetadataCollector};

use crate::config::{Config, Topology};
use crate::llm::OpinionClient;
use crate::mechanisms::{
    LLMOpinionUpdateMechanism, MetricsMechanism, SharedClient, SharedMetadata,
};
use crate::metrics::Metrics;
use crate::world::{AgentState, OpinionWorld};

/// 網生成・ペルソナ/初期意見割当用 RNG ラベル．
const RNG_WORLD_INIT: u64 = 0;
/// socsim エンジン (= ペア一様サンプリング) 用 RNG ラベル．
const RNG_ENGINE: u64 = 1;

/// 初期意見の候補集合 (5 段階; 一様抽選)．
const OPINION_CHOICES: [i8; 5] = [-2, -1, 0, 1, 2];

/// 初期ペルソナのテンプレート集合 (ラウンドロビンで割当; 決定論的)．
const PERSONAS: [&str; 6] = [
    "a cautious skeptic who values evidence",
    "an optimistic early-adopter of new ideas",
    "a traditionalist who trusts established consensus",
    "a contrarian who questions popular claims",
    "a pragmatic moderate weighing both sides",
    "a curious learner forming their first opinion",
];

/// シミュレーション全体の実行結果．
pub struct SimulationResult {
    /// 各ステップ (t=0 を含む) のメトリクス履歴．
    pub metrics_history: Vec<Metrics>,
    /// 各ステップの意見スナップショット (`opinions[t][i]`)．
    pub opinion_history: Vec<Vec<i8>>,
    /// 各エージェントの最終ツイート (opinions.csv の text 列用; 軌跡には残さない)．
    pub final_texts: BTreeMap<AgentId, String>,
    /// 収束したか (意見分散 < tol)．
    pub converged: bool,
    /// 収束 (または最終) ステップ番号．
    pub final_step: usize,
    /// LLM 呼び出しメタデータの集計．
    pub metadata: MetadataCollector,
    /// LLM モデル名 (run_metadata 用)．
    pub llm_model: String,
    /// LLM endpoint (run_metadata 用; primary)．
    pub llm_endpoint: String,
}

/// 世界状態を初期化する (網生成 + ペルソナ/初期意見/メモリ割当)．
///
/// トポロジに応じて `socsim-net` の生成器を使う．全結合は完全グラフ
/// (`erdos_renyi(ids, 1.0, rng)`)．初期意見・ペルソナは init RNG から決定論的に
/// 割り当てる (socsim コア層)．
pub fn init_world(cfg: &Config, rng: &mut SimRng) -> OpinionWorld {
    let ids: Vec<AgentId> = (0..cfg.n_agents as u64).map(AgentId).collect();

    let net = match cfg.topology {
        // 全結合 = 完全グラフ (ER の p=1)．
        Topology::Full => socsim_net::SocialNetwork::erdos_renyi(&ids, 1.0, rng),
        Topology::ErdosRenyi => {
            socsim_net::SocialNetwork::erdos_renyi(&ids, cfg.er_p.clamp(0.0, 1.0), rng)
        }
        Topology::WattsStrogatz => {
            socsim_net::SocialNetwork::watts_strogatz(&ids, cfg.ws_k, cfg.ws_beta, rng)
        }
        Topology::BarabasiAlbert => socsim_net::SocialNetwork::barabasi_albert(&ids, cfg.ba_m, rng),
    };

    let mut agents: BTreeMap<AgentId, AgentState> = BTreeMap::new();
    for (idx, &id) in ids.iter().enumerate() {
        let persona = PERSONAS[idx % PERSONAS.len()].to_string();
        // 初期意見を一様抽選 (決定論的; init RNG ストリーム)．
        let opinion = OPINION_CHOICES[rng.gen_range(0..OPINION_CHOICES.len())];
        agents.insert(id, AgentState::new(persona, opinion));
    }

    OpinionWorld::new(
        net,
        agents,
        cfg.topic.clone(),
        cfg.framing,
        cfg.bias,
        cfg.memory_mode,
        cfg.interact,
        cfg.max_steps as u64,
    )
}

/// オフライン (LLM 不要) の決定論的 mock でシミュレーションを実行する．
///
/// [`crate::reproduce_mock::build_reproduce_client`] の scripted クライアント
/// (in-memory cache) で駆動する．`reproduce --mock` / `run --mock` から使い，
/// ライブ LLM 無しで論文の定性的挙動を構造的に再現する．mock は in-memory cache
/// なので永続キャッシュ保存はスキップされる (`cfg.llm.cache_path` は無視扱い)．
pub fn run_mock(cfg: &Config) -> Result<SimulationResult, String> {
    // mock は永続キャッシュを持たないため，誤って save() を呼ばないよう cache_path
    // を落とした設定で駆動する (in-memory cache は save() が no-op だが明示的に倒す)．
    let mut mock_cfg = cfg.clone();
    mock_cfg.llm.cache_path = None;
    run_with_client(&mock_cfg, crate::reproduce_mock::build_reproduce_client())
}

/// 与えられた [`OpinionClient`] でシミュレーションを実行する．
///
/// 本番は [`build_live_client`] の結果を，テストは
/// [`wrap_client`] でラップした `mock::ScriptedClient` を渡す．LLM クライアントは
/// メカニズムと `Rc<RefCell<…>>` で共有し，実行後にキャッシュ保存・メタデータ集計
/// に使う．
pub fn run_with_client(cfg: &Config, client: OpinionClient) -> Result<SimulationResult, String> {
    let root = cfg.seed.unwrap_or_else(rand::random);

    // 初期世界 (root から派生した init RNG; 決定論的 socsim コア層)．
    let mut init_rng = SimRng::from_seed(derive_seed(root, &[RNG_WORLD_INIT]));
    let world = init_world(cfg, &mut init_rng);

    // LLM モデル/endpoint をメタデータ用に控える．
    let llm_model = client.inner().model().to_string();
    let llm_endpoint = client.inner().endpoint().to_string();

    // クライアント・メタデータを共有 (メカニズムと run ドライバで共用)．
    let shared_client: SharedClient = Rc::new(RefCell::new(client));
    let shared_meta: SharedMetadata = Rc::new(RefCell::new(MetadataCollector::new()));

    let mut sim = SimulationBuilder::new(world)
        .scheduler(Box::new(SequentialScheduler))
        .seed(derive_seed(root, &[RNG_ENGINE]))
        .add_mechanism(Box::new(LLMOpinionUpdateMechanism::new(
            Rc::clone(&shared_client),
            Rc::clone(&shared_meta),
            cfg.llm.clone(),
            cfg.events_per_step,
        )))
        .add_mechanism(Box::new(MetricsMechanism { tol: cfg.tol }))
        .build();

    let mut metrics_history: Vec<Metrics> = Vec::new();
    let mut opinion_history: Vec<Vec<i8>> = Vec::new();

    // 初期状態 (t=0) を記録．
    let init_opinions = sim.world().opinions();
    metrics_history.push(Metrics::compute(&init_opinions, 0));
    opinion_history.push(init_opinions);

    let mut converged = false;
    let mut final_step = 0usize;

    sim.run_observed(|report| {
        let t = report.t as usize;
        let opinions = report.world.opinions();
        metrics_history.push(Metrics::compute(&opinions, t));
        opinion_history.push(opinions);
        converged = *report.scratch.get::<bool>("converged").unwrap_or(&false);
        final_step = t;
    })
    .map_err(|e| format!("シミュレーションの実行に失敗: {e}"))?;

    // キャッシュを保存 (cache_path 指定時; in-memory はスキップ)．
    if cfg.llm.cache_path.is_some() {
        let client = shared_client.borrow();
        client
            .cache()
            .save()
            .map_err(|e| format!("キャッシュ保存に失敗: {e}"))?;
    }

    // 最終ツイートを集める．
    let final_texts: BTreeMap<AgentId, String> = sim
        .world()
        .agents
        .iter()
        .map(|(&id, s)| (id, s.last_text.clone()))
        .collect();

    let metadata = shared_meta.borrow().clone();

    Ok(SimulationResult {
        metrics_history,
        opinion_history,
        final_texts,
        converged,
        final_step,
        metadata,
        llm_model,
        llm_endpoint,
    })
}

/// 意見履歴を long-format CSV (t, agent_id, opinion, text) に保存する．
///
/// `text` 列は最終ステップのみ各エージェントの最終ツイートを埋め，それ以外は空に
/// する (途中ツイートは膨大になるため; 軌跡解析には opinion 列で十分)．
pub fn save_opinions(result: &SimulationResult, output_dir: &str) {
    let path = format!("{}/opinions.csv", output_dir);
    let file = File::create(&path).expect("opinions.csv の作成に失敗");
    let mut wtr = Writer::from_writer(BufWriter::new(file));
    wtr.write_record(["t", "agent_id", "opinion", "text"])
        .expect("ヘッダ書き込みに失敗");
    let last_t = result.opinion_history.len().saturating_sub(1);
    for (t, opinions) in result.opinion_history.iter().enumerate() {
        for (i, &o) in opinions.iter().enumerate() {
            let text = if t == last_t {
                result
                    .final_texts
                    .get(&AgentId(i as u64))
                    .cloned()
                    .unwrap_or_default()
            } else {
                String::new()
            };
            // 改行を空白へ畳んで CSV を 1 行に保つ．
            let text = text.replace(['\n', '\r'], " ");
            wtr.write_record(&[t.to_string(), i.to_string(), o.to_string(), text])
                .expect("レコード書き込みに失敗");
        }
    }
    wtr.flush().expect("フラッシュに失敗");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::wrap_client;
    use socsim_llm::mock::ScriptedClient;
    use socsim_llm::PromptCache;

    fn scripted_client() -> OpinionClient {
        // 聴者プロンプト末尾が整数を要求するので，所感には固定値 "1" を返す．
        // 話者プロンプトには短い文を返す．プロンプト内容で出し分ける．
        let backend = ScriptedClient::new("mock-model", |prompt: &str| {
            if prompt.contains("Answer with a SINGLE integer") {
                "1".to_string()
            } else {
                "I think the topic is worth considering.".to_string()
            }
        });
        wrap_client(backend, PromptCache::in_memory())
    }

    fn test_config() -> Config {
        Config {
            n_agents: 5,
            max_steps: 8,
            events_per_step: 1,
            tol: 1e-9, // 収束で早期停止しないよう厳しめ
            seed: Some(42),
            ..Config::default()
        }
    }

    #[test]
    fn scripted_run_drives_opinions_to_one() {
        let cfg = test_config();
        let result = run_with_client(&cfg, scripted_client()).unwrap();
        // 聴者は常に "1" を採用するので，相互作用が進むほど 1 に寄る．
        let last = result.opinion_history.last().unwrap();
        assert!(last.contains(&1), "少なくとも 1 名は 1 に更新される");
        assert_eq!(result.metrics_history[0].t, 0);
    }

    #[test]
    fn core_is_deterministic_given_mock() {
        let cfg = test_config();
        let a = run_with_client(&cfg, scripted_client()).unwrap();
        let b = run_with_client(&cfg, scripted_client()).unwrap();
        assert_eq!(a.opinion_history, b.opinion_history);
        assert_eq!(a.final_step, b.final_step);
    }
}
