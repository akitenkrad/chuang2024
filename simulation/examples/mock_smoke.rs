//! Mock 駆動のスモーク実行 (ライブ LLM 不要)．
//!
//! ライブ Ollama/OpenAI が使えない環境 (CI・ネットワーク遮断サンドボックス) で
//! 出力パイプライン (run ディレクトリ・metrics.csv・events.jsonl・artifacts/
//! opinions.csv) と Python 可視化を検証するための補助バイナリ．
//! `socsim-llm::mock::ScriptedClient` で決定論的に意見更新を駆動する．
//!
//! ```bash
//! cargo run --release --example mock_smoke -- results
//! ```

use std::env;
use std::fs;

use chuang_opinion_simulation::config::Config;
use chuang_opinion_simulation::llm::wrap_client;
use chuang_opinion_simulation::record::{self, DOMAIN, EXPERIMENT, REPO_ID};
use chuang_opinion_simulation::simulation::{run_with_client, save_opinions};
use runvault::{Run, RunOptions};
use serde::Serialize;
use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;

/// スモーク実行の実験条件．
#[derive(Serialize)]
struct SmokeParameters {
    n_agents: usize,
    max_steps: usize,
    events_per_step: usize,
    tol: f64,
    seed: u64,
}

fn main() {
    let base = env::args().nth(1).unwrap_or_else(|| "results".to_string());
    let seed = 42u64;

    let cfg = Config {
        n_agents: 6,
        max_steps: 12,
        events_per_step: 2,
        tol: 1e-9,
        seed: Some(seed),
        ..Config::default()
    };

    // 聴者プロンプトには擬似的に意見を返す mock．話者にはツイート文を返す．
    // 話者意見に応じてゆるく所感を返し，軌跡に変化を出す．
    let backend = ScriptedClient::new("mock-llama3.2", |prompt: &str| {
        if prompt.contains("Answer with a SINGLE integer") {
            // フレーミングが TRUE のとき肯定方向 (1) へ寄せる擬似挙動．
            if prompt.contains("is TRUE") {
                "1".to_string()
            } else {
                "-1".to_string()
            }
        } else {
            "Sharing my thoughts on the topic today.".to_string()
        }
    });
    let client = wrap_client(backend, PromptCache::in_memory());
    // モデル名と endpoint は Run::start より前に要る (llm ブロックは開始時に確定する)．
    let model = client.inner().model().to_string();
    let endpoint = client.inner().endpoint().to_string();

    let parameters = SmokeParameters {
        n_agents: cfg.n_agents,
        max_steps: cfg.max_steps,
        events_per_step: cfg.events_per_step,
        tol: cfg.tol,
        seed,
    };

    let mut rv = Run::start(
        RunOptions::new(EXPERIMENT, "mock-smoke")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&base)
            .parameters(&parameters)
            .expect("runvault: parameters の組み立てに失敗")
            .seed_pointers(["/seed"])
            .master_seed(seed)
            .llm(record::llm_block(&model, &endpoint, cfg.llm.temperature))
            .replication(record::replication()),
    )
    .expect("runvault: run の開始に失敗");

    let artifacts = rv.dir().join("artifacts");
    fs::create_dir_all(&artifacts).expect("artifacts ディレクトリの作成に失敗");

    let result = run_with_client(&cfg, client).expect("mock run failed");
    save_opinions(&result, &artifacts.to_string_lossy());
    record::log_simulation(&mut rv, &result);
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
    let dir = rv.finish().expect("runvault: run の完了に失敗");
    println!("mock smoke wrote: {}", dir.display());
    println!(
        "final B={:.3} D={:.3} n_clusters={} steps={}",
        last.bias, last.diversity, last.n_clusters, result.final_step
    );
}
