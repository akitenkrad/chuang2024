//! Mock 駆動のスモーク実行 (ライブ LLM 不要)．
//!
//! ライブ Ollama/OpenAI が使えない環境 (CI・ネットワーク遮断サンドボックス) で
//! 出力パイプライン (opinions.csv / metrics.csv / run_metadata.json) と Python
//! 可視化を検証するための補助バイナリ．`socsim-llm::mock::ScriptedClient` で
//! 決定論的に意見更新を駆動し，本番 `run` と同じ writer で結果を書き出す．
//!
//! ```bash
//! cargo run --release --example mock_smoke -- results
//! ```

use std::env;
use std::fs;

use chrono::Local;

use chuang_opinion_simulation::config::Config;
use chuang_opinion_simulation::llm::wrap_client;
use chuang_opinion_simulation::simulation::{
    ensure_output_dir, run_with_client, save_metrics, save_opinions, save_run_metadata,
};
use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;

fn main() {
    let base = env::args().nth(1).unwrap_or_else(|| "results".to_string());
    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
    let output_dir = format!("{base}/{timestamp}");

    let cfg = Config {
        n_agents: 6,
        max_steps: 12,
        events_per_step: 2,
        tol: 1e-9,
        seed: Some(42),
        output_dir: output_dir.clone(),
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

    ensure_output_dir(&cfg.output_dir);
    let result = run_with_client(&cfg, client).expect("mock run failed");
    save_metrics(&result.metrics_history, &cfg.output_dir);
    save_opinions(&result, &cfg.output_dir);
    save_run_metadata(&result, &cfg, &cfg.output_dir);

    // config.json
    let cfg_path = format!("{}/config.json", cfg.output_dir);
    let f = fs::File::create(&cfg_path).unwrap();
    serde_json::to_writer_pretty(f, &cfg.to_run_config_json()).unwrap();

    // latest symlink
    let link = format!("{base}/latest");
    let _ = fs::remove_file(&link);
    #[cfg(unix)]
    let _ = std::os::unix::fs::symlink(&timestamp, &link);

    let last = result.metrics_history.last().unwrap();
    println!("mock smoke wrote: {output_dir}");
    println!(
        "final B={:.3} D={:.3} n_clusters={} steps={}",
        last.bias, last.diversity, last.n_clusters, result.final_step
    );
}
