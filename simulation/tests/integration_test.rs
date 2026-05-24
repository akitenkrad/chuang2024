//! Chuang et al. (2024) LLM 意見力学の統合テスト．
//!
//! **ライブ LLM を一切必要としない**: socsim-llm の `mock::ScriptedClient` で
//! 決定論的に意見更新を駆動し，以下を検証する:
//! ・収束/メトリクス配線 (variance/bias/diversity の計算と記録)
//! ・ネットワーク配線 (トポロジ生成とエージェント数)
//! ・固定 mock を与えたときの socsim コア層の RNG 決定論性
//! ・スクリプト応答に応じた意見更新 (合意/分極)

use chuang_opinion_simulation::config::{Config, Framing, Topology};
use chuang_opinion_simulation::llm::{wrap_client, OpinionClient};
use chuang_opinion_simulation::metrics::{convergence_time, diversity};
use chuang_opinion_simulation::simulation::run_with_client;

use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;

/// 聴者プロンプトには固定整数 `reply`，話者プロンプトには短文を返す mock クライアント．
fn scripted(reply: &'static str) -> OpinionClient {
    let backend = ScriptedClient::new("mock-model", move |prompt: &str| {
        if prompt.contains("Answer with a SINGLE integer") {
            reply.to_string()
        } else {
            "A short tweet about the topic.".to_string()
        }
    });
    wrap_client(backend, PromptCache::in_memory())
}

fn base_config() -> Config {
    Config {
        n_agents: 6,
        max_steps: 30,
        events_per_step: 2,
        tol: 1e-9, // 収束で停止させない (全エージェントが揃うまで回す)
        seed: Some(7),
        topology: Topology::Full,
        framing: Framing::True,
        ..Config::default()
    }
}

// --------------------------------------------------------------------------- //
// 全員が同じ整数を採用 → 合意 (diversity → 0)
// --------------------------------------------------------------------------- //

#[test]
fn unanimous_listener_reply_drives_consensus() {
    let cfg = base_config();
    let result = run_with_client(&cfg, scripted("2")).unwrap();
    let last = result.opinion_history.last().unwrap();
    // 十分なイベント後，全員 (聴者として一度は更新される) が 2 に寄る．
    let d = diversity(last);
    assert!(d < 1.0, "全員 2 を採用すれば多様性は縮小するはず (D={d})");
    // メトリクス配線: t=0 が記録され，末尾まで連続している．
    assert_eq!(result.metrics_history[0].t, 0);
    assert_eq!(
        result.metrics_history.len(),
        result.opinion_history.len(),
        "metrics と opinions の履歴長は一致する"
    );
}

// --------------------------------------------------------------------------- //
// 収束判定の配線: tol を緩めれば収束フラグが立つ
// --------------------------------------------------------------------------- //

#[test]
fn convergence_flag_when_variance_below_tol() {
    let mut cfg = base_config();
    cfg.tol = 0.5; // 合意気味になれば variance < 0.5 で停止
    cfg.max_steps = 200;
    cfg.events_per_step = 4;
    let result = run_with_client(&cfg, scripted("0")).unwrap();
    // 全員が 0 を採用 → variance は 0 へ → 収束停止する．
    assert!(result.converged, "全員中立なら収束フラグが立つべき");
    let variances: Vec<f64> = result.metrics_history.iter().map(|m| m.variance).collect();
    assert!(convergence_time(&variances, cfg.tol).is_some());
}

// --------------------------------------------------------------------------- //
// ネットワーク配線: トポロジごとに正しいノード数
// --------------------------------------------------------------------------- //

#[test]
fn topologies_build_with_correct_node_count() {
    for topo in [
        Topology::Full,
        Topology::WattsStrogatz,
        Topology::BarabasiAlbert,
    ] {
        let mut cfg = base_config();
        cfg.topology = topo;
        cfg.n_agents = 8;
        cfg.max_steps = 3;
        let result = run_with_client(&cfg, scripted("1")).unwrap();
        assert_eq!(
            result.opinion_history[0].len(),
            8,
            "{:?}: 初期意見ベクトルは N 要素",
            topo
        );
    }
}

// --------------------------------------------------------------------------- //
// 決定論性: 同一シード + 同一 mock → 完全再現 (socsim コア層)
// --------------------------------------------------------------------------- //

#[test]
fn core_is_deterministic_given_fixed_mock() {
    let cfg = base_config();
    let a = run_with_client(&cfg, scripted("1")).unwrap();
    let b = run_with_client(&cfg, scripted("1")).unwrap();
    assert_eq!(
        a.opinion_history, b.opinion_history,
        "同一シードは完全再現すべき"
    );
    assert_eq!(a.final_step, b.final_step);
}

// --------------------------------------------------------------------------- //
// 異なるシード → (一般に) 異なる初期意見/ペア選択軌跡
// --------------------------------------------------------------------------- //

#[test]
fn different_seed_changes_trajectory() {
    let mut cfg_a = base_config();
    cfg_a.seed = Some(1);
    let mut cfg_b = base_config();
    cfg_b.seed = Some(999);
    let a = run_with_client(&cfg_a, scripted("-2")).unwrap();
    let b = run_with_client(&cfg_b, scripted("-2")).unwrap();
    // 初期意見は seed 依存なので少なくとも初期状態は (高確率で) 異なる．
    // 厳密同一になる確率は無視できるほど小さいが，安全のため軌跡全体で比較する．
    assert!(
        a.opinion_history != b.opinion_history || a.final_step != b.final_step,
        "異なるシードは (一般に) 異なる軌跡を生む"
    );
}
