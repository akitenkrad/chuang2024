//! socsim フレームワーク上の LLM 意見力学メカニズム．
//!
//! 二層アーキテクチャの **境界** がここにある．下層 (決定論的 socsim コア) は
//! ペア選択を `ctx.rng` (ChaCha20) で行い，上層 (非決定的 LLM レイヤ) は
//! [`OpinionClient`] (キャッシュ付き Ollama→OpenAI フォールバック) 越しの
//! ツイート生成・所感報告・意見数値化を行う．
//!
//! - [`LLMOpinionUpdateMechanism`] (`Interaction` フェーズ): 1 イベント =
//!   1 dyadic interaction．近傍から話者+聴者を一様サンプリングし，話者がツイート
//!   を発話 (LLM) → 聴者が所感を報告 (LLM) → `f_oc` で数値化 → 双方のメモリ更新．
//!   `events_per_step` 回繰り返す．LLM 呼び出しは **すべてこのメカニズムに閉じる**．
//! - [`MetricsMechanism`] (`PostStep` フェーズ): 意見分散が `tol` 未満になったら
//!   [`StepContext::request_stop`] で収束停止を要求する．
//!
//! LLM クライアントと呼び出しメタデータは `Rc<RefCell<…>>` で共有し，run ドライバ
//! が実行後にキャッシュ保存・メタデータ集計に使う (engine はメカニズムを所有する
//! ため，共有参照で取り出す)．

use std::cell::RefCell;
use std::rc::Rc;

use rand::Rng;

use socsim_core::{AgentId, Mechanism, Phase, Result, SocsimError, StepContext, WorldState};
use socsim_llm::MetadataCollector;

use crate::classifier::classify_opinion;
use crate::config::LlmSettings;
use crate::llm::{llm_config, OpinionClient};
use crate::metrics::variance;
use crate::prompts;
use crate::world::{clamp_opinion, OpinionWorld};

/// 共有 LLM クライアント (run ドライバとメカニズムで共有)．
pub type SharedClient = Rc<RefCell<OpinionClient>>;
/// 共有メタデータコレクタ (cache-hit 率などを run 後に集計)．
pub type SharedMetadata = Rc<RefCell<MetadataCollector>>;

/// dyadic な LLM 意見更新メカニズム (`Interaction` フェーズ)．
pub struct LLMOpinionUpdateMechanism {
    client: SharedClient,
    metadata: SharedMetadata,
    settings: LlmSettings,
    events_per_step: usize,
}

impl LLMOpinionUpdateMechanism {
    /// 共有クライアント・メタデータ・LLM 設定・1 ステップあたりイベント数から作る．
    pub fn new(
        client: SharedClient,
        metadata: SharedMetadata,
        settings: LlmSettings,
        events_per_step: usize,
    ) -> Self {
        LLMOpinionUpdateMechanism {
            client,
            metadata,
            settings,
            events_per_step: events_per_step.max(1),
        }
    }

    /// 1 回の dyadic interaction を処理する．`ctx.rng` で話者・聴者を一様抽選し，
    /// LLM でツイート発話 → 所感報告 → 数値化 → メモリ更新を行う．
    fn one_interaction(&self, ctx: &mut StepContext<'_, OpinionWorld>) -> Result<()> {
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        if ids.len() < 2 {
            return Ok(());
        }

        // --- 話者を一様抽選 (決定論的; socsim コア層) ---
        let speaker = ids[ctx.rng.gen_range(0..ids.len())];

        // 話者の近傍から聴者を一様抽選．近傍が無ければ全体から (話者以外) 抽選し，
        // 孤立ノードでも更新が止まらないようにする (全結合では近傍 = 他全員)．
        let mut neighbors: Vec<AgentId> = ctx
            .world
            .net
            .neighbors(speaker)
            .into_iter()
            .filter(|&id| id != speaker)
            .collect();
        if neighbors.is_empty() {
            neighbors = ids.iter().copied().filter(|&id| id != speaker).collect();
        }
        let listener = neighbors[ctx.rng.gen_range(0..neighbors.len())];

        let topic = ctx.world.topic.clone();
        let framing = ctx.world.framing;
        let bias = ctx.world.bias;

        // --- 話者: ツイート発話 (LLM) ---
        let speaker_state = ctx
            .world
            .agents
            .get(&speaker)
            .expect("speaker exists")
            .clone();
        let speaker_prompt = prompts::speaker_prompt(&speaker_state, &topic, framing);
        let tweet = {
            let mut client = self.client.borrow_mut();
            let resp = client
                .complete(&speaker_prompt, &llm_config(&self.settings))
                .map_err(|e| SocsimError::Mechanism(format!("speaker LLM call failed: {e}")))?;
            self.metadata.borrow_mut().record(resp.metadata.clone());
            resp.text
        };

        // --- 聴者: 所感報告 (LLM) ---
        let listener_state = ctx
            .world
            .agents
            .get(&listener)
            .expect("listener exists")
            .clone();
        let listener_prompt =
            prompts::listener_prompt(&listener_state, &tweet, &topic, framing, bias);
        let sentiment = {
            let mut client = self.client.borrow_mut();
            let resp = client
                .complete(&listener_prompt, &llm_config(&self.settings))
                .map_err(|e| SocsimError::Mechanism(format!("listener LLM call failed: {e}")))?;
            self.metadata.borrow_mut().record(resp.metadata.clone());
            resp.text
        };

        // --- f_oc: 所感を 5 段階意見へ数値化 (規則ベース → LLM フォールバック) ---
        let framing_stmt = prompts::framing_statement(&topic, framing);
        let new_opinion = {
            let mut client = self.client.borrow_mut();
            let o = classify_opinion(
                &mut client,
                &self.settings,
                &topic,
                &framing_stmt,
                &sentiment,
            )
            .map_err(|e| SocsimError::Mechanism(format!("classifier LLM call failed: {e}")))?;
            // 分類器フォールバックが呼ばれた場合のメタデータも 1 件記録される可能性が
            // あるが，規則ベースで読めた場合は呼ばれない (キャッシュ消費ゼロ)．
            clamp_opinion(o)
        };

        // --- メモリ更新 (双方) + 聴者の意見更新 ---
        // 話者: 自分が "write" したツイートを記憶．
        if let Some(s) = ctx.world.agents.get_mut(&speaker) {
            s.last_text = tweet.clone();
            s.memory.push(format!("I wrote: {tweet}"));
        }
        // 聴者: 読んだツイートと更新後スタンスを記憶し，意見を更新．
        if let Some(l) = ctx.world.agents.get_mut(&listener) {
            l.memory
                .push(format!("I read: {tweet} -> my stance became {new_opinion}"));
            l.opinion = new_opinion;
        }

        Ok(())
    }
}

impl Mechanism<OpinionWorld> for LLMOpinionUpdateMechanism {
    fn name(&self) -> &str {
        "llm_opinion_update"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Interaction]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, OpinionWorld>) -> Result<()> {
        // 非相互作用統制条件では dyadic 更新を行わない (Phase 3 拡張点)．
        if !ctx.world.interact {
            return Ok(());
        }
        for _ in 0..self.events_per_step {
            self.one_interaction(ctx)?;
        }
        Ok(())
    }
}

/// 収束判定メカニズム (`PostStep` フェーズ)．
///
/// ステップ末に各エージェントの意見軌跡へ現在値を追記し，意見分散が `tol` 未満なら
/// `request_stop` で収束停止を要求する．意見分散を `scratch` にも積む (ドライバ用)．
pub struct MetricsMechanism {
    /// 収束判定の意見分散しきい値．
    pub tol: f64,
}

impl Mechanism<OpinionWorld> for MetricsMechanism {
    fn name(&self) -> &str {
        "metrics"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::PostStep]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, OpinionWorld>) -> Result<()> {
        // 各エージェントの軌跡へ現在意見を追記する．
        for state in ctx.world.agents.values_mut() {
            state.trajectory.push(state.opinion);
        }

        let opinions = ctx.world.opinions();
        let var = variance(&opinions);
        ctx.scratch.insert("variance", var);

        let converged = var < self.tol;
        ctx.scratch.insert("converged", converged);
        if converged {
            ctx.request_stop();
        }
        Ok(())
    }
}
