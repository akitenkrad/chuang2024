//! socsim フレームワーク上の LLM 意見力学の世界状態．
//!
//! エージェントは **固定ノード** であり移動せず，意見・メモリだけが変化する．
//! 相互作用は **ネットワーク近傍** に沿うため `socsim-grid` ではなく
//! `socsim-net::SocialNetwork` を採用する (論文は全結合だが，本再現はトポロジ
//! 効果検証のため ER/WS/BA も生成可能)．全結合は完全グラフ
//! (`erdos_renyi(ids, 1.0, rng)`) で表現する．
//!
//! `#[derive(Clone)]` でスナップショット (save/resume) と非相互作用統制条件の
//! 比較実験に対応する．`agent_ids()` は `BTreeMap` のソート済みキーを返し
//! 決定論を保証する (socsim コア層)．

use std::collections::BTreeMap;

use socsim_core::{AgentId, SimClock, WorldState};
use socsim_net::SocialNetwork;

use crate::config::{ConfirmationBias, Framing, MemoryMode};

/// 1 エージェントの状態 (固定ノード; 移動しない)．
#[derive(Clone, Debug)]
pub struct AgentState {
    /// ペルソナ (政治傾向・年齢・職業など; テキスト)．
    pub persona: String,
    /// 現在意見 `o_i^t ∈ {-2,-1,0,1,2}` (5 段階リッカート尺度)．
    pub opinion: i8,
    /// 動的メモリ `m_i^t` (cumulative は追記; reflective は要約)．
    pub memory: Vec<String>,
    /// 意見軌跡 `⟨o_i⟩` (各イベント後の意見値; ステップ末に記録)．
    pub trajectory: Vec<i8>,
    /// 直近に発話したツイート (デバッグ・出力用; 空なら未発話)．
    pub last_text: String,
}

impl AgentState {
    /// 初期ペルソナ・初期意見からエージェント状態を作る．
    pub fn new(persona: String, opinion: i8) -> Self {
        AgentState {
            persona,
            opinion,
            memory: Vec::new(),
            trajectory: vec![opinion],
            last_text: String::new(),
        }
    }
}

/// 意見空間 `O = {-2,-1,0,1,2}` のクランプ範囲．
pub const OPINION_MIN: i8 = -2;
/// 意見空間の上限．
pub const OPINION_MAX: i8 = 2;

/// 意見値を意見空間 `[-2, 2]` にクランプする．
pub fn clamp_opinion(o: i8) -> i8 {
    o.clamp(OPINION_MIN, OPINION_MAX)
}

/// LLM 意見力学の世界状態．
#[derive(Clone)]
pub struct OpinionWorld {
    /// シミュレーションクロック．
    pub clock: SimClock,
    /// 相互作用網 (全結合=完全グラフ / WS / BA)．
    pub net: SocialNetwork,
    /// 各エージェントの状態 (ソート済みキー)．
    pub agents: BTreeMap<AgentId, AgentState>,
    /// 議論トピック (ground truth 既知)．
    pub topic: String,
    /// フレーミング (True / False)．
    pub framing: Framing,
    /// 確証バイアス (None / Weak / Strong)．
    pub bias: ConfirmationBias,
    /// メモリ方式 (Cumulative / Reflective)．
    pub memory_mode: MemoryMode,
    /// 相互作用するか (false = 非相互作用統制条件)．
    pub interact: bool,
}

impl OpinionWorld {
    /// 構成済みのフィールドから世界状態を組み立てる (網生成・初期化は
    /// [`crate::simulation::init_world`])．
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        net: SocialNetwork,
        agents: BTreeMap<AgentId, AgentState>,
        topic: String,
        framing: Framing,
        bias: ConfirmationBias,
        memory_mode: MemoryMode,
        interact: bool,
        max_steps: u64,
    ) -> Self {
        OpinionWorld {
            clock: SimClock::new(max_steps),
            net,
            agents,
            topic,
            framing,
            bias,
            memory_mode,
            interact,
        }
    }

    /// エージェント数 N．
    pub fn n(&self) -> usize {
        self.agents.len()
    }

    /// 全エージェントの現在意見をソート順 (agent_id 昇順) で返す．
    pub fn opinions(&self) -> Vec<i8> {
        self.agents.values().map(|a| a.opinion).collect()
    }
}

impl WorldState for OpinionWorld {
    fn agent_ids(&self) -> Vec<AgentId> {
        // BTreeMap のキーはソート済み．契約 (sorted) を明示する．
        self.agents.keys().copied().collect()
    }

    fn clock(&self) -> &SimClock {
        &self.clock
    }

    fn clock_mut(&mut self) -> &mut SimClock {
        &mut self.clock
    }
}
