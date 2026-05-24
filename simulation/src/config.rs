//! シミュレーション設定．
//!
//! Chuang et al. (2024) のコアモデル (dyadic LLM 意見力学) と感度分析パラメータ
//! を保持する [`Config`] と，その JSON シリアライズ表現を定義する．意見空間・
//! トポロジ・確証バイアス・フレーミング・メモリ方式などの列挙型もここに集約する．

use serde::Serialize;

// --------------------------------------------------------------------------- //
// トポロジ
// --------------------------------------------------------------------------- //

/// 相互作用網のトポロジ．
///
/// 論文設定は全結合 (all-to-all)．本再現はトポロジ効果検証のため WS / BA も
/// 生成可能とする (`socsim-net` 生成器)．全結合は完全グラフで実装する．
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Topology {
    /// 全結合 (完全グラフ; 論文設定)．
    Full,
    /// Watts–Strogatz 小世界網．
    WattsStrogatz,
    /// Barabási–Albert スケールフリー網．
    BarabasiAlbert,
}

impl Topology {
    pub fn label(&self) -> &'static str {
        match self {
            Topology::Full => "full",
            Topology::WattsStrogatz => "ws",
            Topology::BarabasiAlbert => "ba",
        }
    }
}

/// 文字列から [`Topology`] をパースする．
pub fn parse_topology(s: &str) -> Result<Topology, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "full" | "complete" => Ok(Topology::Full),
        "ws" | "watts_strogatz" | "watts-strogatz" => Ok(Topology::WattsStrogatz),
        "ba" | "barabasi_albert" | "barabasi-albert" => Ok(Topology::BarabasiAlbert),
        _ => Err(format!(
            "不正なトポロジ: \"{}\" (full / ws / ba のいずれか)",
            s
        )),
    }
}

// --------------------------------------------------------------------------- //
// 確証バイアス
// --------------------------------------------------------------------------- //

/// 確証バイアス (system message へ注入する強度)．
///
/// 論文 4.3．バイアスを強めるほど合意が崩れ多様性 D が単調増大する．
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfirmationBias {
    /// 確証バイアスなし (科学的事実へ収束しやすい)．
    None,
    /// 弱い確証バイアス．
    Weak,
    /// 強い確証バイアス (意見分断 fragmentation を誘発)．
    Strong,
}

impl ConfirmationBias {
    pub fn label(&self) -> &'static str {
        match self {
            ConfirmationBias::None => "none",
            ConfirmationBias::Weak => "weak",
            ConfirmationBias::Strong => "strong",
        }
    }

    /// system message へ注入する確証バイアス指示文 (英語; LLM プロンプト用)．
    pub fn instruction(&self) -> &'static str {
        match self {
            ConfirmationBias::None => "",
            ConfirmationBias::Weak => {
                " You have a mild tendency to favour information that agrees with your current view."
            }
            ConfirmationBias::Strong => {
                " You strongly prefer information that confirms your current belief and you discount \
                 anything that contradicts it; you rarely change your mind."
            }
        }
    }
}

/// 文字列から [`ConfirmationBias`] をパースする．
pub fn parse_bias(s: &str) -> Result<ConfirmationBias, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "none" | "off" => Ok(ConfirmationBias::None),
        "weak" => Ok(ConfirmationBias::Weak),
        "strong" | "on" => Ok(ConfirmationBias::Strong),
        _ => Err(format!(
            "不正な確証バイアス: \"{}\" (none / weak / strong)",
            s
        )),
    }
}

// --------------------------------------------------------------------------- //
// フレーミング
// --------------------------------------------------------------------------- //

/// 議論トピックのフレーミング (ground truth の真偽)．
///
/// 論文 4.3．true フレーミングは肯定方向，false フレーミングは否定方向へ
/// バイアスがかかる (フレーミング非対称性)．
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Framing {
    /// 真であるトピック (例: 地球は丸い)．
    True,
    /// 偽であるトピック (例: 地球平面説)．
    False,
}

impl Framing {
    pub fn label(&self) -> &'static str {
        match self {
            Framing::True => "true",
            Framing::False => "false",
        }
    }
}

/// 文字列から [`Framing`] をパースする．
pub fn parse_framing(s: &str) -> Result<Framing, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "true" | "t" => Ok(Framing::True),
        "false" | "f" => Ok(Framing::False),
        _ => Err(format!("不正なフレーミング: \"{}\" (true / false)", s)),
    }
}

// --------------------------------------------------------------------------- //
// メモリ方式
// --------------------------------------------------------------------------- //

/// メモリ更新方式 (論文 4.3)．
///
/// Phase 1/2 では cumulative (経験を逐次追記) を中心に扱う．reflective
/// (反省・要約で一定長を保つ; Park et al. 2023) は拡張点として列挙のみ用意する．
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryMode {
    /// 経験を逐次追記する累積メモリ．
    Cumulative,
    /// 反省・要約で一定長を保つメモリ (Phase 3 拡張点)．
    Reflective,
}

impl MemoryMode {
    pub fn label(&self) -> &'static str {
        match self {
            MemoryMode::Cumulative => "cumulative",
            MemoryMode::Reflective => "reflective",
        }
    }
}

/// 文字列から [`MemoryMode`] をパースする．
pub fn parse_memory(s: &str) -> Result<MemoryMode, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "cumulative" | "cum" => Ok(MemoryMode::Cumulative),
        "reflective" | "ref" => Ok(MemoryMode::Reflective),
        _ => Err(format!(
            "不正なメモリ方式: \"{}\" (cumulative / reflective)",
            s
        )),
    }
}

// --------------------------------------------------------------------------- //
// LLM 設定
// --------------------------------------------------------------------------- //

/// LLM レイヤの設定 (provider / model / temperature / seed / cache)．
///
/// プロバイダ優先順位は «Ollama 第一 → OpenAI フォールバック» 固定．モデル・
/// ホスト・API キーは環境変数で渡す (`OLLAMA_HOST` / `OLLAMA_MODEL` /
/// `OPENAI_API_KEY` / `OPENAI_MODEL`)．`temperature`/`seed` で擬似決定論化する．
#[derive(Debug, Clone)]
pub struct LlmSettings {
    /// 生成温度 (既定 0.0; 再現性のため．論文は 0.7)．
    pub temperature: f32,
    /// 生成シード (バックエンドへ渡す; Ollama は honour，OpenAI は best-effort)．
    pub seed: u64,
    /// プロンプト→応答キャッシュの保存先 (None なら in-memory)．
    pub cache_path: Option<String>,
}

impl Default for LlmSettings {
    fn default() -> Self {
        LlmSettings {
            temperature: 0.0,
            seed: 0,
            cache_path: None,
        }
    }
}

// --------------------------------------------------------------------------- //
// Config
// --------------------------------------------------------------------------- //

/// 単一実行の設定．
#[derive(Debug, Clone)]
pub struct Config {
    /// エージェント数 N．
    pub n_agents: usize,
    /// 議論トピック (ground truth 既知の短い記述)．
    pub topic: String,
    /// フレーミング (true / false)．
    pub framing: Framing,
    /// 確証バイアス (none / weak / strong)．
    pub bias: ConfirmationBias,
    /// メモリ方式 (cumulative / reflective)．
    pub memory_mode: MemoryMode,
    /// 相互作用するか (false = 非相互作用統制条件; Phase 3 拡張点)．
    pub interact: bool,
    /// トポロジ (full / ws / ba)．
    pub topology: Topology,
    /// WS の各ノードの初期次数 k (偶数)．
    pub ws_k: usize,
    /// WS の再配線確率 β．
    pub ws_beta: f64,
    /// BA の新規ノードあたりの結合数 m．
    pub ba_m: usize,
    /// 1 ステップあたりの dyadic interaction 数 (既定 1)．
    pub events_per_step: usize,
    /// 最大ステップ数 T．
    pub max_steps: usize,
    /// 収束判定の意見分散しきい値 (variance < tol で停止)．
    pub tol: f64,
    /// 乱数シード (None の場合はランダム; socsim コア層のみ支配)．
    pub seed: Option<u64>,
    /// LLM レイヤ設定．
    pub llm: LlmSettings,
    /// 結果出力ディレクトリ．
    pub output_dir: String,
}

impl Default for Config {
    /// 論文 §3 に近い標準設定 (N=10, T=100, full, none, false framing, cumulative)．
    fn default() -> Self {
        Config {
            n_agents: 10,
            topic: "flat_earth".to_string(),
            framing: Framing::False,
            bias: ConfirmationBias::None,
            memory_mode: MemoryMode::Cumulative,
            interact: true,
            topology: Topology::Full,
            ws_k: 4,
            ws_beta: 0.1,
            ba_m: 2,
            events_per_step: 1,
            max_steps: 100,
            tol: 1e-6,
            seed: Some(42),
            llm: LlmSettings::default(),
            output_dir: "results".to_string(),
        }
    }
}

/// `config.json` (run 用) のシリアライズ表現．
#[derive(Serialize)]
pub struct RunConfigJson {
    pub command: &'static str,
    pub n_agents: usize,
    pub topic: String,
    pub framing: String,
    pub bias: String,
    pub memory_mode: String,
    pub interact: bool,
    pub topology: String,
    pub ws_k: usize,
    pub ws_beta: f64,
    pub ba_m: usize,
    pub events_per_step: usize,
    pub max_steps: usize,
    pub tol: f64,
    pub seed: Option<u64>,
    pub llm_temperature: f32,
    pub llm_seed: u64,
    pub output_dir: String,
}

impl Config {
    /// `config.json` 用の表現を組み立てる．
    pub fn to_run_config_json(&self) -> RunConfigJson {
        RunConfigJson {
            command: "run",
            n_agents: self.n_agents,
            topic: self.topic.clone(),
            framing: self.framing.label().to_string(),
            bias: self.bias.label().to_string(),
            memory_mode: self.memory_mode.label().to_string(),
            interact: self.interact,
            topology: self.topology.label().to_string(),
            ws_k: self.ws_k,
            ws_beta: self.ws_beta,
            ba_m: self.ba_m,
            events_per_step: self.events_per_step,
            max_steps: self.max_steps,
            tol: self.tol,
            seed: self.seed,
            llm_temperature: self.llm.temperature,
            llm_seed: self.llm.seed,
            output_dir: self.output_dir.clone(),
        }
    }
}
