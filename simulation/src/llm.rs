//! LLM クライアント層 (Ollama 第一 → OpenAI フォールバック + キャッシュ)．
//!
//! 本モジュールは `socsim-llm` の合成 API に対する薄いビルダである．二層
//! アーキテクチャの **上層 (非決定的 LLM レイヤ)** をここに閉じ込め，下層の
//! 決定論的 socsim コアからは [`OpinionClient`] 型エイリアス経由でのみ触れる．
//!
//! # 合成 (Ollama 第一 → OpenAI フォールバック → キャッシュ)
//!
//! ```text
//! CachingClient< FallbackClient< OllamaClient, OpenAiClient > >
//!   └─ cache: PromptCache (prompt → response; 擬似決定論の本体)
//!      └─ primary:   OllamaClient   (OLLAMA_HOST / OLLAMA_MODEL)
//!         secondary: OpenAiClient   (OPENAI_API_KEY / OPENAI_MODEL)
//! ```
//!
//! `FallbackClient` は socsim-llm が提供する (自前実装しない)．「Ollama を試行
//! → 任意のエラーで OpenAI へフォールバック」を担う．`CachingClient` はその上に
//! プロンプト→応答キャッシュを被せ，`temperature=0` / `seed` 固定と合わせて
//! 再実行を擬似決定論化する．キャッシュは `socsim-llm` の features=["live"]
//! 経由で有効化される (Ollama + OpenAI 両バックエンド)．
//!
//! テストでは `socsim-llm::mock::ScriptedClient` を `Box<dyn LlmClient>` として
//! 同じ [`OpinionClient`] に流し込めるよう，[`LlmClient`] を `Box<dyn LlmClient>`
//! に対しても実装する (object-safe)．

use socsim_llm::{
    CachingClient, FallbackClient, LlmClient, LlmConfig, LlmError, LlmResponse, OllamaClient,
    OpenAiClient, PromptCache,
};

use crate::config::LlmSettings;

/// `Box<dyn LlmClient>` を [`LlmClient`] として扱えるようにする newtype ラッパ．
///
/// 孤児規則 (orphan rule) のため `Box<dyn LlmClient>` 自体には直接実装できないので
/// 自前の newtype に転送実装する．これにより本番の
/// `FallbackClient<OllamaClient, OpenAiClient>` も，テスト用の
/// `mock::ScriptedClient` も，同一の `CachingClient<BoxedClient>`
/// ([`OpinionClient`]) に統一できる．
pub struct BoxedClient(pub Box<dyn LlmClient>);

impl LlmClient for BoxedClient {
    fn model(&self) -> &str {
        self.0.model()
    }
    fn endpoint(&self) -> &str {
        self.0.endpoint()
    }
    fn complete(&self, prompt: &str, config: &LlmConfig) -> Result<LlmResponse, LlmError> {
        self.0.complete(prompt, config)
    }
}

/// 本シミュレーションが用いるキャッシュ付きクライアント型．
///
/// バックエンドは [`BoxedClient`] に型消去してあり，本番は
/// `FallbackClient<OllamaClient, OpenAiClient>`，テストは `ScriptedClient` を
/// 注入できる．`complete(&mut self, …)` はキャッシュ更新のため可変借用を取る．
pub type OpinionClient = CachingClient<BoxedClient>;

/// 本番用の «Ollama 第一 → OpenAI フォールバック + キャッシュ» クライアントを
/// 環境変数から構築する．
///
/// - Ollama: `OLLAMA_HOST` (既定 `http://localhost:11434`) / `OLLAMA_MODEL`
///   (既定 `llama3.2:latest`)．
/// - OpenAI: `OPENAI_API_KEY` / `OPENAI_MODEL` (既定 `gpt-4o-mini`)．未設定なら
///   空キーのフォールバックを置く (Ollama が成功すれば呼ばれない; 両方失敗時のみ
///   設定エラーになる)．
/// - キャッシュ: `settings.cache_path` があればその JSON ファイル，なければ
///   in-memory．
pub fn build_live_client(settings: &LlmSettings) -> Result<OpinionClient, LlmError> {
    let ollama = OllamaClient::from_env();
    // OPENAI_API_KEY が無い環境でも Ollama 単独で動かせるよう，from_env が失敗
    // した場合は空キーのプレースホルダを置く (Ollama 失敗時のみ Config エラー)．
    let openai = OpenAiClient::from_env().unwrap_or_else(|_| {
        let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into());
        OpenAiClient::new("", model)
    });

    let fallback = FallbackClient::new(ollama, openai);
    let backend = BoxedClient(Box::new(fallback));

    let cache = match &settings.cache_path {
        Some(path) => PromptCache::open(path)?,
        None => PromptCache::in_memory(),
    };
    Ok(CachingClient::new(backend, cache))
}

/// 任意の [`LlmClient`] (例: `mock::ScriptedClient`) をキャッシュで包んだ
/// [`OpinionClient`] を作る (主にテスト用)．
pub fn wrap_client<C: LlmClient + 'static>(backend: C, cache: PromptCache) -> OpinionClient {
    let boxed = BoxedClient(Box::new(backend));
    CachingClient::new(boxed, cache)
}

/// [`LlmSettings`] から socsim-llm の [`LlmConfig`] を組み立てる．
pub fn llm_config(settings: &LlmSettings) -> LlmConfig {
    LlmConfig::deterministic()
        .with_temperature(settings.temperature)
        .with_seed(settings.seed)
}
