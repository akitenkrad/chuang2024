//! オフライン (LLM 不要) 再現用のスクリプト化クライアント．
//!
//! 論文 (Chuang et al. 2024) の **見出し的知見** を，ライブ LLM 無しで構造的に
//! 再現するための決定論的 mock を提供する．`reproduce` サブコマンドと
//! `run --mock`，および各種テストがこの mock を共用する．
//!
//! 再現する定性的挙動 (論文 4.3 / Fig.4-6):
//! - **確証バイアス無し → 真値方向への合意ドリフト**: LLM エージェントは
//!   フレーミング (TRUE/FALSE) が指す «真» の方向 (TRUE→+2, FALSE→-2) へ
//!   1 段ずつスタンスを寄せる帰納的バイアスを持つ．時間とともに Diversity D が
//!   縮小し，Bias B が真値方向の極へ寄る (合意)．
//! - **強い確証バイアス → 断片化 (合意崩壊)**: 自分の現在スタンスを保持し，
//!   読んだツイートに引きずられない．初期意見分布が温存され Diversity D が
//!   高止まりする (fragmentation)．
//! - **弱い確証バイアス → 中間**: 真値方向へ寄るが，自分のスタンスから 1 段を
//!   超えては動かない (緩やかな合意)．
//!
//! この mock は ground-truth LLM ではなく，論文の定性的結論を再現するための
//! «帰納的バイアスの戯画» である．プロンプト文字列から «現在の Likert»・
//! «フレーミング»・«確証バイアス強度» を読み取って次スタンスを決める．
//! ライブ llama3.2 ではこの戯画ではなく実モデルの応答を用いる (cache 経由)．

use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;

use crate::config::ConfirmationBias;
use crate::llm::{wrap_client, OpinionClient};

/// 確証バイアス指示文 (config.rs の `ConfirmationBias::instruction`) の検出片．
const WEAK_BIAS_MARK: &str = "mild tendency to favour information that agrees";
const STRONG_BIAS_MARK: &str = "strongly prefer information that confirms";

/// 聴者プロンプトを判別するためのマーカ (prompts.rs と一致させる)．
const LISTENER_MARK: &str = "Answer with a SINGLE integer";

/// プロンプトから «現在の Likert スタンス» を読み取る．
///
/// prompts.rs は «(Likert {opinion} on a -2..2 scale)» を埋め込むので，
/// `"Likert "` 直後の整数を解析する．読めなければ 0 (中立) を返す．
fn parse_current_stance(prompt: &str) -> i8 {
    if let Some(idx) = prompt.find("Likert ") {
        let rest = &prompt[idx + "Likert ".len()..];
        let token: String = rest
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '-')
            .collect();
        if let Ok(v) = token.parse::<i8>() {
            return v.clamp(-2, 2);
        }
    }
    0
}

/// プロンプトのフレーミング (TRUE/FALSE) から «真値方向» を返す (+1 / -1)．
///
/// framing_statement は «is TRUE.» / «is FALSE.» を含む．TRUE は +2 方向，
/// FALSE は -2 方向が «真» (= LLM の帰納的バイアスが向かう先)．
fn truth_direction(prompt: &str) -> i8 {
    if prompt.contains("is TRUE") {
        1
    } else {
        -1
    }
}

/// プロンプトから確証バイアス強度を読み取る．
fn detect_bias(prompt: &str) -> ConfirmationBias {
    if prompt.contains(STRONG_BIAS_MARK) {
        ConfirmationBias::Strong
    } else if prompt.contains(WEAK_BIAS_MARK) {
        ConfirmationBias::Weak
    } else {
        ConfirmationBias::None
    }
}

/// 聴者プロンプトに対する «更新後スタンス整数» を決める (mock の中核ロジック)．
///
/// - none: 真値方向へ 1 段ドリフト (合意へ)．
/// - weak: 真値方向へ 1 段ドリフトするが現在値から ±1 まで (緩やか)．none と
///   同じ 1 段だが，すでに極にある場合は動かない点で挙動が分かれる．
/// - strong: 現在スタンスを保持 (断片化; 読んだツイートに引きずられない)．
pub fn reproduce_listener_reply(prompt: &str) -> i8 {
    let current = parse_current_stance(prompt);
    let dir = truth_direction(prompt);
    match detect_bias(prompt) {
        ConfirmationBias::Strong => current,
        ConfirmationBias::Weak => (current + dir).clamp(-2, 2),
        ConfirmationBias::None => (current + dir).clamp(-2, 2),
    }
}

/// 再現用の決定論的スクリプトクライアントを構築する (in-memory cache)．
///
/// 聴者プロンプトには [`reproduce_listener_reply`] の整数を，話者プロンプトには
/// 固定の短文ツイートを返す．`weak` を `strong` と区別するため，実際の挙動差は
/// «極にいるエージェントが動くか» で現れる (none/weak は 1 段，strong は不動)．
pub fn build_reproduce_client() -> OpinionClient {
    let backend = ScriptedClient::new("mock-reproduce", |prompt: &str| {
        if prompt.contains(LISTENER_MARK) {
            reproduce_listener_reply(prompt).to_string()
        } else {
            "Sharing my honest view on the topic.".to_string()
        }
    });
    wrap_client(backend, PromptCache::in_memory())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_negative_and_positive_stance() {
        assert_eq!(parse_current_stance("(Likert -2 on a -2..2 scale)"), -2);
        assert_eq!(parse_current_stance("(Likert 1 on a -2..2 scale)"), 1);
        assert_eq!(parse_current_stance("(Likert 0 on a -2..2 scale)"), 0);
        // 読めなければ中立．
        assert_eq!(parse_current_stance("no likert here"), 0);
    }

    #[test]
    fn truth_direction_follows_framing() {
        assert_eq!(truth_direction("The claim that 'x' is TRUE."), 1);
        assert_eq!(truth_direction("The claim that 'x' is FALSE."), -1);
    }

    #[test]
    fn none_bias_drifts_toward_truth() {
        // TRUE フレーミング・バイアス無し・現在 -1 → +1 段ドリフトで 0．
        let p = "Likert -1 on a -2..2 scale ... is TRUE ... Answer with a SINGLE integer";
        assert_eq!(reproduce_listener_reply(p), 0);
    }

    #[test]
    fn strong_bias_holds_stance() {
        let p = format!(
            "Likert -1 on a -2..2 scale ... is TRUE ...{} ... Answer with a SINGLE integer",
            STRONG_BIAS_MARK
        );
        assert_eq!(reproduce_listener_reply(&p), -1);
    }

    #[test]
    fn drift_saturates_at_extreme() {
        // TRUE・現在 +2 → クランプで +2 のまま．
        let p = "Likert 2 on a -2..2 scale ... is TRUE ... Answer with a SINGLE integer";
        assert_eq!(reproduce_listener_reply(p), 2);
    }
}
