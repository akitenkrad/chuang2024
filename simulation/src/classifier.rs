//! 所感 → 意見の数値化 `f_oc` (論文 4.3)．
//!
//! 聴者の自然言語の所感 `r_j^t` を 5 段階リッカート意見
//! `o_j^t ∈ {-2,-1,0,1,2}` に変換する分類器である．2 段構えで実装する:
//!
//! 1. **規則ベースの先読み**: LLM へ «-2..2 の整数 1 個で答えよ» と指示するため，
//!    多くの応答は単一整数を含む．まず応答テキストから `-2..2` の整数トークンを
//!    抽出して即返す (LLM 追加呼び出し不要; キャッシュ消費ゼロ)．
//! 2. **LLM zero-shot 分類フォールバック**: 整数が読めない自由文の場合のみ，
//!    分類用プロンプトを LLM へ投げ，その応答をふたたび規則ベースで読む．この
//!    呼び出しもキャッシュ対象なので再実行はコスト 0．
//!
//! どちらの段でも最終的に読めなければ中立 `0` を返す (頑健性)．

use socsim_llm::LlmError;

use crate::config::LlmSettings;
use crate::llm::{llm_config, OpinionClient};
use crate::prompts;
use crate::world::{clamp_opinion, OPINION_MAX, OPINION_MIN};

/// 自由文テキストから `[-2, 2]` の整数意見を規則ベースで抽出する．
///
/// テキスト中に現れる最初の `-2..2` 範囲の整数トークン (符号付き) を返す．
/// 範囲外や非数値しか無ければ `None`．`+1` のような符号も解釈する．
pub fn parse_opinion(text: &str) -> Option<i8> {
    // トークンを走査し，符号付き整数として読める最初のトークンを採用する．
    // 句読点を境界として扱うため，数字・符号以外で区切る．
    let cleaned: String = text
        .chars()
        .map(|c| {
            if c.is_ascii_digit() || c == '-' || c == '+' {
                c
            } else {
                ' '
            }
        })
        .collect();
    for tok in cleaned.split_whitespace() {
        if let Ok(v) = tok.parse::<i64>() {
            if (OPINION_MIN as i64..=OPINION_MAX as i64).contains(&v) {
                return Some(v as i8);
            }
        }
    }
    None
}

/// 所感テキストを 5 段階意見へ数値化する (`f_oc`)．
///
/// まず規則ベースで読み，読めなければ LLM zero-shot 分類へフォールバックする．
/// それでも読めなければ中立 `0` を返す．LLM 呼び出しはキャッシュ対象．
pub fn classify_opinion(
    client: &mut OpinionClient,
    settings: &LlmSettings,
    topic: &str,
    framing_statement: &str,
    sentiment: &str,
) -> Result<i8, LlmError> {
    // 1. 規則ベースの先読み．
    if let Some(o) = parse_opinion(sentiment) {
        return Ok(clamp_opinion(o));
    }

    // 2. LLM zero-shot 分類フォールバック (キャッシュ対象)．
    let prompt = prompts::classifier_prompt(topic, framing_statement, sentiment);
    let resp = client.complete(&prompt, &llm_config(settings))?;
    Ok(parse_opinion(&resp.text).map(clamp_opinion).unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_integer() {
        assert_eq!(parse_opinion("2"), Some(2));
        assert_eq!(parse_opinion("-1"), Some(-1));
        assert_eq!(parse_opinion("+1"), Some(1));
    }

    #[test]
    fn parses_integer_in_sentence() {
        assert_eq!(parse_opinion("My stance is -2 on this."), Some(-2));
        assert_eq!(parse_opinion("I would say 0 overall."), Some(0));
    }

    #[test]
    fn rejects_out_of_range_and_text() {
        assert_eq!(parse_opinion("7"), None);
        assert_eq!(parse_opinion("no number here"), None);
    }
}
