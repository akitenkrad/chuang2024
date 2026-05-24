//! LLM プロンプト生成 (ツイート発話 / 所感報告 / 意見分類)．
//!
//! 論文 4.3 の dyadic プロセスに対応する 3 種のプロンプトを組み立てる．プロンプト
//! はキャッシュキー (`hash(prompt + model)`) の素材になるため，同一状態からは
//! 同一プロンプト = 同一応答 (擬似決定論) になるよう決定論的に構築する．

use crate::config::{ConfirmationBias, Framing};
use crate::world::AgentState;

/// 意見値 `[-2,2]` を人間可読なスタンス語へ変換する (プロンプト埋め込み用)．
pub fn stance_word(opinion: i8) -> &'static str {
    match opinion {
        i8::MIN..=-2 => "strongly disagree",
        -1 => "disagree",
        0 => "neutral",
        1 => "agree",
        _ => "strongly agree",
    }
}

/// トピックとフレーミングから議論対象の主張文を作る (ground truth フレーミング)．
pub fn framing_statement(topic: &str, framing: Framing) -> String {
    let topic_h = topic.replace('_', " ");
    match framing {
        Framing::True => format!("The claim that '{topic_h}' is TRUE."),
        Framing::False => format!("The claim that '{topic_h}' is FALSE."),
    }
}

/// メモリ (直近数件) をプロンプト用の短い文字列に畳む．
fn memory_digest(memory: &[String], max_items: usize) -> String {
    if memory.is_empty() {
        return "(no prior memory)".to_string();
    }
    let start = memory.len().saturating_sub(max_items);
    memory[start..].join(" | ")
}

/// 話者プロンプト: 自分のスタンスを述べる短い「ツイート」を生成させる．
pub fn speaker_prompt(speaker: &AgentState, topic: &str, framing: Framing) -> String {
    let statement = framing_statement(topic, framing);
    format!(
        "You are a social-media user with this persona: {persona}.\n\
         Topic under discussion: {statement}\n\
         Your current stance is: {stance} (Likert {opinion} on a -2..2 scale).\n\
         Recent memory: {memory}\n\n\
         Write a single short tweet (max 30 words) stating your view on the topic. \
         Do not include hashtags or quotation marks.",
        persona = speaker.persona,
        statement = statement,
        stance = stance_word(speaker.opinion),
        opinion = speaker.opinion,
        memory = memory_digest(&speaker.memory, 3),
    )
}

/// 聴者プロンプト: 話者のツイートを読み，自分の所感を述べさせる．
///
/// 末尾で «-2..2 の整数 1 個» の出力を促し，分類器の規則ベース先読みを成立させる．
/// 確証バイアス指示を system 相当の前文として注入する．
pub fn listener_prompt(
    listener: &AgentState,
    speaker_text: &str,
    topic: &str,
    framing: Framing,
    bias: ConfirmationBias,
) -> String {
    let statement = framing_statement(topic, framing);
    format!(
        "You are a social-media user with this persona: {persona}.{bias_instr}\n\
         Topic under discussion: {statement}\n\
         Your current stance is: {stance} (Likert {opinion} on a -2..2 scale).\n\
         Recent memory: {memory}\n\n\
         You just read this tweet from another user: \"{speaker_text}\"\n\
         After reflecting on it, report your updated stance on the topic. \
         Answer with a SINGLE integer from -2 (strongly disagree) to 2 (strongly agree), \
         and nothing else.",
        persona = listener.persona,
        bias_instr = bias.instruction(),
        statement = statement,
        stance = stance_word(listener.opinion),
        opinion = listener.opinion,
        memory = memory_digest(&listener.memory, 3),
        speaker_text = speaker_text,
    )
}

/// 分類器プロンプト (`f_oc` のフォールバック): 自由文の所感を整数へ落とす．
pub fn classifier_prompt(topic: &str, framing_statement: &str, sentiment: &str) -> String {
    let topic_h = topic.replace('_', " ");
    format!(
        "Topic: {topic_h}. {framing_statement}\n\
         A person expressed this sentiment about the topic: \"{sentiment}\"\n\
         Classify their stance as a SINGLE integer from -2 (strongly disagree) to \
         2 (strongly agree). Answer with only the integer."
    )
}
