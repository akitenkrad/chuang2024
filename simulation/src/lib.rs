//! Chuang et al. (2024) "Simulating Opinion Dynamics with Networks of LLM-based
//! Agents" の再現実装ライブラリ．
//!
//! socsim フレームワーク上に構築した LLM 駆動の意見力学の公開 API を提供する．
//! 設定 (`config`)・世界状態 (`world`)・LLM クライアント層 (`llm`)・プロンプト
//! 生成 (`prompts`)・意見分類器 `f_oc` (`classifier`)・更新メカニズム
//! (`mechanisms`)・実行ドライバ (`simulation`)・集計メトリクス (`metrics`) を
//! モジュールとして公開し，バイナリ (`chuang`) と統合テストの双方から利用する．
//!
//! # 二層決定論
//!
//! socsim コア層 (ネットワーク・ペア選択・スケジューリング・メトリクス) は seed
//! から bit 単位で決定論的である．LLM レイヤ (ツイート/所感/メモリ生成) は socsim
//! の bit 再現性の **外側** にあり，`socsim-llm` のキャッシュ + `temperature=0` +
//! `seed` 固定で擬似決定論化する．詳細は `crate::llm` を参照．

pub mod classifier;
pub mod config;
pub mod llm;
pub mod mechanisms;
pub mod metrics;
pub mod prompts;
pub mod simulation;
pub mod world;
