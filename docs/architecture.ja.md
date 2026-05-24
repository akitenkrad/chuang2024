# アーキテクチャ

本プロジェクトは Chuang et al. (2024)「Simulating Opinion Dynamics with Networks of LLM-based Agents」の再現実装である．Cargo + uv のモノレポ構成で，Rust クレートがシミュレーションを実行し，Python パッケージが結果を可視化する．本コレクション初の **LLM 駆動** 再現である．

## リポジトリ構成

```
chuang2024/
├── Cargo.toml                  # [workspace] members = ["simulation"]
├── pyproject.toml              # uv workspace (members = ["tools"])
├── simulation/                 # Rust クレート `chuang-opinion-simulation` (bin `chuang`)
│   ├── Cargo.toml              # socsim git 依存: core / engine / net / llm (features=["live"])
│   ├── src/
│   │   ├── main.rs             # clap: run / sweep
│   │   ├── config.rs           # Config + 列挙型 (Topology / ConfirmationBias / Framing / MemoryMode / LlmSettings)
│   │   ├── world.rs            # OpinionWorld (WorldState) + AgentState (opinion / memory / trajectory)
│   │   ├── llm.rs              # 二層 LLM クライアントビルダ (Ollama→OpenAI フォールバック + キャッシュ)
│   │   ├── prompts.rs          # 話者 / 聴者 / 分類器プロンプト
│   │   ├── classifier.rs       # f_oc: 所感 → 意見 (規則ベース → LLM フォールバック)
│   │   ├── mechanisms.rs       # LLMOpinionUpdateMechanism (Interaction) + MetricsMechanism (PostStep)
│   │   ├── metrics.rs          # bias B / diversity D / n_clusters / polarization / convergence_time
│   │   ├── simulation.rs       # init_world + run ドライバ + 出力ライタ
│   │   └── lib.rs              # テスト用モジュール公開
│   ├── examples/mock_smoke.rs  # オフライン (ネットワーク不要) スモーク実行 (CI / サンドボックス用)
│   └── tests/integration_test.rs  # mock 駆動; ライブ LLM 不要
├── tools/                      # Python パッケージ `chuang-tools` (module `chuang_tools`)
│   └── src/chuang_tools/{cli,visualize,visualize_sweep,show_experiment_settings}.py
├── docs/                       # 本ドキュメント (バイリンガル)
└── results/                    # 実行時生成 (gitignore 対象)
```

## 二層決定論

中心的な設計制約 (socsim-mapping §10): LLM は非決定的なので，1 つのレイヤに閉じ込めて擬似決定論化する．

| レイヤ | 担当 | 再現性 |
|---|---|---|
| **決定論的 socsim コア** | ネットワーク生成・話者/聴者サンプリング (`ctx.rng`)・スケジューリング・メトリクス・収束判定 | seed 固定で bit 単位再現 (ChaCha20 `SimRng` + `derive_seed`) |
| **非決定的 LLM レイヤ** | ツイート生成・所感報告・意見分類 | `socsim-llm` のプロンプト→応答キャッシュ + `temperature=0` + `seed` 固定で擬似決定論化 |

RNG ストリーム (コア層のみ):

- `derive_seed(root, &[0])` → world-init RNG (ネットワーク生成・ペルソナ/初期意見割当)．
- `derive_seed(root, &[1])` → engine RNG (メカニズム内のペア一様サンプリング)．

LLM レイヤは `SimRng` の支配下に **ない**．その再現性はキャッシュに由来する: ウォームキャッシュでは同一プロンプトが同一応答を再生する．`run_metadata.json` にモデル・endpoint・温度・seed・cache-hit 率を記録し，実行が何と通信したかを明示する．

## LLM クライアント (`socsim-llm`)

任意クレート `socsim-llm` (feature `live` = `ollama` + `openai`) が部品を提供し，本プロジェクトは `src/llm.rs` でそれを合成する:

```
CachingClient< BoxedClient( FallbackClient< OllamaClient, OpenAiClient > ) >
```

- `FallbackClient` は primary (Ollama) を試み，**任意の** エラーで secondary (OpenAI) へフォールバックする．`socsim-llm` 提供であり自前実装しない．
- `CachingClient` は `PromptCache` (`hash(prompt+model)` → 応答; FNV-1a, JSON ファイル) を被せる．ミス時にキャッシュを更新するため `complete(&mut self, …)` は可変借用を取る．
- `BoxedClient` は `Box<dyn LlmClient>` に `LlmClient` を実装する小さな newtype で，同一の `OpinionClient` 型に本番の `FallbackClient` もテストの `mock::ScriptedClient` も載せられる．
- `OllamaClient::from_env()` は `OLLAMA_HOST` (既定 `http://localhost:11434`) / `OLLAMA_MODEL` (socsim-llm 既定は `llama3.1`; 本プロジェクト CLI は `llama3.2:latest` を既定にする) を読む．`OpenAiClient::from_env()` は `OPENAI_API_KEY` / `OPENAI_MODEL` を読む．

クライアントと `MetadataCollector` はメカニズムと run ドライバで `Rc<RefCell<…>>` 共有する．engine がボックス化メカニズムを所有するため，実行後にドライバがキャッシュ統計を読みキャッシュを保存する．

## WorldState とメカニズム

`OpinionWorld` は `socsim_net::SocialNetwork` と `BTreeMap<AgentId, AgentState>` (ソート済みキー → 決定論的 `agent_ids()`) を持つ．各 `AgentState` はペルソナ (テキスト)・意見 `i8 ∈ [−2,2]`・メモリ (`Vec<String>`)・意見軌跡・最終ツイートを持つ．`#[derive(Clone)]` でスナップショットと (将来の) 非相互作用統制条件に対応する．

トポロジ (`socsim-net` 生成器):

- `full` → `erdos_renyi(ids, 1.0, rng)` の完全グラフ (論文の全結合設定)．
- `ws` → `watts_strogatz(ids, k, beta, rng)`．
- `ba` → `barabasi_albert(ids, m, rng)`．

メカニズム (6 フェーズループ):

| メカニズム | フェーズ | 役割 |
|---|---|---|
| `LLMOpinionUpdateMechanism` | `Interaction` | 1 tick = `events_per_step` 回の dyadic interaction．`ctx.rng` で話者 + 聴者 (話者近傍から) を抽選し，話者がツイート (LLM) → 聴者が所感報告 (LLM) → `f_oc` で `i8` へ数値化 → 双方メモリ更新 → 聴者意見を更新．**LLM 呼び出しはすべてここに閉じる．** |
| `MetricsMechanism` | `PostStep` | 各エージェントの軌跡へ現在意見を追記; 意見分散を計算; 分散 `< tol` で `request_stop()`． |

`Interaction` を選ぶのは，更新が近傍拡散 (聴者が近傍のツイートを読んで変化する) であり，孤立した `Decision` ではなく有界信頼 / DeGroot 更新の LLM 類似物だからである．

## メトリクス

各ステップで意見ベクトル `F_o^t` 上で計算する (`metrics.rs`):

- **bias B** — 意見平均 (論文の `B = mean(F_o^T)`)．
- **diversity D** — 意見分布の標準偏差 (`D = std(F_o^T)`)．
- **variance** — 意見分散 (収束指標)．
- **n_clusters** — 占有された相異なる意見水準の数 (分断量)．
- **polarization** — 意見半径 (2) で正規化した `|opinion|` の平均，∈ `[0,1]`．
- **convergence_time** — 分散 `< tol` となる最初のステップ (sweep サマリで計算)．

`framing_asymmetry` (true/false フレーミング間の `B` の符号差) は条件間比較なので Phase 3 (`reproduce`) に委ねる．

## socsim 基盤

[socsim](https://github.com/akitenkrad/rs-social-simulation-tools) (ライブラリモード, git 依存, `branch = "main"`, `Cargo.lock` で固定):

- `socsim-core` — `WorldState` / `Mechanism` / `Phase` / `StepContext` / `AgentId` / `SimClock` / `SimRng` / `derive_seed`．
- `socsim-engine` — `SimulationBuilder` / `Simulation::run_observed` / `SequentialScheduler`．
- `socsim-net` — `SocialNetwork` と `erdos_renyi` / `watts_strogatz` / `barabasi_albert` 生成器，`neighbors`．
- `socsim-llm` (任意, `features = ["live"]`) — `LlmClient` / `OllamaClient` / `OpenAiClient` / `FallbackClient` / `CachingClient` / `PromptCache` / `LlmConfig` / `CallMetadata` / `MetadataCollector` / `mock::ScriptedClient`．

## 参考文献

- Chuang, Y.-S., et al. (2024). *Simulating Opinion Dynamics with Networks of LLM-based Agents.* Findings of ACL: NAACL 2024, 3326–3346. arXiv:2311.09618.
- Park, J. S., et al. (2023). *Generative Agents: Interactive Simulacra of Human Behavior.* UIST 2023. (reflective メモリ)
- Hegselmann, R., & Krause, U. (2002). *Opinion Dynamics and Bounded Confidence.* JASSS 5(3). (`Interaction` 更新が対応づく有界信頼モデル)

---
*This file was generated by Claude Code.*
