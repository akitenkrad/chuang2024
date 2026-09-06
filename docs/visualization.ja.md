# 可視化

Python パッケージ `chuang-tools` (module `chuang_tools`) は Rust シミュレーションが書いた runvault の run ディレクトリを読み図を生成する．workspace ルートで一度 `uv sync` してから `uv run chuang-tools <サブコマンド>` で呼ぶ．

どの run を見るかは runvault が答える．`--results-dir` を省略すれば `runvault path --latest` に聞く — `results/` を走査して新しそうなディレクトリを当てにいくことはしない．図は run ディレクトリの *隣* (`results/chuang/figures/{run_slug}/`) に置く．`manifest.csv` は `finish()` が確定させたもので，後から足したものはハッシュを持たないためである．

## `visualize` — 単一実行

```bash
uv run chuang-tools visualize
uv run chuang-tools visualize --results-dir "$(runvault path --experiment chuang --latest --subcommand run --standalone)"
uv run chuang-tools visualize --output_dir out
```

run ディレクトリの `artifacts/opinions.csv` と `metrics.csv` を読み，`results/chuang/figures/{run_slug}/` へ出力する．long 形式の `metrics.csv` は `runvault.read.metrics_wide` で 1 ステップ 1 行に倒す:

- `opinion_trajectory.png` — 各エージェントの意見 `o ∈ {−2..2}` の時間推移．整数軌跡の重なりは微小な縦ジッタで分離する．収束は線が 1 水準へ畳まれる様子，分断は複数水準が残存する様子として現れる．
- `metrics_timeseries.png` — 3 パネル: 意見 **分散** (収束指標)・**Bias B** (意見平均; ゼロ破線付き — 真実極へのドリフト)・**Diversity D** (標準偏差 — 意見の広がり)．

## `visualize-sweep` — パラメータスイープ

```bash
uv run chuang-tools visualize-sweep
uv run chuang-tools visualize-sweep --sweep-dir "$(runvault path --experiment chuang --latest --subcommand sweep)"
```

スイープ親 run の子から 1 行 1 試行の表を組み直し (`runvault.read.sweep_events_table`: 条件は子の `parameters`，試行は `terminal` イベント)，`results/chuang/figures/{run_slug}/` へ出力する:

- `sweep_diversity_heatmap.png` — 試行平均した最終 **Diversity D** の，確証バイアス × トポロジ ヒートマップ．論文の核心は D が確証バイアス (`none → weak → strong`) で上昇すること．
- `sweep_bias_heatmap.png` — 最終 **Bias B** の確証バイアス × トポロジ ヒートマップ (0 中心の発散カラーマップ)．
- `sweep_bias_vs_confirmation.png` — 確証バイアス水準ごとの平均 D と平均 B の棒グラフ 2 枚．D はバイアスに対し単調増大するはず．

コンソールにも確証バイアス別の平均 Diversity `D̄` を表示する．

## `reproduce` — 論文の見出し的知見 + 図

```bash
# オフライン (LLM 不要): Rust reproduce を mock で実行してから図を描く
uv run chuang-tools reproduce --run --mock
uv run chuang-tools reproduce --run --mock --quick   # 高速スモーク
# 既存の reproduce ディレクトリを可視化
uv run chuang-tools reproduce --results-dir results/reproduce_20260530_000000
uv run chuang-tools reproduce --json                 # サマリを JSON 出力
```

reproduce の run から要約を組み直し (セル集約と条件別時系列は `metrics.csv`，アンカーの帯と判定は `events.jsonl` の `x.chuang2024.anchor` 行)，観測 vs 論文のアンカー表を表示しつつ `results/chuang/figures/{run_slug}/` へ出力する:

- `bias_control_matrix.png` — 確証バイアス `none / weak / strong` 上の最終 **Diversity D** と **Bias B** をグループ棒グラフで，`interaction` / `no-interaction` アームを並置．見出し的結果 (無バイアス→低 D 合意 / 強バイアス→高 D 断片化) と統制アーム (相互作用を外すと合意しない) を示す．
- `topology_comparison.png` — `full / er / ws / ba` の最終 **Diversity D** と **収束ステップ** (bias `none`, interaction)．
- `control_contrast.png` — 代表 run の **Diversity D** 時系列を，`none` / `strong` バイアスについて `interaction` vs `no-interaction` で重ね描き — 社会的影響 vs 固有ドリフトの時系列対比．

`--run` で新規結果を先に生成する (CI・サンドボックスでは `--mock` を併用しライブ LLM を回避)．省略すると既存ディレクトリを可視化する．

## `show-experiment-settings` — 設定と LLM メタデータ

```bash
uv run chuang-tools show-experiment-settings
uv run chuang-tools show-experiment-settings --results-dir "$(runvault path --experiment chuang --latest --subcommand run --standalone)"
uv run chuang-tools show-experiment-settings --json
```

実行条件 (`config.json` の `parameters`; どのサブコマンドかは `run.json` が答える) と，LLM メタデータ (モデル・provider・温度は `run.json` の `llm` ブロック，呼び出し総数・cache-hit・**cache-hit 率** は run スコープ指標) を整形表示する．legacy の `results/{timestamp}/` (flat な `config.json` / `sweep_config.json` / `run_metadata.json`) もそのまま読める．cache-hit 率は実用的な再現性シグナルである — ウォームキャッシュは同一応答を再生するので，再実行は高ヒット率を報告しライブ LLM 呼び出しはほぼゼロになる．

## 出力の解釈

- **真実収束 (バイアスなし).** 分散と多様性が減衰し，Bias `B` が真実極へドリフトする (`true` フレーミングで正，`false` フレーミングで負)．
- **分断 (強バイアス).** 分散と多様性が高止まりし，`n_clusters` > 1 となる．多様性ヒートマップは `strong` 行が最も明るい．
- **非相互作用統制.** `--control no-interaction` ではエージェントが近傍を見ないため意見が初期分布近くに留まり，多様性は縮小しない．相互作用アームと比べることで「網による意見変化」と「LLM 自身の prior によるドリフト」を分離できる．
- **トポロジ効果.** 疎・不均一なトポロジ (`er` / `ws` / `ba`) は全結合 `full` より収束が遅くクラスタが残りやすい．

絶対値は使用 LLM に依存する (ローカル `llama3.2` ≠ 論文の `gpt-3.5-turbo`) ため，厳密値ではなく符号と傾向を比較する．
