# Visualization

The Python package `chuang-tools` (module `chuang_tools`) reads the runvault run directories the Rust simulation writes and produces figures. Install it once at the workspace root with `uv sync`, then invoke subcommands with `uv run chuang-tools <subcommand>`.

Which run to read is answered by runvault: omit `--results-dir` and the tools call `runvault path --latest`. They never scan `results/` for the newest-looking directory. Figures are written *beside* the run (`results/chuang/figures/{run_slug}/`), never inside it — `manifest.csv` is settled by `finish()`, so anything added afterwards would carry no hash.

## `visualize` — a single run

```bash
uv run chuang-tools visualize
uv run chuang-tools visualize --results-dir "$(runvault path --experiment chuang --latest --subcommand run --standalone)"
uv run chuang-tools visualize --output-dir out
```

Reads `artifacts/opinions.csv` and `metrics.csv` from the run directory and writes to `results/chuang/figures/{run_slug}/`. The long `metrics.csv` is turned back into one row per step by `runvault.read.metrics_wide`:

- `opinion_trajectory.png` — each agent's opinion `o ∈ {−2..2}` over time. A small vertical jitter separates overlapping integer trajectories. Convergence shows as the lines collapsing onto one level; fragmentation shows as several persistent levels.
- `metrics_timeseries.png` — three panels: opinion **variance** (convergence indicator), **Bias B** (mean opinion, with a dashed zero line — drift toward the truthful pole), and **Diversity D** (standard deviation — opinion spread).

## `visualize-sweep` — a parameter sweep

```bash
uv run chuang-tools visualize-sweep
uv run chuang-tools visualize-sweep --sweep-dir "$(runvault path --experiment chuang --latest --subcommand sweep)"
```

Rebuilds the one-row-per-trial table from the sweep parent's children (`runvault.read.sweep_events_table`: each child's `parameters` for the condition, its `terminal` events for the trials) and writes to `results/chuang/figures/{run_slug}/`:

- `sweep_diversity_heatmap.png` — final **Diversity D** averaged over runs, as a confirmation-bias × topology heatmap. The paper's headline finding is that D rises with confirmation bias (`none → weak → strong`).
- `sweep_bias_heatmap.png` — final **Bias B** as a confirmation-bias × topology heatmap (diverging colormap centred at 0).
- `sweep_bias_vs_confirmation.png` — two bar charts: mean D and mean B per confirmation-bias level. D should increase monotonically with bias.

The console also prints the per-bias mean diversity `D̄`.

## `reproduce` — the paper's headline findings + figures

```bash
# Offline (no LLM): run the Rust reproduce with the mock, then draw the figures
uv run chuang-tools reproduce --run --mock
uv run chuang-tools reproduce --run --mock --quick   # fast smoke
# Visualize an existing reproduce directory
uv run chuang-tools reproduce --results-dir "$(runvault path --experiment chuang --latest --subcommand reproduce)"
uv run chuang-tools reproduce --json                 # print the summary as JSON
```

Rebuilds the summary from the reproduce run — the cell aggregates and the per-condition time series from `metrics.csv`, the anchor bands and verdicts from the `x.chuang2024.anchor` rows of `events.jsonl` — prints the observed-vs-paper anchor table, and writes to `results/chuang/figures/{run_slug}/`:

- `bias_control_matrix.png` — final **Diversity D** and **Bias B** as grouped bars over confirmation bias `none / weak / strong`, with the `interaction` and `no-interaction` arms side by side. Shows the headline result (no bias → low D consensus; strong bias → high D fragmentation) and the control arm (no consensus when interaction is removed).
- `topology_comparison.png` — final **Diversity D** and **convergence step** across `full / er / ws / ba` (bias `none`, interaction).
- `control_contrast.png` — the **Diversity D** time series of the representative run, overlaying `interaction` vs `no-interaction` for the `none` and `strong` bias levels — the social-influence-vs-intrinsic-drift contrast over time.

Add `--run` to generate fresh results first (with `--mock` in CI/sandboxes to avoid live LLM); omit it to visualize an existing directory.

## `show-experiment-settings` — settings & LLM metadata

```bash
uv run chuang-tools show-experiment-settings
uv run chuang-tools show-experiment-settings --results-dir "$(runvault path --experiment chuang --latest --subcommand run --standalone)"
uv run chuang-tools show-experiment-settings --json
```

Renders the run configuration (`config.json`, under `parameters`) — which subcommand it belongs to is answered by `run.json` — and the LLM metadata: model, provider and temperature from the `llm` block of `run.json`, and total calls, cache hits and **cache-hit rate** from the run-scope metrics. Legacy `results/{timestamp}/` directories with a flat `config.json` / `sweep_config.json` / `run_metadata.json` still read. The cache-hit rate is the practical reproducibility signal — a warm cache replays identical responses, so a re-run reports a high hit rate and issues few/zero live LLM calls.

## Interpreting the outputs

- **Truthful consensus (no bias).** Variance and diversity decay; bias `B` drifts toward the truthful pole (positive under `true` framing, negative under `false` framing).
- **Fragmentation (strong bias).** Variance and diversity stay high; `n_clusters` > 1; the diversity heatmap is brightest in the `strong` row.
- **Non-interaction control.** With `--control no-interaction` the agents never see neighbours, so opinions stay near their initial spread — diversity does not collapse. Comparing it against the interaction arm separates "opinions change because of the network" from "opinions drift because of the LLM's own prior".
- **Topology effect.** Sparse/heterogeneous topologies (`er` / `ws` / `ba`) tend to converge more slowly and retain more clusters than the all-to-all `full` graph.

Note that absolute numbers depend on the LLM used (local `llama3.2` ≠ the paper's `gpt-3.5-turbo`); compare signs and trends, not exact values.
