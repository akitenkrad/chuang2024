# CLI

The Rust binary is `chuang` (run via `cargo run --release -- …`). It has two subcommands: `run` and `sweep`. The `reproduce` subcommand (paper Table 1 / Fig. 4–6) is deferred to Phase 3.

## LLM environment variables

All LLM calls go through the **Ollama-first → OpenAI-fallback** client. Configure providers via environment variables (never hard-coded):

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server base URL (`/api/chat`). |
| `OLLAMA_MODEL` | `llama3.1` (socsim-llm default) | Ollama model; this project's quick-start uses `llama3.2:latest`. |
| `OPENAI_API_KEY` | (unset) | OpenAI key; if unset, the OpenAI fallback is a no-op placeholder (only an issue if Ollama also fails). |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model used on fallback. |

```bash
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=llama3.2:latest
```

## `run` — a single configuration

```bash
cargo run --release -- run \
    --n-agents 10 --topic flat_earth --framing false --bias none \
    --memory cumulative --topology full \
    --events-per-step 1 --max-steps 100 --tol 1e-6 --seed 42 \
    --temperature 0 --llm-seed 0 --cache-path .llm_cache/cache.json
```

| Flag | Default | Meaning |
|---|---|---|
| `--n-agents` | `10` | number of agents N |
| `--topic` | `flat_earth` | discussion topic (ground-truth known; underscores allowed) |
| `--framing` | `false` | `true` / `false` framing of the claim |
| `--bias` | `none` | confirmation bias: `none` / `weak` / `strong` |
| `--memory` | `cumulative` | memory mode: `cumulative` / `reflective` (reflective is a Phase 3 stub) |
| `--topology` | `full` | `full` (complete graph) / `ws` (Watts–Strogatz) / `ba` (Barabási–Albert) |
| `--ws-k` | `4` | WS initial degree k (even) |
| `--ws-beta` | `0.1` | WS rewiring probability β |
| `--ba-m` | `2` | BA edges per new node m |
| `--events-per-step` | `1` | dyadic interactions per tick (batch knob) |
| `--max-steps` | `100` | maximum steps T |
| `--tol` | `1e-6` | convergence threshold on opinion variance |
| `--seed` | random | core RNG seed (network + pair selection; deterministic core layer) |
| `--temperature` | `0.0` | LLM sampling temperature (paper uses 0.7; default 0 for reproducibility) |
| `--llm-seed` | `0` | LLM sampling seed (Ollama honours it; OpenAI best-effort) |
| `--cache-path` | `.llm_cache/cache.json` | prompt→response cache file (gitignored) |
| `--output-dir` | `results` | base output directory |

Outputs are written to `results/{timestamp}/` and `results/latest` is repointed:

- `config.json` — the full run configuration.
- `opinions.csv` — long format `t, agent_id, opinion[, text]`. The `text` column carries the final tweet per agent on the last step only (intermediate tweets are omitted to keep the file small).
- `metrics.csv` — `t, variance, bias, diversity, n_clusters, polarization`.
- `run_metadata.json` — LLM model / endpoint / temperature / seed / total calls / cache hits / cache-hit rate, plus the determinism note.

## `sweep` — confirmation-bias × framing × topology

```bash
cargo run --release -- sweep \
    --bias-values none,weak,strong \
    --framing-values true,false \
    --topology-values full,ws \
    --memory cumulative --n-agents 10 --runs 5 \
    --max-steps 100 --seed 42 \
    --cache-path .llm_cache/cache.json
```

| Flag | Default | Meaning |
|---|---|---|
| `--bias-values` | `none,weak,strong` | comma-separated confirmation-bias grid |
| `--framing-values` | `true,false` | comma-separated framing grid |
| `--topology-values` | `full` | comma-separated topology grid |
| `--memory` | `cumulative` | single memory mode for the whole sweep |
| `--topic` | `flat_earth` | discussion topic |
| `--n-agents` | `10` | number of agents N |
| `--runs` | `5` | independent trials per condition (each gets an independent derived seed) |
| `--events-per-step` | `1` | dyadic interactions per tick |
| `--max-steps` | `100` | maximum steps T |
| `--tol` | `1e-6` | convergence threshold |
| `--seed` | `42` | root seed (each condition/run derives an independent stream) |
| `--temperature` / `--llm-seed` | `0` / `0` | LLM generation knobs |
| `--cache-path` | `.llm_cache/cache.json` | shared cache across the whole sweep (raises the hit rate) |
| `--output-dir` | `results` | base output directory |

Outputs land in `results/{timestamp}_sweep/`:

- `sweep_summary.csv` — one row per `(bias, framing, topology, run)` with `final_bias`, `final_diversity`, `final_variance`, `n_clusters`, `polarization`, `convergence_time`, `converged`, `cache_hit_rate`.
- `sweep_config.json` — the sweep grid and shared parameters.

The console prints the per-bias mean diversity `D̄` — the paper's headline is that `D̄` increases monotonically `none → weak → strong`.

> **Cost note.** Each step issues two LLM calls (speaker + listener) plus an occasional classifier fallback. A sweep multiplies that by conditions × runs × steps. Keep the cache shared across the sweep so repeated prompts are free, and start with small `--n-agents` / `--max-steps` / `--runs`.

---
*This file was generated by Claude Code.*
