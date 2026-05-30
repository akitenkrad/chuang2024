<p align="center"><img src="docs/assets/hero.svg" width="100%"></p>

**English** | [日本語](README.ja.md)

# Simulating Opinion Dynamics with Networks of LLM-based Agents — Chuang et al. (2024)

A reimplementation of the LLM-agent opinion-dynamics model of Chuang et al. (2024), "Simulating Opinion Dynamics with Networks of LLM-based Agents" (*Findings of ACL: NAACL 2024*, 3326–3346; arXiv:2311.09618). A population of LLM-driven agents sits on a network; each step a speaker–listener pair is sampled, the speaker emits a short "tweet" stating its stance, the listener reviews it and reports an updated stance, and an opinion classifier `f_oc` maps that stance onto a 5-point Likert scale `o ∈ {−2,−1,0,1,2}`. The paper's headline finding is a strong intrinsic bias toward *truthful* consensus that breaks down — fragmenting opinions — once a confirmation bias is injected. This is the first **LLM-driven** replication in this collection: the deterministic [socsim](https://github.com/akitenkrad/rs-social-simulation-tools) core handles the network, pair selection, scheduling and metrics, while the non-deterministic LLM layer is confined to one mechanism and pseudo-determinised via the optional `socsim-llm` crate (prompt→response cache + `temperature=0` + fixed seed).

## Two-layer determinism (read this first)

LLM output is **outside** socsim's bit-reproducibility. The design therefore splits into two layers:

- **Deterministic socsim core** — network generation, speaker/listener sampling (`ctx.rng`, ChaCha20), scheduling, metrics and convergence. Given a seed this reproduces bit-for-bit.
- **Non-deterministic LLM layer** — tweet generation, sentiment report and opinion classification. Pseudo-determinised by `socsim-llm`'s `CachingClient` (a `hash(prompt+model)` → response cache), `temperature=0` and a fixed seed. The provider order is **Ollama first → OpenAI fallback** via `socsim-llm`'s `FallbackClient`.

The cache — not the model — is the reproducibility mechanism: a warm cache replays identical responses, so a rerun is free and stable. Each run writes `run_metadata.json` recording the model, endpoint, temperature, seed and cache-hit rate. Because the local default model (`llama3.2`) differs from the paper's `gpt-3.5-turbo`, reproduction targets are **qualitative** (consensus tendency, sign of the bias `B`, monotone increase of diversity `D` with confirmation bias), not exact numbers.

## Install & Quick start

```bash
# Build the Rust simulation (fetches socsim incl. socsim-llm with the Ollama+OpenAI backends)
cargo build --release

# Make sure a local Ollama is running and a model is pulled, e.g.:
#   ollama pull llama3.2:latest
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=llama3.2:latest
# Optional OpenAI fallback:
#   export OPENAI_API_KEY=sk-...   OPENAI_MODEL=gpt-4o-mini

# Run a small simulation (full graph, no confirmation bias, false framing)
cargo run --release -- run --n-agents 10 --topology full --bias none --framing false --max-steps 100 --seed 42

# Install the Python visualization tools (at the workspace root)
uv sync

# Visualize the most recent run (opinion trajectory + B/D/variance time series)
uv run chuang-tools visualize

# Inspect the run's settings and LLM metadata
uv run chuang-tools show-experiment-settings --results-dir results/latest
```

## Documentation

- [Use cases](docs/usecases.md) — what you can do with this project, with pointers to the rest of the docs.
- [CLI](docs/cli.md) — the Rust CLI: the `run` and `sweep` subcommands and their flags, plus the LLM environment variables.
- [Visualization](docs/visualization.md) — the Python `chuang-tools` and how to interpret the outputs.
- [Architecture](docs/architecture.md) — repository structure, the two-layer determinism, the socsim/`socsim-llm` framework, the mechanism, the metrics, and references.

## Scope

This repository implements the core dyadic LLM opinion-update model on a network, the two-layer LLM client (Ollama→OpenAI fallback + caching), and opinion-convergence metrics, exposed through three Rust subcommands and a Python tool suite:

- `run` — a single configuration, with a `--control no-interaction` arm (agents evolve in isolation, never seeing neighbours) and an offline `--mock` mode.
- `sweep` — a grid over confirmation-bias × framing × topology (`full` / `er` / `ws` / `ba`).
- `reproduce` — a one-command reproduction of the paper's headline findings: the bias × control matrix (no bias → truthful consensus; strong bias → fragmentation; the non-interaction control isolating social influence from the LLM's intrinsic drift) and a topology comparison, with observed-vs-paper anchors written to `reproduce_summary.json`.
- Python `chuang-tools`: `visualize` / `visualize-sweep` / `show-experiment-settings` / `reproduce` (renders the reproduction figures).

The opinion classifier supports a `reflective` memory mode label; the current update path uses the `cumulative` memory documented above. Reflective-memory summarisation is a clean extension point.

## License

MIT

---
*This file was generated by Claude Code.*
