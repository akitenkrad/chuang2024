# Architecture

This project replicates Chuang et al. (2024), "Simulating Opinion Dynamics with Networks of LLM-based Agents". It is a Cargo + uv monorepo: a Rust crate runs the simulation, and a Python package visualizes the results. It is the first **LLM-driven** replication in this collection.

## Repository structure

```
chuang2024/
├── Cargo.toml                  # [workspace] members = ["simulation"]
├── pyproject.toml              # uv workspace (members = ["tools"])
├── simulation/                 # Rust crate `chuang-opinion-simulation` (bin `chuang`)
│   ├── Cargo.toml              # socsim git deps: core / engine / net / llm (features=["live"])
│   ├── src/
│   │   ├── main.rs             # clap: run / sweep
│   │   ├── config.rs           # Config + enums (Topology / ConfirmationBias / Framing / MemoryMode / LlmSettings)
│   │   ├── world.rs            # OpinionWorld (WorldState) + AgentState (opinion / memory / trajectory)
│   │   ├── llm.rs              # two-layer LLM client builder (Ollama→OpenAI fallback + cache)
│   │   ├── prompts.rs          # speaker / listener / classifier prompts
│   │   ├── classifier.rs       # f_oc: sentiment → opinion (rule-based → LLM fallback)
│   │   ├── mechanisms.rs       # LLMOpinionUpdateMechanism (Interaction) + MetricsMechanism (PostStep)
│   │   ├── metrics.rs          # bias B / diversity D / n_clusters / polarization / convergence_time
│   │   ├── simulation.rs       # init_world + run driver + output writers
│   │   └── lib.rs              # module exports for tests
│   ├── examples/mock_smoke.rs  # offline (no-network) smoke run for CI / sandboxes
│   └── tests/integration_test.rs  # mock-driven; needs no live LLM
├── tools/                      # Python package `chuang-tools` (module `chuang_tools`)
│   └── src/chuang_tools/{cli,visualize,visualize_sweep,show_experiment_settings}.py
├── docs/                       # this documentation (bilingual)
└── results/                    # runtime output (gitignored)
```

## Two-layer determinism

The central design constraint (socsim-mapping §10): an LLM is non-deterministic, so it must be confined to one layer and pseudo-determinised.

| Layer | What it owns | Reproducibility |
|---|---|---|
| **Deterministic socsim core** | network generation, speaker/listener sampling via `ctx.rng`, scheduling, metrics, convergence | bit-for-bit given the seed (ChaCha20 `SimRng` + `derive_seed`) |
| **Non-deterministic LLM layer** | tweet generation, sentiment report, opinion classification | pseudo-determinised by `socsim-llm`'s prompt→response cache + `temperature=0` + fixed `seed` |

RNG streams (core layer only):

- `derive_seed(root, &[0])` → world-init RNG (network generation, persona/initial-opinion assignment).
- `derive_seed(root, &[1])` → engine RNG (speaker/listener pair sampling inside the mechanism).

The LLM layer is **not** under `SimRng`. Its reproducibility comes entirely from the cache: with a warm cache, an identical prompt replays an identical response. `run_metadata.json` records model / endpoint / temperature / seed / cache-hit rate so a run logs exactly what it talked to.

## The LLM client (`socsim-llm`)

The optional `socsim-llm` crate (feature `live` = `ollama` + `openai`) provides the building blocks; this project composes them in `src/llm.rs`:

```
CachingClient< BoxedClient( FallbackClient< OllamaClient, OpenAiClient > ) >
```

- `FallbackClient` tries the primary (Ollama) and, on **any** error, falls back to the secondary (OpenAI). This is provided by `socsim-llm` — we do not hand-roll it.
- `CachingClient` wraps it with a `PromptCache` (`hash(prompt+model)` → response, FNV-1a, JSON-file-backed). Its `complete(&mut self, …)` takes a mutable borrow because a miss updates the cache.
- `BoxedClient` is a small newtype implementing `LlmClient` for `Box<dyn LlmClient>`, so the same `OpinionClient` type can carry either the live `FallbackClient` (production) or a `mock::ScriptedClient` (tests / `mock_smoke`).
- `OllamaClient::from_env()` reads `OLLAMA_HOST` (default `http://localhost:11434`) / `OLLAMA_MODEL` (default in `socsim-llm` is `llama3.1`; this project's CLI defaults `OLLAMA_MODEL` to `llama3.2:latest`). `OpenAiClient::from_env()` reads `OPENAI_API_KEY` / `OPENAI_MODEL`.

The client and a `MetadataCollector` are shared between the mechanism and the run driver via `Rc<RefCell<…>>`, because the engine owns the boxed mechanisms; after the run the driver reads the cache stats and saves the cache.

## WorldState and the mechanism

`OpinionWorld` holds a `socsim_net::SocialNetwork` and a `BTreeMap<AgentId, AgentState>` (sorted keys → deterministic `agent_ids()`). Each `AgentState` carries a persona (text), an opinion `i8 ∈ [−2,2]`, a memory (`Vec<String>`), an opinion trajectory, and the last tweet. `#[derive(Clone)]` supports snapshotting and the (future) non-interacting control condition.

Topology (`socsim-net` generators):

- `full` → complete graph via `erdos_renyi(ids, 1.0, rng)` (the paper's all-to-all setting).
- `ws` → `watts_strogatz(ids, k, beta, rng)`.
- `ba` → `barabasi_albert(ids, m, rng)`.

Mechanisms (six-phase loop):

| Mechanism | Phase | Role |
|---|---|---|
| `LLMOpinionUpdateMechanism` | `Interaction` | one tick = `events_per_step` dyadic interactions. Sample speaker + listener (from the speaker's neighbours) via `ctx.rng`; speaker tweets (LLM); listener reviews and reports a stance (LLM); `f_oc` numericises it to `i8`; both update memory; the listener's opinion is updated. **All LLM calls live here.** |
| `MetricsMechanism` | `PostStep` | append each agent's current opinion to its trajectory; compute opinion variance; `request_stop()` when variance `< tol`. |

`Interaction` is chosen because the update is neighbour diffusion (a listener changes after reading a neighbour's tweet) — the LLM analogue of bounded-confidence / DeGroot updating, not an isolated `Decision`.

## Metrics

Computed every step over the opinion vector `F_o^t` (see `metrics.rs`):

- **bias B** — mean opinion (the paper's `B = mean(F_o^T)`).
- **diversity D** — standard deviation of the opinion distribution (`D = std(F_o^T)`).
- **variance** — opinion variance (convergence indicator).
- **n_clusters** — number of distinct occupied opinion levels (fragmentation count).
- **polarization** — mean `|opinion|` normalised by the opinion radius (2), in `[0,1]`.
- **convergence_time** — first step at which variance `< tol` (computed in the sweep summary).

`framing_asymmetry` (sign difference of `B` between true/false framing) is a cross-condition metric, deferred to Phase 3 (`reproduce`).

## socsim framework

[socsim](https://github.com/akitenkrad/rs-social-simulation-tools) (library mode, git dependency, `branch = "main"`, pinned by `Cargo.lock`):

- `socsim-core` — `WorldState` / `Mechanism` / `Phase` / `StepContext` / `AgentId` / `SimClock` / `SimRng` / `derive_seed`.
- `socsim-engine` — `SimulationBuilder`, `Simulation::run_observed`, `SequentialScheduler`.
- `socsim-net` — `SocialNetwork` and the `erdos_renyi` / `watts_strogatz` / `barabasi_albert` generators, `neighbors`.
- `socsim-llm` (optional, `features = ["live"]`) — `LlmClient` / `OllamaClient` / `OpenAiClient` / `FallbackClient` / `CachingClient` / `PromptCache` / `LlmConfig` / `CallMetadata` / `MetadataCollector` / `mock::ScriptedClient`.

## References

- Chuang, Y.-S., et al. (2024). *Simulating Opinion Dynamics with Networks of LLM-based Agents.* Findings of ACL: NAACL 2024, 3326–3346. arXiv:2311.09618.
- Park, J. S., et al. (2023). *Generative Agents: Interactive Simulacra of Human Behavior.* UIST 2023. (reflective memory)
- Hegselmann, R., & Krause, U. (2002). *Opinion Dynamics and Bounded Confidence.* JASSS 5(3). (the bounded-confidence analogue of the `Interaction` update)

---
*This file was generated by Claude Code.*
