# WeatherGenerator

A machine-learning Earth system model. This file holds always-relevant rules for LLM
tools and an index into the reference docs; keep it lean and put detail in `agent_docs/`.

Every change must keep context in sync: update the `agent_docs/` files and AGENTS.md
lines that describe what you changed, in the same change — stale context misleads the
next agent. The index below and the scoped AGENTS.md files say which docs cover what.

## Environment & tooling

- Python 3.12, managed with uv. Deps declared in `pyproject.toml`, locked in `uv.lock`.
- `uv sync` to set up, `uv run <cmd>` to run. Never `pip install` into the env — it bypasses the lock.
- Dev tasks: `scripts/actions.sh {lint|lint-check|type-check|unit-test|toml-check}`; `integration-test*` targets need a GPU.

## Layout

- `src/weathergen/` — core model + training code
- `packages/` — uv-workspace libraries (common, evaluate, metrics, readers_extra)
- `config/` — YAML run configs
- `tests/` — unit tests; `integration_tests/` — GPU integration tests
- `logs/`, `models/`, `plots/`, `results/` — runtime output, gitignored; on HPC these are symlinks into shared storage. Never commit contents; details in `agent_docs/infrastructure.md`.

`src/`, `packages/`, `config/`, `tests/` each have their own AGENTS.md with local
rules. Most tools auto-load it when you work there; if yours doesn't, read it first.

## Documentation (read when relevant)

Full index of `agent_docs/`. Every new doc gets a line here — procedure:
`agent_docs/recipes/add-documentation.md`.

Systems (runtime dataflows):
- `agent_docs/training-step.md` — the end-to-end training step: trainer, model forward, losses. Read before changing any of those; update after.
- `agent_docs/data-pipeline.md` — stream configs → readers → tokenizer → ModelBatch. Read before touching data loading or stream configs; update after.
- `agent_docs/ssl-training.md` — SSL/student-teacher delta: masking, teachers, latent losses. Read before touching SSL or masking code; update after.
- `agent_docs/config-system.md` — config sources, merge precedence, stage configs, runtime mutation. Read before adding/renaming config options; update after changing the merge logic.
- `agent_docs/infrastructure.md` — uv, dependencies, running jobs on HPC vs. developing locally. Read before setting up the env or launching jobs; update after changing tooling or workflow.
- `agent_docs/agentic-setup.md` — how these instruction files work and what belongs in AGENTS.md vs agent_docs/. Read before editing any AGENTS.md or adding documentation.

Recipes (procedures):
- `agent_docs/recipes/add-data-reader.md` — add a reader for a new data source.
- `agent_docs/recipes/add-documentation.md` — add or change agent documentation.

Decisions (rationale):
- `agent_docs/decisions/dashboard-not-in-workspace.md` — why packages/dashboard has its own lockfile.
