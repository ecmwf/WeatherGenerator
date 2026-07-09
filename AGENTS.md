# WeatherGenerator

A machine-learning Earth system model. This file holds always-relevant rules for LLM
tools and an index into the reference docs; keep it lean and put detail in `docs/`.

## Environment & tooling

- Python 3.12, managed with uv. Deps declared in `pyproject.toml`, locked in `uv.lock`.
- `uv sync` to set up, `uv run <cmd>` to run. Never `pip install` into the env — it bypasses the lock.
- Dev tasks: `scripts/actions.sh {lint|lint-check|type-check|unit-test|toml-check}`; `integration-test*` targets need a GPU.

## Layout

- `src/weathergen/` — core model + training code
- `packages/` — uv-workspace libraries (common, evaluate, metrics, readers_extra)
- `config/` — YAML run configs
- `tests/` — unit tests; `integration_tests/` — GPU integration tests

`src/`, `packages/`, `config/`, `tests/` each have their own AGENTS.md with local
rules. Most tools auto-load it when you work there; if yours doesn't, read it first.

## Documentation (read when relevant)

- `docs/infrastructure.md` — uv, dependencies, running jobs on HPC vs. developing locally. Read before setting up the env or launching jobs; update after changing tooling or workflow.
- `docs/agentic_setup.md` — how these instruction files work and what belongs in AGENTS.md vs docs/. Read before editing any AGENTS.md or adding documentation.
