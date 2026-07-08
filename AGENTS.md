# WeatherGenerator

A machine-learning Earth system model. This file holds always-relevant rules for LLM
tools and an index into the reference docs; keep it lean and put detail in `docs/`.

## Environment & tooling

- Python 3.12, managed with uv. Deps declared in `pyproject.toml`, locked in `uv.lock`.
- `uv sync` to set up, `uv run <cmd>` to run. Never `pip install` into the env — it bypasses the lock.

## Documentation (read when relevant)

- `docs/infrastructure.md` — uv, dependencies, running jobs on HPC vs. developing locally. Read before setting up the env or launching jobs.