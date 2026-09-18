# WeatherGenerator

A machine-learning Earth system model: a hierarchical transformer trained on diverse
data streams (ERA5 reanalysis, satellite and in-situ observations, ocean/climate model
output) with self-supervised student-teacher losses (JEPA/DINO-style) alongside
physical prediction losses. Data from all streams is tokenized onto the HEALPix grid,
assimilated into a latent state, optionally rolled out in time autoregressively, and
decoded per stream.

This file holds always-relevant rules for LLM tools and an index into the reference
docs; keep it lean and put detail in `agent_docs/` and the `DOCS-*.md`
reference files.

Every change must keep context in sync: update the `agent_docs/` files, the `DOCS-*.md`
reference files, and the lines in this file that describe what you changed, in the same
change — stale context misleads the next agent. The index below says which docs cover what.
The code may be newer than the docs — where code and docs disagree, trust the code and
update the docs.

## Repositories & branches

- Upstream is `github.com/ecmwf/WeatherGenerator`; development happens on personal
  forks with PRs back to upstream. Branches: `develop` (default; fast-moving, breaking
  changes), `main` (stable, for experiments), `develop-ssl` (fast-moving pretraining
  experiments).
- `../WeatherGenerator-private` (sibling repo): HPC-specific paths, private configs,
  SLURM launch scripts. May contain credentials — never read files that look like
  secrets. Split rule: anything that defines a model or affects training results goes
  in this repo; machine/team-specific paths and settings go in the private repo.
  Never hardcode HPC paths in this repo. Details: `agent_docs/infrastructure.md`.
- Some users also have a `weathergen-research` sibling (experiment planning,
  documentation, literature) with its own instruction files.

## Environment & tooling

- Python 3.12, managed with uv. Deps declared in `pyproject.toml`, locked in `uv.lock`.
- `uv sync` to set up, `uv run <cmd>` to run. Never `pip install` into the env — it bypasses the lock.
- Entry points (`pyproject.toml [project.scripts]`, run as `uv run <cmd>`): `train`,
  `train_continue`, `inference`, `evaluate`, `export`, `plot_train`. CLI flags are
  dash-separated (`--run-id`, not `--run_id`); full CLI in `src/weathergen/utils/cli.py`.
  Invocations, run-id mechanics, and HPC launching: `agent_docs/infrastructure.md`.
- Dev tasks: `scripts/actions.sh {lint|lint-check|type-check|unit-test|toml-check}`; `integration-test*` targets need a GPU.

## Checks & style

- Run `./scripts/actions.sh lint` before considering a change done. CI
  (`.github/workflows/ci.yml`) runs lint-check, toml-check, type-check, unit-test, and
  requires the PR to be linked to a GitHub issue (`scripts/check_gh_issue.py`) — all
  must pass.
- `unit-test` runs pytest on the `*_test.py` files colocated in `src/` (what CI runs);
  the standalone tests in `tests/` run via `uv run --extra cpu pytest tests/` and are
  not currently in CI.
- Line length 100, ruff formatting; type hints required (pyrefly); use `logging`,
  never `print`; match existing naming conventions. Do not commit large binary blobs
  (>2MB commits are rejected — TODO: check this holds on upstream) or the contents of
  runtime output dirs.

## Layout

- `src/weathergen/` — core model + training code. Entry points in `run_train.py`;
  `datasets/` (multi-stream loading, readers, tokenizer, masking), `model/`
  (encoder, engines, attention, EMA), `train/` (trainer, losses, LR schedule, SSL
  teacher), `utils/` (CLI, distributed, logging, plotting)
- `packages/` — uv-workspace libraries (common, evaluate, metrics, readers_extra).
  Also here but not workspace members: `dashboard/` (own lockfile —
  `agent_docs/decisions/dashboard-not-in-workspace.md`) and `science/` (standalone
  analysis scripts)
- `config/` — YAML run configs; `config/streams/` — per-stream config sets
- `tests/` — standalone unit tests; `integration_tests/` — GPU integration tests
- `agent_docs/` — agent-oriented docs indexed below: systems dataflows, `recipes/`
  (procedures), `decisions/` (rationale); how the setup works: `agent_docs/agentic-setup.md`
- `docs/` — human-facing reference (e.g. `docs/evaluate_config_reference.md`)
- `ci/`, `.github/workflows/` — CI definitions
- `logs/`, `models/`, `plots/`, `results/` — runtime output, gitignored; on HPC these are symlinks into shared storage. Never commit contents; details in `agent_docs/infrastructure.md`.

## Documentation (read when relevant)

Full index of `agent_docs/`. Every new doc gets a line here — procedure:
`agent_docs/recipes/add-documentation.md`.

Systems (runtime dataflows):
- `agent_docs/training-step.md` — the end-to-end training step: trainer, model forward, losses. Read before changing any of those; update after.
- `agent_docs/data-pipeline.md` — stream configs → readers → tokenizer → ModelBatch. Read before touching data loading or stream configs; update after.
- `agent_docs/ssl-training.md` — SSL/student-teacher delta: masking, teachers, latent losses. Read before touching SSL or masking code; update after.
- `agent_docs/config-system.md` — config sources, merge precedence, stage configs, runtime mutation. Read before adding/renaming config options; update after changing the merge logic.
- `agent_docs/infrastructure.md` — software stack (cluster base env + uv), SLURM/GH200 hardware, run commands and run IDs, runtime output dirs; local runs not supported yet. Read before setting up an env, launching or continuing runs, or touching run outputs; update after changing tooling or workflow.
- `agent_docs/agentic-setup.md` — how these instruction files work (opt-in loading) and what belongs in AGENT-README.md vs agent_docs/. Read before editing this file or adding documentation.

Recipes (procedures):
- `agent_docs/recipes/add-data-reader.md` — add a reader for a new data source.
- `agent_docs/recipes/add-documentation.md` — add or change agent documentation.

Decisions (rationale):
- `agent_docs/decisions/dashboard-not-in-workspace.md` — why packages/dashboard has its own lockfile.

Directory-scoped reference (`DOCS-*.md`, next to the code): file-by-file detail of one
subsystem and how its scripts function — vs `agent_docs/`, which holds cross-directory
workflows, coupling, and rationale (split defined in `agent_docs/agentic-setup.md`):
- `config/DOCS-Config.md` — config merging, default_config.yml options, variants.
- `config/streams/DOCS-Streams.md` — stream YAML schema, readers, adding a stream.
- `src/weathergen/model/DOCS-model.md` — architecture: encoder, engines.py classes, attention, EMA.
- `src/weathergen/datasets/DOCS-Datasets.md` — data pipeline: samplers, readers, tokenizer, masking.
- `src/weathergen/train/DOCS-Train.md` — trainer, loss system, SSL teacher, checkpointing.
- `packages/DOCS-Packages.md` — workspace packages: common, evaluate, metrics, readers_extra.
