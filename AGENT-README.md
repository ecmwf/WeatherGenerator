# AGENT-README — WeatherGenerator

Entry point for AI coding agents (and humans) working in this repository. It describes the project
structure, the tools and commands to use, and the checks that must pass. More detailed docs live
next to the code they describe and are indexed at the bottom of this file — read the relevant one
before modifying that part of the codebase.

## What this project is

WeatherGenerator is a machine-learning Earth system model: a hierarchical transformer trained on
diverse data streams (ERA5 reanalysis, satellite and in-situ observations, ocean/climate model
output) using self-supervised student-teacher approaches (JEPA/DINO-style) alongside physical
prediction losses. Space is discretized on the HEALPix grid; data from all streams is tokenized
onto HEALPix cells, assimilated into a latent state, optionally rolled out in time
autoregressively, and decoded per stream.

**Branches:** `main` (stable, for experiments), `develop` (fast-moving, breaking changes), `develop-ssl` (fast-moving
pretraining experiments)

## The three sibling repositories

This repo lives beside two others (paths relative to the common parent directory):

| Repo | Purpose |
|---|---|
| `WeatherGenerator/` | This repo. All public code: model, training, evaluation, configs. |
| `WeatherGenerator-private/` | HPC-specific paths, private configs, SLURM launch scripts. **May contain credentials — never read files that look like secrets.** See `../WeatherGenerator-private/AGENT-README.md`. |
| `weathergen-research/` | Only some users may have it. Experiment planning, documentation, literature. Has its own CLAUDE.md. |

Rule of thumb: anything that defines a model or affects training results goes in
`WeatherGenerator`; anything that is a machine/team-specific path or setting goes in
`WeatherGenerator-private`. Never hardcode HPC paths in this repo.

## Repository layout

```
WeatherGenerator/
├── src/weathergen/           # Core code (package "weathergen")
│   ├── run_train.py          # Entry points: train, train_continue, inference
│   ├── datasets/             # Multi-stream data loading, readers, tokenizer, masking
│   ├── model/                # Architecture: encoder, engines, attention, EMA
│   ├── train/                # Trainer, losses, LR schedule, SSL teacher
│   └── utils/                # CLI parsing, distributed, logging, plotting
├── packages/                 # UV workspace packages
│   ├── common/               # Config system (merge_configs), I/O utilities
│   ├── evaluate/             # Evaluation pipeline (Zarr results → metrics/plots)
│   ├── metrics/              # MLFlow integration
│   ├── readers_extra/        # Extra data readers (FESOM, ICON, ...)
│   ├── dashboard/            # Streamlit dashboard (NOT a workspace member)
│   └── science/              # Standalone analysis scripts
├── config/                   # YAML configs: default_config.yml + variants
│   └── streams/              # Per-stream config sets (ERA5, obs, FESOM, ...)
├── tests/                    # Standalone unit tests (plus *_test.py colocated in src/)
├── integration_tests/        # End-to-end GPU tests (small training runs)
├── scripts/actions.sh        # One-stop script for sync/lint/type-check/tests
├── docs/                     # Assorted docs (e.g. evaluate_config_reference.md)
└── ci/, .github/workflows/   # CI definitions
```

## Tooling and everyday commands

The project is managed with **uv** (workspace with multiple packages). Always run things through
`uv run` or `scripts/actions.sh`; never pip-install into a global environment.

```bash
./scripts/actions.sh sync         # Create/update .venv (GPU extra on Linux, CPU on macOS)
./scripts/actions.sh sync-safe    # Slower sync that survives LUSTRE cache corruption (HPC)

# Training / inference (entry points defined in pyproject.toml [project.scripts])
# NB: CLI flags are dash-separated (--run-id, not --run_id)
uv run train --config config/<variant>.yml --run-id <run_id>
uv run train_continue --from-run-id <run_id> --mini-epoch <N>   # resume (also for chained HPC jobs)
uv run inference --from-run-id <run_id>   # inference dates come from test_config / --options
uv run evaluate --config config/evaluate/<cfg>.yml
uv run export ...                 # export inference output
uv run plot_train -fd "{<run_id>: [<job_id>, <label>]}"   # or -fy runs.yml
```

Other shared flags: `--options key.subkey=value ...` (highest-precedence overrides),
`--private-config <path>`, `--base-config <path>` (defaults to `config/default_config.yml`),
`--reuse-run-id` (continue/inference). See `src/weathergen/utils/cli.py` for the full CLI.

**Run IDs:** every training/inference/validation run has an 8-character alphanumeric ID (starts
with a letter; random unless given via `--run-id`). Outputs land in `models/<run_id>/`,
`results/<run_id>/` (under the shared working dir) and `./logs/<run_id>/` (cwd-relative).
Continue training with `train_continue --from-run-id`.

Actual HPC launches (SLURM, job chaining, MLFlow registration) go through
`../WeatherGenerator-private/hpc/launch-slurm.py` — see `../WeatherGenerator-private/hpc/DOCS-HPC.md`.

## Checks to run (before considering any change done)

```bash
./scripts/actions.sh lint         # ruff format + ruff check --fix (auto-fixes)
./scripts/actions.sh lint-check   # CI version: ruff format -n + ruff check + pylint, no fixes
./scripts/actions.sh type-check   # Only run when requested. pyrefly, per package then root (slow: re-syncs envs)
./scripts/actions.sh unit-test    # Only run when requested. pytest on src/ (colocated *_test.py files) — this is what CI runs
uv run --extra cpu pytest tests/  # standalone tests in tests/ (not currently in CI)
./scripts/actions.sh toml-check   # Only run when requested. pyproject.toml consistency across workspace
```

Integration tests need a GPU and pre-synced data (run on CSCS CI, `ci/cscs.yaml`):

```bash
./scripts/actions.sh integration-test-single   # smallest single-stream run
./scripts/actions.sh integration-test          # Outdated. multi-stream
./scripts/actions.sh integration-test-jepa     # Outdated. JEPA/SSL
./scripts/actions.sh integration-test-all
```

**CI (GitHub Actions, `.github/workflows/ci.yml`):** lint-check, toml-check, type-check,
unit-test, and a check that the PR branch is linked to a GitHub issue (branch naming enforced by
`scripts/check_gh_issue.py`). All must pass.

**Code style:** line length 100; ruff formatting; type hints required (pyrefly); use `logging`,
never `print`; match existing naming conventions. Do not commit large binary blobs (>2MB commits
are rejected).

## Configuration system (short version)

Hierarchical OmegaConf YAML. Merge order (ascending priority):
`config/default_config.yml` → private config (from `WeatherGenerator-private`; found via explicit
path, the `WEATHERGEN_PRIVATE_CONF` env var, or auto-detection by `hpc/platform-env.py`) → extra
`--config` files → `--options` CLI overrides.
`validation_config` inherits from `training_config`, `test_config` from `validation_config`.
Use `merge_configs()` from `packages/common/src/weathergen/common/config.py` when combining
configs in code. Details: [config/DOCS-Config.md](config/DOCS-Config.md).

## Detailed docs index

| Doc | Covers |
|---|---|
| [config/DOCS-Config.md](config/DOCS-Config.md) | Config merging, default_config.yml options, variants |
| [config/streams/DOCS-Streams.md](config/streams/DOCS-Streams.md) | Stream YAML schema, readers, adding a stream |
| [src/weathergen/model/DOCS-model.md](src/weathergen/model/DOCS-model.md) | Architecture: encoder, engines.py classes, attention, EMA |
| [src/weathergen/datasets/DOCS-Datasets.md](src/weathergen/datasets/DOCS-Datasets.md) | Data pipeline: samplers, readers, tokenizer, masking |
| [src/weathergen/train/DOCS-Train.md](src/weathergen/train/DOCS-Train.md) | Trainer, loss system, SSL teacher, checkpointing |
| [packages/DOCS-Packages.md](packages/DOCS-Packages.md) | Workspace packages: common, evaluate, metrics, readers_extra |
| `../WeatherGenerator-private/AGENT-README.md` | The private repo: structure, rules |
| `../WeatherGenerator-private/hpc/DOCS-HPC.md` | launch-slurm.py, per-HPC scripts, job chaining |

Keep these docs up to date: if you change something a doc describes (a config key, a class name, a
command), update the doc in the same change.
