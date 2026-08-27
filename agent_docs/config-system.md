# Config system — sources, precedence, runtime shape

Summary: configs are OmegaConf objects (`Config` in
`packages/common/src/weathergen/common/config.py`). A run's config is merged from
base → private → overwrites (ascending precedence) by `load_merge_configs`, gets a
run id, is mutated during the run (step counter, streams), and is saved with every
checkpoint so runs can be continued or inspected.

## Sources and precedence (`config.py:load_merge_configs`)

Ascending precedence (later wins):

1. Base: `config/default_config.yml` (`_DEFAULT_CONFIG_PTH`) — or, when continuing
   (`--from-run-id`), the saved config of that run (`load_run_config`).
2. Private config: platform-dependent paths and secrets; the `secrets` section is
   discarded from the merged result. Found in order (`config.py:_load_private_conf`): explicit
   `--private-config` path → `WEATHERGEN_PRIVATE_CONF` env var → auto-detection by
   running `../WeatherGenerator-private/hpc/platform-env.py`.
3. Overwrites, in order: each `--config` file (paths may be `:`-joined, split
   automatically), then CLI `--options` as an OmegaConf dot-list
   (`from_cli_arglist`).

Special case: an overwrite containing `streams_directory` loads those streams and
replaces any inherited streams (`base_config.streams = None` before merge).

## CLI (`src/weathergen/utils/cli.py`, entry `run_train.py`)

- Stages: `train`, `train_continue`, `inference` (`Stage` enum; positional or
  `WEATHERGEN_STAGE` env var).
- Key args: `--config` (repeatable overwrites), `--base-config`, `--private-config`,
  `--options key=value ...`, `--run-id`/`--reuse-run-id`, `--from-run-id` +
  `--mini-epoch` for continuation.

## Stage configs (`trainer.py:Trainer.init`)

- `training_config` is the source of truth. `validation_config` and `test_config` are
  *deltas* applied on top of it (`get_active_stage_config`), in that chain:
  training → validation → test.
- Entries in `losses`, `model_input`, `target_input` (`cfg_keys_to_filter`) that are
  disabled by an overwrite are removed by `filter_config_by_enabled` — that's the
  mechanism for turning loss terms off per stage.

## Runtime lifecycle

- `set_run_id` assigns/reuses the run id; `config.save(cf, mini_epoch)` writes the
  config into the run directory at init and with every checkpoint.
- The config is mutated during the run: `general.istep` (global step),
  `general.run_history` (continuation lineage), `streams`, `world_size*`. Continuation
  correctness depends on these saved values.

## Coupling & invariants

- New options belong in `config/default_config.yml` with a sensible default — partial
  overwrite configs must stay valid against it.
- Config keys are consumed by name at many sites with `.get(..., default)`; a rename
  silently falls back to the default rather than failing. Grep for the key before
  renaming.
- `losses` / `model_input` / `target_input` are special-cased (`cfg_keys_to_filter`):
  their entries can be disabled per stage; other sections merge wholesale.
- Loss-term names in `losses` are cross-referenced by the trainer, the target-aux
  calculators, and (for SSL) model latent-head names — see `agent_docs/training-step.md`
  and `agent_docs/ssl-training.md`.
- `validate_forecast_policy_and_steps` runs per stage config at init; forecast
  settings must satisfy it for all three stages, not just training.
