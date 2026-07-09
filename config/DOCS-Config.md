# DOCS-Config — The WeatherGenerator configuration system

How run configurations are loaded, merged, and structured, and what lives in `config/`. Part of
the agent docs rooted at [`WeatherGenerator/AGENT-README.md`](../AGENT-README.md).

## How a run config is assembled

All config handling lives in `packages/common/src/weathergen/common/config.py`. The entry points
(`train`, `train_continue`, `inference` in `src/weathergen/run_train.py`) parse CLI args
(`src/weathergen/utils/cli.py`) and call `config.load_merge_configs(...)`, which merges OmegaConf
configs in **ascending order of precedence**:

1. **Base config** — `config/default_config.yml`, unless `--base-config <path>` is given.
   For `train_continue`/`inference` (`--from-run-id`), the base is instead the **saved config of
   the previous run**, loaded from `models/<run_id>/model_<run_id>[_latest|_chkpt<N>].json` via
   `load_run_config()` (with backward-compat fixes applied by `_apply_fixes()`).
2. **Private config** — platform paths and settings from `WeatherGenerator-private` (see below).
   Its `secrets` section is stripped before merging.
3. **Extra configs** — files passed via `--config a.yml b.yml ...`, merged left to right.
   A single path argument may also contain several `:`-separated paths (this is how SLURM passes
   `WEATHERGEN_CONFIG_EXTRA`).
4. **CLI overrides** — `--options key=value nested.key=value ...`, parsed by
   `config.from_cli_arglist()` (OmegaConf dotlist syntax). Highest precedence.

The actual merge is `OmegaConf.merge(base, private, *overwrites)`; when combining configs in code,
use `merge_configs(base_config, update_config)` from `weathergen.common.config` — not dict
updates.

Two special behaviors during loading:

- **Streams**: any config that sets `streams_directory` gets its `streams` populated by
  `load_streams()`, which reads every `*.yml` in that directory (one entry per stream, keyed by
  stream name). If any overwrite config sets `streams_directory`, the streams inherited from the
  base config are dropped first, so the new directory fully replaces them.
- **Time keys**: `_sanitize_time_keys()` rewrites `start_date`/`end_date` and
  `time_window_step`/`time_window_len`/`forecast.time_step` in
  `training_config`/`validation_config`/`test_config` into `${datetime:...}`/`${timedelta:...}`
  resolver interpolations (keeping the raw string in a hidden `_<key>` backup key).

Other CLI flags: `--run-id` (assign instead of auto-generating an 8-char id), `--private-config`
(explicit private config path), `--mini-epoch` (checkpoint to resume, `-1` = latest),
`--reuse-run-id` (continue writing into the previous run's directories). Note:
`train_continue --finetune-forecast` exists in the CLI but is currently a no-op in
`run_train.py`; use an overlay config such as `config/config_forecasting_finetuning.yml` instead.

The fully merged config is saved with each checkpoint as JSON under
`<path_shared_working_dir>/models/<run_id>/`; results go to
`<path_shared_working_dir>/results/<run_id>/` (`get_path_model()` / `get_path_run()`).

## Files in `config/`

`default_config.yml` is the base every run starts from. The other `config_*.yml` files are
**variants merged on top of it** via `--config` (some are near-complete configs, some are thin
overlays) — list the directory to see what is available; each file's header comments describe its
purpose.

Subdirectories:

- `config/streams/` — one subdirectory per stream set (`era5_1deg/`, `era5_nppatms_synop/`,
  `fesom/`, ...), each containing per-stream YAML files. Selected via the top-level
  `streams_directory` key. Stream-level options are documented in
  [`config/streams/DOCS-Streams.md`](streams/DOCS-Streams.md).
- `config/evaluate/` — configs for the evaluation package: `eval_config.yml` /
  `eval_config_default.yml` (plotting/scoring options per run and stream),
  `config_zarr2cf.yaml` (Zarr → CF-NetCDF variable/unit mapping),
  `config_zarr2verif.yaml` (variable mapping for verification against MetNor observations).
- `config/profiling/` — `annotations.json`: nvtx-style annotation targets (module/function lists)
  for profiling runs.

## Structure of `default_config.yml`

### Top-level: model architecture

Flat keys at the root define the model. Prefixes map to the encoder/decoder pipeline
(`src/weathergen/model/`):

- `embed_*` — per-stream embedding (`embed_unembed_mode`, `embed_dropout_rate`).
- `ae_local_*` — local assimilation engine (self-attention within HEALPix cells): `dim_embed`,
  `num_blocks`, `num_heads`, `dropout_rate`, `with_qk_lnorm`, plus query setup
  (`ae_local_num_queries`, `ae_local_queries_per_cell`).
- `ae_adapter_*` — local→global adapter layer.
- `ae_global_*` — global assimilation engine (attention across cells): also
  `att_dense_rate`, `block_factor`, `mlp_hidden_factor`, `trailing_layer_norm`.
- `ae_aggregation_*` — optional query-aggregation blocks (`ae_aggregation_num_blocks: 0`
  disables).
- `decoder_type` (`PerceiverIOCoordConditioning` or `Linear`) and `pred_*` — per-stream target
  prediction heads.
- `fe_*` — forecast engine (temporal rollout): `fe_num_blocks`, `fe_num_heads`,
  `fe_layer_norm_after_blocks`, `fe_impute_latent_noise_std`, `forecast_att_dense_rate`.
- `healpix_level` — spatial resolution (5 → 12,288 cells); `rope_2D`, `num_class_tokens`,
  `num_register_tokens`.
- Precision/runtime: `with_mixed_precision`, `with_flash_attention`, `compile_model`,
  `with_fsdp`, `attention_dtype`, `mixed_precision_dtype`, `norm_type`, `qk_norm_type`,
  `norm_eps`, `mlp_norm_eps`.
- `latent_noise_*` — optional VAE-style latent noise (KL weight etc.).
- `freeze_modules` — regex of module names to freeze (used by finetuning overlays);
  `load_chkpt` — dict for partial checkpoint loading.
- `streams_directory` / `streams` — see above (`streams: ???` is mandatory-missing until
  resolved from the directory).
- `zarr_store` — output store type (`"zip"` or `"zarr"`).

### Sections

- `general` — run identity and mutable state: `run_id` (`???` until `set_run_id()` fills it),
  `run_history`, `istep`, `rank`/`world_size`, `desc`, `multiprocessing_method`.
- `train_logging` — logging frequencies in batches (`terminal`, `metrics`, `checkpoint`),
  `log_grad_norms`; variants may add `track_performance_metrics`, `collapse_monitoring`.
- `data_loading` — `num_workers`, `rng_seed`, `repeat_data_in_mini_epoch`, `memory_pinning`.
- `training_config` — the training stage (details below).
- `validation_config` — overrides on top of training (see inheritance).
- `test_config` — overrides on top of validation; used by default for `inference`. Not present in
  `default_config.yml` (empty = identical to validation).
- `wgtags` — free-form MLFlow tags (`org`, `issue`, `exp`, ...); primitive values only, keep
  strings short.

### `training_config` keys

- `training_mode` — list of modes: `"masking"`, `"student_teacher"`, `"latent_loss"`.
- `num_mini_epochs`, `samples_per_mini_epoch`, `shuffle`.
- `start_date` / `end_date` — datetimes; `time_window_step` / `time_window_len` — timedeltas
  (see resolvers below).
- `learning_rate_scheduling` — `lr_start`/`lr_max`/`lr_final_decay`/`lr_final`, warmup/cooldown
  step counts and policies, `parallel_scaling_policy`.
- `optimizer` — `grad_clip`, `weight_decay`, `adamw: {beta1, beta2, eps}`.
- `losses` — dict of named loss terms; each has `type` (`LossPhysical`,
  `LossLatentSSLStudentTeacher`, ...), optional `weight` and `enabled`, a `loss_fcts` dict
  (`mse`, `JEPA`, `DINO`, `iBOT`, ... with per-fct weights, head configs, and
  `target_source_correspondence`), and optionally `target_and_aux_calc` (below).
- `model_input` — dict of named input-sampling strategies; each entry has `masking_strategy`
  (`"random"`, `"healpix"`, `"forecast"`), `num_samples`, `num_steps_input`, and a
  `masking_strategy_config`. The per-GPU batch size is the **sum of `num_samples`** over enabled
  `model_input` entries (`get_batch_size_from_config()` in `src/weathergen/train/utils.py`).
  SSL configs additionally define `target_input` for teacher-side views.
- `forecast` — `time_step` (timedelta), `num_steps` (int or per-mini-epoch list), `offset`
  (0 = autoencoding/denoising, 1 = forecasting), `policy` (`"fixed"`, `"sequential"`); validated
  by `validate_forecast_policy_and_steps()` in `common/config.py`.

### `target_and_aux_calc`: string-or-dict pattern

Lives **inside a loss entry** in `losses` (not directly under `training_config`). It selects how
targets/aux data for that loss are computed (`src/weathergen/train/target_and_aux_*.py`) and
follows a common pattern: a plain string for defaults, or a one-key dict for parameters:

```yaml
# defaults
target_and_aux_calc: "Physical"          # or "EMATeacher"

# parameterized
target_and_aux_calc:
  EMATeacher:
    ema_ramp_up_ratio: 0.09
    ema_halflife_in_thousands: 250000
    model_param_overrides: { ... }       # config overrides applied to the teacher model

target_and_aux_calc:
  FrozenTeacher:
    teacher_run_id: "p43hxwic"
    teacher_mini_epoch: -1
```

Follow this string-or-dict pattern when adding new modular components.

### Stage inheritance: training → validation → test

Implemented in `Trainer.init()` (`src/weathergen/train/trainer.py`) via
`get_active_stage_config()` (`src/weathergen/train/utils.py`):

- effective **validation** config = `merge_configs(training_config, validation_config)`
- effective **test** config = `merge_configs(<effective validation>, test_config)`

So `validation_config`/`test_config` only need the keys they change (dates,
`samples_per_mini_epoch`, ...). After each merge, entries under `losses`, `model_input`, and
`target_input` with `enabled: False` are dropped — so a stage can switch off an inherited loss or
input strategy by overriding just `enabled`. Validation-only keys include `validate_with_ema`
(separate EMA of weights for validation), `output` (how many predicted samples to write to disk),
and `validate_before_training`.

## Custom OmegaConf resolvers

Registered at import time in `packages/common/src/weathergen/common/config.py`:

- `${datetime:<value>}` → `numpy.datetime64` (via `pd.to_datetime`; accepts strings or YAML ints
  like `20001010000000`).
- `${timedelta:<value>}` → `numpy.timedelta64[ms]` (strings via `pd.to_timedelta`, e.g.
  `06:00:00` or `6h`; bare numbers are interpreted as **seconds**).

You normally never write these by hand: `_sanitize_time_keys()` wraps `start_date`, `end_date`,
`time_window_step`, `time_window_len`, `forecast.time_step` (and stream `frequency`)
automatically, storing the original string under `_<key>` so saved configs stay plain strings
(`_strip_interpolation()`).

## Private config (WeatherGenerator-private)

Platform-specific paths and secrets live in the separate `WeatherGenerator-private` repo, never
here. `_load_private_conf()` locates it by, in order: an explicit `--private-config` path, the
`WEATHERGEN_PRIVATE_CONF` environment variable, or by running
`WeatherGenerator-private/hpc/platform-env.py` for HPC auto-detection. It supplies keys such as
`path_shared_working_dir` (root for `models/`, `results/`) and dataset paths (`data_paths`).
**Never hardcode HPC paths in this repo** — anything machine-specific belongs in the private
config, which always overrides the base config in the merge.
