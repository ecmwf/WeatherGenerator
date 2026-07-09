# DOCS-Train: Training Code (`src/weathergen/train/`)

Purpose: developer/agent documentation for the WeatherGenerator training loop, loss system,
target-and-aux calculators (incl. SSL teachers), LR scheduling, and collapse monitoring.

Part of the agent docs rooted at `AGENT-README.md`.
For the model architecture side, see `src/weathergen/model/DOCS-model.md`.

Line anchors below are approximate; verify with grep before editing.

---

## 1. File-by-file overview

| File | Purpose |
|------|---------|
| `trainer.py` | `Trainer` (L61): full training/validation/inference orchestration. Owns model, optimizer, data loaders, loss calculators, target-and-aux calculators, EMA model, logging. |
| `trainer_base.py` | `TrainerBase` (L25): static setup helpers `init_torch()` (L30, device + multiprocessing method) and `init_ddp()` (L66, process group init from torchrun/SLURM env vars, broadcasts run_id to all ranks). |
| `loss_calculator.py` | `LossCalculator` (L28): instantiates and dispatches to loss modules, sums weighted per-term losses, keeps loss histories for logging. |
| `loss_modules/` | Loss module implementations (see section 3). |
| `lr_scheduler.py` | `LearningRateScheduler` (L22): three-phase warmup → decay → cooldown schedule with parallel scaling of `lr_max`. |
| `collapse_monitor.py` | `CollapseMonitor` (L51): SSL representation-collapse metrics (RankMe effective rank, singular values, per-dimension variance, DINO prototype entropy, EMA beta). |
| `target_and_aux_module_base.py` | `TargetAuxOutput` dataclass (L21, the target container API), `TargetAndAuxModuleBase` (L65), `PhysicalTargetAndAux` (L85, passthrough physical targets from the batch). |
| `target_and_aux_ssl_teacher.py` | SSL teachers: `EncoderTeacher` base (L34), `EMATeacher` (L85), `FrozenTeacher` (L112), `get_target_postprocessing()` (L173, per-SSL-loss target post-processing modules). |
| `target_and_aux_utils.py` | `get_target_aux_calculator()` (L11): factory that parses the string-or-dict `target_and_aux_calc` config and builds the calculator. |
| `teacher_utils.py` | `prepare_encoder_teacher()` (L60, strip model to encoder-only + fresh SSL latent heads), `load_encoder_from_checkpoint()` (L97, load only `encoder.*`/`latent_pre_norm*` weights), `_create_teacher_heads()` (L28). |
| `utils.py` | `Stage` literals (`TRAIN`/`VAL`/`TEST`), `get_batch_size_from_config()` (L139, batch size = sum of enabled `model_input.*.num_samples`), `get_target_idxs_from_cfg()` (L153), `get_active_stage_config()`/`filter_config_by_enabled()` (L165/L179, stage-config inheritance + `enabled:` filtering), `extract_batch_metadata()` (L130), `NoOpGradScaler` (L196). |
| `loss_modules/loss_module_base.py` | `LossValues` dataclass (L24: `loss`, `losses_all`, `stddev_all`) and abstract `LossModuleBase` (L42). |
| `loss_modules/loss_module_physical.py` | `LossPhysical` (L83) and `DynamicLossEMA` (L34, Samudra-2-style dynamic inverse-MSE channel weighting). |
| `loss_modules/loss_module_ssl.py` | `LossLatentSSLStudentTeacher` (L25) with `jepa_loss` (L222), `ibot_loss` (L249), `dino_loss` (L269). |
| `loss_modules/loss_functions.py` | Primitive loss functions: `lp_loss` (L126), `mse` (L206), `rss` (L227), `rmse` (L248), `mae` (L269), `kernel_crps` (L70), location weight `cosine_latitude` (L290), timestep weight `gamma_decay` (L295), softmax cross-entropy helpers for DINO/iBOT (L301+). |

---

## 2. Training flow

### Entry points

`src/weathergen/run_train.py` defines console entry points (see `pyproject.toml` scripts):
- `train()` → `main([Stage.train, ...])` → `run_train(args)`
- `train_continue()` → `run_continue(args)`
- `inference()` → `run_inference(args)`

The CLI (`src/weathergen/utils/cli.py`) uses argparse subcommands `train | train_continue | inference`
(`Stage` StrEnum, `get_main_parser()`). Flags are dash-separated:
`--config` (multiple, ascending precedence), `--run-id`, `--options key.sub=value ...`,
`--base-config`, `--private-config`; for `train_continue`/`inference` additionally
`--from-run-id` (required), `--mini-epoch` (default -1 = latest), `--reuse-run-id`, and
`--finetune-forecast` (train_continue only). There are no `--start`/`--end` inference flags;
inference dates come from `test_config` in the config files / `--options`.

Each entry function:
1. Merges configs via `config.load_merge_configs()` (ascending precedence: base config — the
   saved run config when continuing — then private config, extra `--config` files, and CLI
   `--options`), then `config.set_run_id()`.
2. `Trainer.init_torch()` (device, multiprocessing method from `general.multiprocessing_method`)
   and `Trainer.init_ddp(cf)` (sets `cf.world_size/rank/local_rank/with_ddp`).
3. `init_loggers(run_id)`, appends to `cf.general.run_history`.
4. Creates `Trainer(cf.train_logging)` and calls `trainer.run(cf, devices[, from_run_id, mini_epoch])`
   or `trainer.inference(...)`. On exception with `world_size == 1`, drops into `pdb.post_mortem`.

### Trainer setup (`Trainer.init`, trainer.py L100; `Trainer.run`, L250)

- `init()`: merges latent-noise defaults into `cf`; builds the three stage configs:
  `training_cfg` = `cf.training_config` filtered by `enabled:` flags; `validation_cfg` =
  training merged with `cf.validation_config`; `test_cfg` = validation merged with
  `cf.test_config` (`get_active_stage_config`, filter keys: `losses`, `model_input`,
  `target_input`). Derives per-GPU batch sizes from `model_input` sample counts. Creates
  output dirs, `TrainLogger`, `CollapseMonitor`, optional `ThroughputTracker`.
- `run()`: creates `MultiStreamDataSampler` datasets for TRAIN and VAL, plain
  `torch.utils.data.DataLoader`s (`batch_size=None`, batching handled by the sampler).
- Model: `init_model_and_shard()` (`weathergen/model/model_interface.py`) builds the model and
  applies FSDP2 sharding when `cf.with_ddp and cf.with_fsdp` (default config: `with_fsdp: True`).
  FSDP2 state dicts are DTensor-sharded; `_get_full_model_state_dict()` (L658) and
  `_get_full_optimizer_state_dict()` (L674) gather full tensors on rank 0 for checkpointing.
- Mixed precision: forward passes run under `torch.autocast(dtype=cf.mixed_precision_dtype,
  enabled=cf.with_mixed_precision)`; default `mixed_precision_dtype: bf16`. A
  `torch.amp.GradScaler` is used only if `training_config.optimizer.grad_scaling` (default True);
  otherwise `NoOpGradScaler`.
- Optimizer: fused AdamW, with betas/eps rescaled by total batch size (SDE scaling rule,
  trainer.py L323-340). `LearningRateScheduler` steps per batch.
- Optional validation EMA (see section 4) and target-and-aux calculators for both TRAIN and VAL.

### Mini-epoch loop (`Trainer.run`, L391)

```python
for mini_epoch in range(mini_epoch_base, training_cfg.num_mini_epochs):
    self.train(mini_epoch)      # one pass of samples_per_mini_epoch
    self.validate(mini_epoch, self.validation_cfg, ...)
    self.save_model(mini_epoch)
self.save_model(num_mini_epochs)  # final
```

`mini_epoch_base` is recovered from `cf.general.istep` when continuing a run. Optionally,
`validation_config.validate_before_training: bool | int` runs a validation pass (or N samples)
before training starts.

`train()` (L434) per batch: autocast forward → target-and-aux `compute()` per loss term →
`loss_calculator.compute_loss()` → backward with grad scaling → `clip_grad_norm_`
(`training_config.optimizer.grad_clip`) → optimizer step → `lr_scheduler.step()` →
target-and-aux `update_state_post_opt_step()` (EMA teacher update) → validation-EMA update →
collapse monitoring → periodic logging and `_latest` checkpointing
(`train_logging.terminal/metrics/checkpoint` frequencies). `cf.general.istep` increments per batch.

`validate()` (L573) runs `torch.no_grad()` over the validation loader (through `ema_model.forward_eval`
if validation EMA is enabled), computes losses via `loss_calculator_val`, and writes the first
`output.num_samples * batch_size` batches to disk via `write_output()`
(`weathergen/utils/validation_io.py`).

`inference()` (L193) is init + a single `validate()` pass with `test_cfg` (`stage=VAL` dataset);
no optimizer/scheduler is created.

### Output paths

All rooted at `path_shared_working_dir` from the private config
(`packages/common/src/weathergen/common/config.py`):

- Checkpoints: `get_path_model()` (config.py L688) → `<shared>/models/<run_id>/` with files
  `<run_id>_chkpt<NNNNN>.chkpt` (5-digit, per mini-epoch), `<run_id>_latest.chkpt` (mid-epoch
  saves, `save_model(-1)`). Written atomically via a `_tmp` file + rename; rank 0 only.
  The run config is saved alongside each checkpoint (`config.save(cf, mini_epoch)`).
- Run results: `get_path_run()` (config.py L683) → `<shared>/results/<run_id>/` — training
  metrics/logs from `TrainLogger` and validation output zarr stores
  `validation_chkpt<NNNNN>_rank<NNNN>.zarr` (`get_path_results()`, config.py L698).
- Text logs: `./logs/<run_id>/` relative to the working directory
  (`init_loggers` in `packages/common/src/weathergen/common/logger.py` L99).

---

## 3. Loss system

### Dispatch

Losses are configured per stage under `training_config.losses` (inherited/overridable in
`validation_config`/`test_config`; entries can be disabled with `enabled: False`). Each entry is a
named *loss term*:

```yaml
training_config:
  losses:
    physical:                       # arbitrary term name
      type: LossPhysical            # class name in weathergen.train.loss_modules
      weight: 1.0                   # term weight (default 1.0)
      loss_fcts: { mse: {}, }       # passed as **kwargs to the module
      # target_and_aux_calc: "Physical"   (default; see section 4)
```

`LossCalculator.__init__` instantiates `getattr(LossModules, params.type)(cf, mode_cfg, stage,
device, **params.loss_fcts)` for every term. `compute_loss()` looks up the matching
`targets_and_aux[loss_term_name]` (produced by that term's target-and-aux calculator), calls each
module's `compute_loss(preds, targets, metadata)` and accumulates
`loss += weight * loss_values.loss`. Terms with `weight: 0.0` are skipped. Modules return a
`LossValues(loss, losses_all, stddev_all)`; histories are kept on the calculator for logging.

The exported modules (`loss_modules/__init__.py`) — a `type:` value must be one of:

- **`LossPhysical`** (`loss_module_physical.py`): loss on decoded physical variables, averaged as
  `Mean_streams(Mean_timesteps(Mean_loss_fcts(...)))`. Its `loss_fcts` keys name functions in
  `loss_functions.py`: `mse`, `mae`, `rmse`, `rss` (all via generic `lp_loss`), `kernel_crps`
  (ensemble CRPS, requires ens_size > 1), each with optional per-function `weight`. Supports:
  per-stream `loss_weight` and static `target_channel_weights` (train stage only; validation is
  always unweighted), location weighting via stream `location_weight: cosine_latitude`, forecast
  timestep weighting via `forecast.timestep_weight: {gamma_decay: {decay_factor: ...}}`, spoofed
  targets masked out, NaN targets masked, and optional **dynamic channel weighting** via
  `loss_fcts.dynamic_loss: {window: 100, L: 20.0}` (`DynamicLossEMA`, EMA of inverse per-channel
  MSE, train only).
- **`LossLatentSSLStudentTeacher`** (`loss_module_ssl.py`): DINO/iBOT/JEPA-style latent losses on
  `preds.latent[0]` vs teacher targets (`targets.latent`). `loss_fcts` keys must be `"JEPA"`,
  `"iBOT"`, or `"DINO"`, each with `weight` and `loss_extra_args` (e.g. `student_temp`), plus
  head/target-processing params (`out_dim`, `head`, `teacher_temp`, `teacher_style`,
  `center_momentum` for iBOT/DINO) and `target_source_correspondence` mapping target views to
  source views. JEPA = masked L1 in latent space; iBOT = masked patch + class-token softmax
  cross-entropy; DINO = local→global + global→global class-token softmax cross-entropy. See
  `config/config_dinov2.yml` and `config/config_jepa.yml` for complete examples.

Multiple terms can be mixed (e.g. physical + SSL); each term gets its own target-and-aux
calculator, and `get_target_idxs_from_cfg()` (utils.py L153) selects which target samples of the
batch each term consumes based on `target_source_correspondence`.

---

## 4. Target-and-aux calculators

Loss modules never look at the raw batch for targets; a per-loss-term *target-and-aux calculator*
produces a `TargetAuxOutput` (physical targets, latent targets, aux outputs). Selected via
`losses.<term>.target_and_aux_calc`, parsed by `get_target_aux_calculator()`
(`target_and_aux_utils.py` L11) with the **string-or-dict pattern**:

```yaml
# string form: defaults
target_and_aux_calc: "Physical"

# dict form: single key = type, value = params
target_and_aux_calc:
  EMATeacher:
    ema_halflife_in_thousands: 1e-3     # EMA halflife (in thousands of samples)
    ema_ramp_up_ratio: 0.09             # EMAModel rampup_ratio
    model_param_overrides: { ae_global_num_blocks: 10 }  # teacher architecture overrides
    teacher_run_id: abc12345            # optional warm start: load encoder weights
    teacher_mini_epoch: -1              #   from this checkpoint (-1 = latest)
```

Implemented types:

- **`Physical`** → `PhysicalTargetAndAux` (`target_and_aux_module_base.py` L85): collects target
  tokens/times/coords/spoof flags per stream and forecast step from the batch. Default when
  `target_and_aux_calc` is unspecified.
- **`EMATeacher`** (`target_and_aux_ssl_teacher.py` L85): SSL teacher = EMA of the student. Builds
  a separate (never DDP/FSDP-wrapped) model instance, strips it to encoder-only with fresh SSL
  latent heads (`prepare_encoder_teacher`), and wraps it in `EMAModel`
  (`weathergen/model/ema.py`). `update_state_post_opt_step()` performs the EMA update after each
  optimizer step. **Not supported with FSDP2** (assert in `target_and_aux_utils.py` L38 — set
  `with_fsdp: False`).
- **`FrozenTeacher`** (L112): frozen pre-trained encoder loaded via
  `FrozenTeacher.from_pretrained()`; params `teacher_run_id`, `teacher_mini_epoch`. Never updated.

Teacher forward runs under `torch.no_grad()`; outputs pass through per-loss target
post-processing (`get_target_postprocessing()`, L173): `JEPATargetProcessing`,
`iBOTPatchTargetProcessing`, `DINOTargetProcessing` (centering/softmax; from
`weathergen/model/ssl_target_processing.py`).

Base-class hooks (all no-ops unless overridden): `compute()`, `update_state_pre_backward()`,
`update_state_post_opt_step()`, `reset()`, `to_device()`. The Trainer calls the update hooks on
**both** the train and val calculators every training step.

### Two independent EMAs

1. **Training EMA (SSL teacher)** — `losses.<term>.target_and_aux_calc: EMATeacher` (above).
2. **Validation EMA** — `validation_config.validate_with_ema` (trainer.py L287-311): a separate
   `EMAModel` of the full model used only for validation forward passes and as the state dict
   saved by `save_model()` when present (checkpoints then contain EMA weights!).

```yaml
validation_config:
  validate_with_ema:
    enabled: True                   # default True if the block exists
    ema_ramp_up_ratio: 0.09
    ema_halflife_in_thousands: 1e-3
```

Both use the same `EMAModel` class but independent hyperparameters and instances.

---

## 5. Collapse monitor and LR scheduler

**`CollapseMonitor`** (`collapse_monitor.py` L51): detects SSL representation collapse. Configured
under `train_logging.collapse_monitoring`: `enabled`, `compute_frequency`, `log_frequency`, and
per-metric blocks under `metrics:` (`effective_rank`, `singular_values`, `dimension_variance`,
`prototype_entropy`, `ema_beta`), each with `tensor_source: student|teacher|both` and
`forecast_aggregation: all|aggregate_only|per_step_only` for latent sequences. Metrics: RankMe
effective rank, top-k singular values + concentration ratio, per-dimension variance min/mean/max,
DINO prototype entropy, and current EMA teacher beta (via `EMATeacher.get_current_beta`). Computed
in `Trainer.train()` on `should_compute()` steps, logged on `should_log()` steps.

**`LearningRateScheduler`** (`lr_scheduler.py` L22): three phases — warmup → decay → cooldown —
configured by `training_config.learning_rate_scheduling`: `lr_start`, `lr_max`, `lr_final_decay`,
`lr_final`, `num_steps_warmup`, `num_steps_cooldown` (decay steps = total − warmup − cooldown,
auto-adjusted if too small), `policy_warmup: linear|cosine`, `policy_decay:
linear|exponential|cosine|sqrt|constant` (use `constant` for warmup→constant→cooldown),
`policy_cooldown: linear`, and `parallel_scaling_policy: const|sqrt|linear` which scales `lr_max`
by (total batch size). Steps once per batch; on run continuation it replays `istep` steps to
restore state.

---

## 6. Gotchas for modifying training code

- Batch size is *not* a config scalar: it is the sum of enabled `model_input.*.num_samples`
  (`get_batch_size_from_config`). Effective batch size = per-GPU × `world_size_original`.
- Stage configs inherit: training → validation → test; anything under `losses`, `model_input`,
  `target_input` can be switched off per stage with `enabled: False`.
- `EMATeacher` requires `with_fsdp: False`; FSDP2 checkpoints go through DTensor
  `full_tensor()` gathering — keep `_get_full_model_state_dict` in sync if changing sharding.
- With `validate_with_ema` enabled, saved checkpoints contain the **EMA** weights, not the raw
  student (`_get_full_model_state_dict`, trainer.py L658).
- Adding a loss module: subclass `LossModuleBase`, return `LossValues`, export it in
  `loss_modules/__init__.py` (the config `type:` string is resolved with `getattr` on that
  package), and make sure a matching target-and-aux calculator exists for its targets.
- Adding a target-and-aux calculator: subclass `TargetAndAuxModuleBase` and register it in
  `get_target_aux_calculator()` (`target_and_aux_utils.py`), following the string-or-dict config
  pattern.
