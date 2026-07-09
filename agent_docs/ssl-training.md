# SSL / student-teacher training — the delta over a plain training step

Summary: SSL runs (JEPA-style) change three things relative to
`agent_docs/training-step.md`: the sampler composes masked/multi-view samples
(`training_mode`), targets come from a teacher encoder instead of observations
(target-aux calculators), and losses compare latent predictions
(`LossLatentSSLStudentTeacher`). Representation collapse is monitored separately.

## Sample composition (`multi_stream_data_sampler.py:_get_batch`)

- `mode_cfg.training_mode` selects what enters sources vs targets: `"masking"` →
  masked inputs + `target_coords` as sources, `target_values` as targets;
  `"student_teacher"` / `"latent_loss"` → `network_input` on both sides.
- Masking: `datasets/masking.py:Masker` — `masking_strategy` (+
  `masking_strategy_config`, per-target overridable), healpix-cell (`hl_mask`) and
  `geodesic_disk` variants. `source_to_target` maps multiple student views to one
  teacher target.

## Teachers (`train/target_and_aux_utils.py:get_target_aux_calculator`)

Selected per loss term via `losses.<name>.target_and_aux_calc`:

- `Physical` (default) — observation targets, no teacher.
- `EMATeacher` (`target_and_aux_ssl_teacher.py`) — EMA copy of the student encoder
  (`model/ema.py:EMAModel`); weights update in `update_state_post_opt_step`, i.e.
  after each optimizer step. Optional warm start from `teacher_run_id`.
  Asserts `not cf.with_fsdp` — EMATeacher currently unsupported with FSDP2.
- `FrozenTeacher` — pretrained encoder loaded from a checkpoint
  (`from_pretrained`), stripped to the encoder with fresh latent heads
  (`teacher_utils.py:prepare_encoder_teacher`), never updated.

Target computation (`EncoderTeacher.compute`): teacher forward under `no_grad` →
`get_latent_prediction(0)` → per-loss postprocessing
(`model/ssl_target_processing.py`, via `get_target_postprocessing`) →
`TargetAuxOutput.latent`.

## Student side

- `model.py:Model.predict_latent` emits `LatentState` (register/class/patch tokens)
  plus one prediction per entry in `model.latent_heads`, keyed by name.
- Loss: `losses.<name>.type: LossLatentSSLStudentTeacher`
  (`train/loss_modules/loss_module_ssl.py`) compares student latent predictions to
  teacher targets.
- Separate from teachers: `validation_config.validate_with_ema` maintains an EMA
  model just for validation forwards (`trainer.py:run`).

## Collapse monitoring (`train/collapse_monitor.py`)

`CollapseMonitor` (configured under `train_logging.collapse_monitoring`) computes
representation-collapse metrics from preds/targets every N steps in the train loop
and logs them with the regular metrics.

## Coupling & invariants

- The loss-term name is the key that ties everything: `losses.<name>` ↔ latent-head
  name in `model.latent_heads` ↔ `postprocess_targets[<name>]` in the teacher ↔
  `targets_and_auxs[<name>]` in the trainer. `EncoderTeacher.__init__` finds its
  config by scanning `losses` for `type == LossLatentSSLStudentTeacher`. Rename or
  add an SSL loss term → all four places.
- `training_mode` strings are matched by substring in both the sampler (`_get_batch`,
  batch-validity rules in `__iter__`) and stage configs; a new mode needs both sides.
- Teacher EMA updates happen post-optimizer-step for train *and* val calculators
  (both lists are iterated in `Trainer.train`) — a new calculator hook must be added
  to both.
- EMATeacher × FSDP2 is asserted out; use FrozenTeacher or DDP-only for sharded runs.
