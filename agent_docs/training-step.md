# Training step — life of a batch

Summary: `run_train.py` parses the stage and config, then `Trainer.run`
(`src/weathergen/train/trainer.py`) builds samplers, model, optimizer, and loss
machinery, and loops over mini-epochs of `train()` + `validate()` + `save_model()`.
Inside a step: a pre-assembled `ModelBatch` moves to GPU → autocast forward through
`Model.forward` → per-loss-term targets from target-aux calculators → weighted loss →
scaled backward, clip, step, LR update → EMA/teacher state updates. Batching happens in
the dataset, not the DataLoader.

## Setup (`Trainer.run`, trainer.py)

- Entry: `run_train.py:train` → `main` → `run_train` → config merge (see
  `agent_docs/config-system.md`) → `Trainer.run(cf, devices)`.
- `Trainer.init`: derives `validation_cfg`/`test_cfg` from `training_config` via
  `get_active_stage_config`; filters disabled loss/model_input/target_input entries
  (`filter_config_by_enabled`, keys in `train/utils.py:cfg_keys_to_filter`).
- Data: two `MultiStreamDataSampler`s (TRAIN/VAL), wrapped in `DataLoader` with
  `batch_size=None` — the sampler yields complete `ModelBatch` objects (see
  `agent_docs/data-pipeline.md`).
- Model: `model_interface.py:init_model_and_shard` (creates `Model` + `ModelParams`,
  applies DDP/FSDP per `cf.with_ddp`/`cf.with_fsdp`).
- Per loss term in `mode_cfg.losses`: a target-aux calculator
  (`target_and_aux_utils.py:get_target_aux_calculator`) and, jointly, one
  `LossCalculator` per stage.
- Optimizer: AdamW with batch-size-scaled betas/eps (kappa = total batch size;
  SDE scaling rule, see comment at trainer.py:323). `LearningRateScheduler` steps
  per batch. GradScaler unless `optimizer.grad_scaling: false`.

## Step anatomy (`Trainer.train`)

1. `batch.pin_memory()` (if `data_loading.memory_pinning`) → `batch.to_device`.
2. Under `torch.autocast(dtype=cf.mixed_precision_dtype, enabled=cf.with_mixed_precision)`:
   - `preds = model(model_params, batch.get_source_samples())`
   - per loss term: `target_aux.compute(istep, batch.get_target_samples(target_idxs), ...)`
     (teacher forwards run here, under `no_grad`).
3. Outside autocast: `loss_calculator.compute_loss(preds, targets_and_aux, metadata)` —
   weighted sum over loss terms; each term instantiates a `train/loss_modules/` class by
   its config `type`.
4. `update_state_pre_backward` on all calculators (train and val) → `zero_grad` →
   `grad_scaler.scale(loss).backward()` → `unscale_` → `clip_grad_norm_` →
   `grad_scaler.step` → `update` → `lr_scheduler.step()`.
5. `update_state_post_opt_step` (EMA teacher weights update here), optional
   validation-EMA update, throughput tracker, collapse monitor, periodic logging,
   checkpoint every `train_logging.checkpoint` steps (saved as `_latest`).
6. `cf.general.istep += 1`; after the epoch, `dataset.advance()`.

## Model forward (`model.py:Model.forward`)

1. `EncoderModule` (`encoder.py`): EmbeddingEngine → LocalAssimilationEngine →
   Local2Global(Sum|Assimilation)Engine → QueryAggregationEngine →
   GlobalAssimilationEngine → latent tokens + posteriors.
2. Tokens reshaped to (batch, input_steps, ...) and summed over input steps.
3. Per output step: `forecast_engine` advances the latent state; `predict_decoders`
   maps latent → physical per stream (target-coord embedding → per-stream
   target_token_engines → pred_heads; varlen attention over ragged coords);
   `predict_latent` records `LatentState` + per-head SSL latent predictions.
4. Pushforward (`training_config.forecast.pushforward`): intermediate steps advance
   without grad and skip decoding.

## Checkpointing (`Trainer.save_model`)

FSDP: per-param `full_tensor()` gather, rank-0 CPU copy. Written to a `_tmp` file then
atomically renamed. Config saved alongside (`config.save`). Filename:
`<run_id>_chkpt<NNNNN>` or `<run_id>_latest`.

## Coupling & invariants

- Loss-term names are a shared key across four places: `training_config.losses.<name>`
  ↔ target-aux calculators dict ↔ `LossCalculator.loss_calculators` ↔
  `get_target_idxs_from_cfg`. Renaming a loss term touches all of them; for SSL terms
  the model's latent-head names must match too (see `agent_docs/ssl-training.md`).
- Batching lives in `MultiStreamDataSampler`, not the DataLoader (`batch_size=None`).
  Changing batch-size semantics → check `worker_workset`, `__len__`, LR-scheduler step
  count, and the kappa-scaled optimizer betas (trainer.py:326).
- `cf.general.istep` is the global step counter, persisted in the run config;
  run-continuation math also needs `world_size_original` (trainer.py:367).
- `cf.with_flash_attention` requires `cf.with_mixed_precision` (assert in run_train.py).
- Loss is computed outside autocast; only forward + target computation run under it.
- The config is mutated at runtime (istep, streams, run_history) despite the intent
  that it be read-only (TODO at trainer.py:346) — don't cache derived values across
  steps without checking.
