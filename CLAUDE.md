# WeatherGenerator — Claude Code Guide

## Running training

```bash
uv run train --base-config config/config_diffusion.yml
```

Request an interactive GPU node first with `agpu`, then run the command above.

## Debugging training failures

### General approach

1. Run training and capture the full traceback — the first error is usually the root cause; later errors are often cascades.
2. Read the file at the crashing line before editing anything.
3. Verify the fix in isolation if possible (small unit test or `python -c "..."`) before re-running the full job.

### Common error patterns

#### `KeyError` / `AttributeError` in loss modules

`preds.latent` is a list indexed by output step. The shape depends on `output_offset` and `num_steps` in the forecast config:

- Index 0 may hold `{"posteriors": ...}` (encoder posteriors) rather than a latent state.
- Always guard with `.get("key") is not None` instead of `if pl:` — a non-empty posteriors dict is truthy but does not contain `"latent_state"`.

```python
# safe
pred_tokens_all = [pl["latent_state"].z_pre_norm for pl in preds.latent if pl.get("latent_state") is not None]
```

#### `TypeError: unexpected keyword argument` in data sampler

`MultiStreamDataSampler._build_stream_data` has several call sites. If you add a parameter, check every call site — the method signature and each caller must agree.

#### `AssertionError: ada_ln_aux should not be provided when diffusion model conditioning is disabled`

`ForecastingEngine.forward()` in `fe_diffusion_model=True` mode asserts `ada_ln_aux is None`. Pass `ada_ln_aux=None` in `DiffusionForecastEngine.denoise()` until conditioning is wired into the network blocks.

#### `bdb.BdbQuit` — process exits silently

A `breakpoint()` call was left in the code. Running non-interactively (batch job, subprocess) causes Python's debugger to immediately quit the process. Search for and remove stray `breakpoint()` calls before submitting jobs:

```bash
grep -rn "breakpoint()" src/
```

### Key data flow: DiffusionForecastEngine

```
MultiStreamDataSampler._get_batch()
  → source_batch  (X_t in source_tokens_cells)
  → target_batch  (X_{t+1} in source_tokens_cells, when mode=="diffusion_forecast")

trainer.train():
  1. target_aux.pre_compute(source_batch, target_batch)   ← runs BEFORE model.forward
        encodes X_{t+1} via frozen encoder
        writes tokens into source_batch.samples[0].meta_info["ERA5"].params["diffusion_target_tokens"]
  2. preds = model(source_batch)
        DiffusionForecastEngine.training_forward():
          y = meta_info["ERA5"].params["diffusion_target_tokens"]  # X_{t+1}
          c = tokens                                               # X_t (conditioning)
          adds EDM noise → calls denoise(x=y+n, c=c, sigma)
  3. targets_and_auxs = target_aux.compute(target_batch)
        reuses _pending_tokens set by pre_compute (no second encoder pass)
        returns diffusion_latent = encoded X_{t+1}
  4. loss(preds, targets_and_auxs)
        compares denoised prediction against diffusion_latent
```

### Channel/normalization mismatch trap

`source_exclude` and `target_exclude` differ for ERA5 streams (`skt` vs `slor`/`sdor`). Never reuse `output_data` (target-normalized, target-channels) as a drop-in for source input. When building X_{t+1} as a source-side input, collect it explicitly as `"source"` type:

```python
future_input_data = [collect_datasources(stream_ds, idx + step_delta, "source", self.rng)]
```

### Config levers for the diffusion training mode

In `config/config_diffusion.yml`:

```yaml
training_mode: ["masking", "diffusion_forecast"]  # enables X_{t+1} target conditioning
num_steps: 1
offset: 1   # target batch is one step ahead of source batch
```

With `offset: 1, num_steps: 1`: `output_steps=2`, `output_idxs=[1]`. The posteriors slot lives at index 0 of `preds.latent`; the diffusion latent state is at index 1.

## Project layout (relevant to training)

```
src/weathergen/
  model/
    diffusion.py          # DiffusionForecastEngine — training_forward, denoise, inference_forward
    engines.py            # ForecastingEngine (the underlying transformer)
  train/
    trainer.py            # main training loop; pre_compute hook lives here
    target_and_aux_diffusion.py   # DiffusionLatentTargetEncoder (frozen encoder + pre_compute)
    target_and_aux_module_base.py # base class with no-op pre_compute
    loss_modules/
      loss_module_latent_diffusion.py  # latent-space EDM loss
  datasets/
    multi_stream_data_sampler.py  # batch construction; diffusion_forecast mode here
config/
  config_diffusion.yml    # main config for diffusion experiments
```
