---
name: weathergen-inference-diagnostics
description: Run sampling/ODE diagnostics (per-step maps and spectra along the denoising trajectory) or the latent-RMSE-vs-lead-time curve on an already-trained WeatherGenerator run, and attach a backbone decoder to a run that has none. Use when asked to diagnose a diffusion or flow-matching sampler, to plot latent RMSE during rollout, or when inference fails with "assert len(outputs_physical) == 1", "Empty preds but non-empty targets", or a request for physical output from a model trained with a latent-only loss.
---

# Inference diagnostics on a trained run

Both diagnostics run inside an ordinary `uv run inference` job on an existing run — never
retraining, never a separate `evaluate` step. Every flag below is passed via `--options`.

Both features must exist in the *checked-out working copy*. Job launchers that submit a snapshot of
the run's original training code will not have them — run these interactively.

## Which one is being asked for

| Ask | Section | Needs a physical decoder? |
|---|---|---|
| latent RMSE vs lead time, rollout error curve | A | no |
| maps / spectra / `x0_hat` / trajectory inspection | B | **yes** |

Start from A when the run has no decoder — it is the diagnostic that works unconditionally.

## A. Latent-RMSE rollout curve

RMSE between the rolled-out latent and the encoded truth latent, per lead step. Pure latent space,
so it works on latent-only runs as-is.

```bash
uv run inference --from-run-id <RUN-ID> --options \
  test_config.start_date=<START> test_config.end_date=<END> \
  test_config.samples_per_mini_epoch=1 test_config.output.num_samples=0 \
  test_config.latent_rollout_rmse=True \
  training_config.forecast.num_steps=16 diffusion_rollout=True \
  fe_diffusion_num_ensemble_members=1 fe_diffusion_num_steps=10 \
  'validation_config.validation_noise_levels=[]' \
  data_loading.num_workers=0
```

For the flow-matching engine replace the two `fe_diffusion_*` flags with
`fe_flow_num_ensemble_members=<N> fe_flow_sampler=ode fm_num_steps=10 fm_sde_sigma=0.0`.

- Take `<START>`/`<END>` from the run's own `validation_config` in `models/<RUN-ID>/model_<RUN-ID>.json`.
- `output.num_samples=0` skips the zarr write: it is not needed for the curve, it costs hundreds of
  MB per sample, and it is the code path that requires a decoder.
- Asserts `forecast.offset == 0`.
- Output: `results/<inference-run-id>/line_plots/compare_rmse_global_<inference-run-id>_latent.png`
  plus a JSON sidecar of the values.
- 1 sample / 1 member is the smoke test. Scale `samples_per_mini_epoch` for a real average; several
  ensemble members can occupy an entire large GPU, and fragmentation OOM is fixed with
  `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (without it the process may hang at a
  `(Pdb++)` post-mortem prompt instead of exiting).
- Interpret against the climatology floor of **that** latent space (√2·σ_anom of its own encoder).
  A floor measured on a different encoder does not transfer.

## B. Sampling (ODE) diagnostics

Decodes intermediate sampler states and renders, per ODE step, `x_t`, `x0_hat = D_t(x_t)`,
`decode(z)` and the ground truth, as maps and power spectra. `decode(z)` vs truth is the control
that separates decoder error from sampler error.

```bash
uv run inference --from-run-id <RUN-ID> --options \
  test_config.start_date=<START> test_config.end_date=<END> \
  test_config.samples_per_mini_epoch=1 test_config.output.num_samples=1 \
  training_config.forecast.num_steps=1 'validation_config.validation_noise_levels=[]' \
  diffusion_rollout=True fe_diffusion_num_ensemble_members=1 fe_diffusion_num_steps=10 \
  diag_ode_maps=True diag_stream=<STREAM> 'diag_channels=["<CH1>", "<CH2>"]' \
  diag_latent_channels=128 data_loading.num_workers=0
```

| flag | meaning | default |
|---|---|---|
| `diag_ode_maps` | master switch | `False` |
| `diag_stream` | stream to decode | `ERA5` |
| `diag_channels` | physical channels to plot | `["2t", "q_850"]` |
| `diag_ode_every_n_steps` | record every n-th ODE step (two decoder passes each) | `1` |
| `diag_latent_channels` | latent channels in the latent spectra | `128` |

`diag_stream` must be a stream of the run's config, and each `diag_channels` entry must appear in
that stream's `val_target_channels`:

```bash
python3 -c "
import json
d = json.load(open('models/<RUN-ID>/model_<RUN-ID>.json'))
print(list(d['streams']))
print(d['streams']['<STREAM>']['val_target_channels'])
"
```

Output: `results/<inference-run-id>/plots/ode_diagnostics/{maps,spectra}/`.

Prerequisites, and what happens when they fail: a non-diffusion run, a stage other than
`inference`, a forecast engine without a `.diagnostics` hook, or a `diag_stream` absent from
`cf.streams` each log a warning and silently disable the diagnostic. A missing or duplicated
`LossPhysical` term asserts instead.

## Failure modes

Both of these mean the run has no usable physical decoder, not that a flag is wrong:

- `AssertionError` at `assert len(outputs_physical) == 1` (`utils/validation_io.py`) — the active
  test config has no `LossPhysical` term.
- `AssertionError: Empty preds but non-empty targets` — a `LossPhysical` term exists but the model
  built no decoder, so nothing was predicted while targets were still loaded.

A run trained with a latent-only loss has no decoder weights in its checkpoint and builds none at
load time: `Model.__init__` creates `embed_target_coords` / `target_token_engines` / `pred_heads`
only when `LossPhysical` appears in `training_config.losses` or `validation_config.losses`
(`test_config` is never inspected). Adding the loss alone therefore yields a *randomly initialised*
decoder, which is worse than none — the plots look plausible and mean nothing.

To decode anyway, borrow the decoder of the backbone the run was initialised from: read
`references/decoder-overlay.md` and generate the overlay with
`scripts/make_decoder_overlay.py`. If no suitable backbone exists, say so and offer section A
instead rather than producing physical plots from an untrained decoder.
