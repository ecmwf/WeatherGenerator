---
name: weathergen-pair-inference
description: Run a rollout inference that pairs one WeatherGenerator run's backbone (its encoder + forecast engine) with a *different* run's trained physical decoder — e.g. a diffusion-training run + a decoder-finetune's decoder from a sibling. Use when asked to run inference / a 40-step rollout / spectra on a "backbone-decoder pair", to test whether a decoder finetune generalises to a backbone it was not trained on, or to run physical-space inference on a latent-only run using a *specific other run's* decoder rather than its own backbone's.
---

# Backbone-decoder pair inference

This runs `uv run inference` on a **backbone** run (supplies the encoder + forecast engine via
`--from-run-id`) with the **decoder** of some **other** run attached through a decoder-overlay
config (`load_decoder_chkpt`). Nothing is retrained.

It builds on the `weathergen-decoder-overlay` skill's machinery (`make_decoder_overlay.py`,
`references/decoder-overlay.md`), but the intent is different:

| | `weathergen-decoder-overlay` | this skill |
|---|---|---|
| decoder source | the run's **own** `load_chkpt` backbone (default) | an **arbitrary other run** you name (`--backbone <DECODER-RUN>`) |
| typical use | make a latent-only run produce *any* physical output | **compare** decoders / lineages: does run B's finetuned decoder work on run A's forecast engine? |

Example: backbone `c3nvsqu8` (latent-only diffusion training from `cw6a4szu@8`) + decoder
`nu2d0ew0` (a decoder-finetune of the *sibling* `bwmk4ktc`). Both share the `cw6a4szu@8` lineage,
so the decoder shapes match and the pairing is meaningful.

## ⚠️ Branch check FIRST — this skill is broken on `mh/fix/decoder-load-not-on-continuation`

That branch changed `init_model_and_shard` (`model_interface.py`) to skip the
`load_decoder_chkpt` overlay whenever `run_id_contd is not None`. The intent was to stop a
**training** `train_continue` from reverting a decoder-finetune's decoder back to the borrowed one
on every chained-job restart — a real bug that hit all 12 dcfts of the 2026-09 sweep, 4-6 times each.

But `run_id_contd` is **not** a train_continue signal. `run_train.py` passes
`trainer.inference(cf, devices, args.from_run_id, args.mini_epoch)`, and `--from-run-id` is
mandatory for inference — so on that branch the overlay is skipped on **every** inference run, and
this skill silently produces output from the backbone's own decoder instead of `<DECODER-RUN>`'s.

**How to detect it.** The run still completes, exit 0, no warning. In `logs/<run>/log.txt`:

| | working | broken |
|---|---|---|
| overlay log line | `Loading decoder weights from id=<DECODER-RUN> ...` | `Run is a continuation, decoder not loaded separately; ...` |
| tensor count line | `Loaded 99/99` (or `115/131`) `decoder tensors` | **absent** |

The missing `Loaded N/M decoder tensors` line is the tell — it is also the verification signal
step 6 below relies on, so the check that would catch this disappears along with the overlay.

**What to do instead.** Prefer loading a decoder-finetune run *directly* as the main model, which
needs no overlay at all: a dcft checkpoint is already complete (e.g. `ei90hx2k` = 1315 tensors =
1002 encoder + 198 forecast_engine + 115 decoder), and its forecast engine is bitwise identical to
the diffusion run it came from (198/198 — `.*fe.*` is frozen during dcft). So:

```bash
uv run --offline inference --from-run-id <DCFT-RUN> --options sigma_max=50 ...
```

`sigma_max=50` is **mandatory**: dcft configs carry `sigma [0.4, 0.41]`, and `diffusion.py:560`
does `sigma_max_eff = min(cf.sigma_max, sigma_max_train)`, so without it the ODE starts at 0.41 —
essentially already denoised. Note a pure latent-diffusion run has **zero** decoder tensors
(`model.py:484` builds a decoder only when `"LossPhysical" in loss_terms`), which is precisely why
the overlay exists: it is the only way to get physical output from a diffusion run *before* a dcft
of it exists. Use this skill's overlay path only for genuine mix-and-match — decoder A on
forecast engine B — and on a branch where the overlay still runs.

## 1. Pre-flight — confirm the pair is valid

```bash
cd /e/project1/weatherai/hauschulz1/WeatherGenerator

# backbone must be decoder-free (latent-only): expect [] and load_decoder_chkpt: None
uv run python -c "
import json, torch
b='<BACKBONE>'; d='<DECODER-RUN>'
mj=json.load(open(f'models/{b}/model_{b}.json'))
print(b, 'load_chkpt', mj.get('load_chkpt'), '| load_decoder_chkpt', mj.get('load_decoder_chkpt'))
sb=torch.load(f'models/{b}/{b}_latest.chkpt', map_location='meta', mmap=True, weights_only=True)
print(b, 'decoder keys:', len([k for k in sb if k.startswith(('embed_target_coords','target_token_engines','pred_heads'))]))
sd=torch.load(f'models/{d}/{d}_latest.chkpt', map_location='meta', mmap=True, weights_only=True)
dk=[k for k in sd if k.startswith(('embed_target_coords','target_token_engines','pred_heads'))]
print(d, 'decoder keys:', len(dk))
import re
print(d, 'decoded stream(s):', {re.search(r'\\.(ERA5|[A-Z0-9_]+)\\.', k).group(1) for k in dk if re.search(r'\\.(ERA5|[A-Z0-9_]+)\\.', k)})
"
```

Require: backbone has **0** decoder keys; decoder-run has a full decoder keyed to **one** stream
(usually `ERA5`); the decoder-run reached its final mini_epoch. Same `load_chkpt.run_id` lineage
between backbone and the decoder-run's own source is the usual reason the shapes line up — if the
lineages differ, read `weathergen-decoder-overlay/references/decoder-overlay.md` §"Block 2" and
check `T`/`C` shapes explicitly before spending GPU time (a channel/width mismatch is a hard load
error; a stream-*name* mismatch is silent and gives you random-decoder output).

**Branch check:** `git merge-base --is-ancestor 6bc62d46 HEAD` — the `MLP.lnorm` fix must be
present (see `weathergen-dcft`/`CLAUDE.md`), or the checkpoint load `KeyError`s or hangs.

## 2. Generate the overlay config

```bash
uv run python .claude/skills/weathergen-decoder-overlay/scripts/make_decoder_overlay.py \
  --model <BACKBONE> --backbone <DECODER-RUN> \
  -o config/inference_decoder_overlay_<BACKBONE>_<DECODER-RUN>.yml
```

`--model` = backbone (inference is run *as* this run); `--backbone` = the run whose decoder is
borrowed. Name the file `<BACKBONE>_<DECODER-RUN>` (the RUN-LOG convention), not the script's
default of just the decoder run. The script prints the decoded stream and which of the backbone's
own streams got `reconstruct: false`.

**The generated `.yml` must carry a commented-out example launch command in its header**, so the
file is self-documenting and can be run without re-deriving the flags. `make_decoder_overlay.py`
emits this automatically; if you hand-write or hand-edit an inference config, add the same block.
It shows the `launch-slurm.py --stage inference` form with `--wgen-dir` (use the current checkout,
not the run's stored code snapshot -- this replaces the old `--code-from-home`), the standard
`--options` set, and the `sigma_max` caveat for decoder-finetuned runs.

**No top-level key may appear in both the overlay and `--options` when submitting through
`launch-slurm.py`.** MLflow logs each top-level config key as one immutable param, so writing it
twice fails submission with
`INVALID_PARAMETER_VALUE: Parameter with key <key> was already logged`. The overlay owns
**`streams`, `test_config`, `validation_config`** and `load_decoder_chkpt`, so the generated file
carries `samples_per_mini_epoch` / `output.num_samples` inside `test_config` and
`validation_noise_levels: []` inside `validation_config` -- edit them there, never on the CLI. It
fails one key at a time, so fixing `test_config` just surfaces `validation_config` next; audit the
whole `--options` list against the yml's top-level keys before resubmitting. `--options` is for
keys the overlay does not define (`sigma_max`, `training_config.*`, `diffusion_rollout`,
`fe_diffusion_*`, `data_loading.*`). `--no-register` skips the tracker and sidesteps all of this,
but only use it for expendable runs. The bare `srun ... uv run inference` form in section 5
registers nothing, so overrides in `--options` are fine there.

## 3. ALWAYS ask the user about rollout frequency

The overlay copies the decoded stream verbatim from the decoder-run's config, whose resolved
`ERA5` block **often has no `frequency` field**. Before verifying/launching, ask the user with
`AskUserQuestion` (header "Frequency"):

| Option | Action |
|---|---|
| **(a) Default ERA5 frequency** | Leave the overlay as generated — do **not** add a `frequency` field. |
| **(b) 6-hourly** | Add `frequency: 06:00:00` to the `ERA5:` stream block (4-space indent, as a sibling of `name:`/`channel_weights:`), just before the top-level `test_config:` key, with a one-line `# added by hand` comment. |
| **(c) Custom** | Ask the user for the exact value (e.g. `03:00:00`, `12:00:00`), then add `frequency: <value>` the same way as (b). |

Do not skip this question or assume 6-hourly — the correct value depends on what lead-time-per-step
the user wants for the rollout.

## 4. Verify the overlay before burning GPU time

```bash
uv run python -c "
import yaml
f='config/inference_decoder_overlay_<BACKBONE>_<DECODER-RUN>.yml'
d=yaml.safe_load(open(f)); st=d['streams']
recon=[k for k,v in st.items() if not (isinstance(v,dict) and v.get('reconstruct') is False)]
print('reconstructed stream(s):', recon, '(expect exactly one)')
print('ERA5.frequency         :', st.get('ERA5',{}).get('frequency'), '(matches the user choice above)')
p=d['validation_config']['losses']['physical']
print('LossPhysical           :', p['type'], 'weight', p['weight'], '(expect weight 0.0)')
print('load_decoder_chkpt     :', d['load_decoder_chkpt'])
print('offset / num_steps_input:', d['test_config']['forecast']['offset'], '/', d['test_config']['model_input']['forecasting']['num_steps_input'])
"
```

Expect: exactly one reconstructed stream, one `LossPhysical` at `weight: 0.0`,
`load_decoder_chkpt.run_id == <DECODER-RUN>`, `offset/num_steps_input == 1/1`, and `frequency`
matching the answer to step 3.

**Then confirm the overlay actually ran** — the config being right does not mean the code applied
it (see the branch warning at the top). After the run starts:

```bash
grep -E "Loading decoder weights from id=|Loaded [0-9]+/[0-9]+ decoder tensors|not loaded separately" logs/<RUN-ID>/log.txt
```

A healthy overlay prints both `Loading decoder weights from id=<DECODER-RUN>` and
`Loaded N/M decoder tensors` with **N > 0**. `Loaded 0` raises a `RuntimeError` since the
2026-09-07 fix, but a *skipped* overlay prints neither line and is silent — treat the absence of
`Loaded N/M decoder tensors` as a failed run, not as a passing one.

## 5. Run it on a GPU node — NOT the login node

The user runs these on a GPU node (`agpu4_weatherai` in their shell). The **interactive** alias
(`srun ... --pty bash -i`) cannot be driven from a tool call — it needs a TTY. Use the
**non-interactive** equivalent (same flags, no `--pty bash -i`, command appended). This has been
verified to work from a tool call: it queues briefly, gets a node, runs, and returns.

A 40-step inference takes ~10-20 min, past the 2-minute foreground limit — run it **in the
background** (`run_in_background: true`), or write it to a small script and background that.

```bash
srun --ntasks-per-node=1 --cpus-per-task=72 --mem=0 --time=1:30:00 --gres=gpu:1 \
     --partition booster --account=weatherai \
  bash -lc 'cd /e/project1/weatherai/hauschulz1/WeatherGenerator && \
  uv run inference --from-run-id <BACKBONE> \
    --config ./config/inference_decoder_overlay_<BACKBONE>_<DECODER-RUN>.yml \
    --options \
    test_config.samples_per_mini_epoch=1 \
    test_config.output.num_samples=1 \
    data_loading.num_workers=0 \
    training_config.forecast.num_steps=40 \
    "validation_config.validation_noise_levels=[]" \
    diffusion_rollout=True \
    fe_diffusion_num_ensemble_members=1'
```

- **`samples_per_mini_epoch=1` is only valid for this bare-`srun` form** (`--ntasks-per-node=1`,
  world_size 1). Through `launch-slurm.py` a `--nodes 1` job has **4 ranks**, and
  `multi_stream_data_sampler.py:238` computes
  `len = ((samples // world_size) // (batch_size * workers)) * (batch_size * workers)` -- so 1
  sample becomes `1 // 4 == 0`, every rank gets an empty slice, and the run finishes in minutes
  with `iter_start=0, iter_end=0, len=0`, a `nan` metric and no output. `check_samples` does not
  catch it: it validates against the date range and still prints "Sufficient available samples".
  Use a multiple of `world_size * batch_size * max(1, num_workers)` (the generator defaults to 8).
- `num_steps=40`, 1 sample is the smoke-test shape (matches every overlay inference in `RUN-LOG.md`).
  Change `forecast.num_steps` / `samples_per_mini_epoch` / `output.num_samples` for a bigger run.
- Test window is the backbone's own `validation_config` date range; with 1 sample only the first
  timestamp is used. Read it if the user asks:
  `python3 -c "import json;vc=json.load(open('models/<BACKBONE>/model_<BACKBONE>.json'))['validation_config'];print(vc['start_date'],vc['end_date'])"`
- Run one at a time — a shared GPU has OOM'd overlay inference before (`ssu6rncm` in `RUN-LOG.md`).
- For an unattended / much longer run, `launch-slurm.py --stage inference --from-run-id <BACKBONE>
  --config <overlay>` submits a proper batch job instead.

## 5b. Batches — parallel SLURM pipeline (preferred for >2 runs, or inference→evaluate)

Do **not** loop `srun` sequentially for a batch. `launch-slurm.py --pipeline <yaml> --code-from-home`
submits each stage as its own `sbatch` job; with `parallelize: true` the inference stages have no
cross-deps and SLURM runs them **concurrently** on separate nodes, and an `evaluation` stage that
lists them in `run_ids` gets an automatic `--dependency=afterany:<all of them>` barrier.

```yaml
# scratchpad/pipe_<name>.yml
parallelize: true
stages:
  - name: infnu                       # <= 15 chars, [a-zA-Z0-9] only (no _ or -)
    stage: inference
    from_run_id: bwmk4ktc
    nodes: 1
    slurm_args: ["--ntasks=1", "--time=02:00:00"]   # --ntasks=1 -> 1 GPU / 1 rank
    config_files: [config/inference_decoder_overlay_bwmk4ktc_nu2d0ew0.yml]
    options:
      - test_config.samples_per_mini_epoch=1
      - test_config.output.num_samples=1
      - training_config.forecast.num_steps=40
      - diffusion_rollout=True
      - fe_diffusion_num_ensemble_members=1
      - "validation_config.validation_noise_levels=[]"
      - data_loading.num_workers=0
  - name: inffw                        # ... one stage per pair, all independent -> parallel
    stage: inference
    from_run_id: bwmk4ktc
    nodes: 1
    slurm_args: ["--ntasks=1", "--time=02:00:00"]
    config_files: [config/inference_decoder_overlay_bwmk4ktc_fw8wb0v1.yml]
    options: [ ... ]
  - name: evalblk1
    stage: evaluation
    eval_config: config/evaluate/eval_spectral_block1.yml
    run_ids: [STAGE.infnu, STAGE.inffw, ku6i70v3]   # STAGE.* refs + hardcoded ids mix freely
```

```bash
../WeatherGenerator-private/hpc/launch-slurm.py --pipeline scratchpad/pipe_<name>.yml --code-from-home
```

- **`--code-from-home` is required** whenever the run depends on uncommitted working-copy code
  (e.g. the `latent_rollout_rmse` chunking in `multi_stream_data_sampler.py`). Without it, an
  inference stage runs from the *training run's code snapshot*, not `~/WeatherGenerator`. The
  configs (`config_files` + the auto-generated `config_command_line.yaml` holding `options`) are
  always copied from the working tree, uncommitted included.
- **run_ids** are printed in the launch summary (`… 'infnu' | run_id=<id> | job_ids=[…]`) and
  under `logs/<id>/` — grab them there, not from `results/`.
- Limits: **≤ 8 stages** per pipeline (`STAGES_MAX_ALLOWED`); Block 1 (5 inf + 1 eval) fits, Block
  1 + 2 (6 inf + 2 eval) = exactly 8. Bigger sweeps → split into multiple pipelines.
- `_launch_inference` defaults to `--ntasks=4` (4-rank FSDP inference); `slurm_args: ["--ntasks=1"]`
  overrides it to a single GPU. Multi-rank is fine only when `samples_per_mini_epoch ≥ ntasks`
  (each rank needs ≥ 1 sample — otherwise ranks get 0 samples and do nothing).
- **Memory:** `hpc/jupiter/weathergen_slurm.sh` already exports `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  A heavy `latent_rollout_rmse` combo (25 ODE steps × 4 samples) can still OOM on sample 2 — the
  per-sample preds / target-aux GPU latents were the leak; if it recurs, drop
  `latent_rollout_rmse_chunk` to 1, cut `forecast.num_steps`, or run 4 parallel 1-sample stages
  and average the JSON sidecars.
- **`--no-register`** for expendable diagnostic runs — skips the experiment-tracker upload.
  Needed as a workaround: `launch-slurm.py`'s mlflow `log_config` crashes (`RestException:
  INVALID_PARAMETER_VALUE: Parameter with key validation_config was already logged`) when an
  `options` dotlist sets a `validation_config.*` key AND a `config_files` overlay also has a
  `validation_config:` block — it collides on the flattened `validation_config` param and the
  launcher dies *before submitting the remaining stages*. Either `--no-register`, or bake the
  setting into the overlay file instead of passing it via `options` (e.g. put
  `validation_noise_levels: []` inside the overlay's `validation_config:`).
- Verified 2026-09-03 (pipeline `ogudhpvw`): two `stage: inference` runs with `latent_rollout_rmse`
  ran in parallel, `--code-from-home` picked up the chunked truth-encoding + memory fix.

## 6. During and after

- In the run log, confirm `Loading decoder weights from id=<DECODER-RUN>` and **no `Missing keys`**
  naming `target_token_engines` / `pred_heads` / `embed_target_coords` (a silent stream-name
  mismatch shows up here).
- A fresh `run_id` is minted. Read it from the **inference's own stdout** — it logs
  `… results/<run_id>/…` / `Saved … to results/<run_id>/`. Do **not** identify it by diffing
  `ls results/` before/after: `results/` is a shared dir where hundreds of other users' run dirs
  appear mid-run, and a diff will grab the wrong one (this has bitten a batch script). **Log it in
  `RUN-LOG.md` immediately** per `CLAUDE.md`:
  backbone run, decoder run + mini_epoch, overlay config path, branch/commit, test date range,
  chosen `frequency`, and `LossPhysical.<stream>.mse.avg` from
  `results/<run_id>/<run_id>_train_metrics.json`. Output (metrics json, `plots/`, a large
  validation zip) lands in `results/<run_id>/`.
