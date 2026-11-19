# Fine-tuning sdnbfpve towards the operational analysis (f79g20uw, inference ft0)

sdnbfpve fine-tuned towards the operational analysis on its own years: train
2016-01-02..2022-12-31, validate 2023-10..12. The final model is **f79g20uw, mini-epoch 8**;
its 6-month inference over sdnbfpve's preprint dates (as sd0) is **ft0**.

## Code

Branch `shmh40/dev-ssl-diffusion/finetune-analysis`, the exact code these runs used:

* `develop-ssl-diffusion-v1` at caf2a256 (#2760, 2026-08-14),
* + the ssl-diffusion inference fixes up to be4a6a0e "Fix all ranks taking the same samples"
  (as on `mk/dev-ssl-diff/inference-branch`: no target writing, empty/NaN target checks),
* + `iluise/develop-ssl-diffusion-v1/eval-no-target` (merged in full, tip 5ed2b279),
* + on this branch: the real-time/operational-analysis readers, NaN-filling of target
  channels missing from a store (`fill_missing_channels`), freeze/eval-mode patterns under
  DDP (`eval_modules`), export of prediction-only output; these configs and `uv.lock`.

## Set up (once, on an ag login node)

```bash
git clone -b shmh40/dev-ssl-diffusion/finetune-analysis git@github.com:ecmwf/WeatherGenerator.git
git clone <WeatherGenerator-private>       # main, next to it: ../WeatherGenerator-private
cd WeatherGenerator
UV_PYTHON_PREFERENCE=only-managed uv sync --frozen --extra gpu     # uv >= 0.11; builds .venv
```

Run everything from this `WeatherGenerator` directory. Checkpoints and results go to the
shared working directory of `../WeatherGenerator-private/hpc/hpc2020/config/paths.yml`
(`/ec/res4/scratch/ecm8774/weathergen/shared_work/`), where `sdnbfpve` is.

## Fine-tune: three jobs, each from the previous run's mini-epoch 8

Each job prints its new run id at the top of its log (`finetune-<job>.log`).

```bash
C=config/finetune_analysis_2022
sbatch $C/ag_finetune.sbatch $C/stage1_decoder.yml sdnbfpve -1                      # stage 1: decoder, 4 GPUs, ~3 h
sbatch -N 4 -t 24:00:00 $C/ag_finetune.sbatch $C/stage2_latent.yml <STAGE1_RUN> 8   # stage 2: 16 GPUs, ~16 h
sbatch $C/ag_finetune.sbatch $C/stage1_decoder.yml <STAGE2_RUN> 8                    # refit: decoder, 4 GPUs, ~3 h -> final
```

Original runs: stage 1 `xnbrogph`, stage 2 `gi0v1zz7`, refit `f79g20uw`. `stage3_rollout.yml`
(12-step rollout) was not used for f79g20uw.

Quick test first (1 GPU, ~10 min):

```bash
sbatch -t 00:30:00 --gpus-per-node=1 --ntasks-per-node=1 $C/ag_finetune.sbatch $C/stage1_decoder.yml sdnbfpve -1 \
    training_config.num_mini_epochs=1 training_config.samples_per_mini_epoch=16 validation_config.samples_per_mini_epoch=8 \
    training_config.learning_rate_scheduling.num_steps_warmup=4 training_config.learning_rate_scheduling.num_steps_cooldown=4
```

## Inference: 40 x 6 h forecasts, 2023-06-03 .. 2023-12-09 (as sd0)

```bash
sbatch --array=1-95%10 config/finetune_analysis_2022/ag_inference.sbatch <REFIT_RUN> 8 ft0
```

95 array tasks x 8 forecasts (start times every 6 h), 4 GPUs each, into
`<shared>/results/ft0NNNN/`. Link them into one run (`ft0`) with `link_results.sh` for scoring.

## What the model sees

* Inputs (from the sdnbfpve checkpoint): the latest operational analysis available at the
  end of the 6-h input window (`ERA5_in`: for the window [T, T+6h) the analysis valid at T)
  and the satellite and surface observations in that window. The first output is valid at
  T+6h; exported forecasts are dated T.
* Target (`streams/target.yml`): the operational analysis
  `aifs-od-an-oper-0001-mars-n320-2016-2026-6h-v4-for-single-v3`, with weights z 100, t 20,
  msl 50, 2t 30, 10u/10v 2, the rest 1 (strd/ssrd 0).
* The IASI stores end on 2023-05-31: the 2023-06..12 inference (like sd0) has no IASI.
* `streams_era5stats/` and `streams_sdnbinputs/`: alternative stream sets used only for
  scoring comparisons, not for f79g20uw.
