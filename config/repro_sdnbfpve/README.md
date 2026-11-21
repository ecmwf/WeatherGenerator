# Reproducing sdnbfpve

sdnbfpve is an 8-step (48 h) forecast model, fine-tuned on operational IFS analysis inputs with ERA5 targets. It is the last of four training stages. This directory holds the exact config for each stage, copied from commit `1298747f` (the code all four stages ran). The values passed as `--options` at launch are written into the stage files. Each stage file was checked against the config saved in its run's checkpoint, and the copied stream definitions against the streams saved there.

| Stage | Config | Original run | Nodes (GPUs) | Starts from | Handed over at step |
|---|---|---|---|---|---|
| 1. Temporal JEPA pre-train | `stage1_jepa_pretrain.yml` (base config) | wduc34ev | 2 (8) | scratch | 44,300 (stopped by time limit) |
| 2. 2-step forecast | `stage2_forecast_2step.yml` | rrvj7vjv | 2 (8) | stage 1 @ 44,300 | 23,920 (stopped by time limit) |
| 3. 8-step rollout, encoder frozen | `stage3_rollout_8step.yml` | nhv6tkln | 8 (32) | stage 2 @ 23,920 | 2,048 (finished) |
| 4. Operational-analysis fine-tune | `stage4_operan_finetune.yml` | sdnbfpve | 8 (32) | stage 3 @ 2,048 | 1,224 (see below) |

Batch size is 1 per GPU on JUPITER GH200, so keep the GPU counts. Stages 1–2 scale the learning rate with sqrt(#GPUs), and stages 3–4 need a global batch of 32. Stages 2–4 are overlays applied with `train_continue` on top of the previous stage's saved config.

- `streams/`: stream definitions for each stage, referenced by `streams_directory` in the stage files.
- `resolved/`: the full config each run actually trained with, as saved in its checkpoint. Use these as the reference when checking a refactored code base. They include values the code fills in at runtime, such as the per-channel `target_channel_weights` list derived from `channel_weights`.

## Hand-overs between stages

- **Stages 1 and 2 did not finish their schedules.** They were configured for 96 and 24 mini-epochs. Each ran 4 chained 12 h jobs and was killed by the time limit, and the next stage started from the latest checkpoint at that point: step 44,300 for stage 1 and step 23,920 for stage 2. Both hand-overs fall in the constant-learning-rate phase. Stage 2 stopped 48 steps before its 512-step cooldown would have started, so neither model was cooled down. To reproduce this, run past the target step and start the next stage from the checkpoint nearest to it. For stage 2, take a checkpoint at or before step 23,968.
- **Stage 3** finished its 8 mini-epochs (2,048 steps), and stage 4 started from its final checkpoint.
- **Stage 4** finished its 4 mini-epochs (1,024 steps). A second chained job then resumed from the step-968 checkpoint and ran one more 256-step mini-epoch at lr ≈ 4e-9, ending at step 1,224. Those are the published weights. The difference from the 1,024-step model is negligible.

## Launch commands (original code, WeatherGenerator-private launcher)

```bash
L=../WeatherGenerator-private/hpc/launch-slurm.py
C=config/repro_sdnbfpve

uv run $L --base-config $C/stage1_jepa_pretrain.yml --nodes 2 --chain-jobs 4
uv run $L --from-run-id <S1> --mini-epoch <N1> --config $C/stage2_forecast_2step.yml --nodes 2 --chain-jobs 4
uv run $L --from-run-id <S2> --mini-epoch <N2> --config $C/stage3_rollout_8step.yml --nodes 8 --chain-jobs 1
uv run $L --from-run-id <S3> --mini-epoch 8    --config $C/stage4_operan_finetune.yml --nodes 8 --chain-jobs 2
```

`<N1>` and `<N2>` are the checkpoints closest to steps 44,300 and 23,920. Read the step from `general.istep` in `models/<id>/model_<id>_chkptNNNNN.json`. Choose by step, not by mini-epoch number, because the numbering skips at every resume.
