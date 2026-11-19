#!/usr/bin/env bash
#SBATCH --job-name=wg-ft-inference
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --account=ecwegen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=inference-%A_%a.log
#
# 40 x 6 h forecasts of a fine-tuned run over sdnbfpve's preprint dates (as runs sd0/ft0):
# array task i = 8 forecasts from source windows starting 2023-06-01T00 + i*48 h, every 6 h;
# tasks 1-95 = 2023-06-03 .. 2023-12-09. Run from the WeatherGenerator checkout root:
#
#   sbatch --array=1-95%10 config/finetune_analysis_2022/ag_inference.sbatch RUN_ID [MINI_EPOCH] [PREFIX]
#
# Output: <shared working dir>/results/<PREFIX><task, 4 digits>/validation_chkpt00000_rank000{0..3}.zip
set -euo pipefail
FROM_RUN_ID=${1:?run to forecast with}; MINI_EPOCH=${2:-8}; PREFIX=${3:-ft0}

export WEATHERGEN_PRIVATE_CONF=$PWD/../WeatherGenerator-private/hpc/hpc2020/config/paths.yml
export LD_LIBRARY_PATH=/usr/local/apps/gcc/15.2.0/lib64:${LD_LIBRARY_PATH:-}
export PYTORCH_ALLOC_CONF=expandable_segments:True

i=${SLURM_ARRAY_TASK_ID:?run as an array job}
RUN_ID=$(printf '%s%04d' "$PREFIX" "$i")
src=$(date -u --date="2023-06-01T00:00 UTC + $((i * 48)) hours" '+%Y-%m-%dT%H:%M')
test_start=$(date -u --date="$src UTC - 6 hours" '+%Y-%m-%dT%H:%M')
test_end=$(date -u --date="$test_start UTC + 306 hours" '+%Y-%m-%dT%H:%M')
echo "$RUN_ID: $FROM_RUN_ID:$MINI_EPOCH, test interval [$test_start, $test_end)"

.venv/bin/torchrun --nproc-per-node=4 src/weathergen/run_train.py inference \
    --from-run-id "$FROM_RUN_ID" --mini-epoch "$MINI_EPOCH" --run-id "$RUN_ID" \
    --private-config "$WEATHERGEN_PRIVATE_CONF" \
    --options \
        data_loading.num_workers=0 data_loading.num_workers_validation=0 data_loading.persistent_workers=false \
        test_config.samples_per_mini_epoch=8 test_config.output.num_samples=8 test_config.shuffle=false \
        test_config.start_date="$test_start" test_config.end_date="$test_end" \
        test_config.forecast.num_steps=40 test_config.forecast.chunk_size=1 \
        test_config.compute_full_validation_loss=false test_config.skip_target_values=true
