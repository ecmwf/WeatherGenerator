#!/usr/bin/env bash
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

# ag (GH200) twin of inference_cw6a4szu_40step_array_jupiter.sbatch: IDENTICAL
# sampling/date logic, only the platform layer differs --
#   * ECMWF ag account/qos instead of the JSC account/partition;
#   * the self-contained aarch64 uv venv + three runtime exports instead of the
#     module / UCX / MASTER_ADDR block;
#   * torchrun for the 4-way single-node DDP instead of srun --ntasks=4;
#   * this checkpoint is really cw6a4szu (the Jupiter file runs xq65fhca).
#
# Eight 40-step inference samples per array task. Task i writes run cw6iiiii for
# the 8 source windows starting 2023-06-01T12:00 + i * 48 hours
# (NUM_SAMPLES_PER_TASK * INPUT_STEP_HOURS), so tasks tile consecutive dates
# without overlap.
#
# Submit from the WeatherGenerator root:
#   sbatch scripts/inference_cw6a4szu_40step_array_ag.sbatch

#SBATCH --qos=ng
#SBATCH --account=ecwegen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --array=0-15%16
#SBATCH --job-name=wg-cw8-f40
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

FROM_RUN_ID=xq65fhca
MINI_EPOCH=8
RUN_PREFIX="${FROM_RUN_ID:0:4}"
FORECAST_STEPS=40
FORECAST_CHUNK_SIZE=1                 # value proven interactively on ag for this checkpoint
FIRST_SOURCE_START=2023-06-01T12:00
INPUT_STEP_HOURS=6
NUM_SAMPLES_PER_TASK=8
# 1 input step + 1 selected window + (40 forecast steps + offset 1) * 6 h,
# widened by NUM_SAMPLES_PER_TASK windows so all 8 samples survive check_samples.
LOCAL_TEST_RANGE_HOURS=258

task_id=${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array task.}
run_id=$(printf '%s%04d' "$RUN_PREFIX" "$task_id")
# Advance each task by exactly its own NUM_SAMPLES_PER_TASK windows
# (INPUT_STEP_HOURS each) so consecutive tasks tile without overlap or grid drift.
source_start=$(date -u --date="$FIRST_SOURCE_START UTC + $((task_id * INPUT_STEP_HOURS * NUM_SAMPLES_PER_TASK)) hours" '+%Y-%m-%dT%H:%M')
source_end=$(date -u --date="$source_start UTC + $INPUT_STEP_HOURS hours" '+%Y-%m-%dT%H:%M')
test_start=$(date -u --date="$source_start UTC - $INPUT_STEP_HOURS hours" '+%Y-%m-%dT%H:%M')
test_end=$(date -u --date="$test_start UTC + $((LOCAL_TEST_RANGE_HOURS + NUM_SAMPLES_PER_TASK * INPUT_STEP_HOURS)) hours" '+%Y-%m-%dT%H:%M')

weathergen_home=${SLURM_SUBMIT_DIR:?Submit this script from the WeatherGenerator root.}
RESULTS_ROOT="$weathergen_home/results"
expected_result="$RESULTS_ROOT/$run_id/validation_chkpt00000_rank0000.zip"
if [[ -e "$expected_result" ]]; then
    echo "Refusing to overwrite existing inference output: $expected_result" >&2
    exit 2
fi
if [[ ! -x "$weathergen_home/.venv/bin/torchrun" ]]; then
    echo "WeatherGenerator venv not found: $weathergen_home/.venv" >&2
    exit 2
fi
private_config="$(dirname "$weathergen_home")/WeatherGenerator-private/hpc/hpc2020/config/paths.yml"
if [[ ! -f "$private_config" ]]; then
    echo "Missing ag private config: $private_config" >&2
    exit 2
fi

# ag (aarch64) runtime -- the three fixes from the N320 port: resolve the
# checkpoint via the private config (platform-env.py has no ag rule), give
# flash-attn its GLIBCXX (gcc/15.2.0), and de-fragment the allocator (the var
# was renamed from PYTORCH_CUDA_ALLOC_CONF in torch 2.9).
export WEATHERGEN_PRIVATE_CONF="$private_config"
export LD_LIBRARY_PATH="/usr/local/apps/gcc/15.2.0/lib64:${LD_LIBRARY_PATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd "$weathergen_home"
echo "run_id=$run_id source_interval=[$source_start, $source_end)"
echo "test_interval=[$test_start, $test_end)"
.venv/bin/torchrun --nproc-per-node=4 \
    src/weathergen/run_train.py inference \
    --from-run-id "$FROM_RUN_ID" \
    --mini-epoch "$MINI_EPOCH" \
    --run-id "$run_id" \
    --private-config "$private_config" \
    --options \
        data_loading.num_workers=0 \
        data_loading.num_workers_validation=0 \
        data_loading.persistent_workers=false \
        test_config.samples_per_mini_epoch="$NUM_SAMPLES_PER_TASK" \
        test_config.output.num_samples="$NUM_SAMPLES_PER_TASK" \
        test_config.shuffle=false \
        test_config.start_date="$test_start" \
        test_config.end_date="$test_end" \
        test_config.forecast.num_steps="$FORECAST_STEPS" \
        test_config.forecast.chunk_size="$FORECAST_CHUNK_SIZE" \
        test_config.compute_full_validation_loss=false \
        test_config.skip_target_values=true

test -s "$expected_result"
