#!/bin/bash -x
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

# One independent 40-step inference sample per array task.
# Task i writes run cw6iiiii for source window
# 2023-06-01T12:00 + i * 12 hours. Each task uses a local test date range that
# contains exactly that one valid input window.
#
# Submit from the WeatherGenerator root with:
#   sbatch scripts/inference_cw6a4szu_40step_array.sbatch
#
# Jupiter normal is node-exclusive: every one-GPU array task reserves one full
# four-GPU GH200 node. The shared partition can avoid this, but is limited to 1h.

#SBATCH --account=e-ext-2025e01-128
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --chdir=.
#SBATCH --partition=booster
#SBATCH --array=0-15%16
#SBATCH --output=logs/weathergen-%x.%j.out
#SBATCH --error=logs/weathergen-%x.%j.err

echo "shell $SHELL"
if [ -f "$HOME/.local/bin/env" ]; then
    . "$HOME/.local/bin/env"
fi

. "$HOME/.bashrc"
echo "LMOD_ROOT $LMOD_ROOT"
# Does not get loaded automatically. Unsure why.
. $LMOD_ROOT/lmod/init/bash
# Load basic modules from software stack
module --force purge
module load Stages/2025

module load GCC/13.3.0
module load GCCcore/.13.3.0

ml git/2.45.1
ml Python/3.12.3


# set some environment variables to ensure that GPU-devices are used properly
export UCX_TLS="^cma"
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=0,1,2,3

# so processes know who to talk to
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
if [ "$SYSTEMNAME" = juwelsbooster ] \
       || [ "$SYSTEMNAME" = juwels ] \
       || [ "$SYSTEMNAME" = jurecadc ] \
       || [ "$SYSTEMNAME" = jusuf ]; then
    # Allow communication over InfiniBand cells on JSC machines.
    MASTER_ADDR="$MASTER_ADDR"i
fi
echo "MASTER_ADDR: $MASTER_ADDR"
set -euo pipefail

FROM_RUN_ID=xq65fhca
MINI_EPOCH=14
RUN_PREFIX=xq65
FORECAST_STEPS=40
FORECAST_CHUNK_SIZE=1
RESULTS_ROOT=/e/scratch/weatherai/shared_work/results
FIRST_SOURCE_START=2023-06-01T12:00
INPUT_STEP_HOURS=6
NUM_SAMPLES_PER_TASK=8
# SOURCE_CADENCE_HOURS=12
# 1 input step + 1 selected window + (40 forecast steps + offset 1) * 6 h.
LOCAL_TEST_RANGE_HOURS=258

task_id=${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array task.}
run_id=$(printf '%s%05d' "$RUN_PREFIX" "$task_id")
source_start=$(date -u --date="$FIRST_SOURCE_START UTC + $((task_id * 4 * NUM_SAMPLES_PER_TASK)) hours" '+%Y-%m-%dT%H:%M')
source_end=$(date -u --date="$source_start UTC + $INPUT_STEP_HOURS hours" '+%Y-%m-%dT%H:%M')
test_start=$(date -u --date="$source_start UTC - $INPUT_STEP_HOURS hours" '+%Y-%m-%dT%H:%M')
test_end=$(date -u --date="$test_start UTC + $((LOCAL_TEST_RANGE_HOURS + NUM_SAMPLES_PER_TASK * INPUT_STEP_HOURS)) hours" '+%Y-%m-%dT%H:%M')
expected_result="$RESULTS_ROOT/$run_id/validation_chkpt00000_rank0000.zip"

if [[ -e "$expected_result" ]]; then
    echo "Refusing to overwrite existing inference output: $expected_result" >&2
    exit 2
fi

# The Santis virtual environment bundles Python 3.12 and CUDA 12.6 PyTorch.
# Do not load the former Jupiter modules or force its Mellanox UCX device names.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Slurm executes a spooled copy of this script, so BASH_SOURCE[0] points into
# /var/spool/slurmd rather than this checkout. Submit from the repository root.
weathergen_home=${SLURM_SUBMIT_DIR:?Submit this script from the WeatherGenerator root.}
if [[ ! -f "$weathergen_home/.venv/bin/activate" ]]; then
    echo "WeatherGenerator virtual environment not found: $weathergen_home/.venv" >&2
    exit 2
fi
private_config="$(dirname "$weathergen_home")/WeatherGenerator-private/hpc/jupiter/config/paths.yml"
if [[ ! -f "$private_config" ]]; then
    echo "Missing Jupiter private config: $private_config" >&2
    exit 2
fi
cd "$weathergen_home"
# shellcheck disable=SC1091
source .venv/bin/activate

echo "run_id=$run_id source_interval=[$source_start, $source_end)"
echo "test_interval=[$test_start, $test_end)"
srun --ntasks=4 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    python -u src/weathergen/run_train.py inference \
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
