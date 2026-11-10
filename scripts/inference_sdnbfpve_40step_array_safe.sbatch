#!/usr/bin/env bash
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

# One independent 40-step inference sample per array task, from model sdnbfpve.
# Task i writes run <prefix>000ii (e.g. abc00000 ... abc00003 with the default --array=0-3) for source window
# 2023-06-03T12:00 + i * 12 hours. Each task uses a local test date range that
# contains exactly that one valid input window.
#
# Set up for the Zurich hackathon (account wghack-course2026-cscs, results in the
# course shared_work directory, as in WeatherGenerator-private's Santis paths.yml).
# <prefix> is a unique 3-character prefix (lowercase letters/digits) per participant.
#
# sdnbfpve only has a "latest" checkpoint (no numbered mini-epochs), so
# --mini-epoch is omitted and the CLI default (-1 = latest) is used.
# Its ERA5_in stream reads the operational analysis dataset
# aifs-od-an-oper-0001-mars-n320-2016-2025-6h-v1-for-single-v2.zarr.
#
# Submit from the WeatherGenerator root with:
#   sbatch scripts/inference_sdnbfpve_40step_array_safe.sbatch <prefix>
# To test a single task first:
#   sbatch --array=0 scripts/inference_sdnbfpve_40step_array_safe.sbatch <prefix>
#
# Santis normal is node-exclusive: every one-GPU array task reserves one full
# four-GPU GH200 node. The shared partition can avoid this, but is limited to 1h.

#SBATCH -A wghack-course2026-cscs
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --array=0-3
#SBATCH --job-name=wg-sdn-f40
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

FROM_RUN_ID=sdnbfpve
RUN_PREFIX=${1:-}
if [[ ! "$RUN_PREFIX" =~ ^[a-z][a-z0-9]{2}$ ]]; then
    echo "Usage: sbatch scripts/inference_sdnbfpve_40step_array_safe.sbatch <prefix>" >&2
    echo "<prefix>: your unique 3 characters, lowercase letters/digits, starting with a letter." >&2
    exit 2
fi
FORECAST_STEPS=40
FORECAST_CHUNK_SIZE=4
RESULTS_ROOT=/iopsstor/scratch/cscs/course_29fe58c1/shared_work/results
FIRST_SOURCE_START=2023-06-03T12:00
SOURCE_CADENCE_HOURS=12
INPUT_STEP_HOURS=6
# 1 input step + 1 selected window + (40 forecast steps + offset 1) * 6 h.
LOCAL_TEST_RANGE_HOURS=258

task_id=${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array task.}
run_id=$(printf '%s%05d' "$RUN_PREFIX" "$task_id")
source_start=$(date -u --date="$FIRST_SOURCE_START UTC + $((task_id * SOURCE_CADENCE_HOURS)) hours" '+%Y-%m-%dT%H:%M')
source_end=$(date -u --date="$source_start UTC + $INPUT_STEP_HOURS hours" '+%Y-%m-%dT%H:%M')
test_start=$(date -u --date="$source_start UTC - $INPUT_STEP_HOURS hours" '+%Y-%m-%dT%H:%M')
test_end=$(date -u --date="$test_start UTC + $LOCAL_TEST_RANGE_HOURS hours" '+%Y-%m-%dT%H:%M')
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
private_config="$(dirname "$weathergen_home")/WeatherGenerator-private/hpc/santis/config/paths.yml"
if [[ ! -f "$private_config" ]]; then
    echo "Missing Santis private config: $private_config" >&2
    exit 2
fi
cd "$weathergen_home"
# shellcheck disable=SC1091
source .venv/bin/activate

echo "run_id=$run_id source_interval=[$source_start, $source_end)"
echo "test_interval=[$test_start, $test_end)"
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    python -u src/weathergen/run_train.py inference \
    --from-run-id "$FROM_RUN_ID" \
    --run-id "$run_id" \
    --private-config "$private_config" \
    --options \
        data_loading.num_workers=1 \
        data_loading.persistent_workers=false \
        test_config.samples_per_mini_epoch=1 \
        test_config.shuffle=false \
        test_config.start_date="$test_start" \
        test_config.end_date="$test_end" \
        test_config.output.num_samples=1 \
        test_config.forecast.num_steps="$FORECAST_STEPS" \
        test_config.forecast.chunk_size="$FORECAST_CHUNK_SIZE" \
        test_config.compute_full_validation_loss=false

test -s "$expected_result"
