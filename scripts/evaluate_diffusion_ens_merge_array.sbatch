#!/bin/bash -x
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

# Phase 2: runs the three "jsonmerge" virtual run_ids defined in
# config/config_eval_diffusion_ens.yml (deterministic_mean, diffusion_ensemble_mean,
# diffusion_member_mean), reading cheap cached per-case JSON scores produced by
# scripts/evaluate_diffusion_ens_percase_array.sbatch (phase 1) - run that first.
#
# Submit from the WeatherGenerator root with:
#   sbatch scripts/evaluate_diffusion_ens_merge_array.sbatch
#
# TIP: before running the full array, sanity-check one run_id cheaply with e.g.:
#   uv run evaluate --config config/config_eval_diffusion_ens.yml \
#     --options evaluation.regions=[global] \
#     --run-ids deterministic_mean
# (adjust merge_run_ids to a short list temporarily to verify it works end-to-end).
#
# Tune --time/--mem/--cpus-per-task below once you've seen actual resource usage.

#SBATCH --account=e-ext-2025e01-128
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --chdir=.
#SBATCH --partition=booster
#SBATCH --array=0-2
#SBATCH --output=logs/weathergen-eval-%x.%j.out
#SBATCH --error=logs/weathergen-eval-%x.%j.err

echo "shell $SHELL"
if [ -f "$HOME/.local/bin/env" ]; then
    . "$HOME/.local/bin/env"
fi

. "$HOME/.bashrc"
echo "LMOD_ROOT $LMOD_ROOT"
. $LMOD_ROOT/lmod/init/bash
module --force purge
module load Stages/2025
module load GCC/13.3.0
module load GCCcore/.13.3.0
ml git/2.45.1
ml Python/3.12.3

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

set -euo pipefail

RUN_IDS=(deterministic_mean diffusion_ensemble_mean diffusion_member_mean)
task_id=${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array task.}
RUN_ID="${RUN_IDS[$task_id]}"

weathergen_home=${SLURM_SUBMIT_DIR:?Submit this script from the WeatherGenerator root.}
cd "$weathergen_home"

echo "Evaluating merged run_id: $RUN_ID"
# --offline: compute nodes have no internet access, so uv must not try to resolve/download packages
uv run --offline evaluate --config config/config_eval_diffusion_ens.yml --run-ids "$RUN_ID"
