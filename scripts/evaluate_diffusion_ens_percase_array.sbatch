#!/bin/bash -x
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

# Phase 1: compute & cache per-case "rmse" scores (one Slurm task per run_id, each
# reading only its own zarr store) for every id behind the merged run_ids in
# config/config_eval_diffusion_ens.yml. Run this BEFORE
# evaluate_diffusion_ens_merge_array.sbatch (phase 2), which only reads these cached
# JSON files and would otherwise have to re-read every raw zarr store at once.
#
# Submit from the WeatherGenerator root with:
#   sbatch scripts/evaluate_diffusion_ens_percase_array.sbatch
#
# NB. diffusion_ensemble_mean needs the dj* ids cached with ensemble: "mean" instead
# of the default "all" used here. Re-run this array for just the dj* ids with:
#   sbatch --array=44-92 --export=ENSEMBLE=mean scripts/evaluate_diffusion_ens_percase_array.sbatch
# AFTER merging diffusion_member_mean (phase 2), since it overwrites the same cache.

#SBATCH --account=e-ext-2025e01-128
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --chdir=.
#SBATCH --partition=booster
#SBATCH --array=0-92%20
#SBATCH --output=logs/weathergen-eval-percase-%x.%A_%a.out
#SBATCH --error=logs/weathergen-eval-percase-%x.%A_%a.err

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

# Keep in sync with the *deterministic_run_ids / *diffusion_run_ids anchors in
# config/config_eval_diffusion_ens.yml (indices 0-43 deterministic, 44-92 diffusion).
RUN_IDS=(
    zz060500 zz060506 zz060512 zz060518 zz060600 zz060606 zz060612 zz060618
    zz060700 zz060706 zz060712 zz060718 zz060800 zz060806 zz060812 zz060818
    zz060900 zz060906 zz060912 zz060918 zz061000 zz061006 zz061012 zz061018
    zz061100 zz061106 zz061112 zz061118 zz061200 zz061206 zz061212 zz061218
    zz061300 zz061306 zz061312 zz061318 zz061400 zz061406 zz061412 zz061418
    zz061500 zz061506 zz061512 zz061518
    dj060318 dj060400 dj060406 dj060412 dj060418
    dj060500 dj060506 dj060512 dj060518 dj060600 dj060606 dj060612 dj060618
    dj060700 dj060706 dj060712 dj060718 dj060800 dj060806 dj060812 dj060818
    dj060900 dj060906 dj060912 dj060918 dj061000 dj061006 dj061012 dj061018
    dj061100 dj061106 dj061112 dj061118 dj061200 dj061206 dj061212 dj061218
    dj061300 dj061306 dj061312 dj061318 dj061400 dj061406 dj061412 dj061418
    dj061500 d1061506 dj061512 dj061518
)
# RUN_IDS=( kz2n{0000..0121} )
task_id=${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array task.}
RUN_ID="${RUN_IDS[$task_id]}"
ENSEMBLE="${ENSEMBLE:-all}"

weathergen_home=${SLURM_SUBMIT_DIR:?Submit this script from the WeatherGenerator root.}
cd "$weathergen_home"

echo "Caching per-case scores for run_id: $RUN_ID (ensemble=$ENSEMBLE)"
uv run --offline evaluate --config config/config_eval_diffusion_ens.yml \
    --run-ids "$RUN_ID" \
    --options "default_streams.ERA5.evaluation.ensemble=$ENSEMBLE"
