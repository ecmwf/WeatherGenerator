#!/usr/bin/env bash
# Per-lead-window gradient diagnostics (scripts/grad_diag.py): one independent process per GPU (WORLD_SIZE=1),
# each on its own training dates; no weight updates. Submit from the WeatherGenerator root, e.g. a smoke test:
#   sbatch --ntasks=1 --time=00:40:00 --export=ALL,BASHRCSOURCED=Y,OUT_DIR=...,NUM_SAMPLES=1,WINDOWS=anchor,8,80 \
#       scripts/grad_diag_jupiter.sbatch
# and the full run with --ntasks=4 (default) and NUM_SAMPLES=16. Optional: LOAD_RUN_ID, LOAD_MINI_EPOCH.

#SBATCH --account=e-ext-2025e01-128
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
#SBATCH --chdir=.
#SBATCH --output=logs/grad_diag.%j.out
#SBATCH --error=logs/grad_diag.%j.err

set -euo pipefail
source "$HOME/.bashrc"
source "$LMOD_ROOT/lmod/init/bash"
module --force purge
module load Stages/2025 GCC/13.3.0 GCCcore/.13.3.0
ml git/2.45.1
ml Python/3.12.3

OUT_DIR=${OUT_DIR:?Set OUT_DIR}
NUM_SAMPLES=${NUM_SAMPLES:?Set NUM_SAMPLES (per process)}
WINDOWS=${WINDOWS:-anchor,8,12,16,20,28,40,60,80}
LOAD_RUN_ID=${LOAD_RUN_ID:-axhby5w8}
LOAD_MINI_EPOCH=${LOAD_MINI_EPOCH:-9}

home=${SLURM_SUBMIT_DIR:?}
venv=/e/project1/e-ext-2025e01-128/hickman3/wg_base/develop-ssl-diffusion-v1-inference/WeatherGenerator/.venv
private_config="$(dirname "$home")/WeatherGenerator-private/hpc/jupiter/config/paths.yml"
cd "$home"
source "$venv/bin/activate"
export PYTHONPATH="$home/src:$home/packages/common/src:$home/packages/readers_extra/src:$home/packages/evaluate/src:$home/packages/metrics/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16
mkdir -p "$OUT_DIR"
echo "code=$(git rev-parse --short HEAD)$(git diff --quiet || echo '+dirty') load=$LOAD_RUN_ID/$LOAD_MINI_EPOCH samples/proc=$NUM_SAMPLES windows=$WINDOWS out=$OUT_DIR"

# one process per GPU, each a world of size 1
srun --ntasks="$SLURM_NTASKS" --cpus-per-task="$SLURM_CPUS_PER_TASK" bash -c '
    export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
    python -u scripts/grad_diag.py --out-dir "'"$OUT_DIR"'" --proc $SLURM_PROCID --num-samples '"$NUM_SAMPLES"' \
        --windows '"$WINDOWS"' --load-run-id '"$LOAD_RUN_ID"' --load-mini-epoch '"$LOAD_MINI_EPOCH"' \
        --private-config "'"$private_config"'" > "'"$OUT_DIR"'/proc$SLURM_PROCID.log" 2>&1
'
ls -la "$OUT_DIR"
