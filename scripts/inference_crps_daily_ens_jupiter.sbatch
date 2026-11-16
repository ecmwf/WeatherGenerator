#!/usr/bin/env bash
# Daily-initialised MC-dropout ensemble forecasts (e.g. 3 months of 00 UTC forecasts for quaver comparison with
# AIFS-ENS). Each array task runs 4 independent single-GPU processes (WORLD_SIZE=1), one init each, bound to the
# GPU's own Grace NUMA node. Day index d = 4*(task-1) + local rank, init = FIRST_DAY + d days at 00 UTC.
#
# Init convention (as in training): a "00 UTC" forecast uses the input window [00:00, 06:00): the 00 UTC analysis
# (ERA5_in nominal_time_mapping treats it as available at 05 UTC, the latest analysis before the window ends) and
# the observations in [00:00, 06:00). Each 6-h output step holds hourly fields; step k covers init + 6k ..
# init + 6k + 5 h (step 1 = 06:00-11:00), so fields valid at 06/12/18/00 UTC are the first hourly field of steps
# 1, 2, 3, 4, ... Note for comparisons: the forecast has seen observations up to 06 UTC.
# Run ids: RUN_PREFIX + 4-digit day number.
#
# Submit from the WeatherGenerator root, e.g. 1 Jun - 31 Aug 2023 (92 days = 23 tasks), crps8long, 8 members:
#   sbatch --array=1-23 --export=ALL,BASHRCSOURCED=Y,RUN_PREFIX=cq23,FIRST_DAY=2023-06-01,NDAYS=92 \
#       scripts/inference_crps_daily_ens_jupiter.sbatch
# Optional: FROM_RUN_ID (axhby5w8), MINI_EPOCH (10), ENS (8), FORECAST_STEPS (40), EXTRA_OPTIONS.

#SBATCH --account=e-ext-2025e01-128
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
#SBATCH --chdir=.
#SBATCH --output=logs/weathergen-%x.%A_%a.out
#SBATCH --error=logs/weathergen-%x.%A_%a.err

set -euo pipefail
source "$HOME/.bashrc"
source "$LMOD_ROOT/lmod/init/bash"
module --force purge
module load Stages/2025 GCC/13.3.0 GCCcore/.13.3.0
ml git/2.45.1
ml Python/3.12.3

FROM_RUN_ID=${FROM_RUN_ID:-axhby5w8}
MINI_EPOCH=${MINI_EPOCH:-10}
ENS=${ENS:-8}
FORECAST_STEPS=${FORECAST_STEPS:-40}
RUN_PREFIX=${RUN_PREFIX:?Set RUN_PREFIX (4 characters)}
FIRST_DAY=${FIRST_DAY:?Set FIRST_DAY, e.g. 2023-06-01}
NDAYS=${NDAYS:?Set NDAYS}
[[ ${#RUN_PREFIX} -eq 4 ]] || { echo "RUN_PREFIX must have 4 characters" >&2; exit 2; }
task_id=${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array task.}
export FROM_RUN_ID MINI_EPOCH ENS FORECAST_STEPS RUN_PREFIX FIRST_DAY NDAYS task_id
export EXTRA_OPTIONS="${EXTRA_OPTIONS:-}"

home=${SLURM_SUBMIT_DIR:?Submit from the WeatherGenerator root.}
venv=/e/project1/e-ext-2025e01-128/hickman3/wg_base/develop-ssl-diffusion-v1-inference/WeatherGenerator/.venv
private_config="$(dirname "$home")/WeatherGenerator-private/hpc/jupiter/config/paths.yml"
[[ -f "$venv/bin/activate" && -f "$private_config" ]]
cd "$home"
source "$venv/bin/activate"
export PYTHONPATH="$home/src:$home/packages/common/src:$home/packages/readers_extra/src:$home/packages/evaluate/src:$home/packages/metrics/src"
export home private_config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import weathergen.model.model as m; assert m.__file__.startswith('$home'), m.__file__"
cpu_nodes=($(numactl -H | awk '/^node [0-9]+ cpus: [0-9]/ {print $2}'))
[[ ${#cpu_nodes[@]} -ge 4 ]] || { echo "expected 4 CPU NUMA nodes, got ${cpu_nodes[*]}" >&2; exit 2; }
export CPU_NODES="${cpu_nodes[*]}"
echo "code=$(git rev-parse --short HEAD)$(git diff --quiet || echo '+dirty') from_run_id=$FROM_RUN_ID mini_epoch=$MINI_EPOCH ens=$ENS steps=$FORECAST_STEPS task=$task_id first_day=$FIRST_DAY ndays=$NDAYS cpu_nodes=$CPU_NODES"

srun --ntasks=4 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash -c '
    set -euo pipefail
    i=$SLURM_LOCALID
    d=$(( 4 * (task_id - 1) + i ))
    if (( d >= NDAYS )); then echo "local rank $i: day $d beyond NDAYS, nothing to do"; exit 0; fi
    run_id=$(printf "%s%04d" "$RUN_PREFIX" $((d + 1)))
    init=$(date -u --date="$FIRST_DAY UTC + $d days" "+%Y-%m-%dT%H:%M")
    source_start=$init
    test_start=$(date -u --date="$source_start UTC - 6 hours" "+%Y-%m-%dT%H:%M")
    test_end=$(date -u --date="$test_start UTC + $(( FORECAST_STEPS * 6 + 18 + 6 )) hours" "+%Y-%m-%dT%H:%M")
    if [[ -e /e/scratch/weatherai/shared_work/results/$run_id/validation_chkpt00000_rank0000.zip ]]; then
        echo "local rank $i: $run_id exists, refusing to overwrite" >&2; exit 2
    fi
    nodes=($CPU_NODES)
    export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
    echo "local rank $i: day $d init $init (inputs $source_start..+5h) run_id $run_id test [$test_start, $test_end) numa ${nodes[$i]}"
    read -r -a extra <<< "$EXTRA_OPTIONS"
    numactl --cpunodebind=${nodes[$i]} --membind=${nodes[$i]} python -u src/weathergen/run_train.py inference \
        --from-run-id "$FROM_RUN_ID" --mini-epoch "$MINI_EPOCH" --run-id "$run_id" --private-config "$private_config" \
        --options data_loading.num_workers=0 data_loading.num_workers_validation=0 data_loading.persistent_workers=false \
            test_config.samples_per_mini_epoch=1 test_config.output.num_samples=1 test_config.shuffle=false \
            test_config.start_date="$test_start" test_config.end_date="$test_end" \
            test_config.forecast.num_steps="$FORECAST_STEPS" test_config.forecast.chunk_size=1 \
            test_config.compute_full_validation_loss=false test_config.skip_target_values=true \
            test_config.forecast.ensemble_size="$ENS" "${extra[@]}" \
        > "logs/weathergen-${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}_${task_id}.${run_id}.log" 2>&1
    test -s /e/scratch/weatherai/shared_work/results/$run_id/validation_chkpt00000_rank0000.zip
    echo "local rank $i: $run_id done"
'
