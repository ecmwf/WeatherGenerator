#!/usr/bin/env bash
# 40-step MC-dropout ensemble inference for the long-rollout fine-tunes A-C of nhv6tkln (pipeline
# hoxd15yg), on the same 15 test windows as nhv6tkln (nhv600NN) and the deterministic runs
# (zmmj/okfe/cbqs00NN). ENS members (test_config.forecast.ensemble_size) roll out with the forecast
# engine's dropout active, as in training; ENS=1 is deterministic. EXTRA_OPTIONS are appended to
# --options. There is no venv for this worktree: it runs the inference checkout's venv with this
# worktree's sources first on PYTHONPATH.
# Submit from the WeatherGenerator root (BASHRCSOURCED=Y: /etc/bashrc fails under set -u when
# sbatch is called from a non-interactive shell), e.g. C with 4 members on window 1:
#   sbatch --array=1 --export=ALL,BASHRCSOURCED=Y,FROM_RUN_ID=cbqstu8f,ENS=4,RUN_PREFIX=cbe4 \
#       scripts/inference_long_rollout_ens_40step_array_jupiter.sbatch

#SBATCH --account=e-ext-2025e01-128
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --chdir=.
#SBATCH --partition=booster
#SBATCH --array=1-15%15
#SBATCH --output=logs/weathergen-%x.%A_%a.out
#SBATCH --error=logs/weathergen-%x.%A_%a.err

set -euo pipefail

source "$HOME/.bashrc"
source "$LMOD_ROOT/lmod/init/bash"
module --force purge
module load Stages/2025 GCC/13.3.0 GCCcore/.13.3.0
ml git/2.45.1
ml Python/3.12.3

export UCX_TLS="^cma"
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"

FROM_RUN_ID=${FROM_RUN_ID:?Set FROM_RUN_ID to zmmj3eby, okfe03ir, cbqstu8f, klbklwf7, axhby5w8, q4gwmov3, nyscsarv, ci7g92ye or ps71a3lb.}
# Final checkpoints: A ran 8 mini-epochs (as nhv6tkln), B and C 5, crps8 (klbklwf7) 25, crps8long
# (axhby5w8) 10, combo_crps (q4gwmov3) and its control (nyscsarv) 4. decoder_crps (ci7g92ye) and
# crps8_restart (ps71a3lb) default to chkpt 3 and 9: their final-index chkpts were re-saved from
# _latest by redundant chain parts. For the CRPS runs, MINI_EPOCH selects another checkpoint.
case "$FROM_RUN_ID" in
    zmmj3eby) MINI_EPOCH=8 ;;
    okfe03ir|cbqstu8f) MINI_EPOCH=5 ;;
    klbklwf7) MINI_EPOCH=${MINI_EPOCH:-25} ;;
    axhby5w8) MINI_EPOCH=${MINI_EPOCH:-10} ;;
    q4gwmov3|nyscsarv) MINI_EPOCH=${MINI_EPOCH:-4} ;;
    ci7g92ye) MINI_EPOCH=${MINI_EPOCH:-3} ;;
    ps71a3lb) MINI_EPOCH=${MINI_EPOCH:-9} ;;
    # crps8_w8 (8-step gradient window): 10 mini-epochs in 3 chain parts, chkpt 10 is the true final
    rs5zhuu1) MINI_EPOCH=${MINI_EPOCH:-10} ;;
    # crps20d (20-day pushforward): chkpt 9, its chkpt 10 was re-saved from _latest by redundant chain parts
    spqvzngg) MINI_EPOCH=${MINI_EPOCH:-9} ;;
    # base model nhv6tkln (deterministic with ENS=1, e.g. to make maps; its scored runs are nhv600NN)
    nhv6tkln) MINI_EPOCH=${MINI_EPOCH:-8} ;;
    # crps20d_warm (crps20d recipe from crps8long's final checkpoint, 5 mini-epochs): final chkpt 5
    qiuftfm4) MINI_EPOCH=${MINI_EPOCH:-5} ;;
    *)
        echo "FROM_RUN_ID must be zmmj3eby, okfe03ir, cbqstu8f, klbklwf7, axhby5w8, q4gwmov3, nyscsarv, ci7g92ye, ps71a3lb, rs5zhuu1, spqvzngg, nhv6tkln or qiuftfm4; got '$FROM_RUN_ID'." >&2
        exit 2
        ;;
esac
ENS=${ENS:?Set ENS, the number of ensemble members (1: deterministic).}
RUN_PREFIX=${RUN_PREFIX:?Set RUN_PREFIX, 4 characters for the output run ids.}
[[ ${#RUN_PREFIX} -eq 4 ]] || {
    echo "RUN_PREFIX must have 4 characters; got '$RUN_PREFIX'." >&2
    exit 2
}
read -r -a extra_options <<< "${EXTRA_OPTIONS:-}"
FORECAST_STEPS=${FORECAST_STEPS:-40}
FORECAST_CHUNK_SIZE=1
RESULTS_ROOT=/e/scratch/weatherai/shared_work/results
# FIRST_SOURCE_START (opt-in) shifts the 15 windows, e.g. 2022-06-01T00:00 for an in-sample check
FIRST_SOURCE_START=${FIRST_SOURCE_START:-2023-06-01T00:00}
INPUT_STEP_HOURS=6
NUM_SAMPLES_PER_TASK=8
# With forecast offset 1, this retains eight source windows per task (258 h for 40 steps).
LOCAL_TEST_RANGE_HOURS=$((FORECAST_STEPS * INPUT_STEP_HOURS + 18))

task_id=${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array task.}
run_id=$(printf '%s%04d' "$RUN_PREFIX" "$task_id")
source_start=$(date -u --date="$FIRST_SOURCE_START UTC + $((task_id * INPUT_STEP_HOURS * NUM_SAMPLES_PER_TASK)) hours" '+%Y-%m-%dT%H:%M')
source_end=$(date -u --date="$source_start UTC + $INPUT_STEP_HOURS hours" '+%Y-%m-%dT%H:%M')
test_start=$(date -u --date="$source_start UTC - $INPUT_STEP_HOURS hours" '+%Y-%m-%dT%H:%M')
test_end=$(date -u --date="$test_start UTC + $((LOCAL_TEST_RANGE_HOURS + NUM_SAMPLES_PER_TASK * INPUT_STEP_HOURS)) hours" '+%Y-%m-%dT%H:%M')
expected_result="$RESULTS_ROOT/$run_id/validation_chkpt00000_rank0000.zip"

[[ ! -e "$RESULTS_ROOT/$run_id" ]] || {
    echo "Refusing to overwrite existing results directory: $RESULTS_ROOT/$run_id" >&2
    exit 2
}

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

weathergen_home=${SLURM_SUBMIT_DIR:?Submit this script from the WeatherGenerator root.}
venv=/e/project1/e-ext-2025e01-128/hickman3/wg_base/develop-ssl-diffusion-v1-inference/WeatherGenerator/.venv
private_config="$(dirname "$weathergen_home")/WeatherGenerator-private/hpc/jupiter/config/paths.yml"
[[ -f "$venv/bin/activate" ]]
[[ -f "$private_config" ]]

cd "$weathergen_home"
source "$venv/bin/activate"
export PYTHONPATH="$weathergen_home/src:$weathergen_home/packages/common/src:$weathergen_home/packages/readers_extra/src:$weathergen_home/packages/evaluate/src:$weathergen_home/packages/metrics/src"
python -c "import weathergen.model.model as m; assert m.__file__.startswith('$weathergen_home'), m.__file__"

echo "code=$(git rev-parse --short HEAD)$(git diff --quiet || echo '+dirty') from_run_id=$FROM_RUN_ID mini_epoch=$MINI_EPOCH ens=$ENS steps=$FORECAST_STEPS run_id=$run_id extra=[${extra_options[*]:-}]"
echo "source_interval=[$source_start, $source_end) test_interval=[$test_start, $test_end)"

# NUMA_BIND=1 (opt-in, used for 80 steps): the GH200 HBM is also a CPU-less NUMA node, so host
# memory that overflows a Grace node can land in GPU memory. Bind host memory to the Grace nodes
# and log host/HBM usage every minute to logs/mem_<run_id>.<job>.log.
launcher=()
if [[ "${NUMA_BIND:-0}" == 1 ]]; then
    numactl -H
    cpu_nodes=$(numactl -H | awk '/^node [0-9]+ cpus: [0-9]/ {print $2}' | paste -sd, -)
    [[ -n "$cpu_nodes" ]]
    launcher=(numactl --membind="$cpu_nodes")
    echo "NUMA_BIND: host memory bound to nodes $cpu_nodes"
    mem_log="logs/mem_${run_id}.${SLURM_JOB_ID}.log"
    (
        while true; do
            date '+%F %T'
            numastat -m | grep -E 'Node 0|^MemUsed|^MemFree'
            nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
            sleep 60
        done
    ) > "$mem_log" 2>&1 &
    mem_logger=$!
    trap 'kill "$mem_logger" 2>/dev/null || true' EXIT
fi

srun --ntasks=4 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    "${launcher[@]}" python -u src/weathergen/run_train.py inference \
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
        test_config.skip_target_values=true \
        test_config.forecast.ensemble_size="$ENS" \
        "${extra_options[@]}"

test -s "$expected_result"
