#!/bin/bash
# =============================================================================
# JEPA Prototype Hyperparameter Sweep
# =============================================================================
#
# Launches N experiments with randomly sampled hyperparameters on top of
# config_jepa.yml. Each experiment runs on 1 node.
#
# Usage:
#   bash sweep_jepa_prototype.sh [NUM_EXPERIMENTS]   # default: 10
#
# Fixed design choices (not swept):
#   - Full encoder architecture from config_jepa.yml
#   - JEPA-only loss (L1), window_offset=1, 2d_rope=True
#   - Warmup 1024 steps, EMA ramp-up ratio 0.05
#   - 32 mini-epochs, 4096 samples each
#
# Swept parameters:
#   lr_max              log-uniform [1e-5, 2e-4]
#   student mask_rate   uniform     [0.50, 0.75]
#   predictor blocks    choice      {3, 4, 6}
#   predictor idim      choice      {512, 768}
#   EMA halflife (k)    log-uniform [0.05, 1.0]
#
# =============================================================================

set -euo pipefail

NUM_EXPERIMENTS=${1:-10}

# --- Paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LAUNCHER="${PROJECT_ROOT}/WeatherGenerator-private/hpc/launch-slurm.py"
BASE_CONFIG="${PROJECT_ROOT}/WeatherGenerator/config/config_jepa.yml"
LOG_FILE="${SCRIPT_DIR}/sweep_jepa_prototype_log.csv"

# --- Validate ---
[[ -f "$LAUNCHER" ]] || { echo "ERROR: launch-slurm.py not found at $LAUNCHER"; exit 1; }
[[ -f "$BASE_CONFIG" ]] || { echo "ERROR: config_jepa.yml not found at $BASE_CONFIG"; exit 1; }

# --- CSV log header ---
echo "run_id,lr_max,mask_rate,pred_blocks,pred_idim,ema_halflife_k" > "$LOG_FILE"

# --- Sampling helper (one python call per experiment) ---
sample_params() {
    python3 -c "
import random, math, string

run_id = 'j' + ''.join(random.choices(string.ascii_lowercase + string.digits, k=7))

lr     = math.exp(random.uniform(math.log(1e-6), math.log(1e-4)))
mask   = random.uniform(0.50, 0.75)
blocks = random.choice([3, 4, 6])
idim   = random.choice([512, 768])
ema    = math.exp(random.uniform(math.log(0.05), math.log(1.0)))

print(f'{run_id},{lr:.2e},{mask:.2f},{blocks},{idim},{ema:.3f}')
"
}

# --- Main ---
echo "========================================="
echo " JEPA Prototype Hyperparameter Sweep"
echo " Experiments : $NUM_EXPERIMENTS"
echo " Base config : $BASE_CONFIG"
echo " Log file    : $LOG_FILE"
echo "========================================="
echo ""

for i in $(seq 1 "$NUM_EXPERIMENTS"); do
    PARAMS=$(sample_params)
    IFS=',' read -r RUN_ID LR_MAX MASK_RATE PRED_BLOCKS PRED_IDIM EMA_HL <<< "$PARAMS"

    echo "--- Experiment $i/$NUM_EXPERIMENTS [$RUN_ID] ---"
    echo "  lr_max         = $LR_MAX"
    echo "  mask_rate      = $MASK_RATE"
    echo "  pred_blocks    = $PRED_BLOCKS"
    echo "  pred_idim      = $PRED_IDIM"
    echo "  ema_halflife_k = $EMA_HL"

    echo "$PARAMS" >> "$LOG_FILE"

    "$LAUNCHER" \
        --run-id "$RUN_ID" \
        --base-config "$BASE_CONFIG" \
        --nodes 1 \
        --no-register \
        --options \
            "training_config.learning_rate_scheduling.lr_max=$LR_MAX" \
            "training_config.learning_rate_scheduling.num_steps_warmup=1024" \
            "training_config.model_input.random_easy.masking_strategy_config.rate=$MASK_RATE" \
            "training_config.losses.student-teacher.loss_fcts.JEPA.num_blocks=$PRED_BLOCKS" \
            "training_config.losses.student-teacher.loss_fcts.JEPA.intermediate_dim=$PRED_IDIM" \
            "training_config.losses.student-teacher.target_and_aux_calc.EMATeacher.ema_halflife_in_thousands=$EMA_HL" \
            "training_config.losses.student-teacher.target_and_aux_calc.EMATeacher.ema_ramp_up_ratio=0.05" \
            "wgtags.exp=jepa_proto_sweep"

    echo ""
done

echo "========================================="
echo " All $NUM_EXPERIMENTS experiments submitted"
echo " Parameters logged to: $LOG_FILE"
echo "========================================="

