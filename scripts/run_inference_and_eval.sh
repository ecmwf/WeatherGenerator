#!/bin/bash
#
# Run inference on one or more training run IDs, then evaluate all results
# with global RMSE and FROCT metrics.
#
# Usage:
#   ./scripts/run_inference_and_eval.sh <train_run_id1> [train_run_id2 ...]
#   ./scripts/run_inference_and_eval.sh --dry-run <train_run_id1> ...
#
# The script:
#   1. Runs inference for each training run ID
#   2. Collects the inference run IDs
#   3. Generates a temporary eval config from the template
#   4. Runs evaluation on all inference runs together

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMPLATE_CONFIG="$SCRIPT_DIR/eval_config_template.yml"

DRY_RUN=false

# Parse flags
TRAIN_RUN_IDS=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        -*) echo "Unknown flag: $arg" >&2; exit 1 ;;
        *) TRAIN_RUN_IDS+=("$arg") ;;
    esac
done

if [[ ${#TRAIN_RUN_IDS[@]} -eq 0 ]]; then
    echo "Usage: $0 [--dry-run] <train_run_id1> [train_run_id2 ...]" >&2
    exit 1
fi

if [[ ! -f "$TEMPLATE_CONFIG" ]]; then
    echo "ERROR: Template config not found: $TEMPLATE_CONFIG" >&2
    exit 1
fi

cd "$REPO_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EVAL_CONFIG="config/evaluate/eval_config_${TIMESTAMP}.yml"
mkdir -p ./logs config/evaluate

declare -A INFERENCE_RUN_IDS

for TRAIN_RUN_ID in "${TRAIN_RUN_IDS[@]}"; do
    echo "==> Running inference for training run: $TRAIN_RUN_ID"
    LOG_FILE="./logs/inference_${TRAIN_RUN_ID}_${TIMESTAMP}.log"

    if $DRY_RUN; then
        echo "    [dry-run] uv run --offline inference --from-run-id $TRAIN_RUN_ID ..."
        echo "    [dry-run] Log would be saved to: $LOG_FILE"
        INFERENCE_RUN_IDS["$TRAIN_RUN_ID"]="dry-run-$TRAIN_RUN_ID"
        continue
    fi

    echo "    Logging to: $LOG_FILE"
    uv run --offline inference \
        --from-run-id "$TRAIN_RUN_ID" \
        --options test_config.output.num_samples=2 \
                  test_config.forecast.num_steps=20 \
                  test_config.samples_per_mini_epoch=32 2>&1 | tee "$LOG_FILE" || {
        echo "ERROR: Inference failed for training run '$TRAIN_RUN_ID'" >&2
        echo "       See log: $LOG_FILE" >&2
        exit 1
    }

    # Extract run ID from: "Finished inference run with id: <run_id>"
    INFER_RUN_ID=$(grep "Finished inference run with id:" "$LOG_FILE" \
        | sed 's/.*Finished inference run with id: //' \
        | sed 's/[^a-z0-9].*//' \
        | tail -1)

    if [[ -z "$INFER_RUN_ID" ]]; then
        echo "ERROR: Could not extract inference run ID for training run '$TRAIN_RUN_ID'" >&2
        echo "       Check the log: $LOG_FILE" >&2
        exit 1
    fi

    echo "==> Inference run ID: $INFER_RUN_ID (from training run: $TRAIN_RUN_ID)"
    INFERENCE_RUN_IDS["$TRAIN_RUN_ID"]="$INFER_RUN_ID"
done

# Build train_id:infer_id pairs
PAIRS=()
for TRAIN_ID in "${!INFERENCE_RUN_IDS[@]}"; do
    PAIRS+=("${TRAIN_ID}:${INFERENCE_RUN_IDS[$TRAIN_ID]}")
done

# Generate eval config: copy template and inject run_ids
uv run python3 -c "
import sys
import yaml

template_path, *pairs = sys.argv[1:]

with open(template_path) as f:
    config = yaml.safe_load(f)

config['run_ids'] = {}
for pair in pairs:
    train_id, infer_id = pair.split(':')
    config['run_ids'][infer_id] = {
        'label': f'Inference of {train_id}',
        'results_base_dir': f'./results/{infer_id}/',
    }

print(yaml.dump(config, default_flow_style=False, allow_unicode=True))
" "$TEMPLATE_CONFIG" "${PAIRS[@]}" > "$EVAL_CONFIG"

echo ""
echo "==> Generated eval config: $EVAL_CONFIG"
echo "==> Evaluating ${#INFERENCE_RUN_IDS[@]} run(s)..."

if $DRY_RUN; then
    echo "    [dry-run] uv run evaluate --config $EVAL_CONFIG"
    echo ""
    echo "Generated config:"
    cat "$EVAL_CONFIG"
    exit 0
fi

uv run evaluate --config "$EVAL_CONFIG"
