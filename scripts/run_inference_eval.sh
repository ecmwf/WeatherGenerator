#!/bin/bash
#
# Run inference on one or more training run IDs, then evaluate all results.
#
# Usage (from WeatherGenerator/):
#   ./scripts/run_inference_eval.sh <train_run_id1> [train_run_id2 ...]
#
# Each inference run generates a new run ID. These are collected into a
# temporary eval config (inheriting settings from the base config) and
# evaluated together at the end.

set -euo pipefail

BASE_EVAL_CONFIG="config/evaluate/eval_era5.yml"

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <train_run_id1> [train_run_id2 ...]" >&2
    exit 1
fi

TEMP_CONFIG=$(mktemp ./tmp/eval_config_XXXXXX.yml)

declare -A INFERENCE_RUN_IDS

for TRAIN_RUN_ID in "$@"; do
    echo "==> Inference for training run: $TRAIN_RUN_ID"

    INFERENCE_OUTPUT=$(uv run --offline inference \
        --from-run-id "$TRAIN_RUN_ID" \
        --options test_config.output.num_samples=2 \
                  test_config.forecast.num_steps=20 \
                  test_config.samples_per_mini_epoch=32 2>&1)

    echo "$INFERENCE_OUTPUT"

    # Extract run ID from: "Finished inference run with id: <run_id>"
    # sed strips the ANSI color reset code and any trailing noise
    INFER_RUN_ID=$(echo "$INFERENCE_OUTPUT" \
        | grep "Finished inference run with id:" \
        | sed 's/.*Finished inference run with id: //' \
        | sed 's/[^a-z0-9].*//' \
        | tail -1)

    if [[ -z "$INFER_RUN_ID" ]]; then
        echo "ERROR: Could not extract inference run ID for training run '$TRAIN_RUN_ID'" >&2
        exit 1
    fi

    echo "==> Inference run ID: $INFER_RUN_ID"
    INFERENCE_RUN_IDS["$TRAIN_RUN_ID"]="$INFER_RUN_ID"
done

# Build train_id:infer_id pairs to pass to Python
PAIRS=()
for TRAIN_ID in "${!INFERENCE_RUN_IDS[@]}"; do
    PAIRS+=("${TRAIN_ID}:${INFERENCE_RUN_IDS[$TRAIN_ID]}")
done

# Generate a temp eval config: inherit all settings from base, replace run_ids
uv run python3 -c "
import sys, yaml

base_config, *pairs = sys.argv[1:]

with open(base_config) as f:
    config = yaml.safe_load(f)

config['run_ids'] = {}
for pair in pairs:
    train_id, infer_id = pair.split(':')
    config['run_ids'][infer_id] = {
        'label': f'Inference of {train_id}',
        'results_base_dir': f'./results/{infer_id}',
    }

print(yaml.dump(config, default_flow_style=False, allow_unicode=True))
" "$BASE_EVAL_CONFIG" "${PAIRS[@]}" > "$TEMP_CONFIG"

echo "==> Evaluating ${#INFERENCE_RUN_IDS[@]} run(s)..."
uv run evaluate --config "$TEMP_CONFIG"

