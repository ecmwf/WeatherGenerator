#!/usr/bin/env bash
# Launch finetune-forecast runs from multiple pretraining checkpoints.
#
# This script ablates the number of pretraining epochs by launching a
# separate finetune-forecast job for each checkpoint epoch.
#
# The SLURM copy directory created by launch-slurm.py uses only the
# first 8 characters of the run-id (no stage suffix like "-pretrain").
# When --from-run-id includes a suffix, launch-slurm.py looks for
# "slurm_weathergen_{from_run_id}_dir" which doesn't exist.  This
# script creates a temporary symlink so that the launcher can find the
# source directory.
#
# Usage:
#   ./scripts/launch_pretrain_ablation.sh <pretrain_run_id> [options]
#   ./scripts/launch_pretrain_ablation.sh abcd1234-pretrain
#   ./scripts/launch_pretrain_ablation.sh abcd1234-pretrain --dry-run
#   ./scripts/launch_pretrain_ablation.sh abcd1234-pretrain --epochs "16 32"
#   ./scripts/launch_pretrain_ablation.sh abcd1234-pretrain --chain-jobs 2
#
# Options:
#   --dry-run          Print commands without executing them
#   --epochs "E1 E2"   Override default ablation epochs (default: "16 32 48 64")
#   --chain-jobs N     Number of chained SLURM jobs per ablation (default: 3)
#   --config PATH      Extra config file(s) to pass to launch-slurm.py
#   --slurm-script P   Override the SLURM script passed to launch-slurm.py
#   --                 Remaining args are forwarded to launch-slurm.py
#
# Requirements:
#   - WeatherGenerator-private must be a sibling directory of WeatherGenerator
#   - The pretraining SLURM copy directory must exist

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PRIVATE_DIR="$(cd "$REPO_DIR/../WeatherGenerator-private" && pwd)"
LAUNCHER="$PRIVATE_DIR/hpc/launch-slurm.py"

# Defaults
EPOCHS=(16 32 48 64)
DRY_RUN=false
CHAIN_JOBS=3
EXTRA_CONFIGS=()
SLURM_SCRIPT=""
EXTRA_LAUNCHER_ARGS=()
FINETUNE_CONFIG="$REPO_DIR/config/config_jepa_forecasting_finetuning.yml"

generate_random_suffix() {
    if command -v python3 &>/dev/null; then
        python3 -c "import secrets, string; alphabet=string.ascii_lowercase+string.digits; print(''.join(secrets.choice(alphabet) for _ in range(8)))"
    else
        # Fallback without python: 8 hex chars from bash RNG.
        printf "%08x\n" "$(((RANDOM << 16) | RANDOM))"
    fi
}

# --- Parse arguments ---
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <pretrain_run_id> [--dry-run] [--epochs \"E1 E2 ...\"] [--chain-jobs N] [--config PATH ...] [-- extra_args...]" >&2
    exit 1
fi

PRETRAIN_RUN_ID="$1"; shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true; shift ;;
        --epochs)
            read -ra EPOCHS <<< "$2"; shift 2 ;;
        --chain-jobs)
            CHAIN_JOBS="$2"; shift 2 ;;
        --config)
            EXTRA_CONFIGS+=("$2"); shift 2 ;;
        --slurm-script)
            SLURM_SCRIPT="$2"; shift 2 ;;
        --)
            shift; EXTRA_LAUNCHER_ARGS+=("$@"); break ;;
        *)
            EXTRA_LAUNCHER_ARGS+=("$1"); shift ;;
    esac
done

# --- Resolve the SLURM copy directory ---
# launch-slurm.py resolves the slurm dir from the private config's
# path_shared_slurm_dir (or falls back to the parent of WeatherGenerator).
# We replicate that logic here so we can create the symlink.
SLURM_ROOT="$REPO_DIR/.."
if command -v python3 &>/dev/null; then
    # Try to read path_shared_slurm_dir from the private config
    _maybe_root=$(python3 -c "
import sys, importlib.util
spec = importlib.util.spec_from_file_location('platform_env', '$PRIVATE_DIR/hpc/platform-env.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
conf = mod.private_config()
print(conf.get('path_shared_slurm_dir', ''))
" 2>/dev/null || true)
    if [[ -n "$_maybe_root" && -d "$_maybe_root" ]]; then
        SLURM_ROOT="$_maybe_root"
    fi
fi

# The 8-char base id (strip everything from the first hyphen onward)
BASE_ID="${PRETRAIN_RUN_ID%%-*}"
EXISTING_DIR="$SLURM_ROOT/slurm_weathergen_${BASE_ID}_dir"
SYMLINK_DIR="$SLURM_ROOT/slurm_weathergen_${PRETRAIN_RUN_ID}_dir"

CREATED_SYMLINK=false

if [[ "$PRETRAIN_RUN_ID" != "$BASE_ID" ]]; then
    # The full run-id has a suffix — we may need a symlink
    if [[ -d "$EXISTING_DIR" && ! -e "$SYMLINK_DIR" ]]; then
        echo "Creating symlink: $SYMLINK_DIR -> $EXISTING_DIR"
        if ! $DRY_RUN; then
            ln -s "$(basename "$EXISTING_DIR")" "$SYMLINK_DIR"
            CREATED_SYMLINK=true
        else
            echo "  [dry-run] ln -s $(basename "$EXISTING_DIR") $SYMLINK_DIR"
        fi
    elif [[ -e "$SYMLINK_DIR" ]]; then
        echo "SLURM dir already exists: $SYMLINK_DIR"
    else
        echo "WARNING: Neither $EXISTING_DIR nor $SYMLINK_DIR found." >&2
        echo "         The launcher may fail to locate the source directory." >&2
    fi
fi

# --- Cleanup helper ---
cleanup() {
    if $CREATED_SYMLINK && [[ -L "$SYMLINK_DIR" ]]; then
        echo "Cleaning up symlink: $SYMLINK_DIR"
        rm -f "$SYMLINK_DIR"
    fi
}
trap cleanup EXIT

# --- Launch one finetune-forecast per epoch ---
echo ""
echo "=== Pretraining epoch ablation ==="
echo "  Pretrain run-id : $PRETRAIN_RUN_ID"
echo "  Base id         : $BASE_ID"
echo "  Epochs          : ${EPOCHS[*]}"
echo "  Chain jobs      : $CHAIN_JOBS"
echo "  Finetune config : $FINETUNE_CONFIG"
echo "  Dry run         : $DRY_RUN"
echo ""

for EPOCH in "${EPOCHS[@]}"; do
    # Add a random suffix to avoid collisions on repeated ablation launches.
    RUN_ID="${BASE_ID}-ft${EPOCH}-$(generate_random_suffix)"

    echo "--- Epoch $EPOCH -> run-id: $RUN_ID ---"

    CMD=("$LAUNCHER"
        --from-run-id "$PRETRAIN_RUN_ID"
        --run-id "$RUN_ID"
        --mini-epoch "$EPOCH"
        --chain-jobs "$CHAIN_JOBS"
        --config "$FINETUNE_CONFIG"
    )

    if [[ ${#EXTRA_CONFIGS[@]} -gt 0 ]]; then
        for cfg in "${EXTRA_CONFIGS[@]}"; do
            CMD+=(--config "$cfg")
        done
    fi

    if [[ -n "$SLURM_SCRIPT" ]]; then
        CMD+=(--slurm-script "$SLURM_SCRIPT")
    fi

    if [[ ${#EXTRA_LAUNCHER_ARGS[@]} -gt 0 ]]; then
        CMD+=("${EXTRA_LAUNCHER_ARGS[@]}")
    fi

    if $DRY_RUN; then
        echo "  [dry-run] ${CMD[*]}"
    else
        echo "  Launching: ${CMD[*]}"
        "${CMD[@]}"
    fi
    echo ""
done

echo "=== All ablation jobs submitted ==="
