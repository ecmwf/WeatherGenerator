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
REPO_PARENT="$(cd "$REPO_DIR/.." && pwd)"

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

resolve_slurm_root() {
    local default_root="$REPO_PARENT"
    local hpc_config_path
    local parsed_root

    if [[ -x "$PRIVATE_DIR/hpc/platform-env.py" ]]; then
        hpc_config_path="$("$PRIVATE_DIR/hpc/platform-env.py" hpc-config 2>/dev/null || true)"
        if [[ -n "$hpc_config_path" && -f "$hpc_config_path" ]]; then
            parsed_root="$(
                awk -F': *' '
                    /^[[:space:]]*path_shared_slurm_dir[[:space:]]*:/ {
                        print $2
                        exit
                    }
                ' "$hpc_config_path" \
                | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e "s/^['\"]//" -e "s/['\"]$//"
            )"
            if [[ -n "$parsed_root" && -d "$parsed_root" ]]; then
                printf "%s\n" "$parsed_root"
                return
            fi
        fi
    fi

    printf "%s\n" "$default_root"
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
SLURM_ROOT="$(resolve_slurm_root)"

# The 8-char base id (strip everything from the first hyphen onward)
BASE_ID="${PRETRAIN_RUN_ID%%-*}"
SYMLINK_DIR="$SLURM_ROOT/slurm_weathergen_${PRETRAIN_RUN_ID}_dir"

CREATED_SYMLINK=false

if [[ "$PRETRAIN_RUN_ID" != "$BASE_ID" ]]; then
    # The full run-id has a suffix.
    # Ensure launch-slurm.py can resolve:
    #   ${SLURM_ROOT}/slurm_weathergen_${PRETRAIN_RUN_ID}_dir/WeatherGenerator
    # by aliasing to the current workspace parent (contains WeatherGenerator).
    if [[ -L "$SYMLINK_DIR" ]]; then
        current_target="$(cd "$SYMLINK_DIR" 2>/dev/null && pwd -P || true)"
        if [[ "$current_target" == "$REPO_PARENT" ]]; then
            echo "Using existing compatibility symlink: $SYMLINK_DIR -> $REPO_PARENT"
        else
            echo "Replacing compatibility symlink: $SYMLINK_DIR -> $REPO_PARENT"
            if ! $DRY_RUN; then
                rm -f "$SYMLINK_DIR"
                ln -s "$REPO_PARENT" "$SYMLINK_DIR"
                CREATED_SYMLINK=true
            else
                echo "  [dry-run] rm -f $SYMLINK_DIR && ln -s $REPO_PARENT $SYMLINK_DIR"
            fi
        fi
    elif [[ -e "$SYMLINK_DIR" ]]; then
        if [[ -d "$SYMLINK_DIR/WeatherGenerator" ]]; then
            echo "Using existing directory for from-run-id source: $SYMLINK_DIR"
        else
            echo "ERROR: Existing path is incompatible: $SYMLINK_DIR" >&2
            echo "       Expected directory containing WeatherGenerator/." >&2
            exit 1
        fi
    else
        echo "Creating compatibility symlink: $SYMLINK_DIR -> $REPO_PARENT"
        if ! $DRY_RUN; then
            ln -s "$REPO_PARENT" "$SYMLINK_DIR"
            CREATED_SYMLINK=true
        else
            echo "  [dry-run] ln -s $REPO_PARENT $SYMLINK_DIR"
        fi
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
        --dir "$SLURM_ROOT"
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
