#!/usr/bin/env bash
# run_cepdiag.sh — compute CEPDIAG metrics and plots for a WeatherGenerator run
#
# Usage:
#   bash scripts/run_cepdiag.sh <run_id> [<datesel> [<group>]]
#
# Examples:
#   bash scripts/run_cepdiag.sh af25nepk
#   bash scripts/run_cepdiag.sh ww9atcoz all ww9atcoz   # explicit group name
#
# Environment overrides:
#   CEPDIAG_METRICS   space-separated list of metrics (default: bias rmse crps)
#   CEPDIAG_GROUP     group name (default: <run_id>)
#
# Prerequisites (one-time setup):
#   1. cepdiag repo cloned at ~/repos/cepdiag
#   2. Symlink created:  ln -sf ~/repos/cepdiag/cep ~/repos/cepdiag/py
#   3. lxml installed:   python3 -m ensurepip && python3 -m pip install lxml
#   4. Staged forecast files already in  results/<run_id>/cepdiag/eval/stage/
#   5. Staged ERA5 files already in      results/<run_id>/cepdiag/eval/stage/
#      (generate with: zarr_to_cepdiag.py and era5_to_cepdiag.py)
#
# The conf file is expected at:
#   results/<run_id>/cepdiag/cepdiag_mofc.conf
# Metrics and plots land in:
#   results/<run_id>/cepdiag/eval/metrics/
#   results/<run_id>/cepdiag/eval/plots/
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_ID="${1:?Usage: $0 <run_id> [<datesel> [<group>]]}"
DATESEL="${2:-all}"
# group: positional arg > env var > run_id (default)
GROUP="${3:-${CEPDIAG_GROUP:-$RUN_ID}}"

CEPDIAG_CEP="$HOME/repos/cepdiag/cep"
CONF="$REPO_DIR/results/$RUN_ID/cepdiag/cepdiag_mofc.conf"

# --------------------------------------------------------------------------
# Sanity checks
# --------------------------------------------------------------------------
if [[ ! -f "$CONF" ]]; then
    echo "ERROR: conf file not found: $CONF" >&2
    exit 1
fi
if [[ ! -d "$CEPDIAG_CEP" ]]; then
    echo "ERROR: cepdiag cep/ directory not found: $CEPDIAG_CEP" >&2
    exit 1
fi

# Read params and group from conf
PARAMS=$(grep -A20 '^\[staging\]' "$CONF" | grep '^params' | head -1 | sed 's/.*=\s*//' | tr -d ' ')
VERIF=$(grep -A20 '^\[staging\]' "$CONF" | grep '^verana' | head -1 | sed 's/.*=\s*//' | tr -d ' ')
HCMODE=$(grep "hcmode" "$CONF" | head -1 | sed 's/.*=\s*//' | tr -d ' ')

if [[ -z "$PARAMS" ]]; then
    echo "ERROR: could not read params from $CONF" >&2
    exit 1
fi

# Metrics to compute (must be known to CEPDIAG's metrics/factory.py)
METRICS="${CEPDIAG_METRICS:-bias rmse crps}"

IFS=',' read -ra PARAM_LIST <<< "$PARAMS"

echo "==================================================================="
echo "  CEPDIAG run for: $RUN_ID"
echo "  conf       : $CONF"
echo "  params     : ${PARAM_LIST[*]}"
echo "  metrics    : $METRICS"
echo "  verif      : $VERIF"
echo "  group      : $GROUP"
echo "  datesel    : $DATESEL"
echo "  hcmode     : ${HCMODE:-false}"
echo "==================================================================="

# Create output directories if needed
mkdir -p "$REPO_DIR/results/$RUN_ID/cepdiag/eval/metrics"
mkdir -p "$REPO_DIR/results/$RUN_ID/cepdiag/eval/plots"

cd "$CEPDIAG_CEP"

# --------------------------------------------------------------------------
# Step 1: Compute metrics
# --------------------------------------------------------------------------
echo
echo "--- Step 1: computing metrics ---"
for metric in $METRICS; do
    for param in "${PARAM_LIST[@]}"; do
        echo "  calc  $metric  $param"
        python3 plot_map.py -c "$CONF" --calc force --plot no \
            "$metric" "$param" "$GROUP" "$VERIF" "$DATESEL" 2>&1 \
            | grep -v "might be archived in spectral space" \
            | grep -v "^$" \
            | sed 's/^/    /'
    done
done

# --------------------------------------------------------------------------
# Step 2: Generate plots
# --------------------------------------------------------------------------
echo
echo "--- Step 2: generating plots ---"
for metric in $METRICS; do
    for param in "${PARAM_LIST[@]}"; do
        echo "  plot  $metric  $param"
        python3 plot_map.py -c "$CONF" --calc no --plot force \
            "$metric" "$param" "$GROUP" "$VERIF" "$DATESEL" 2>&1 \
            | grep -v "might be archived in spectral space" \
            | grep "^/" \
            | sed 's/^/    /'
    done
done

echo
echo "Done. Plots written to:"
echo "  $REPO_DIR/results/$RUN_ID/cepdiag/eval/plots/"
