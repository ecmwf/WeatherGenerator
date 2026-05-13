#!/bin/bash
#SBATCH --job-name=wg-eval
#SBATCH --nodes=1
#SBATCH --mem=368G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=0
#SBATCH --time=11:59:59
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --switches=1

module load gcc/12.2.0
module load cuda/12.2

if [ $# -lt 3 ]; then
    echo "Usage: $0 <config_file.yml> <run_id> <label>"
    exit 1
fi

CONFIG_FILE="$1"
RUN_ID="$2"
LABEL="$3"
TMP_CONFIG=$(mktemp)

# Copy everything up to (but not including) the first 'run_ids:' line
awk '/^run_ids:/ {exit} {print}' "$CONFIG_FILE" > "$TMP_CONFIG"

# Add the new run_ids section
echo "run_ids:" >> "$TMP_CONFIG"
echo "  $RUN_ID:" >> "$TMP_CONFIG"
echo "    label: \"$LABEL\"" >> "$TMP_CONFIG"

# Append the rest of the config after the original run_ids section (if any)
awk 'found {print} /^run_ids:/ {found=1}' "$CONFIG_FILE" | \
awk 'NR>3' >> "$TMP_CONFIG"

echo "Modified config:"
cat "$TMP_CONFIG"

echo "Starting Evaluation Job at $(date)"
uv run --offline evaluate --config "$TMP_CONFIG"
echo "Finished Evaluation Job at $(date)"