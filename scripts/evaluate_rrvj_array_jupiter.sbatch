#!/usr/bin/env bash
# Score completed rrvj, ps9k, or i325 40-step inference shards one at a time.
# Submit from the WeatherGenerator root:
#   sbatch scripts/evaluate_rrvj_array_jupiter.sbatch
#   sbatch --export=ALL,EVAL_FAMILY=ps9k scripts/evaluate_rrvj_array_jupiter.sbatch
#   sbatch --export=ALL,EVAL_FAMILY=i325 scripts/evaluate_rrvj_array_jupiter.sbatch

#SBATCH --account=e-ext-2025e01-128
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=10:00:00
#SBATCH --array=1-15%15
#SBATCH --job-name=wg-eval-f40
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

source "$HOME/.bashrc"
source "$LMOD_ROOT/lmod/init/bash"
module --force purge
module load Stages/2025 GCC/13.3.0 GCCcore/.13.3.0 Python/3.12.3

weathergen_home=${SLURM_SUBMIT_DIR:?Submit this script from the WeatherGenerator root.}
private_config="$weathergen_home/../WeatherGenerator-private/hpc/jupiter/config/paths.yml"
config="$weathergen_home/config/evaluate/eval_config_40step_score.yml"
family=${EVAL_FAMILY:-rrvj}

case "$family" in
    rrvj|ps9k|i325) ;;
    *)
        echo "EVAL_FAMILY must be rrvj, ps9k, or i325; got '$family'." >&2
        exit 2
        ;;
esac

[[ -f "$weathergen_home/.venv/bin/activate" ]]
[[ -f "$private_config" ]]
[[ -f "$config" ]]

run_id=$(printf '%s%04d' "$family" "${SLURM_ARRAY_TASK_ID:?Missing array task ID}")
for rank in 0 1 2 3; do
    archive="$weathergen_home/results/$run_id/validation_chkpt00000_rank$(printf '%04d' "$rank").zip"
    [[ -s "$archive" ]] || {
        echo "Skipping $run_id: missing or empty $archive" >&2
        exit 0
    }
done

cd "$weathergen_home"
source .venv/bin/activate
export WEATHERGEN_PRIVATE_CONF="$private_config"

.venv/bin/evaluate --config "$config" --run-ids "$run_id"