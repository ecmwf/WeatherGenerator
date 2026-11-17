#!/usr/bin/env bash
# Score completed 40-step inference shards of the best long-rollout models without stored targets
# (anemoi-target reader), as evaluate_long_rollout_abc_array_jupiter.sbatch does for A-C:
#   ch7e: combo_det ch7py82d chkpt 4 (deterministic): rmse
#   c8lv: crps8long axhby5w8 chkpt 10; ci4f: decoder_crps ci7g92ye chkpt 3; c8lx, c8rx: crps8long and
#         crps8_restart (ps71a3lb) chkpt 9 over 80 steps. All 4 MC-dropout members: rmse/crps/spread/ssr over members into
#         evaluation/, and rmse of the 4-member mean into evaluation_ens_mean/
# Submit from the WeatherGenerator root (BASHRCSOURCED=Y: /etc/bashrc fails under set -u when
# sbatch is called from a non-interactive shell):
#   sbatch --array=1 --export=ALL,BASHRCSOURCED=Y,EVAL_FAMILY=ch7e scripts/evaluate_long_rollout_best_array_jupiter.sbatch
#SBATCH --account=e-ext-2025e01-128
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=10:00:00
#SBATCH --array=1-15%15
#SBATCH --job-name=wg-eval-best
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

source "$HOME/.bashrc"
source "$LMOD_ROOT/lmod/init/bash"
module --force purge
module load Stages/2025 GCC/13.3.0 GCCcore/.13.3.0 Python/3.12.3

weathergen_home=${SLURM_SUBMIT_DIR:?Submit this script from the WeatherGenerator root.}
private_config="$weathergen_home/../WeatherGenerator-private/hpc/jupiter/config/paths.yml"
family=${EVAL_FAMILY:?Set EVAL_FAMILY to ch7e, c8lv, ci4f, c8lx or c8rx.}
case "$family" in
    ch7e) templates=(eval_config_40step_best_det_template.yml) ;;
    c8lv|ci4f|c8rx) templates=(eval_config_40step_best_ens_template.yml eval_config_40step_best_ensmean_template.yml) ;;
    # c8lx: crps8long over 80 steps, all 81 ERA5 channels (per-step member MSE for the 80-step
    # lead-balance growth table of the 20-day training)
    c8lx) templates=(eval_config_80step_allch_ens_template.yml eval_config_40step_best_ensmean_template.yml) ;;
    *)
        echo "EVAL_FAMILY must be ch7e, c8lv, ci4f, c8lx or c8rx; got '$family'." >&2
        exit 2
        ;;
esac
# ENSMEAN_ONLY=1 (ensemble families): score only the 4-member-mean RMSE into evaluation_ens_mean/
if [[ "${ENSMEAN_ONLY:-0}" == 1 && "$family" != ch7e ]]; then
    templates=(eval_config_40step_best_ensmean_template.yml)
fi
# CH_GROUP=1..11 (c8lx): per-member RMSE of one group of the 81 channels into evaluation_g<N>/
# (~78 GB of host memory per channel at 80 steps x 4 members x 8 samples, so 7-8 per job)
channels=""
if [[ -n "${CH_GROUP:-}" ]]; then
    [[ "$family" == c8lx ]] || { echo "CH_GROUP is for EVAL_FAMILY=c8lx." >&2; exit 2; }
    case "$CH_GROUP" in
        1) channels='["10u", "10v", "2d", "2t", "hcc", "lcc", "mcc", "msl"]' ;;
        2) channels='["q_10", "q_1000", "q_150", "q_200", "q_250", "q_300", "q_400", "q_500"]' ;;
        3) channels='["q_600", "q_700", "q_850", "q_925", "skt", "sp", "ssrd", "strd"]' ;;
        4) channels='["t_10", "t_100", "t_1000", "t_150", "t_200", "t_250", "t_300", "t_400"]' ;;
        5) channels='["t_50", "t_500", "t_600", "t_700", "t_850", "t_925", "tcc"]' ;;
        6) channels='["u_10", "u_100", "u_1000", "u_150", "u_200", "u_250", "u_300"]' ;;
        7) channels='["u_400", "u_50", "u_500", "u_600", "u_700", "u_850", "u_925"]' ;;
        8) channels='["v_10", "v_100", "v_1000", "v_150", "v_200", "v_250", "v_300"]' ;;
        9) channels='["v_400", "v_50", "v_500", "v_600", "v_700", "v_850", "v_925"]' ;;
        10) channels='["z_10", "z_100", "z_1000", "z_150", "z_200", "z_250", "z_300"]' ;;
        11) channels='["z_400", "z_50", "z_500", "z_600", "z_700", "z_850", "z_925"]' ;;
        *) echo "CH_GROUP must be 1-11; got '$CH_GROUP'." >&2; exit 2 ;;
    esac
    templates=(eval_config_80step_rmse_group_template.yml)
fi

[[ -f "$weathergen_home/.venv/bin/activate" ]]
[[ -f "$private_config" ]]

run_id=$(printf '%s%04d' "$family" "${SLURM_ARRAY_TASK_ID:?Missing array task ID}")
for rank in 0 1 2 3; do
    archive="$weathergen_home/results/$run_id/validation_chkpt00000_rank$(printf '%04d' "$rank").zip"
    [[ -s "$archive" ]] || {
        echo "Missing or empty $archive" >&2
        exit 1
    }
done

cd "$weathergen_home"
source .venv/bin/activate
export WEATHERGEN_PRIVATE_CONF="$private_config"

for template in "${templates[@]}"; do
    [[ -f "config/evaluate/$template" ]]
    tmp_config=$(mktemp "$weathergen_home/config/evaluate/.${run_id}.XXXXXX.yml")
    sed -e "s/__RUN_ID__/$run_id/g" -e "s/__GROUP__/${CH_GROUP:-}/g" \
        -e "s/__CHANNELS__/$channels/g" "config/evaluate/$template" > "$tmp_config"
    echo "run_id=$run_id template=$template"
    .venv/bin/evaluate --config "$tmp_config"
    rm -f "$tmp_config"
done
