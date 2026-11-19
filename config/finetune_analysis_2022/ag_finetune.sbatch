#!/usr/bin/env bash
#SBATCH --job-name=wg-finetune
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --account=ecwegen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=36
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=finetune-%j.log
#
# One stage of the operational-analysis fine-tune of sdnbfpve on ag (GH200).
# Run from the WeatherGenerator checkout root; see config/finetune_analysis_2022/README.md.
#
#   sbatch config/finetune_analysis_2022/ag_finetune.sbatch CONFIG FROM_RUN_ID FROM_EPOCH [key=value ...]
set -euo pipefail
CONFIG=${1:?config}; FROM_RUN_ID=${2:?run to continue from}; FROM_EPOCH=${3:?its mini-epoch}; shift 3

export WEATHERGEN_PRIVATE_CONF=$PWD/../WeatherGenerator-private/hpc/hpc2020/config/paths.yml
export LD_LIBRARY_PATH=/usr/local/apps/gcc/15.2.0/lib64:${LD_LIBRARY_PATH:-}   # flash-attn on aarch64
export PYTORCH_ALLOC_CONF=expandable_segments:True
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_ADDR="$(getent ahostsv4 "$MASTER_ADDR" | awk 'NR == 1 {print $1}')"
RUN_ID="$(python3 -c 'import secrets, string as s
print(secrets.choice(s.ascii_lowercase) + "".join(secrets.choice(s.ascii_lowercase + s.digits) for _ in range(7)))')"
echo "run id $RUN_ID: $CONFIG from $FROM_RUN_ID:$FROM_EPOCH on $SLURM_NTASKS GPUs, options: $*"

srun --cpu-bind=none .venv/bin/train_continue \
    --from-run-id "$FROM_RUN_ID" --mini-epoch "$FROM_EPOCH" \
    --config "$CONFIG" --run-id "$RUN_ID" \
    --private-config "$WEATHERGEN_PRIVATE_CONF" \
    --options world_size_original="$SLURM_NTASKS" "$@" \
    < /dev/null
echo "done: run id $RUN_ID (checkpoint ..._chkpt00008.chkpt in the shared models directory)"
