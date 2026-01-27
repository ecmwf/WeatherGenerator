#!/bin/bash

#SBATCH --job-name=train
#SBATCH --output=./logs/output_%j.txt
#SBATCH --error=./logs/error_%j.txt
#SBATCH --exclusive --mem=450G
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH -A ch17
#SBATCH --output=logs/weathergen-%x.%j.out
#SBATCH --error=logs/weathergen-%x.%j.err


UENV_IMAGE="prgenv-gnu/25.6:v2"

FROM_RUN_ID="$1"
#RUN_ID="$2"

export FROM_RUN_ID
#export RUN_ID 

echo "Top-level from_run_id: $FROM_RUN_ID"
echo "Top-level run_id: $RUN_ID"

echo "=== Checking for uenv image: $UENV_IMAGE ==="

IMAGE_EXISTS=false

if [ "$IMAGE_EXISTS" = false ]; then
    if uenv image inspect "$UENV_IMAGE" &>/dev/null; then
        IMAGE_EXISTS=true
    fi
fi

if [ "$IMAGE_EXISTS" = false ]; then
    echo "========================================"
    echo "ERROR: uenv image '$UENV_IMAGE' not found!"
    echo "========================================"
    echo ""
    echo "The image needs to be pulled before use."
    echo ""
    echo "Steps to fix:"
    echo ""
    echo "  1. On the santis login node, run:"
    echo "     uenv image pull $UENV_IMAGE"
    echo ""
    echo "  2. Wait for download to complete (this may take a few minutes)"
    echo ""
    echo "  3. Verify the image is available:"
    echo "     uenv image ls"
    echo ""
    echo "  4. Re-submit your SLURM job"
    echo ""
    echo "========================================"
    exit 1
fi

echo "✓ Image '$UENV_IMAGE' found"
echo ""

FROM_RUN_ID="$1"
#RUN_ID="$2"

uenv run "$UENV_IMAGE" --view=modules -- bash << 'EOF'

module load aws-ofi-nccl/1.16.0

export NCCL_NET="AWS Libfabric"
export MPICH_GPU_SUPPORT_ENABLED=0
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_PROTO=^LL128

export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=16384
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RX_MATCH_MODE=software
export FI_MR_CACHE_MONITOR=userfaultfd

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
export MASTER_PORT=29514

# disable core dumps
ulimit -c 0
ulimit -t unlimited

export CC=/usr/bin/gcc
export NCCL_DEBUG=INFO

echo "Starting job."
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Number of Tasks: $SLURM_NTASKS"
echo "from_run_id: $FROM_RUN_ID"
#echo "run_id: $RUN_ID"
echo "WEATHERGEN_HOME: $WEATHERGEN_HOME"
echo "WEATHERGEN_CONFIG_EXTRA: $WEATHERGEN_CONFIG_EXTRA"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
date


#cd $WEATHERGEN_HOME
source .venv/bin/activate

srun uv run --offline inference --from_run_id "$FROM_RUN_ID" --samples=16 --start_date=2023-10-01 --end_date=2023-12-01 --options forecast_steps=80
#srun uv run inference --from_run_id "$FROM_RUN_ID" --run_id "$RUN_ID" --samples 16 --start_date=2023-10-01 --end_date=2023-12-01 --options forecast_steps=80

echo "Finished job."
sstat -j $SLURM_JOB_ID.batch   --format=JobID,MaxVMSize
date
EOF

