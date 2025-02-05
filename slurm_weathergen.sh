#!/bin/bash -x
#SBATCH --account=ab0995
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --chdir=.
#SBATCH --output=logs/atmorep-%x.%j.out
#SBATCH --error=logs/atmorep-%x.%j.err

# import modules at dkrz
module purge
module load nvhpc/24.7-gcc-11.2.0 git/2.43.3-gcc-11.2.0
source .venv/bin/activate 

export CUDA_HOME=/sw/spack-levante/nvhpc-23.9-xpxqeo/Linux_x86_64/24.7/cuda/12.5
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Set environment variables to point to GCC
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# Ensure GCC is first in your PATH
export PATH=/usr/bin:$PATH

export UCX_TLS="^cma"
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0,1,2,3

# so processes know who to talk to
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
echo "MASTER_ADDR: $MASTER_ADDR"

export NCCL_DEBUG=TRACE
echo "nccl_debug: $NCCL_DEBUG"

# work-around for flipping links issue on JUWELS-BOOSTER
export NCCL_IB_TIMEOUT=250
export UCX_RC_TIMEOUT=16s
export NCCL_IB_RETRY_CNT=50

# export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export TORCHDYNAMO_VERBOSE=1  # Get detailed Dynamo logs
# export TORCH_COMPILE_DEBUG=1  # Debug compilation issues
# export TORCH_LOGS="+dynamo,+inductor"
# export TORCHDYNAMO_VERBOSE=1
# export NCCL_DEBUG=TRACE

echo "======== ENVIRONMENT ========"
echo "CUDA_HOME: $CUDA_HOME"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "Compiler: $(which gcc)"
echo "NVCC: $(which nvcc)"
echo "============================="

echo "Starting job."
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Number of Tasks: $SLURM_NTASKS"
date

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

srun --label --cpu-bind=v --accel-bind=v ${SLURM_SUBMIT_DIR}/.venv/bin/python -u train.py > output/output_${SLURM_JOBID}.txt

echo "Finished job."
sstat -j $SLURM_JOB_ID.batch   --format=JobID,MaxVMSize
date
