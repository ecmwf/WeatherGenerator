#!/bin/bash
#SBATCH -p all
#SBATCH -A zam
#SBATCH --time=01:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
##SBATCH --mem=0
#SBATCH --job-name WeatherGenerator
#SBATCH --output=./results/slurm-%j-%x.out

# Print current datetime:
echo "START" $(date +"%Y-%m-%d %H:%M:%S")

# Print node list:
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES"
echo "SLURM_NODELIST=$SLURM_NODELIST"

# Note: the following srun commands assume that pyxis plugin is installed on a SLURM cluster.
# https://github.com/NVIDIA/pyxis


# Set number of threads to use for parallel regions:
export OMP_NUM_THREADS=1

# Set MLPerf variables:
export DATESTAMP=$(date +"%y%m%d%H%M%S%N")
export EXP_ID=1

# Run the command:
# Note: MASTER_ADDR and MASTER_PORT variables are set automatically by pyxis.

module purge
module try-load GCC Apptainer-Tools
module load GCC CUDA NCCL

##export NCCL_DEBUG=INFO 
export NCCL_SOCKET_IFNAME=ib0 
export GLOO_SOCKET_IFNAME=ib0 


export sif_path="/p/project1/hclimrep/weather_generator_apptainer_images/weather_generator_jsc_pytorch-24.12-py3-arm64.sif"



export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
if [ "$SYSTEMNAME" = juwelsbooster ] \
       || [ "$SYSTEMNAME" = juwels ] \
       || [ "$SYSTEMNAME" = jurecadc ] \
       || [ "$SYSTEMNAME" = jusuf ]; then
    # Allow communication over InfiniBand cells on JSC machines.
    MASTER_ADDR="$MASTER_ADDR"i
fi
MASTER_PORT=25678

srun --cpu-bind=none --export=ALL env -u CUDA_VISIBLE_DEVICES \
    apptainer exec \
    --nv \
    --bind .:/workspace/WeatherGenerator\
    --bind /p/scratch/hclimrep/shared/weather_generator_data:/data \
    "$sif_path" \
    bash -c "\
        export PYTHONPATH='/workspace/WeatherGenerator/src/weathergen'; \
        python -u /workspace/WeatherGenerator/src/run_train.py"
