#!/bin/bash

#SBATCH --job-name=continuous_train
#SBATCH --output=./runlogs/output_%j.txt
#SBATCH --error=./runlogs/error_%j.txt
#SBATCH --mem=120G
#SBATCH --partition=normal
#SBATCH --gres=gpu:4
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=4
#SBATCH --time=12:00:00

#SBATCH --environment=wgen-pytorch
#SBATCH -A a-a01


export MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
echo "MASTER_ADDR : $MASTER_ADDR"

export MASTER_PORT=29514

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export NCCL_DEBUG=INFO

cd $HOME/projects/Meteoswiss/WeatherGenerator/

source $HOME/projects/Meteoswiss/WeatherGenerator/.venv/bin/activate
source /users/ktezcan/projects/Meteoswiss/WeatherGenerator/.venv/bin/activate
pip install -e . --no-dependencies


# srun bash -c "source /users/ktezcan/projects/Meteoswiss/WeatherGenerator/.venv/bin/activate && \
#      python /users/ktezcan/projects/Meteoswiss/WeatherGenerator/personal/print_slurm_env_vars.py"

srun bash -c "source /users/ktezcan/projects/Meteoswiss/WeatherGenerator/.venv/bin/activate && \
     python /users/ktezcan/projects/Meteoswiss/WeatherGenerator/src/weathergen/__init__.py \
     --private_config /users/ktezcan/projects/Meteoswiss/WeatherGenerator/personal/paths.yaml \
     --config /users/ktezcan/projects/Meteoswiss/WeatherGenerator/personal/train_maskedtoken/overwrite_config.yaml"