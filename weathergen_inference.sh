#!/bin/bash

#SBATCH --job-name=train
#SBATCH --output=./logs/output_%j.txt
#SBATCH --error=./logs/error_%j.txt
#SBATCH --exclusive --mem=450G
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH -A ch17
#SBATCH --output=logs/weathergen-%x.%j.out
#SBATCH --error=logs/weathergen-%x.%j.err

source .venv/bin/activate


srun uv --offline run inference --from-run-id t2a0vosm  --options test_config.start_date=202301010000 test_config.end_date=202312310000 test_config.output.num_samples=1 test_config.samples_per_mini_epoch=1 test_config.forecast.num_steps=650 test_config.output.streams=[ERA5]
