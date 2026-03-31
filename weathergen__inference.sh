#!/bin/bash -x
#SBATCH --account=weatherai
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --chdir=.
#SBATCH --partition=booster
#SBATCH --output=logs/weathergen-%x.%j.out
#SBATCH --error=logs/weathergen-%x.%j.err

source .venv/bin/activate

srun  uv --offline run inference --from-run-id $1 --samples 1 --start-date 202204010000 --end-date 202312010000 --options training_config.forecast.num_steps=1440 model_path="/e/scratch/weatherai/shared_work/models" training_config.forecast.forecast_chunk_size=100 zarr_store=zip streams_directory="./config/streams/era5_1deg_forecasting/"
