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

srun uv --offline run inference --from-run-id co1h3rd5 --options validation_config.samples_per_mini_epoch=1 validation_config.output.num_samples=1 'validation_config.output.streams=[ERA5_winter, ERA5_summer, ERA5_spring, ERA5_autumn]' training_config.forecast.num_steps=1440 training_config.forecast.forecast_chunk_size=100 validation_config.start_date=2022-10-01T00:00

