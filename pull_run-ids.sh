#!/bin/bash

# 1. Ensure the user passed an argument
if [ -z "$1" ]; then
    echo "Error: Missing argument. Please provide a Run ID."
    echo "Usage: $0 <run-id>"
    exit 1
fi

RUN_ID="$1"

# 2. Safety Check: Verify it is an 8-character alphanumeric string
if [[ ! "$RUN_ID" =~ ^[a-zA-Z0-9]{8}$ ]]; then
    echo "Warning: '$RUN_ID' is not a standard 8-character alphanumeric run ID."
    read -p "Are you sure you want to proceed? (y/N): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Sync aborted."
        exit 1
    fi
fi

echo "========================================================"
echo "🚀 Starting pull-sync for Run ID: $RUN_ID"
echo "========================================================"

# Task 1: Pull specific model checkpoint files from remote
echo "--> Pulling models (latest files and base config JSON)..."
rsync -avuP --mkpath \
    --include="model_${RUN_ID}.json" \
    --include="*latest*" \
    --include="*08*" \
    --include="*chkpt00000.json" \
    --exclude="*" \
    "santis:/users/mkarlbau/repos/WeatherGenerator/models/${RUN_ID}/" \
    "./models/${RUN_ID}/"
    # "ecm334637@transfer1.bsc.es:/home/ecm/ecm334637/repos/WeatherGenerator/models/${RUN_ID}/" \

echo "--------------------------------------------------------"

# Task 2: Pull all results from remote, excluding images/gifs
echo "--> Pulling results (excluding PNGs and GIFs)..."
rsync -avuP --mkpath \
    --exclude="*.png*" \
    --exclude="*.gif*" \
    --exclude="*.zip*" \
    "santis:/users/mkarlbau/repos/WeatherGenerator/results/${RUN_ID}/" \
    "./results/${RUN_ID}/"
    # "ecm334637@transfer1.bsc.es:/home/ecm/ecm334637/repos/WeatherGenerator/results/${RUN_ID}/" \

echo "--------------------------------------------------------"

# Task 3: Pull slurm directory from remote, ignoring symlinks
echo "--> Pulling slurm directory (ignoring symlinks)..."
rsync -avuP --mkpath --no-links \
    --exclude="*.venv*" \
    --exclude="*profiling*" \
    "santis:/iopsstor/scratch/cscs/thunter/slurm/slurm_weathergen_${RUN_ID}_dir/" \
    "/e/scratch/weatherai/slurm/slurm_weathergen_${RUN_ID}_dir/"
    # "ecm334637@transfer1.bsc.es:/gpfs/scratch/ehpc01/wegen_data2026/slurm/slurm_weathergen_${RUN_ID}_dir/" \

echo "========================================================"
echo "✅ All pull tasks finished for $RUN_ID!"
echo "========================================================"
