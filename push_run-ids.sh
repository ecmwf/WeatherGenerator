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
echo "🚀 Starting sync for Run ID: $RUN_ID"
echo "========================================================"

# Task 1: Sync specific model checkpoint files
echo "--> Syncing models (latest files and base config JSON)..."
rsync -avuP --mkpath \
    --include="model_${RUN_ID}.json" \
    --include="*latest*" \
    --include="*chkpt00000.json" \
    --exclude="*" \
    "./models/${RUN_ID}/" \
    "santis:/users/mkarlbau/repos/WeatherGenerator/models/${RUN_ID}/"
    # --include="*88*" \

echo "--------------------------------------------------------"

# Task 2: Sync all results, excluding images/gifs
echo "--> Syncing results (excluding PNGs and GIFs)..."
rsync -avuP --mkpath \
    --exclude="*.png*" \
    --exclude="*.gif*" \
    --exclude="*.zip*" \
    "./results/${RUN_ID}/" \
    "santis:/users/mkarlbau/repos/WeatherGenerator/results/${RUN_ID}/"

echo "--------------------------------------------------------"

# Task 3: Push slurm directory to remote, ignoring symlinks
echo "--> Pushing slurm directory (ignoring symlinks)..."
rsync -avuP --mkpath --no-links \
    --exclude="*.venv*" \
    --exclude="*profiling*" \
    "/e/scratch/weatherai/slurm/slurm_weathergen_${RUN_ID}_dir/" \
    "santis:/iopsstor/scratch/cscs/thunter/slurm/slurm_weathergen_${RUN_ID}_dir/"
    # "ecm334637@transfer1.bsc.es:/gpfs/scratch/ehpc01/wegen_data2026/slurm/slurm_weathergen_${RUN_ID}_dir/"

echo "========================================================"
echo "✅ All sync tasks finished for $RUN_ID!"
echo "========================================================"
