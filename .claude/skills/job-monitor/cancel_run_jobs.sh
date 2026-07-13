#!/bin/bash
# Cancel all queued/running SLURM jobs of ONE WeatherGenerator run.
# The only sanctioned way for the job monitor to cancel jobs: raw scancel is
# denied in the monitor session's permissions, so cancellation is structurally
# limited to jobs named weathergen_<run_id>[-_]* .
# Usage: cancel_run_jobs.sh <run_id>
set -euo pipefail

run_id="${1:?usage: cancel_run_jobs.sh <run_id>}"
if [[ ! "$run_id" =~ ^[a-z][a-z0-9]{7}$ ]]; then
    echo "invalid run_id (expected 8-char id): $run_id" >&2
    exit 1
fi

ids=$(squeue -h -u "$USER" -o "%i %j" | awk -v p="^weathergen_${run_id}[-_]" '$2 ~ p {print $1}')
if [[ -z "$ids" ]]; then
    echo "no queued/running jobs for run $run_id"
    exit 0
fi
echo "cancelling jobs of run $run_id:" $ids
# shellcheck disable=SC2086
scancel $ids
