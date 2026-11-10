#!/bin/bash
#SBATCH --job-name=evaluate-mpi
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=evaluate-mpi.%j.out
#SBATCH --qos=np

# Chunked evaluation over MPI: 4 ranks per node (rank 0 master, the others workers), each
# rank using 32 cores through loky (get_num_workers reads SLURM_CPUS_PER_TASK, capped at
# max_workers, default 36). 4 x 32 fills a 128-core Atos node and the master takes 1/4 of one.
# Usage, from the repo root: sbatch packages/evaluate/scripts/evaluate_distributed.slurm <config.yml>
# ECMWF's sbatch wrapper rejects arguments after the script, so extra evaluate options go
# through the environment:
#   EVAL_OPTIONS="distributed.chunks.forecast_step=3 evaluation.summary_dir=./plots_mpi" \
#       sbatch --export=ALL packages/evaluate/scripts/evaluate_distributed.slurm config.yml
#
# The environment must be ready before submitting (uv run is not used in the job: it would
# rebuild the editable packages from every rank at once into the shared .venv):
#   module load prgenv/gnu openmpi && uv sync --extra mpi

set -euo pipefail
CONFIG=${1:?config file}

module load prgenv/gnu openmpi 2>/dev/null || true
# mpi4py built against the site's Open MPI
export MPI4PY_LIBMPI=/usr/local/apps/hpcx-openmpi/2.21.3/GNU/8.5/ec-hpcx-ompi/lib/libmpi.so.40

# sbatch runs a copy of this script from its spool directory, so the repo is not found from
# $0: submit from the repo root (the directory uv run needs to see pyproject.toml in).
cd "$SLURM_SUBMIT_DIR"
if [ -n "${EVAL_OPTIONS:-}" ]; then
    # shellcheck disable=SC2086  # the options are meant to split on spaces
    srun --mpi=pmix .venv/bin/python -m weathergen.evaluate.distributed.mpi --config "$CONFIG" --options $EVAL_OPTIONS
else
    srun --mpi=pmix .venv/bin/python -m weathergen.evaluate.distributed.mpi --config "$CONFIG"
fi
