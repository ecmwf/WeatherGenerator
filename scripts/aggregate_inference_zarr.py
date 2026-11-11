#!/usr/bin/env python3
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Merge the one-sample outputs of an inference array into one run.

Merges runs <prefix>00000 ... <prefix>000NN (NN = num_shards - 1) of
scripts/inference_sdnbfpve_40step_array_safe.sbatch into the run <prefix>merge.
Paths are those of the Zurich hackathon (course shared_work directory).

Nothing is copied: the evaluation reader supports several rank files per run
(validation_chkpt00000_rank0000.zip, rank0001.zip, ...), each holding local sample 0,
and assigns them consecutive global sample indices. So shard i becomes a relative
symlink rank000i.zip -> ../<prefix>0000i/validation_chkpt00000_rank0000.zip. This
takes seconds and no disk space, and is fine on a login node. Do not delete the
<prefix>0000i runs afterwards, since the merged run points at them.

Usage:
    uv run python scripts/aggregate_inference_zarr.py <prefix>
    uv run python scripts/aggregate_inference_zarr.py <prefix> --num-shards 4
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

RESULTS_ROOT = Path("/iopsstor/scratch/cscs/course_29fe58c1/shared_work/results")
MODELS_ROOT = Path("/iopsstor/scratch/cscs/course_29fe58c1/shared_work/models")
MINI_EPOCH = 0
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("prefix", help="3-character run-id prefix used for the inference array.")
    parser.add_argument(
        "--num-shards",
        type=int,
        default=4,
        help="number of array tasks to merge, <prefix>00000 onwards (default 4)",
    )
    parser.add_argument(
        "--output-run-id",
        help="run id of the merged run (default: <prefix>merge)",
    )
    return parser.parse_args()


def archive_path(run_id: str) -> Path:
    return RESULTS_ROOT / run_id / f"validation_chkpt{MINI_EPOCH:05d}_rank0000.zip"


def write_config(output_run_id: str, prefix: str, num_shards: int) -> None:
    """Copy the first shard's saved inference config for use by evaluation."""
    first_run_id = f"{prefix}00000"
    source = MODELS_ROOT / first_run_id / f"model_{first_run_id}_chkpt{MINI_EPOCH:05d}.json"
    if not source.is_file():
        raise FileNotFoundError(f"Missing inference config: {source}")

    config = json.loads(source.read_text(encoding="utf-8"))
    config["general"]["run_id"] = output_run_id
    config["test_config"]["samples_per_mini_epoch"] = num_shards
    config["test_config"]["output"]["num_samples"] = num_shards

    destination_dir = MODELS_ROOT / output_run_id
    destination_dir.mkdir()
    destination = destination_dir / f"model_{output_run_id}_chkpt{MINI_EPOCH:05d}.json"
    destination.write_text(json.dumps(config), encoding="utf-8")


def main() -> None:
    args = parse_args()
    prefix, num_shards = args.prefix, args.num_shards
    if len(prefix) != 3 or not prefix[0].isalpha() or not prefix.isalnum():
        raise ValueError("prefix must be three alphanumeric characters starting with a letter.")
    output_run_id = args.output_run_id or f"{prefix}merge"

    shard_ids = [f"{prefix}{index:05d}" for index in range(num_shards)]
    archives = [archive_path(run_id) for run_id in shard_ids]
    missing = [path for path in archives if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing shard output(s):\n" + "\n".join(map(str, missing)))

    output_dir = RESULTS_ROOT / output_run_id
    if output_dir.exists() or (MODELS_ROOT / output_run_id).exists():
        raise FileExistsError(f"Output run ID is already in use: {output_run_id}")

    output_dir.mkdir()
    try:
        for rank, (run_id, archive) in enumerate(zip(shard_ids, archives, strict=True)):
            link = output_dir / f"validation_chkpt{MINI_EPOCH:05d}_rank{rank:04d}.zip"
            link.symlink_to(Path("..") / run_id / archive.name)
            logger.info(f"{link.name} -> {run_id} (sample {rank})")
        write_config(output_run_id, prefix, num_shards)
    except Exception:
        logger.error(f"Aggregation failed. Inspect or remove {output_dir} before retrying.")
        raise

    logger.info(f"Merged {num_shards} runs into {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
