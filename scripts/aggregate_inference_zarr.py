#!/usr/bin/env python3
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Merge the NUM_SHARDS one-sample outputs of an inference array into one run.

Set RUN_PREFIX to the array's run-id prefix (e.g. "cw6" for cw600000..cw600015,
"sdn" for sdn00000..sdn00015) before running.

Usage:
    python scripts/aggregate_inference_zarr.py cw6merge
    python scripts/aggregate_inference_zarr.py sdnmerge   # with RUN_PREFIX = "sdn"
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import zarr
from zarr.storage import ZipStore

RESULTS_ROOT = Path("/iopsstor/scratch/cscs/thunter/shared_work/results")
MODELS_ROOT = Path("/iopsstor/scratch/cscs/thunter/shared_work/models")
RUN_PREFIX = "cw6"
NUM_SHARDS = 16
MINI_EPOCH = 0
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_run_id", help="Eight-character run ID for the merged output.")
    return parser.parse_args()


def archive_path(run_id: str) -> Path:
    return RESULTS_ROOT / run_id / f"validation_chkpt{MINI_EPOCH:05d}_rank0000.zip"


def copy_group(source: zarr.Group, destination: zarr.Group) -> None:
    """Recursively copy Zarr arrays, groups, and attributes."""
    for name, array in source.arrays():
        values = np.asarray(array)
        destination.create_array(
            name,
            data=values,
            chunks="auto" if values.size == 0 else array.chunks,
            attributes=dict(array.attrs),
        )

    for name, group in source.groups():
        child = destination.create_group(name, attributes=dict(group.attrs))
        copy_group(group, child)


def copy_sample(source: zarr.Group, destination: zarr.Group, sample_index: int) -> None:
    """Copy source sample 0 to its contiguous sample index in the merged archive."""
    if list(source.group_keys()) != ["0"]:
        raise ValueError(
            f"Expected one source sample named '0', found {list(source.group_keys())}."
        )

    source_sample = source["0"]
    destination_sample = destination.create_group(
        str(sample_index), attributes=dict(source_sample.attrs)
    )
    for stream_name, source_stream in source_sample.groups():
        destination_stream = destination_sample.create_group(
            stream_name, attributes=dict(source_stream.attrs)
        )

        # Chunked offset-one output contains forecast groups 1..40. ZarrIO looks
        # at fstep 0 to identify offset-one output, so retain an empty fstep 0.
        if "0" not in source_stream:
            destination_stream.create_group("0")

        for fstep, source_fstep in source_stream.groups():
            destination_fstep = destination_stream.create_group(
                fstep, attributes=dict(source_fstep.attrs)
            )
            copy_group(source_fstep, destination_fstep)


def write_config(output_run_id: str) -> None:
    """Copy the first shard's saved inference config for use by evaluation."""
    first_run_id = f"{RUN_PREFIX}00000"
    source = MODELS_ROOT / first_run_id / f"model_{first_run_id}_chkpt{MINI_EPOCH:05d}.json"
    if not source.is_file():
        raise FileNotFoundError(f"Missing inference config: {source}")

    config = json.loads(source.read_text(encoding="utf-8"))
    config["general"]["run_id"] = output_run_id
    config["test_config"]["samples_per_mini_epoch"] = NUM_SHARDS
    config["test_config"]["output"]["num_samples"] = NUM_SHARDS

    destination_dir = MODELS_ROOT / output_run_id
    destination_dir.mkdir()
    destination = destination_dir / f"model_{output_run_id}_chkpt{MINI_EPOCH:05d}.json"
    destination.write_text(json.dumps(config), encoding="utf-8")


def main() -> None:
    output_run_id = parse_args().output_run_id
    if len(output_run_id) != 8 or not output_run_id[0].isalpha() or not output_run_id.isalnum():
        raise ValueError(
            "output_run_id must be eight alphanumeric characters starting with a letter."
        )

    shard_ids = [f"{RUN_PREFIX}{index:05d}" for index in range(NUM_SHARDS)]
    archives = [archive_path(run_id) for run_id in shard_ids]
    missing = [path for path in archives if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing shard output(s):\n" + "\n".join(map(str, missing)))

    output_dir = RESULTS_ROOT / output_run_id
    output = output_dir / f"validation_chkpt{MINI_EPOCH:05d}_rank0000.zip"
    partial_output = output.with_suffix(".partial.zip")
    if output_dir.exists() or (MODELS_ROOT / output_run_id).exists() or partial_output.exists():
        raise FileExistsError(f"Output run ID is already in use: {output_run_id}")

    output_dir.mkdir()
    try:
        with ZipStore(partial_output, mode="w") as destination_store:
            destination_root = zarr.group(store=destination_store)
            for sample_index, (run_id, archive) in enumerate(zip(shard_ids, archives, strict=True)):
                logger.info(f"Copying {run_id} -> sample {sample_index}")
                with ZipStore(archive, mode="r") as source_store:
                    source_root = zarr.open_group(store=source_store, mode="r")
                    copy_sample(source_root, destination_root, sample_index)

        partial_output.replace(output)
        write_config(output_run_id)
    except Exception:
        logger.error(f"Aggregation failed. Inspect or remove {output_dir} before retrying.")
        raise

    logger.info(f"Merged {NUM_SHARDS} samples into {output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
