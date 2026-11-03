#!/usr/bin/env python3
"""Copy a Zarr 2 anemoi store, keeping only a range of time steps of its `data` chunks.

Everything except the chunk files of `data` is copied unchanged, including the full `dates`
array, the root attributes and `.zmetadata`, so the copy describes the same dataset as the
source and the same training dates select the same windows. Chunks of `data` outside the
range are left out; Zarr returns the fill value for them, so only reads inside the range are
meaningful. Chunk files are copied byte for byte, so reads inside the range return exactly
the source's data.

Used by ZRB-E002 to get a Zarr 2 store (and a squashfs image of it) of one month without
copying the whole dataset:

  stage_v2_time_slice.py SOURCE.zarr DEST.zarr --first-step 378336 --last-step 379127

`--dry-run` only counts the files and bytes that would be copied.
"""

# ruff: noqa: T201  (command-line tool: prints its report)

import argparse
import itertools
import json
import math
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zarr


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", type=Path, help="Zarr 2 store to copy from.")
    parser.add_argument("dest", type=Path, help="New store; must not exist.")
    parser.add_argument("--first-step", type=int, required=True, help="First time step kept.")
    parser.add_argument("--last-step", type=int, required=True, help="Last time step kept.")
    parser.add_argument("--workers", type=int, default=16, help="Parallel chunk copies.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _chunk_keys(zarray: dict, first_step: int, last_step: int) -> list[str]:
    """Keys of the `data` chunks holding the time steps first_step..last_step."""
    shape, chunks = zarray["shape"], zarray["chunks"]
    separator = zarray.get("dimension_separator") or "."
    time_chunks = range(first_step // chunks[0], last_step // chunks[0] + 1)
    other_chunks = [range(math.ceil(s / c)) for s, c in zip(shape[1:], chunks[1:], strict=True)]
    return [
        separator.join(str(i) for i in index)
        for index in itertools.product(time_chunks, *other_chunks)
    ]


def _copy_metadata(source: Path, dest: Path) -> int:
    """Copy everything but the chunk files of `data`; return the number of files copied."""
    dest.mkdir(parents=True)
    for entry in source.iterdir():
        if entry.name == "data":
            continue
        if entry.is_dir():
            shutil.copytree(entry, dest / entry.name, copy_function=shutil.copyfile)
        else:
            shutil.copyfile(entry, dest / entry.name)
    (dest / "data").mkdir()
    for name in (".zarray", ".zattrs"):
        if (source / "data" / name).exists():
            shutil.copyfile(source / "data" / name, dest / "data" / name)
    return sum(1 for path in dest.rglob("*") if path.is_file())


def _check(source: Path, dest: Path, steps: list[int]) -> None:
    """Fail unless dest returns the source's bytes at the given time steps."""
    src = zarr.open_array(store=str(source), path="data", mode="r")
    dst = zarr.open_array(store=str(dest), path="data", mode="r")
    dates = zarr.open_array(store=str(source), path="dates", mode="r")
    for step in steps:
        if np.asarray(src[step]).tobytes() != np.asarray(dst[step]).tobytes():
            raise RuntimeError(f"Copy differs from the source at time step {step}")
        print(f"step {step} ({dates[step]}): identical")


def main() -> None:
    args = _parse_args()
    if args.last_step < args.first_step:
        raise ValueError("--last-step is before --first-step")
    zarray = json.loads((args.source / "data" / ".zarray").read_text())
    if zarray.get("zarr_format") != 2:
        raise ValueError(f"{args.source}/data is not a Zarr 2 array")
    if args.last_step >= zarray["shape"][0]:
        raise ValueError(f"--last-step past the {zarray['shape'][0]} time steps of the source")
    print(
        f"data: shape {zarray['shape']}, chunks {zarray['chunks']}, "
        f"dtype {zarray['dtype']}, compressor {zarray.get('compressor')}, "
        f"filters {zarray.get('filters')}"
    )

    keys = _chunk_keys(zarray, args.first_step, args.last_step)
    sources = [args.source / "data" / key for key in keys]
    present = [path for path in sources if path.exists()]
    num_bytes = sum(path.stat().st_size for path in present)
    print(
        f"{len(present)} of {len(keys)} chunk files present "
        f"(missing ones are fill values), {num_bytes / 2**30:.1f} GiB"
    )
    if args.dry_run:
        return
    if args.dest.exists():
        raise FileExistsError(args.dest)

    started = time.perf_counter()
    num_metadata_files = _copy_metadata(args.source, args.dest)

    def copy(path: Path) -> None:
        target = args.dest / "data" / path.relative_to(args.source / "data")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(copy, present))
    seconds = time.perf_counter() - started
    print(
        f"copied {num_metadata_files} metadata files and {len(present)} chunk files, "
        f"{num_bytes / 2**30:.1f} GiB in {seconds:.0f} s "
        f"({num_bytes / 2**20 / seconds:.0f} MiB/s)"
    )
    _check(args.source, args.dest, [args.first_step, args.last_step])


if __name__ == "__main__":
    main()
