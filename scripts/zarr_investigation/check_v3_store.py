#!/usr/bin/env python3
"""Check a sharded Zarr 3 anemoi store converted from a Zarr 2 source.

  check_v3_store.py SOURCE_V2.zarr STORE_V3.zarr --start-date 2022-03-01T00:00 \
      --shard-steps 96 [--num-steps 24 | --all-steps]

For the shard-size comparison (ZRB-E003), the stores must differ from the source only in
the shard size. The script checks, without writing anything:

1. Layout of `data`: Zarr format 3, dtype, inner chunks equal to the source chunks, shards of
   --shard-steps time steps, inner codecs (blosc cname, clevel, shuffle) equal to the source
   compressor.
2. Completeness: every shard file of the time range exists, and its index lists every inner
   chunk (time step) inside the array.
3. Metadata: what `restore_zarr_attrs.py --check-only` checks (dates, start/end date and
   frequency against the time axis, variables and statistics against the data), plus: `dates`
   equals the source dates from --start-date, small arrays (statistics, coordinates) and
   `variables` equal the source's, consolidated metadata (if any) matches `data`.
4. Data: time steps compared bit for bit with the source at the same date: first and last
   step, the steps on both sides of shard boundaries, and random ones (--num-steps in total),
   or every step with --all-steps.
5. If anemoi-datasets is importable: `open_dataset` on the store returns the same dates and
   data.

Prints one line per check and ends with OK or the list of problems (exit code 1).
"""

# ruff: noqa: T201  (command-line tool: prints its report)

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import zarr
from numpy.typing import NDArray
from restore_zarr_attrs import check_metadata, source_mismatch

DATA = "data"
_MISSING = np.uint64(2**64 - 1)
_SHUFFLE_V2 = {0: "noshuffle", 1: "shuffle", 2: "bitshuffle"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", type=Path, help="Zarr 2 store the conversion started from.")
    parser.add_argument("store", type=Path, help="Converted Zarr 3 store.")
    parser.add_argument(
        "--start-date", required=True, help="Date of the store's first step, e.g. 2022-03-01T00:00."
    )
    parser.add_argument(
        "--shard-steps", type=int, required=True, help="Expected time steps per shard."
    )
    steps = parser.add_mutually_exclusive_group()
    steps.add_argument("--num-steps", type=int, default=24, help="Time steps compared.")
    steps.add_argument("--all-steps", action="store_true", help="Compare every time step.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the random steps.")
    return parser.parse_args()


class Report:
    def __init__(self) -> None:
        self.problems: list[str] = []

    def ok(self, message: str) -> None:
        print(f"ok    {message}")

    def fail(self, message: str) -> None:
        print(f"FAIL  {message}")
        self.problems.append(message)

    def expect(self, what: str, actual: object, expected: object) -> bool:
        if actual == expected:
            self.ok(f"{what}: {actual}")
            return True
        self.fail(f"{what} is {actual}, expected {expected}")
        return False


def _chunk_key(metadata: dict, coords: tuple[int, ...]) -> str:
    """Relative path of a chunk (here: shard) file, per the array's chunk_key_encoding."""
    encoding = metadata["chunk_key_encoding"]
    if encoding["name"] == "v2":
        separator = encoding.get("configuration", {}).get("separator", ".")
        return separator.join(map(str, coords))
    separator = encoding.get("configuration", {}).get("separator", "/")
    return separator.join(["c", *map(str, coords)])


def check_layout(report: Report, source_zarray: dict, metadata: dict, shard_steps: int) -> dict:
    """Check `data`'s zarr.json against the source; return the sharding configuration."""
    report.expect("zarr_format", metadata.get("zarr_format"), 3)
    report.expect("data_type", metadata.get("data_type"), np.dtype(source_zarray["dtype"]).name)
    codecs = metadata.get("codecs", [])
    if len(codecs) != 1 or codecs[0].get("name") != "sharding_indexed":
        report.fail(f"codecs are {[c.get('name') for c in codecs]}, expected [sharding_indexed]")
        return {}
    sharding = codecs[0]["configuration"]
    shard_shape = metadata["chunk_grid"]["configuration"]["chunk_shape"]
    report.expect("inner chunk shape", sharding["chunk_shape"], source_zarray["chunks"])
    report.expect("shard shape", shard_shape, [shard_steps, *source_zarray["chunks"][1:]])

    compressor = source_zarray.get("compressor") or {}
    inner = {c["name"]: c.get("configuration", {}) for c in sharding.get("codecs", [])}
    blosc = inner.get("blosc")
    if compressor.get("id") != "blosc" or blosc is None:
        report.fail(f"inner codecs {list(inner)}, source compressor {compressor.get('id')}")
    else:
        report.expect("blosc cname", blosc.get("cname"), compressor.get("cname"))
        report.expect("blosc clevel", blosc.get("clevel"), compressor.get("clevel"))
        report.expect(
            "blosc shuffle", blosc.get("shuffle"), _SHUFFLE_V2.get(compressor.get("shuffle"))
        )
    if source_zarray.get("filters"):
        report.fail(f"source has filters {source_zarray['filters']}; not compared")
    return sharding


def check_completeness(
    report: Report, store: Path, metadata: dict, sharding: dict, num_steps: int
) -> None:
    """Every shard file exists and its index lists every inner chunk inside the array."""
    shard_shape = metadata["chunk_grid"]["configuration"]["chunk_shape"]
    inner_shape = sharding["chunk_shape"]
    shape = metadata["shape"]
    index_codecs = [c["name"] for c in sharding.get("index_codecs", [])]
    if index_codecs not in (["bytes"], ["bytes", "crc32c"]):
        report.fail(f"shard index codecs {index_codecs}: completeness not checked")
        return
    per_shard = [s // c for s, c in zip(shard_shape, inner_shape, strict=True)]
    index_bytes = math.prod(per_shard) * 16 + (4 if "crc32c" in index_codecs else 0)
    at_end = sharding.get("index_location", "end") == "end"
    grid = [math.ceil(s / c) for s, c in zip(shape, shard_shape, strict=True)]
    if grid[1:] != [1] * (len(grid) - 1):
        report.fail(f"shard grid {grid}: expected shards to split only the time axis")
        return

    missing_files, missing_chunks, files, total_bytes = [], 0, 0, 0
    for shard in range(grid[0]):
        path = store / DATA / _chunk_key(metadata, (shard, 0, 0, 0))
        if not path.exists():
            missing_files.append(shard)
            continue
        size = path.stat().st_size
        files += 1
        total_bytes += size
        with path.open("rb") as f:
            f.seek(size - index_bytes if at_end else 0)
            raw = f.read(math.prod(per_shard) * 16)
        index = np.frombuffer(raw, dtype="<u8").reshape(per_shard[0], 2)
        steps_in_array = min(per_shard[0], num_steps - shard * per_shard[0])
        absent = (index[:steps_in_array] == _MISSING).all(axis=1)
        missing_chunks += int(absent.sum())
        present = index[:steps_in_array][~absent]
        if len(present) and int((present[:, 0] + present[:, 1]).max()) > size:
            report.fail(f"shard {shard}: index points past the end of the file")
    if missing_files:
        report.fail(
            f"{len(missing_files)} of {grid[0]} shard files missing (shards {missing_files[:5]})"
        )
    else:
        report.ok(f"all {grid[0]} shard files present ({total_bytes / 2**30:.1f} GiB)")
    if missing_chunks:
        report.fail(f"{missing_chunks} time steps missing from the shard indices")
    elif files:
        report.ok(f"every time step present in the shard indices ({num_steps} steps)")


def check_metadata_vs_source(report: Report, source_path: Path, store: Path, offset: int) -> None:
    source = zarr.open_group(str(source_path), mode="r", zarr_format=2)
    group = zarr.open_group(str(store), mode="r", zarr_format=3, use_consolidated=False)
    for problem in source_mismatch(source_path, store, time_sliced=True):
        report.fail(problem)
    problems = check_metadata(store)
    for problem in problems:
        report.fail(f"metadata: {problem}")
    if not problems:
        report.ok("metadata matches the data (restore_zarr_attrs.check_metadata)")

    num_steps = group[DATA].shape[0]
    if "dates" in group:
        expected = source["dates"][offset : offset + num_steps]
        if np.array_equal(group["dates"][:], expected):
            report.ok(f"dates equal the source's ({expected[0]} .. {expected[-1]})")
        else:
            report.fail("dates differ from the source dates at --start-date")
    else:
        report.fail("no 'dates' array")

    source_attrs, attrs = source.attrs.asdict(), group.attrs.asdict()
    missing = sorted(set(source_attrs) - set(attrs))
    if missing:
        report.fail(f"root attributes missing from the store: {missing}")
    else:
        report.ok(f"all {len(source_attrs)} source root attributes present")
    if "variables" in source_attrs:
        same = attrs.get("variables") == source_attrs["variables"]
        report.expect("variables equal the source's", same, True)

    names, bad = [], []
    for name, array in source.arrays():
        if name in (DATA, "dates"):
            continue
        names.append(name)
        if name not in group:
            bad.append(f"array '{name}' missing")
        elif not np.array_equal(group[name][:], array[:], equal_nan=True):
            bad.append(f"array '{name}' differs from the source")
    for problem in bad:
        report.fail(problem)
    if not bad:
        report.ok(f"arrays equal the source's: {', '.join(sorted(names))}")

    root = json.loads((store / "zarr.json").read_text())
    consolidated = (root.get("consolidated_metadata") or {}).get("metadata", {})
    if DATA in consolidated:
        own = json.loads((store / DATA / "zarr.json").read_text())
        keys = ("shape", "chunk_grid", "codecs")
        if all(consolidated[DATA].get(k) == own.get(k) for k in keys):
            report.ok("consolidated metadata matches data/zarr.json")
        else:
            report.fail("consolidated metadata for 'data' is stale (shape/shards/codecs)")


def _steps_to_compare(num_steps: int, shard_steps: int, count: int, seed: int) -> list[int]:
    steps = {0, num_steps - 1}
    for boundary in range(shard_steps, num_steps, shard_steps)[:4]:
        steps |= {boundary - 1, boundary}
    rng = np.random.default_rng(seed)
    while len(steps) < min(count, num_steps):
        steps.add(int(rng.integers(num_steps)))
    return sorted(steps)


def _same_bits(a: NDArray, b: NDArray) -> bool:
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and np.array_equal(a.view(np.uint8), b.view(np.uint8))
    )


def check_data(
    report: Report, source: zarr.Array, data: zarr.Array, offset: int, steps: list[int]
) -> None:
    started, differ = time.monotonic(), []
    for k, step in enumerate(steps):
        if not _same_bits(data[step], source[offset + step]):
            differ.append(step)
        if (k + 1) % 50 == 0:
            print(f"      {k + 1}/{len(steps)} steps, {time.monotonic() - started:.0f} s")
    if differ:
        report.fail(f"{len(differ)} of {len(steps)} steps differ from the source: {differ[:10]}")
    else:
        report.ok(
            f"{len(steps)} steps bit-identical to the source "
            f"({time.monotonic() - started:.0f} s; steps {steps[:6]}...)"
        )


def check_anemoi(report: Report, store: Path, data: zarr.Array, dates: NDArray) -> None:
    try:
        from anemoi.datasets import open_dataset
    except ImportError:
        print("skip  anemoi-datasets not installed: open_dataset not checked")
        return
    ds = open_dataset(str(store))
    report.expect("anemoi len", len(ds), data.shape[0])
    report.expect("anemoi first date", ds.dates[0], dates[0])
    step = data.shape[0] // 2
    if _same_bits(np.asarray(ds[step]), data[step]):
        report.ok(f"anemoi ds[{step}] equals the store's data")
    else:
        report.fail(f"anemoi ds[{step}] differs from the store's data")


def main() -> None:
    args = _parse_args()
    report = Report()
    print(f"store  {args.store}\nsource {args.source}\n")

    source = zarr.open_group(str(args.source), mode="r", zarr_format=2)
    source_zarray = json.loads((args.source / DATA / ".zarray").read_text())
    metadata = json.loads((args.store / DATA / "zarr.json").read_text())
    group = zarr.open_group(str(args.store), mode="r", zarr_format=3, use_consolidated=False)
    data = group[DATA]
    num_steps = data.shape[0]

    dates = source["dates"][:]
    start = np.datetime64(args.start_date)
    offset = int(np.searchsorted(dates, start))
    if offset == len(dates) or dates[offset] != start:
        raise SystemExit(f"--start-date {args.start_date} is not one of the source dates")
    if offset + num_steps > len(dates):
        raise SystemExit(f"{num_steps} steps from {args.start_date} run past the source end")
    print(f"{num_steps} steps = source steps {offset}..{offset + num_steps - 1}\n")

    sharding = check_layout(report, source_zarray, metadata, args.shard_steps)
    if sharding:
        check_completeness(report, args.store, metadata, sharding, num_steps)
    check_metadata_vs_source(report, args.source, args.store, offset)
    steps = (
        list(range(num_steps))
        if args.all_steps
        else _steps_to_compare(num_steps, args.shard_steps, args.num_steps, args.seed)
    )
    check_data(report, source[DATA], data, offset, steps)
    check_anemoi(report, args.store, data, dates[offset : offset + num_steps])

    if report.problems:
        print(f"\nFAILED: {len(report.problems)} problem(s)")
        for problem in report.problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print(f"\nOK: {args.store.name} (shards of {args.shard_steps} steps)")


if __name__ == "__main__":
    main()
