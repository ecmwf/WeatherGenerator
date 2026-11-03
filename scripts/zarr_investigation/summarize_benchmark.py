#!/usr/bin/env python3
"""Summarise benchmark_anemoi_reader.py runs and check that they read the same data.

  summarize_benchmark.py RUN.jsonl [RUN.jsonl ...]

A run is one output name: the per-task files of a multi-task run (<name>.rank<k>.jsonl) are
merged. One row per run, in argument order:

- windows: timed windows over all tasks (warmup and empty windows excluded);
- median / p90 ms: per-window `_get` time over all tasks;
- MiB/s: decoded bytes of all tasks over the slowest task's total time, i.e. what the node
  delivered;
- checksums: windows whose data hash equals that of the same window index in the first run
  that has it (needs --checksum in both runs).
"""

# ruff: noqa: T201  (command-line tool: prints its report)

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load(paths: list[Path]) -> dict[str, dict]:
    runs: dict[str, dict] = defaultdict(lambda: {"headers": [], "windows": [], "summaries": []})
    for path in paths:
        name = re.sub(r"\.rank\d+$", "", path.stem)
        run = runs[name]
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if "header" in record:
                run["headers"].append(record["header"])
            elif "window" in record:
                run["windows"].append(record["window"])
            elif "summary" in record:
                run["summaries"].append(record["summary"])
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()
    runs = _load(args.paths)

    reference: dict[int, str] = {}
    print(
        f"{'run':<28} {'tasks':>5} {'windows':>7} {'median':>8} {'p90':>8} "
        f"{'MiB/s':>7} {'empty':>5}  checksums"
    )
    for name, run in runs.items():
        missing = [s for s in run["summaries"] if "total_seconds" not in s]
        if not run["summaries"] or missing:
            print(f"{name:<28} incomplete: {len(run['summaries'])} summaries, {len(missing)} empty")
            continue
        timed = [
            w["duration_ns"] / 1e6 for w in run["windows"] if not w["warmup"] and w["num_points"]
        ]
        mib = sum(s["decoded_mib_per_second"] * s["total_seconds"] for s in run["summaries"])
        seconds = max(s["total_seconds"] for s in run["summaries"])
        empty = sum(s["num_empty"] for s in run["summaries"])

        matches, compared = 0, 0
        for window in run["windows"]:
            digest = window.get("blake2b")
            if digest is None:
                continue
            expected = reference.setdefault(window["idx"], digest)
            compared += 1
            matches += expected == digest
        print(
            f"{name:<28} {len(run['summaries']):>5} {len(timed):>7} "
            f"{np.median(timed):>8.1f} {np.percentile(timed, 90):>8.1f} "
            f"{mib / seconds:>7.0f} {empty:>5}  {matches}/{compared}"
        )

    print()
    for name, run in runs.items():
        array = run["headers"][0]["array"] if run["headers"] else {}
        codecs = array.get("codecs") or array.get("compressor")
        print(
            f"{name}: chunks {array.get('chunks')}, shards {array.get('shards')}, "
            f"codecs {json.dumps(codecs)}"
        )


if __name__ == "__main__":
    main()
