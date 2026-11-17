#!/usr/bin/env python3
"""Pool shard-level 40-step RMSE score JSONs for cross-model ratio plotting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr


FAMILIES = ("ps9k", "rrvj", "i325")
SHARDS = range(1, 16)
STREAM = "ERA5"
REGION = "global"
METRIC = "rmse"
CHECKPOINT = 0


def score_path(results_dir: Path, family: str, shard: int) -> Path:
    run_id = f"{family}{shard:04d}"
    return results_dir / run_id / "evaluation" / f"{run_id}_{STREAM}_{REGION}_{METRIC}_chkpt{CHECKPOINT:05d}.json"


def load_score(path: Path) -> xr.DataArray:
    with path.open() as handle:
        payload = json.load(handle)

    scores = payload.get("scores", [payload])
    if len(scores) != 1:
        raise ValueError(f"Expected one score version in {path}, found {len(scores)}.")

    score = xr.DataArray.from_dict(scores[0])
    required_dims = {"sample", "forecast_step", "channel", "ens"}
    if set(score.dims) != required_dims or "init_times" not in score.coords:
        raise ValueError(f"Unexpected score layout in {path}: dims={score.dims}, coords={list(score.coords)}")
    if score.init_times.dims != ("sample",):
        raise ValueError(f"init_times is not indexed by sample in {path}: {score.init_times.dims}")
    return score


def load_family(results_dir: Path, family: str) -> tuple[xr.DataArray, list[str]]:
    shards: list[xr.DataArray] = []
    input_paths: list[str] = []
    sample_offset = 0

    for shard in SHARDS:
        path = score_path(results_dir, family, shard)
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing score file: {path}")

        score = load_score(path)
        shards.append(
            score.assign_coords(sample=np.arange(sample_offset, sample_offset + score.sizes["sample"]))
        )
        input_paths.append(str(path))
        sample_offset += score.sizes["sample"]

    combined = xr.concat(shards, dim="sample", coords="minimal", compat="override", join="exact")
    init_times = np.asarray(combined.init_times.values, dtype=np.int64)
    if len(np.unique(init_times)) != len(init_times):
        raise ValueError(f"Duplicate initialization times found for {family}.")
    return combined, input_paths


def select_common_init_times(score: xr.DataArray, common_init_times: np.ndarray) -> xr.DataArray:
    by_init_time = score.swap_dims({"sample": "init_times"}).drop_vars("sample")
    return by_init_time.sel(init_times=common_init_times)


def pool_rmse(score: xr.DataArray, common_init_times: np.ndarray) -> xr.DataArray:
    selected = select_common_init_times(score, common_init_times)
    # Every inference uses the same fixed global grid, so each per-init RMSE has equal weight.
    pooled = np.sqrt((selected**2).mean(dim="init_times", skipna=True)).expand_dims(sample=[0])
    pooled = pooled.assign_coords(init_times=("sample", common_init_times[:1]))
    pooled.attrs = dict(score.attrs)
    return pooled


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    results_dir = repository_root / "results"
    output_root = results_dir / "score_summary_40step"

    family_scores: dict[str, xr.DataArray] = {}
    input_paths: dict[str, list[str]] = {}
    for family in FAMILIES:
        family_scores[family], input_paths[family] = load_family(results_dir, family)

    common_init_times = np.array(
        sorted(
            set.intersection(
                *(set(score.init_times.values.astype(np.int64).tolist()) for score in family_scores.values())
            )
        ),
        dtype=np.int64,
    )
    if len(common_init_times) == 0:
        raise ValueError("The three families have no common initialization times.")

    for family, score in family_scores.items():
        pooled = pool_rmse(score, common_init_times)
        run_id = f"{family}0001"
        output_dir = output_root / family
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{run_id}_{STREAM}_{REGION}_{METRIC}_chkpt{CHECKPOINT:05d}.json"
        payload = {
            "scores": [pooled.to_dict()],
            "pooled_init_times": [
                str(np.datetime64(int(value), "ns")) for value in common_init_times
            ],
            "input_score_files": input_paths[family],
        }
        with output_path.open("w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Wrote {output_path} from {len(common_init_times)} common initialization times.")


if __name__ == "__main__":
    main()