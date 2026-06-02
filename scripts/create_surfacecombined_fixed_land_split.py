#!/usr/bin/env python3
"""Create fixed-land spatial train/validation splits for SurfaceCombined observations."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numcodecs
import numpy as np
import numpy.typing as npt
import zarr
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

DEFAULT_SOURCE = Path(
    "/e/data1/slmet/ml_training/observations-ea-ofb-0001-1979-2025-combined-surface-v3.zarr"
)
DEFAULT_OUTPUT_DIR = Path("/e/data1/slmet/ml_training")
DEFAULT_TRAIN_NAME = (
    "observations-ea-ofb-0001-1979-2022-combined-surface-v3-"
    "fixed-land-spatial80-lsm09-min10.zarr"
)
DEFAULT_VALIDATION_NAME = (
    "observations-ea-ofb-0001-2023-combined-surface-v3-"
    "fixed-land-heldout20-lsm09-min10.zarr"
)
DEFAULT_MANIFEST_NAME = (
    "observations-ea-ofb-0001-combined-surface-v3-"
    "fixed-land-spatial-split-lsm09-min10-seed42.json"
)

BASE_DATETIME = np.datetime64("1970-01-01T00:00:00", "ns")
HOUR = np.timedelta64(1, "h")
INDEX_NAME = "idx_197001010000_1"
COORD_KEY_DTYPE = np.dtype("V8")
EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class SplitConfig:
    source: str
    train_output: str
    validation_output: str
    manifest: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    reportypes: list[int]
    lsm_threshold: float
    min_latitude: float
    max_latitude: float
    min_validation_rows_per_coord: int
    train_fraction: float
    min_train_validation_distance_km: float
    seed: int
    chunk_rows: int


@dataclass(frozen=True)
class WriteSummary:
    output: str
    rows: int
    first_date: str | None
    last_date: str | None
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create SurfaceCombined fixed-land train/validation zarr stores by splitting exact "
            "2023 station-proxy coordinates, optionally pruning train coordinates that are too "
            "close to validation coordinates."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-output", type=Path, default=None)
    parser.add_argument("--validation-output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--train-start", default="1979-01-01T00:00:00")
    parser.add_argument("--train-end", default="2023-01-01T00:00:00")
    parser.add_argument("--validation-start", default="2023-01-01T00:00:00")
    parser.add_argument("--validation-end", default="2024-01-01T00:00:00")
    parser.add_argument(
        "--reportypes",
        type=int,
        nargs="+",
        default=[16001, 16002, 16004, 16065, 16076],
    )
    parser.add_argument("--lsm-threshold", type=float, default=0.9)
    parser.add_argument(
        "--min-latitude",
        type=float,
        default=-90.0,
        help="Exclude rows with latitude below this value before splitting/writing.",
    )
    parser.add_argument(
        "--max-latitude",
        type=float,
        default=90.0,
        help="Exclude rows with latitude above this value before splitting/writing.",
    )
    parser.add_argument("--min-validation-rows-per-coord", type=int, default=10)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument(
        "--min-train-validation-distance-km",
        type=float,
        default=0.0,
        help=(
            "Minimum geodesic distance between validation coordinates and retained train "
            "coordinates. If positive, train coordinates closer than this to any validation "
            "coordinate are removed after the random split."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=0,
        help="Rows per source chunk. Defaults to the source data chunk length.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--count-outputs-in-dry-run",
        action="store_true",
        help="In dry-run mode, also scan the train/validation windows to count output rows.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def to_datetime(value: str) -> np.datetime64:
    return np.datetime64(value, "ns")


def hour_offset(value: np.datetime64) -> int:
    return int((value - BASE_DATETIME) / HOUR)


def coord_keys_from_lat_lon(lat_lon: npt.NDArray) -> npt.NDArray:
    coords = np.ascontiguousarray(lat_lon.astype(np.float32, copy=False))
    return coords.view(COORD_KEY_DTYPE).reshape(-1)


def coords_from_keys(keys: Iterable[bytes]) -> list[list[float]]:
    sorted_keys = sorted(keys)
    if not sorted_keys:
        return []
    buffer = b"".join(sorted_keys)
    coords = np.frombuffer(buffer, dtype=np.float32).reshape(-1, 2)
    return [[float(lat), float(lon)] for lat, lon in coords]


def coords_array_from_keys(keys: Iterable[bytes]) -> npt.NDArray:
    sorted_keys = sorted(keys)
    if not sorted_keys:
        return np.empty((0, 2), dtype=np.float32)
    buffer = b"".join(sorted_keys)
    return np.frombuffer(buffer, dtype=np.float32).reshape(-1, 2).copy()


def make_key_array(keys: Iterable[bytes]) -> npt.NDArray:
    sorted_keys = sorted(keys)
    if not sorted_keys:
        return np.empty(0, dtype=COORD_KEY_DTYPE)
    return np.frombuffer(b"".join(sorted_keys), dtype=COORD_KEY_DTYPE).copy()


def json_safe(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):  # noqa: TID251
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def resolve_outputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    train_output = args.train_output or args.output_dir / DEFAULT_TRAIN_NAME
    validation_output = args.validation_output or args.output_dir / DEFAULT_VALIDATION_NAME
    manifest = args.manifest or args.output_dir / DEFAULT_MANIFEST_NAME
    return train_output, validation_output, manifest


def require_columns(colnames: list[str], names: Iterable[str]) -> dict[str, int]:
    missing = [name for name in names if name not in colnames]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return {name: colnames.index(name) for name in names}


def coarse_row_bounds_for_window(
    index: zarr.Array, start: np.datetime64, end: np.datetime64
) -> tuple[int, int]:
    start_hour = hour_offset(start)
    end_hour = hour_offset(end)
    if start_hour < 0 or end_hour >= index.shape[0]:
        raise ValueError(
            f"Window [{start}, {end}) maps to hours [{start_hour}, {end_hour}), "
            f"outside index length {index.shape[0]}"
        )
    coarse_start_hour = max(start_hour - 1, 0)
    return int(index[coarse_start_hour]), int(index[end_hour])


def split_ranges(start: int, end: int, chunk_rows: int) -> Iterable[tuple[int, int]]:
    for chunk_start in range(start, end, chunk_rows):
        yield chunk_start, min(chunk_start + chunk_rows, end)


def base_filter(
    data_chunk: npt.NDArray,
    columns: dict[str, int],
    reportypes: npt.NDArray,
    lsm_threshold: float,
    min_latitude: float,
    max_latitude: float,
) -> npt.NDArray:
    return (
        np.isin(data_chunk[:, columns["reportype"]].astype(np.int64), reportypes)
        & (data_chunk[:, columns["lsm"]] >= lsm_threshold)
        & (data_chunk[:, columns["lat"]] >= min_latitude)
        & (data_chunk[:, columns["lat"]] <= max_latitude)
        & np.isfinite(data_chunk[:, columns["lat"]])
        & np.isfinite(data_chunk[:, columns["lon"]])
    )


def collect_candidate_coord_counts(
    data: zarr.Array,
    dates: zarr.Array,
    columns: dict[str, int],
    row_start: int,
    row_end: int,
    date_start: np.datetime64,
    date_end: np.datetime64,
    chunk_rows: int,
    reportypes: npt.NDArray,
    lsm_threshold: float,
    min_latitude: float,
    max_latitude: float,
) -> dict[bytes, int]:
    counts: dict[bytes, int] = {}
    progress = tqdm(
        total=row_end - row_start,
        unit="rows",
        unit_scale=True,
        desc="scan 2023 candidate coords",
    )
    for start, stop in split_ranges(row_start, row_end, chunk_rows):
        chunk = data[start:stop]
        date_chunk = dates[start:stop][:, 0]
        mask = base_filter(chunk, columns, reportypes, lsm_threshold, min_latitude, max_latitude)
        mask &= (date_chunk >= date_start) & (date_chunk < date_end)
        if np.any(mask):
            keys = coord_keys_from_lat_lon(chunk[mask][:, [columns["lat"], columns["lon"]]])
            unique_keys, unique_counts = np.unique(keys, return_counts=True)
            for key, count in zip(unique_keys, unique_counts, strict=True):
                key_bytes = bytes(key)
                counts[key_bytes] = counts.get(key_bytes, 0) + int(count)
        progress.update(stop - start)
    progress.close()
    return counts


def split_candidate_keys(
    candidate_counts: dict[bytes, int],
    min_rows: int,
    train_fraction: float,
    seed: int,
) -> tuple[set[bytes], set[bytes], dict[str, int]]:
    candidate_keys = np.array(
        [key for key, count in candidate_counts.items() if count >= min_rows], dtype=object
    )
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(candidate_keys))
    train_count = int(np.floor(len(candidate_keys) * train_fraction))
    train_keys = set(candidate_keys[permutation[:train_count]].tolist())
    validation_keys = set(candidate_keys[permutation[train_count:]].tolist())
    retained_2023_rows = sum(candidate_counts[key] for key in train_keys | validation_keys)
    total_2023_rows = sum(candidate_counts.values())
    stats = {
        "candidate_coords_before_min_rows": len(candidate_counts),
        "candidate_coords_after_min_rows": len(candidate_keys),
        "candidate_2023_rows_before_min_rows": total_2023_rows,
        "candidate_2023_rows_after_min_rows": retained_2023_rows,
        "train_coords": len(train_keys),
        "validation_coords": len(validation_keys),
    }
    return train_keys, validation_keys, stats


def nearest_distances_km(
    query_coords: npt.NDArray,
    reference_coords: npt.NDArray,
    batch_size: int = 256,
) -> npt.NDArray:
    if len(query_coords) == 0:
        return np.empty(0, dtype=np.float64)
    if len(reference_coords) == 0:
        return np.full(len(query_coords), np.inf, dtype=np.float64)

    reference_latitude = np.deg2rad(reference_coords[:, 0].astype(np.float64))
    reference_longitude = np.deg2rad(reference_coords[:, 1].astype(np.float64))
    reference_cos_latitude = np.cos(reference_latitude)
    nearest = np.empty(len(query_coords), dtype=np.float64)

    for start in range(0, len(query_coords), batch_size):
        stop = min(start + batch_size, len(query_coords))
        query = query_coords[start:stop].astype(np.float64, copy=False)
        query_latitude = np.deg2rad(query[:, 0])
        query_longitude = np.deg2rad(query[:, 1])
        latitude_delta = query_latitude[:, None] - reference_latitude[None, :]
        longitude_delta = query_longitude[:, None] - reference_longitude[None, :]
        haversine_term = (
            np.sin(latitude_delta / 2.0) ** 2
            + np.cos(query_latitude)[:, None]
            * reference_cos_latitude[None, :]
            * np.sin(longitude_delta / 2.0) ** 2
        )
        distance = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.minimum(haversine_term, 1.0)))
        nearest[start:stop] = distance.min(axis=1)
    return nearest


def apply_train_validation_distance_threshold(
    train_keys: set[bytes],
    validation_keys: set[bytes],
    candidate_counts: dict[bytes, int],
    min_distance_km: float,
) -> tuple[set[bytes], set[bytes], dict[str, object]]:
    if min_distance_km <= 0.0:
        return train_keys, validation_keys, {}

    sorted_train_keys = sorted(train_keys)
    train_coords = coords_array_from_keys(sorted_train_keys)
    validation_coords = coords_array_from_keys(validation_keys)
    nearest_validation_distance = nearest_distances_km(train_coords, validation_coords)
    keep_train = nearest_validation_distance >= min_distance_km
    retained_train_keys = {
        sorted_train_keys[index] for index in np.flatnonzero(keep_train).tolist()
    }
    pruned_train_keys = train_keys - retained_train_keys
    train_rows_before_prune = sum(candidate_counts[key] for key in train_keys)
    train_rows_after_prune = sum(candidate_counts[key] for key in retained_train_keys)
    validation_rows = sum(candidate_counts[key] for key in validation_keys)
    retained_distances = nearest_validation_distance[keep_train]
    distance_stats = {
        "distance_prune_side": "train",
        "distance_threshold_km": min_distance_km,
        "train_coords_before_distance_prune": len(train_keys),
        "validation_coords_before_distance_prune": len(validation_keys),
        "train_coords_pruned_by_distance": len(pruned_train_keys),
        "train_coords_after_distance_prune": len(retained_train_keys),
        "validation_coords_after_distance_prune": len(validation_keys),
        "train_2023_rows_before_distance_prune": train_rows_before_prune,
        "train_2023_rows_after_distance_prune": train_rows_after_prune,
        "train_2023_rows_pruned_by_distance": train_rows_before_prune - train_rows_after_prune,
        "validation_2023_rows_after_distance_prune": validation_rows,
        "candidate_2023_rows_after_distance_prune": train_rows_after_prune + validation_rows,
        "min_train_validation_distance_km_before_prune": (
            float(nearest_validation_distance.min()) if len(nearest_validation_distance) else None
        ),
        "min_train_validation_distance_km_after_prune": (
            float(retained_distances.min()) if len(retained_distances) else None
        ),
    }
    return retained_train_keys, validation_keys, distance_stats


def open_output_group(path: Path, overwrite: bool) -> zarr.Group:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {path}. Pass --overwrite to replace it.")
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return zarr.open_group(path, mode="w", zarr_format=2)


def clone_compressor(array: zarr.Array) -> numcodecs.abc.Codec | None:
    return copy.deepcopy(array.compressor)


def create_output_arrays(
    group: zarr.Group,
    source_data: zarr.Array,
    source_dates: zarr.Array,
    source_index: zarr.Array,
) -> tuple[zarr.Array, zarr.Array, zarr.Array]:
    data = group.create_dataset(
        "data",
        shape=(0, source_data.shape[1]),
        chunks=source_data.chunks,
        dtype=source_data.dtype,
        compressor=clone_compressor(source_data),
    )
    dates = group.create_dataset(
        "dates",
        shape=(0, 1),
        chunks=source_dates.chunks,
        dtype=source_dates.dtype,
        compressor=clone_compressor(source_dates),
    )
    index = group.create_dataset(
        INDEX_NAME,
        shape=source_index.shape,
        chunks=source_index.chunks,
        dtype=source_index.dtype,
        compressor=clone_compressor(source_index),
    )
    for key, value in source_data.attrs.items():
        data.attrs[key] = value
    for key, value in source_dates.attrs.items():
        dates.attrs[key] = value
    for key, value in source_index.attrs.items():
        index.attrs[key] = value
    return data, dates, index


def build_index_from_hour_counts(
    hour_counts: npt.NDArray, exact_hour_counts: npt.NDArray
) -> npt.NDArray:
    if len(hour_counts) == 0:
        return hour_counts.astype(np.int64)
    cumulative_before_hour = np.empty_like(hour_counts, dtype=np.int64)
    cumulative_before_hour[0] = 0
    if len(hour_counts) > 1:
        cumulative_before_hour[1:] = np.cumsum(hour_counts[:-1], dtype=np.int64)
    hourly_index = cumulative_before_hour + exact_hour_counts
    return hourly_index


def count_selected_rows(
    data: zarr.Array,
    dates: zarr.Array,
    columns: dict[str, int],
    row_start: int,
    row_end: int,
    date_start: np.datetime64,
    date_end: np.datetime64,
    chunk_rows: int,
    reportypes: npt.NDArray,
    lsm_threshold: float,
    min_latitude: float,
    max_latitude: float,
    coord_keys: npt.NDArray,
    description: str,
) -> int:
    total = 0
    progress = tqdm(total=row_end - row_start, unit="rows", unit_scale=True, desc=description)
    for start, stop in split_ranges(row_start, row_end, chunk_rows):
        chunk = data[start:stop]
        date_chunk = dates[start:stop][:, 0]
        mask = base_filter(chunk, columns, reportypes, lsm_threshold, min_latitude, max_latitude)
        mask &= (date_chunk >= date_start) & (date_chunk < date_end)
        if np.any(mask):
            candidate_rows = np.flatnonzero(mask)
            keys = coord_keys_from_lat_lon(
                chunk[candidate_rows][:, [columns["lat"], columns["lon"]]]
            )
            total += int(np.isin(keys, coord_keys).sum())
        progress.update(stop - start)
    progress.close()
    return total


def append_selected_rows(
    output_data: zarr.Array,
    output_dates: zarr.Array,
    data_rows: npt.NDArray,
    date_rows: npt.NDArray,
    position: int,
) -> int:
    row_count = data_rows.shape[0]
    if row_count == 0:
        return position
    new_position = position + row_count
    output_data.resize((new_position, output_data.shape[1]))
    output_dates.resize((new_position, 1))
    output_data[position:new_position] = data_rows
    output_dates[position:new_position] = date_rows
    return new_position


def update_output_attrs(
    data: zarr.Array,
    split_name: str,
    source: Path,
    config: SplitConfig,
    rows: int,
    first_date: np.datetime64 | None,
    last_date: np.datetime64 | None,
) -> None:
    data.attrs["split_name"] = split_name
    data.attrs["split_source"] = str(source)
    data.attrs["split_config"] = asdict(config)
    data.attrs["split_rows"] = rows
    data.attrs["start_date"] = str(first_date) if first_date is not None else None
    data.attrs["end_date"] = str(last_date) if last_date is not None else None


def write_split_dataset(
    output_path: Path,
    split_name: str,
    source_path: Path,
    source_data: zarr.Array,
    source_dates: zarr.Array,
    source_index: zarr.Array,
    columns: dict[str, int],
    row_start: int,
    row_end: int,
    date_start: np.datetime64,
    date_end: np.datetime64,
    chunk_rows: int,
    reportypes: npt.NDArray,
    lsm_threshold: float,
    min_latitude: float,
    max_latitude: float,
    coord_keys: npt.NDArray,
    config: SplitConfig,
    overwrite: bool,
) -> WriteSummary:
    start_time = time.monotonic()
    group = open_output_group(output_path, overwrite=overwrite)
    output_data, output_dates, output_index = create_output_arrays(
        group, source_data, source_dates, source_index
    )
    hour_counts = np.zeros(source_index.shape[0], dtype=np.int64)
    exact_hour_counts = np.zeros(source_index.shape[0], dtype=np.int64)
    position = 0
    first_date: np.datetime64 | None = None
    last_date: np.datetime64 | None = None

    progress = tqdm(
        total=row_end - row_start, unit="rows", unit_scale=True, desc=f"write {split_name}"
    )
    for start, stop in split_ranges(row_start, row_end, chunk_rows):
        chunk = source_data[start:stop]
        dates_chunk = source_dates[start:stop]
        flat_dates_chunk = dates_chunk[:, 0]
        mask = base_filter(chunk, columns, reportypes, lsm_threshold, min_latitude, max_latitude)
        mask &= (flat_dates_chunk >= date_start) & (flat_dates_chunk < date_end)
        if np.any(mask):
            candidate_rows = np.flatnonzero(mask)
            keys = coord_keys_from_lat_lon(
                chunk[candidate_rows][:, [columns["lat"], columns["lon"]]]
            )
            selected_rows = candidate_rows[np.isin(keys, coord_keys)]
        else:
            selected_rows = np.empty(0, dtype=np.int64)
        if len(selected_rows) > 0:
            selected_dates = dates_chunk[selected_rows]
            flat_dates = selected_dates[:, 0]
            selected_hours = ((flat_dates - BASE_DATETIME) / HOUR).astype(np.int64)
            valid_hours = (selected_hours >= 0) & (selected_hours < len(hour_counts))
            hour_counts += np.bincount(
                selected_hours[valid_hours], minlength=len(hour_counts)
            ).astype(np.int64)
            exact_hour_mask = flat_dates == (
                BASE_DATETIME + selected_hours.astype("timedelta64[h]")
            )
            exact_selected_hours = selected_hours[valid_hours & exact_hour_mask]
            exact_hour_counts += np.bincount(
                exact_selected_hours, minlength=len(exact_hour_counts)
            ).astype(np.int64)
            first_date = flat_dates[0] if first_date is None else first_date
            last_date = flat_dates[-1]
            position = append_selected_rows(
                output_data, output_dates, chunk[selected_rows], selected_dates, position
            )
        progress.update(stop - start)
    progress.close()

    output_index[:] = build_index_from_hour_counts(hour_counts, exact_hour_counts)
    update_output_attrs(
        output_data, split_name, source_path, config, position, first_date, last_date
    )
    elapsed = time.monotonic() - start_time
    LOGGER.info("Wrote %s rows to %s in %.1f seconds", position, output_path, elapsed)
    return WriteSummary(
        output=str(output_path),
        rows=position,
        first_date=str(first_date) if first_date is not None else None,
        last_date=str(last_date) if last_date is not None else None,
        elapsed_seconds=elapsed,
    )


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, default=json_safe)
        handle.write("\n")


def maybe_copy_config_yaml(source: Path, outputs: Iterable[Path]) -> None:
    source_config = source / "config.yaml"
    if not source_config.exists():
        return
    for output in outputs:
        shutil.copy2(source_config, output / "config.yaml")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    train_output, validation_output, manifest_path = resolve_outputs(args)
    source = zarr.open(args.source, mode="r")
    source_data = source["data"]
    source_dates = source["dates"]
    source_index = source[INDEX_NAME]
    colnames = list(source_data.attrs["colnames"])
    columns = require_columns(colnames, ["lat", "lon", "reportype", "lsm"])
    reportypes = np.array(args.reportypes, dtype=np.int64)
    chunk_rows = args.chunk_rows or int(source_data.chunks[0])
    if args.min_train_validation_distance_km < 0.0:
        raise ValueError("--min-train-validation-distance-km must be non-negative")
    if args.min_latitude > args.max_latitude:
        raise ValueError("--min-latitude must be less than or equal to --max-latitude")

    train_start = to_datetime(args.train_start)
    train_end = to_datetime(args.train_end)
    validation_start = to_datetime(args.validation_start)
    validation_end = to_datetime(args.validation_end)
    train_start_row, train_end_row = coarse_row_bounds_for_window(
        source_index, train_start, train_end
    )
    validation_start_row, validation_end_row = coarse_row_bounds_for_window(
        source_index, validation_start, validation_end
    )

    config = SplitConfig(
        source=str(args.source),
        train_output=str(train_output),
        validation_output=str(validation_output),
        manifest=str(manifest_path),
        train_start=args.train_start,
        train_end=args.train_end,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        reportypes=args.reportypes,
        lsm_threshold=args.lsm_threshold,
        min_latitude=args.min_latitude,
        max_latitude=args.max_latitude,
        min_validation_rows_per_coord=args.min_validation_rows_per_coord,
        train_fraction=args.train_fraction,
        min_train_validation_distance_km=args.min_train_validation_distance_km,
        seed=args.seed,
        chunk_rows=chunk_rows,
    )

    LOGGER.info("Source rows: %s", source_data.shape[0])
    LOGGER.info("Train source row window: [%s, %s)", train_start_row, train_end_row)
    LOGGER.info(
        "Validation/candidate source row window: [%s, %s)",
        validation_start_row,
        validation_end_row,
    )
    candidate_counts = collect_candidate_coord_counts(
        source_data,
        source_dates,
        columns,
        validation_start_row,
        validation_end_row,
        validation_start,
        validation_end,
        chunk_rows,
        reportypes,
        args.lsm_threshold,
        args.min_latitude,
        args.max_latitude,
    )
    train_keys, validation_keys, split_stats = split_candidate_keys(
        candidate_counts,
        args.min_validation_rows_per_coord,
        args.train_fraction,
        args.seed,
    )
    train_keys, validation_keys, distance_stats = apply_train_validation_distance_threshold(
        train_keys,
        validation_keys,
        candidate_counts,
        args.min_train_validation_distance_km,
    )
    split_stats.update(distance_stats)
    split_stats["train_coords"] = len(train_keys)
    split_stats["validation_coords"] = len(validation_keys)
    train_key_array = make_key_array(train_keys)
    validation_key_array = make_key_array(validation_keys)
    LOGGER.info("Split stats: %s", split_stats)

    manifest: dict[str, object] = {
        "config": asdict(config),
        "split_stats": split_stats,
        "train_coordinates": coords_from_keys(train_keys),
        "validation_coordinates": coords_from_keys(validation_keys),
        "outputs": {},
    }

    if args.dry_run:
        if args.count_outputs_in_dry_run:
            train_count = count_selected_rows(
                source_data,
                source_dates,
                columns,
                train_start_row,
                train_end_row,
                train_start,
                train_end,
                chunk_rows,
                reportypes,
                args.lsm_threshold,
                args.min_latitude,
                args.max_latitude,
                train_key_array,
                "count train rows",
            )
            validation_count = count_selected_rows(
                source_data,
                source_dates,
                columns,
                validation_start_row,
                validation_end_row,
                validation_start,
                validation_end,
                chunk_rows,
                reportypes,
                args.lsm_threshold,
                args.min_latitude,
                args.max_latitude,
                validation_key_array,
                "count validation rows",
            )
            manifest["dry_run_counts"] = {
                "train_rows": train_count,
                "validation_rows": validation_count,
            }
            LOGGER.info("Dry-run counts: train=%s validation=%s", train_count, validation_count)
        LOGGER.info("Dry run only; no zarr stores written")
        write_manifest(manifest_path, manifest)
        LOGGER.info("Wrote manifest to %s", manifest_path)
        return

    train_summary = write_split_dataset(
        train_output,
        "train",
        args.source,
        source_data,
        source_dates,
        source_index,
        columns,
        train_start_row,
        train_end_row,
        train_start,
        train_end,
        chunk_rows,
        reportypes,
        args.lsm_threshold,
        args.min_latitude,
        args.max_latitude,
        train_key_array,
        config,
        overwrite=args.overwrite,
    )
    validation_summary = write_split_dataset(
        validation_output,
        "validation",
        args.source,
        source_data,
        source_dates,
        source_index,
        columns,
        validation_start_row,
        validation_end_row,
        validation_start,
        validation_end,
        chunk_rows,
        reportypes,
        args.lsm_threshold,
        args.min_latitude,
        args.max_latitude,
        validation_key_array,
        config,
        overwrite=args.overwrite,
    )
    maybe_copy_config_yaml(args.source, [train_output, validation_output])
    manifest["outputs"] = {
        "train": asdict(train_summary),
        "validation": asdict(validation_summary),
    }
    write_manifest(manifest_path, manifest)
    LOGGER.info("Wrote manifest to %s", manifest_path)


if __name__ == "__main__":
    main()