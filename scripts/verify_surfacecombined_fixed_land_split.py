#!/usr/bin/env python3
"""Verify fixed-land SurfaceCombined spatial split outputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt
import zarr

INDEX_NAME = "idx_197001010000_1"
BASE_DATETIME = np.datetime64("1970-01-01T00:00:00", "ns")
HOUR = np.timedelta64(1, "h")
COORD_KEY_DTYPE = np.dtype("V8")
EARTH_RADIUS_KM = 6371.0088


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify generated fixed-land SurfaceCombined train/validation splits."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-start", default="1979-01-01T00:00:00")
    parser.add_argument("--train-end", default="2023-01-01T00:00:00")
    parser.add_argument("--validation-start", default="2023-01-01T00:00:00")
    parser.add_argument("--validation-end", default="2024-01-01T00:00:00")
    parser.add_argument("--min-latitude", type=float, required=True)
    parser.add_argument("--max-latitude", type=float, required=True)
    parser.add_argument("--lsm-threshold", type=float, required=True)
    parser.add_argument("--min-train-validation-distance-km", type=float, required=True)
    parser.add_argument("--reportypes", type=int, nargs="+", required=True)
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=2_000_000,
        help="Rows per chunk for streaming verification scans.",
    )
    parser.add_argument(
        "--train-sample-windows",
        type=int,
        default=48,
        help="Random train windows to sample in addition to start/middle/end windows.",
    )
    parser.add_argument(
        "--train-sample-window-size",
        type=int,
        default=200_000,
        help="Rows per sampled train window.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def coord_keys_from_lat_lon(lat_lon: npt.NDArray[np.float32]) -> npt.NDArray[np.void]:
    coords = np.ascontiguousarray(lat_lon.astype(np.float32, copy=False))
    return coords.view(COORD_KEY_DTYPE).reshape(-1)


def coords_array_from_list(coords: list[list[float]]) -> npt.NDArray[np.float32]:
    if not coords:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(coords, dtype=np.float32)


def nearest_distances_km(
    query_coords: npt.NDArray[np.float32],
    reference_coords: npt.NDArray[np.float32],
    batch_size: int = 256,
) -> npt.NDArray[np.float64]:
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


def build_index_from_hour_counts(
    hour_counts: npt.NDArray[np.int64], exact_hour_counts: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    cumulative_before_hour = np.empty_like(hour_counts, dtype=np.int64)
    cumulative_before_hour[0] = 0
    if len(hour_counts) > 1:
        cumulative_before_hour[1:] = np.cumsum(hour_counts[:-1], dtype=np.int64)
    return cumulative_before_hour + exact_hour_counts


def key_set(keys: npt.NDArray[np.void]) -> set[bytes]:
    return {bytes(key) for key in keys}


def scan_dates_and_index(
    group: zarr.Group,
    date_start: np.datetime64,
    date_end: np.datetime64,
    chunk_rows: int,
) -> dict[str, object]:
    data = group["data"]
    dates = group["dates"]
    index = group[INDEX_NAME]
    hour_counts = np.zeros(index.shape[0], dtype=np.int64)
    exact_hour_counts = np.zeros(index.shape[0], dtype=np.int64)
    outside_window = 0
    nat_count = 0
    first_date: np.datetime64 | None = None
    last_date: np.datetime64 | None = None

    for start in range(0, dates.shape[0], chunk_rows):
        stop = min(start + chunk_rows, dates.shape[0])
        date_chunk = dates[start:stop][:, 0]
        if len(date_chunk) == 0:
            continue
        if first_date is None:
            first_date = date_chunk[0]
        last_date = date_chunk[-1]
        nat_mask = np.isnat(date_chunk)
        nat_count += int(nat_mask.sum())
        outside_window += int(((date_chunk < date_start) | (date_chunk >= date_end)).sum())
        selected_hours = ((date_chunk - BASE_DATETIME) / HOUR).astype(np.int64)
        valid_hours = (selected_hours >= 0) & (selected_hours < len(hour_counts)) & ~nat_mask
        hour_counts += np.bincount(selected_hours[valid_hours], minlength=len(hour_counts)).astype(
            np.int64
        )
        exact_hour_mask = date_chunk == (BASE_DATETIME + selected_hours.astype("timedelta64[h]"))
        exact_selected_hours = selected_hours[valid_hours & exact_hour_mask]
        exact_hour_counts += np.bincount(
            exact_selected_hours, minlength=len(exact_hour_counts)
        ).astype(np.int64)

    computed_index = build_index_from_hour_counts(hour_counts, exact_hour_counts)
    stored_index = index[:]
    require(np.array_equal(computed_index, stored_index), "hourly index does not match dates")
    require(int(stored_index[-1]) == data.shape[0], "final index entry does not equal row count")
    require(np.all(stored_index[1:] >= stored_index[:-1]), "hourly index is not monotonic")
    require(outside_window == 0, f"found {outside_window} dates outside expected window")
    require(nat_count == 0, f"found {nat_count} NaT dates")
    return {
        "rows": int(data.shape[0]),
        "first_date": str(first_date),
        "last_date": str(last_date),
        "outside_window": outside_window,
        "nat_count": nat_count,
        "index_last": int(stored_index[-1]),
    }


def scan_validation_rows(
    group: zarr.Group,
    columns: dict[str, int],
    validation_keys: npt.NDArray[np.void],
    train_keys: npt.NDArray[np.void],
    min_latitude: float,
    max_latitude: float,
    lsm_threshold: float,
    reportypes: npt.NDArray[np.int64],
    chunk_rows: int,
) -> dict[str, object]:
    data = group["data"]
    dates = group["dates"]
    unique_keys: set[bytes] = set()
    min_lat = math.inf
    max_lat = -math.inf
    min_lsm = math.inf
    bad_reportype = 0
    bad_lsm = 0
    bad_lat = 0
    bad_finite = 0
    bad_allowed_coord = 0
    bad_forbidden_coord = 0

    for start in range(0, data.shape[0], chunk_rows):
        stop = min(start + chunk_rows, data.shape[0])
        chunk = data[start:stop]
        _ = dates[start:stop]
        lat = chunk[:, columns["lat"]]
        lon = chunk[:, columns["lon"]]
        lsm = chunk[:, columns["lsm"]]
        reportype = chunk[:, columns["reportype"]].astype(np.int64)
        min_lat = min(min_lat, float(np.nanmin(lat)))
        max_lat = max(max_lat, float(np.nanmax(lat)))
        min_lsm = min(min_lsm, float(np.nanmin(lsm)))
        finite_mask = np.isfinite(lat) & np.isfinite(lon)
        bad_finite += int((~finite_mask).sum())
        bad_lat += int(((lat < min_latitude) | (lat > max_latitude)).sum())
        bad_lsm += int((lsm < lsm_threshold).sum())
        bad_reportype += int((~np.isin(reportype, reportypes)).sum())
        keys = coord_keys_from_lat_lon(chunk[:, [columns["lat"], columns["lon"]]])
        unique_keys.update(bytes(key) for key in np.unique(keys))
        bad_allowed_coord += int((~np.isin(keys, validation_keys)).sum())
        bad_forbidden_coord += int(np.isin(keys, train_keys).sum())

    require(bad_finite == 0, "validation: found non-finite lat/lon rows")
    require(bad_lat == 0, "validation: found rows outside latitude bounds")
    require(bad_lsm == 0, "validation: found rows below lsm threshold")
    require(bad_reportype == 0, "validation: found rows with unexpected reportype")
    require(bad_allowed_coord == 0, "validation: found rows outside validation coordinate set")
    require(bad_forbidden_coord == 0, "validation: found rows in train coordinate set")
    require(
        unique_keys == key_set(validation_keys),
        "validation: unique coords do not match manifest",
    )
    return {
        "min_lat": min_lat,
        "max_lat": max_lat,
        "min_lsm": min_lsm,
        "unique_coords": len(unique_keys),
        "bad_allowed_coord": bad_allowed_coord,
        "bad_forbidden_coord": bad_forbidden_coord,
    }


def sample_train_rows(
    group: zarr.Group,
    columns: dict[str, int],
    train_keys: npt.NDArray[np.void],
    validation_keys: npt.NDArray[np.void],
    date_start: np.datetime64,
    date_end: np.datetime64,
    min_latitude: float,
    max_latitude: float,
    lsm_threshold: float,
    reportypes: npt.NDArray[np.int64],
    sample_windows: int,
    window_size: int,
    seed: int,
) -> dict[str, object]:
    data = group["data"]
    dates = group["dates"]
    rng = np.random.default_rng(seed)
    max_start = max(data.shape[0] - window_size, 0)
    starts = {0, max_start, data.shape[0] // 2}
    if sample_windows > 0:
        starts.update(int(x) for x in rng.integers(0, max_start + 1, size=sample_windows))

    rows_checked = 0
    min_lat = math.inf
    max_lat = -math.inf
    min_lsm = math.inf
    unique_keys: set[bytes] = set()

    for start in sorted(starts):
        stop = min(start + window_size, data.shape[0])
        chunk = data[start:stop]
        date_chunk = dates[start:stop][:, 0]
        rows_checked += stop - start
        require(np.isnat(date_chunk).sum() == 0, "train sample: NaT dates found")
        require(
            np.all((date_chunk >= date_start) & (date_chunk < date_end)),
            "train sample: date outside train window",
        )
        lat = chunk[:, columns["lat"]]
        lon = chunk[:, columns["lon"]]
        lsm = chunk[:, columns["lsm"]]
        reportype = chunk[:, columns["reportype"]].astype(np.int64)
        min_lat = min(min_lat, float(np.nanmin(lat)))
        max_lat = max(max_lat, float(np.nanmax(lat)))
        min_lsm = min(min_lsm, float(np.nanmin(lsm)))
        require(np.all(np.isfinite(lat) & np.isfinite(lon)), "train sample: non-finite lat/lon")
        require(
            np.all((lat >= min_latitude) & (lat <= max_latitude)),
            "train sample: latitude outside bounds",
        )
        require(np.all(lsm >= lsm_threshold), "train sample: lsm below threshold")
        require(np.all(np.isin(reportype, reportypes)), "train sample: unexpected reportype")
        keys = coord_keys_from_lat_lon(chunk[:, [columns["lat"], columns["lon"]]])
        require(np.all(np.isin(keys, train_keys)), "train sample: coord outside train manifest set")
        require(
            not np.any(np.isin(keys, validation_keys)),
            "train sample: held-out validation coord present",
        )
        unique_keys.update(bytes(key) for key in np.unique(keys))

    return {
        "rows_checked": int(rows_checked),
        "windows_checked": len(starts),
        "unique_coords_seen": len(unique_keys),
        "min_lat": min_lat,
        "max_lat": max_lat,
        "min_lsm": min_lsm,
    }


def check_schema(
    source_group: zarr.Group,
    output_group: zarr.Group,
    output_name: str,
    source_path: Path,
    output_path: Path,
) -> None:
    source_data = source_group["data"]
    source_dates = source_group["dates"]
    source_index = source_group[INDEX_NAME]
    data = output_group["data"]
    dates = output_group["dates"]
    index = output_group[INDEX_NAME]
    require(
        set(output_group.array_keys()) == {"data", "dates", INDEX_NAME},
        f"{output_name}: unexpected array keys",
    )
    require(data.shape[1] == source_data.shape[1], f"{output_name}: data column count mismatch")
    require(data.dtype == source_data.dtype, f"{output_name}: data dtype mismatch")
    require(dates.dtype == source_dates.dtype, f"{output_name}: dates dtype mismatch")
    require(index.shape == source_index.shape, f"{output_name}: index shape mismatch")
    require(index.dtype == source_index.dtype, f"{output_name}: index dtype mismatch")
    require(
        list(data.attrs["colnames"]) == list(source_data.attrs["colnames"]),
        f"{output_name}: colnames mismatch",
    )
    require(
        np.array_equal(np.asarray(data.attrs["means"]), np.asarray(source_data.attrs["means"])),
        f"{output_name}: means mismatch",
    )
    require(
        np.array_equal(np.asarray(data.attrs["vars"]), np.asarray(source_data.attrs["vars"])),
        f"{output_name}: vars mismatch",
    )
    require(data.attrs["split_name"] == output_name, f"{output_name}: split_name attr mismatch")
    require(
        data.attrs["split_source"] == str(source_path),
        f"{output_name}: split_source attr mismatch",
    )
    require(data.attrs["split_rows"] == data.shape[0], f"{output_name}: split_rows attr mismatch")
    require((output_path / "config.yaml").exists(), f"{output_name}: config.yaml was not copied")


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=UserWarning, module="zarr")

    with args.manifest.open() as handle:
        manifest = json.load(handle)

    source = zarr.open_group(args.source, mode="r")
    train = zarr.open_group(args.train, mode="r")
    validation = zarr.open_group(args.validation, mode="r")
    source_data = source["data"]
    colnames = list(source_data.attrs["colnames"])
    columns = {name: colnames.index(name) for name in ["lat", "lon", "reportype", "lsm"]}
    reportypes = np.asarray(args.reportypes, dtype=np.int64)

    check_schema(source, train, "train", args.source, args.train)
    check_schema(source, validation, "validation", args.source, args.validation)

    require(
        manifest["config"]["min_latitude"] == args.min_latitude,
        "manifest min_latitude mismatch",
    )
    require(
        manifest["config"]["max_latitude"] == args.max_latitude,
        "manifest max_latitude mismatch",
    )
    require(
        manifest["config"]["min_train_validation_distance_km"]
        == args.min_train_validation_distance_km,
        "manifest min distance mismatch",
    )
    require(
        manifest["outputs"]["train"]["rows"] == train["data"].shape[0],
        "manifest train rows mismatch",
    )
    require(
        manifest["outputs"]["validation"]["rows"] == validation["data"].shape[0],
        "manifest validation rows mismatch",
    )
    require(
        manifest["split_stats"]["validation_2023_rows_after_distance_prune"]
        == validation["data"].shape[0],
        "validation rows stat mismatch",
    )

    train_coords = coords_array_from_list(manifest["train_coordinates"])
    validation_coords = coords_array_from_list(manifest["validation_coordinates"])
    train_keys = coord_keys_from_lat_lon(train_coords)
    validation_keys = coord_keys_from_lat_lon(validation_coords)
    train_key_bytes = key_set(train_keys)
    validation_key_bytes = key_set(validation_keys)

    require(
        len(train_coords) == manifest["split_stats"]["train_coords"],
        "train coord count mismatch",
    )
    require(
        len(validation_coords) == manifest["split_stats"]["validation_coords"],
        "validation coord count mismatch",
    )
    require(
        len(train_key_bytes & validation_key_bytes) == 0,
        "train/validation coordinate overlap",
    )
    require(
        float(train_coords[:, 0].min()) >= args.min_latitude
        and float(train_coords[:, 0].max()) <= args.max_latitude,
        "train manifest coords outside lat bounds",
    )
    require(
        float(validation_coords[:, 0].min()) >= args.min_latitude
        and float(validation_coords[:, 0].max()) <= args.max_latitude,
        "validation manifest coords outside lat bounds",
    )

    nearest = nearest_distances_km(train_coords, validation_coords)
    min_distance = float(nearest.min()) if len(nearest) else math.inf
    require(
        min_distance >= args.min_train_validation_distance_km,
        f"min train-validation distance below threshold: {min_distance}",
    )
    require(
        abs(min_distance - manifest["split_stats"]["min_train_validation_distance_km_after_prune"])
        < 1e-9,
        "manifest min distance stat mismatch",
    )

    train_dates_and_index = scan_dates_and_index(
        train,
        np.datetime64(args.train_start, "ns"),
        np.datetime64(args.train_end, "ns"),
        args.chunk_rows,
    )
    validation_dates_and_index = scan_dates_and_index(
        validation,
        np.datetime64(args.validation_start, "ns"),
        np.datetime64(args.validation_end, "ns"),
        args.chunk_rows,
    )
    validation_full_scan = scan_validation_rows(
        validation,
        columns,
        validation_keys,
        train_keys,
        args.min_latitude,
        args.max_latitude,
        args.lsm_threshold,
        reportypes,
        args.chunk_rows,
    )
    train_sample_scan = sample_train_rows(
        train,
        columns,
        train_keys,
        validation_keys,
        np.datetime64(args.train_start, "ns"),
        np.datetime64(args.train_end, "ns"),
        args.min_latitude,
        args.max_latitude,
        args.lsm_threshold,
        reportypes,
        args.train_sample_windows,
        args.train_sample_window_size,
        args.seed,
    )

    summary = {
        "manifest_train_coords": int(len(train_coords)),
        "manifest_train_lat_range": [
            float(train_coords[:, 0].min()),
            float(train_coords[:, 0].max()),
        ],
        "manifest_validation_coords": int(len(validation_coords)),
        "manifest_validation_lat_range": [
            float(validation_coords[:, 0].min()),
            float(validation_coords[:, 0].max()),
        ],
        "min_train_validation_distance_km": min_distance,
        "train_dates_and_index": train_dates_and_index,
        "train_sample_scan": train_sample_scan,
        "validation_dates_and_index": validation_dates_and_index,
        "validation_full_scan": validation_full_scan,
    }
    sys.stdout.write("VERIFICATION PASSED\n")
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()