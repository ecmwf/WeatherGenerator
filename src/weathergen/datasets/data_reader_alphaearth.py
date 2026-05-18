# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import zarr
from numpy.typing import NDArray

_logger = logging.getLogger(__name__)


class DataReaderAlphaEarthGeoinfo:
    """Read AlphaEarth embeddings and expose them as per-observation geoinfo features."""

    def __init__(self, filename: Path, config: dict) -> None:
        self.filename = filename
        self.config = config
        self.z = zarr.open(filename, mode="r")
        self.data = self.z["data"]
        self.dates = np.asarray(self.z["dates"][:]).astype("datetime64[ns]")

        self.patch_mode = str(config.get("patch_mode", "center")).lower()
        self.max_distance_deg = float(config.get("max_distance_deg", 0.05))
        self.lookup_cell_size_deg = float(
            config.get("lookup_cell_size_deg", self.max_distance_deg)
        )
        self.stats_sample_size = int(config.get("stats_sample_size", 2048))
        self.missing_value = config.get("missing_value", "mean")
        self.prefix = str(config.get("prefix", "alphaearth"))

        if self.max_distance_deg <= 0.0:
            raise ValueError("AlphaEarth geoinfo max_distance_deg must be greater than zero")
        if self.lookup_cell_size_deg <= 0.0:
            raise ValueError("AlphaEarth geoinfo lookup_cell_size_deg must be greater than zero")

        if len(self.data.shape) != 5:
            raise ValueError(
                "AlphaEarth geoinfo data must have shape "
                "(station, date, channel, y, x), got "
                f"{self.data.shape}"
            )

        metadata = self.z["metadata"][:]
        self.station_coords = np.column_stack(
            [metadata["lat"], self._normalize_longitudes(metadata["lon"])]
        ).astype(np.float32)

        self.num_channels = int(self.data.shape[2])
        self.patch_y = int(self.data.shape[3])
        self.patch_x = int(self.data.shape[4])
        center = config.get("patch_center", [self.patch_y // 2, self.patch_x // 2])
        self.center_y = int(center[0])
        self.center_x = int(center[1])

        self.channel_names = self._build_channel_names()
        self.feature_size = len(self.channel_names)
        self._station_bins = self._build_station_bins()
        self._coord_cache: dict[tuple[float, float], int] = {}
        self.mean, self.stdev = self._compute_stats()

        _logger.info(
            "Loaded AlphaEarth geoinfos from %s with %s features, patch_mode=%s, "
            "max_distance_deg=%s",
            filename,
            self.feature_size,
            self.patch_mode,
            self.max_distance_deg,
        )

    @staticmethod
    def _normalize_longitudes(longitudes: NDArray[np.floating]) -> NDArray[np.float32]:
        return (((longitudes + 180.0) % 360.0) - 180.0).astype(np.float32)

    def _build_channel_names(self) -> list[str]:
        if self.patch_mode in ("center", "mean"):
            return [f"{self.prefix}_{channel_idx:03d}" for channel_idx in range(self.num_channels)]
        if self.patch_mode == "flatten":
            return [
                f"{self.prefix}_{channel_idx:03d}_y{y_idx}_x{x_idx}"
                for channel_idx in range(self.num_channels)
                for y_idx in range(self.patch_y)
                for x_idx in range(self.patch_x)
            ]
        raise ValueError(
            f"Unknown AlphaEarth geoinfo patch_mode {self.patch_mode}. "
            "Expected one of: center, mean, flatten."
        )

    def _lat_bin(self, lat: float) -> int:
        return int(np.floor((lat + 90.0) / self.lookup_cell_size_deg))

    def _lon_bin(self, lon: float) -> int:
        return int(np.floor((lon + 180.0) / self.lookup_cell_size_deg)) % self._num_lon_bins

    @property
    def _num_lon_bins(self) -> int:
        return int(np.ceil(360.0 / self.lookup_cell_size_deg))

    @property
    def _num_lat_bins(self) -> int:
        return int(np.ceil(180.0 / self.lookup_cell_size_deg)) + 1

    def _build_station_bins(self) -> dict[tuple[int, int], list[int]]:
        station_bins: dict[tuple[int, int], list[int]] = defaultdict(list)
        for station_idx, (lat, lon) in enumerate(self.station_coords):
            if np.isnan(lat) or np.isnan(lon):
                continue
            station_bins[(self._lat_bin(float(lat)), self._lon_bin(float(lon)))].append(
                station_idx
            )
        return dict(station_bins)

    def _compute_stats(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if not bool(self.config.get("normalize", True)) or self.stats_sample_size <= 0:
            return (
                np.zeros(self.feature_size, dtype=np.float32),
                np.ones(self.feature_size, dtype=np.float32),
            )

        num_stations = int(self.data.shape[0])
        sample_size = min(self.stats_sample_size, num_stations)
        station_indices = np.linspace(0, num_stations - 1, sample_size, dtype=np.int64)
        sample_features = [
            self._read_features(station_indices, date_idx)
            for date_idx in range(int(self.data.shape[1]))
        ]
        sample = np.concatenate(sample_features, axis=0)
        mean = np.mean(sample, axis=0, dtype=np.float64).astype(np.float32)
        stdev = np.std(sample, axis=0, dtype=np.float64).astype(np.float32)
        stdev[np.isclose(stdev, 0.0)] = 1.0
        return mean, stdev

    def _read_features(
        self, station_indices: NDArray[np.int64], date_idx: int
    ) -> NDArray[np.float32]:
        if self.patch_mode == "center":
            features = self.data.oindex[
                station_indices, date_idx, slice(None), self.center_y, self.center_x
            ]
        elif self.patch_mode == "mean":
            patch = self.data.oindex[station_indices, date_idx, :, :, :].astype(np.float32)
            features = patch.mean(axis=(-1, -2))
        elif self.patch_mode == "flatten":
            patch = self.data.oindex[station_indices, date_idx, :, :, :]
            features = patch.reshape((len(station_indices), self.feature_size))
        else:
            raise ValueError(f"Unknown AlphaEarth geoinfo patch_mode {self.patch_mode}")

        return np.asarray(features, dtype=np.float32)

    def _date_indices(self, datetimes: NDArray[np.datetime64]) -> NDArray[np.int64]:
        datetimes = np.asarray(datetimes).astype("datetime64[ns]")
        right = np.searchsorted(self.dates, datetimes, side="left")
        right = np.clip(right, 0, len(self.dates) - 1)
        left = np.clip(right - 1, 0, len(self.dates) - 1)

        left_delta = np.abs(datetimes - self.dates[left])
        right_delta = np.abs(datetimes - self.dates[right])
        return np.where(right_delta < left_delta, right, left).astype(np.int64)

    def _station_indices(self, coords: NDArray[np.float32]) -> NDArray[np.int64]:
        station_indices = np.full(coords.shape[0], -1, dtype=np.int64)
        search_radius = int(np.ceil(self.max_distance_deg / self.lookup_cell_size_deg))

        coords = np.asarray(coords, dtype=np.float32)
        lats = coords[:, 0]
        lons = self._normalize_longitudes(coords[:, 1])

        for coord_idx, (lat, lon) in enumerate(zip(lats, lons, strict=True)):
            if np.isnan(lat) or np.isnan(lon):
                continue

            cache_key = (float(lat), float(lon))
            cached_station_idx = self._coord_cache.get(cache_key)
            if cached_station_idx is not None:
                station_indices[coord_idx] = cached_station_idx
                continue

            lat_bin = self._lat_bin(float(lat))
            lon_bin = self._lon_bin(float(lon))
            candidate_indices = []
            for lat_offset in range(-search_radius, search_radius + 1):
                candidate_lat_bin = lat_bin + lat_offset
                if candidate_lat_bin < 0 or candidate_lat_bin >= self._num_lat_bins:
                    continue
                for lon_offset in range(-search_radius, search_radius + 1):
                    candidate_lon_bin = (lon_bin + lon_offset) % self._num_lon_bins
                    candidate_indices.extend(
                        self._station_bins.get((candidate_lat_bin, candidate_lon_bin), [])
                    )

            if not candidate_indices:
                self._coord_cache[cache_key] = -1
                continue

            candidates = self.station_coords[np.asarray(candidate_indices, dtype=np.int64)]
            dlat = candidates[:, 0] - lat
            dlon = np.abs(candidates[:, 1] - lon)
            dlon = np.minimum(dlon, 360.0 - dlon)
            distances = np.sqrt(dlat * dlat + dlon * dlon)
            nearest = int(np.argmin(distances))
            if distances[nearest] <= self.max_distance_deg:
                station_indices[coord_idx] = candidate_indices[nearest]
            self._coord_cache[cache_key] = int(station_indices[coord_idx])

        return station_indices

    def _missing_features(self, num_rows: int) -> NDArray[np.float32]:
        if isinstance(self.missing_value, str):
            if self.missing_value == "mean":
                return np.broadcast_to(self.mean, (num_rows, self.feature_size)).copy()
            if self.missing_value in ("zero", "zeros"):
                return np.zeros((num_rows, self.feature_size), dtype=np.float32)
            raise ValueError(
                f"Unknown AlphaEarth geoinfo missing_value {self.missing_value}. "
                "Expected 'mean', 'zero', or a numeric fill value."
            )

        return np.full((num_rows, self.feature_size), float(self.missing_value), dtype=np.float32)

    def get(
        self, coords: NDArray[np.float32], datetimes: NDArray[np.datetime64]
    ) -> NDArray[np.float32]:
        features = self._missing_features(coords.shape[0])
        if coords.shape[0] == 0:
            return features

        station_indices = self._station_indices(coords)
        valid_station_mask = station_indices >= 0
        if not valid_station_mask.any():
            return features

        date_indices = self._date_indices(datetimes)
        for date_idx in np.unique(date_indices[valid_station_mask]):
            row_mask = valid_station_mask & (date_indices == date_idx)
            features[row_mask] = self._read_features(station_indices[row_mask], int(date_idx))

        return features
