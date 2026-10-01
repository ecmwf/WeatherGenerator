# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""External target sources for the zarr I/O path.

By default the I/O workers read targets from the ``target`` group of the
WeatherGenerator zarr store.  A :class:`TargetSource` replaces that: the
workers only read predictions, times and row layout, and the source fills
the targets afterwards in the parent process.

:class:`AnemoiTargetSource` reads targets from an anemoi dataset.  Every
valid time needed by a batch of results is read once (consecutive samples
and forecast steps share most of them), contiguous dates are read in one
slice, and the reads run on a thread pool.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import anemoi.datasets
import numpy as np
from numpy.typing import NDArray
from omegaconf import ListConfig

_logger = logging.getLogger(__name__)

_ANEMOI_STREAM_TYPES = ("anemoi", "anemoi_operan", "anemoi_rt")


@dataclass(slots=True)
class TargetRequest:
    """Rows of one ``(sample, fstep, sub-step)`` target that a source must fill.

    Attributes
    ----------
    result_idx, target_idx
        Location of the target in ``results[result_idx][1][target_idx]``.
    n_rows
        Number of rows (prediction points) in the target.
    times
        Valid time of each row (shape ``(n_rows,)``) or a single valid time
        shared by all rows.
    coords
        ``(n, 2)`` lat/lon array the rows index into.  Requests usually share
        one array (the stream's prediction coordinates), so grid matching
        runs once per array, not once per request.
    rows
        Row indices into *coords*, or ``None`` for ``coords[:n_rows]``.
    """

    result_idx: int
    target_idx: int
    n_rows: int
    times: NDArray
    coords: NDArray
    rows: NDArray | None = None


class TargetSource(ABC):
    """Fills targets that the I/O workers did not read from the zarr store."""

    @abstractmethod
    def fill_targets(self, requests: list[TargetRequest], channels: list[str]) -> list[NDArray]:
        """Return one ``(n_rows, len(channels))`` float32 array per request."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_xyz(lat: NDArray, lon: NDArray) -> NDArray:
    lat_r = np.deg2rad(lat.astype(np.float64))
    lon_r = np.deg2rad(lon.astype(np.float64))
    cos_lat = np.cos(lat_r)
    return np.stack([cos_lat * np.cos(lon_r), cos_lat * np.sin(lon_r), np.sin(lat_r)], axis=1)


def _wrap_lon(lon: NDArray) -> NDArray:
    """Normalise longitudes to ``[-180, 180)``."""
    return ((lon.astype(np.float64) + 180.0) % 360.0) - 180.0


class AnemoiGridMatcher:
    """Maps prediction coordinates to anemoi grid-point indices.

    Prediction rows are usually the anemoi grid itself, repeated once per
    valid time, in which case the mapping is ``row % n_grid`` and no search
    is needed.  Otherwise each row is matched to its nearest grid point with
    a KD-tree that is built once and reused.
    """

    def __init__(self, lats: NDArray, lons: NDArray, atol: float = 1e-4) -> None:
        self._lats = np.asarray(lats, dtype=np.float64)
        self._lons = _wrap_lon(np.asarray(lons))
        self._atol = atol
        self._tree = None
        self._tree_lock = threading.Lock()
        self._cache: dict[bytes, NDArray] = {}

    @property
    def n_grid(self) -> int:
        return self._lats.shape[0]

    def indices(self, coords: NDArray) -> NDArray:
        """Return the grid index of each ``(lat, lon)`` row in *coords*."""
        key = hashlib.blake2b(np.ascontiguousarray(coords).view(np.uint8), digest_size=16).digest()
        cached = self._cache.get(key)
        if cached is None:
            cached = self._identity_indices(coords)
            if cached is None:
                cached = self._nearest_indices(coords)
            self._cache[key] = cached
        return cached

    def _identity_indices(self, coords: NDArray) -> NDArray | None:
        n_rows = coords.shape[0]
        if n_rows == 0 or n_rows % self.n_grid != 0:
            return None
        blocks = coords.reshape(-1, self.n_grid, 2)
        same_lat = np.allclose(blocks[..., 0], self._lats, atol=self._atol)
        if not same_lat:
            return None
        lon_diff = np.abs(_wrap_lon(blocks[..., 1]) - self._lons)
        # Treat -180 and 180 as the same longitude.
        if not np.all(np.minimum(lon_diff, 360.0 - lon_diff) <= self._atol):
            return None
        return np.tile(np.arange(self.n_grid, dtype=np.int64), n_rows // self.n_grid)

    def _nearest_indices(self, coords: NDArray) -> NDArray:
        from scipy.spatial import cKDTree

        with self._tree_lock:
            if self._tree is None:
                _logger.info(
                    f"Prediction points differ from the anemoi grid; building a KD-tree "
                    f"over {self.n_grid} grid points for nearest-neighbour matching."
                )
                self._tree = cKDTree(_unit_xyz(self._lats, self._lons))
        _, idx = self._tree.query(_unit_xyz(coords[:, 0], coords[:, 1]), k=1, workers=-1)
        return np.asarray(idx, dtype=np.int64)


class _DateSliceCache:
    """Thread-safe LRU of per-date ``(n_grid, n_vars)`` arrays, bounded in bytes."""

    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max_bytes
        self._bytes = 0
        self._data: OrderedDict[tuple, NDArray] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: tuple) -> NDArray | None:
        with self._lock:
            arr = self._data.get(key)
            if arr is not None:
                self._data.move_to_end(key)
            return arr

    def put(self, key: tuple, arr: NDArray) -> None:
        if arr.nbytes > self._max_bytes:
            return
        with self._lock:
            if key in self._data:
                return
            self._data[key] = arr
            self._bytes += arr.nbytes
            while self._bytes > self._max_bytes:
                _, old = self._data.popitem(last=False)
                self._bytes -= old.nbytes


# ---------------------------------------------------------------------------
# Anemoi target source
# ---------------------------------------------------------------------------


class AnemoiTargetSource(TargetSource):
    """Reads targets from an anemoi dataset.

    Parameters
    ----------
    filenames
        One or more anemoi dataset paths.  Several files are combined by
        ``anemoi.datasets.open_dataset``.
    ensemble_member
        Member to read when the dataset has an ensemble dimension.
    n_threads
        Maximum threads used to read date slices in parallel.
    max_read_bytes
        Upper bound on data decompressed by in-flight reads (all threads).
        Anemoi stores one chunk per date holding every variable, so a date
        costs ``n_variables x n_ens x n_grid x 4`` bytes to read even when
        only a few variables are selected.  This bounds both the number of
        threads and the number of dates per slice.
    cache_bytes
        Size of the LRU of already-read dates, reused across calls (e.g. by
        the fstep-serial LocalStore path).
    """

    def __init__(
        self,
        filenames: list[str],
        ensemble_member: int = 0,
        n_threads: int | None = None,
        max_read_bytes: int = 2 * 1024**3,
        cache_bytes: int = 1024**3,
    ) -> None:
        self.filenames = [str(f) for f in filenames]
        self.ensemble_member = ensemble_member
        self.n_threads = n_threads or min(16, os.cpu_count() or 4)
        self.max_read_bytes = max_read_bytes
        self._cache = _DateSliceCache(cache_bytes)
        self._datasets: dict[tuple[str, ...], tuple] = {}
        self._open_lock = threading.Lock()
        self._grid: AnemoiGridMatcher | None = None
        self._dates: NDArray | None = None
        self._warned_missing: set[str] = set()

    @classmethod
    def from_inference_config(
        cls, inference_cfg: dict, stream: str, overrides: dict | None = None
    ) -> AnemoiTargetSource | None:
        """Build a source for *stream*, or return ``None`` if it is not an anemoi stream."""
        streams = inference_cfg.get("streams", {})
        if isinstance(streams, list | ListConfig):
            stream_info = next((s for s in streams if s.get("name") == stream), {})
        else:
            stream_info = streams.get(stream, {})
        if stream_info.get("type") not in _ANEMOI_STREAM_TYPES or not stream_info.get("filenames"):
            return None
        data_path = Path(inference_cfg.get("data_path_anemoi", ""))
        filenames = [str(data_path / f) for f in stream_info["filenames"]]
        _logger.info(f"Anemoi target source for stream {stream}: {filenames}")
        return cls(filenames, **dict(overrides or {}))

    # ---- dataset access ---------------------------------------------------

    def _open(self, select: list[str] | None = None):
        source = self.filenames[0] if len(self.filenames) == 1 else self.filenames
        if select is None:
            return anemoi.datasets.open_dataset(source)
        return anemoi.datasets.open_dataset(source, select=select)

    def _ensure_metadata(self) -> None:
        if self._grid is not None:
            return
        with self._open_lock:
            if self._grid is not None:
                return
            ds = self._open()
            self._all_variables = list(ds.variables)
            self._dates = np.asarray(ds.dates).astype("datetime64[s]")
            self._grid = AnemoiGridMatcher(np.asarray(ds.latitudes), np.asarray(ds.longitudes))
            _logger.info(
                f"Opened anemoi dataset {self.filenames}: {len(self._dates)} dates "
                f"({self._dates[0]} .. {self._dates[-1]}), {self._grid.n_grid} grid points, "
                f"{len(self._all_variables)} variables."
            )

    def _dataset_for(self, variables: tuple[str, ...]):
        """Return ``(ds, var_idx)`` with only *variables* selected (cached).

        ``var_idx`` gives the position of each requested variable in the
        dataset's variable axis (``select`` need not preserve the order).
        """
        with self._open_lock:
            entry = self._datasets.get(variables)
            if entry is None:
                try:
                    ds = self._open(select=list(variables))
                except Exception as exc:
                    _logger.warning(
                        f"Anemoi 'select' failed ({type(exc).__name__}: {exc}); "
                        f"reading all variables instead."
                    )
                    ds = self._open()
                var_idx = np.array([list(ds.variables).index(v) for v in variables])
                entry = (ds, var_idx)
                self._datasets[variables] = entry
            return entry

    def _date_indices(self, times: NDArray) -> NDArray:
        times_s = np.asarray(times).astype("datetime64[s]")
        idx = np.searchsorted(self._dates, times_s)
        idx_clipped = np.minimum(idx, len(self._dates) - 1)
        missing = self._dates[idx_clipped] != times_s
        if np.any(missing):
            raise ValueError(
                f"Valid time(s) {np.unique(times_s[missing])[:5]} not found in anemoi dataset "
                f"{self.filenames} (available {self._dates[0]} .. {self._dates[-1]})."
            )
        return idx

    # ---- reading -----------------------------------------------------------

    def _read_plan(self, date_idxs: NDArray, bytes_per_date: int) -> tuple[list[NDArray], int]:
        """Split sorted unique date indices into contiguous runs; return ``(runs, n_threads)``.

        At most ``max_read_bytes`` worth of dates are in flight: that many
        dates are shared between up to ``n_threads`` threads, each reading
        runs of ``max_run`` consecutive dates.
        """
        max_dates_in_flight = max(1, self.max_read_bytes // max(1, bytes_per_date))
        n_threads = max(1, min(self.n_threads, max_dates_in_flight))
        max_run = max(1, max_dates_in_flight // n_threads)
        breaks = np.flatnonzero(np.diff(date_idxs) != 1) + 1
        runs = []
        for run in np.split(date_idxs, breaks):
            runs.extend(run[i : i + max_run] for i in range(0, len(run), max_run))
        return runs, min(n_threads, len(runs))

    def _read_run(
        self, ds, var_idx: NDArray, out_cols: NDArray, n_channels: int, run: NDArray
    ) -> dict[int, NDArray]:
        """Read one contiguous run of dates.

        Returns ``{date_idx: (n_grid, n_channels)}`` float32 arrays laid out
        like the requested channels, NaN where a channel is not in the dataset.
        """
        lo, hi = int(run[0]), int(run[-1]) + 1
        block = np.asarray(ds[lo:hi])  # (n_dates, n_vars, n_ens, n_grid)
        member = min(self.ensemble_member, block.shape[2] - 1)
        out = {}
        for i in range(hi - lo):
            arr = np.full((block.shape[3], n_channels), np.nan, dtype=np.float32)
            arr[:, out_cols] = block[i, var_idx, member, :].T
            out[lo + i] = arr
        return out

    # ---- public API -------------------------------------------------------

    def fill_targets(self, requests: list[TargetRequest], channels: list[str]) -> list[NDArray]:
        self._ensure_metadata()
        assert self._grid is not None and self._dates is not None

        present = [ch for ch in channels if ch in self._all_variables]
        for ch in sorted(set(channels) - set(present) - self._warned_missing):
            _logger.warning(
                f"Channel '{ch}' is not in anemoi dataset {self.filenames}; its target is NaN."
            )
            self._warned_missing.add(ch)
        out_cols = np.array([channels.index(ch) for ch in present], dtype=np.int64)
        variables = tuple(present)
        cache_ns = tuple(channels)

        targets = [np.full((r.n_rows, len(channels)), np.nan, dtype=np.float32) for r in requests]
        if not variables or not requests:
            return targets

        # Grid matching once per distinct coordinate array.
        grid_idx_by_coords: dict[int, NDArray] = {}

        def grid_indices(req: TargetRequest) -> NDArray:
            key = id(req.coords)
            if key not in grid_idx_by_coords:
                grid_idx_by_coords[key] = self._grid.indices(req.coords)
            full = grid_idx_by_coords[key]
            return full[: req.n_rows] if req.rows is None else full[req.rows]

        # Group every request's rows by anemoi date: date -> [(request, rows, grid_idx)].
        by_date: dict[int, list[tuple[int, NDArray | slice, NDArray]]] = defaultdict(list)
        for ri, req in enumerate(requests):
            if req.n_rows == 0:
                continue
            grid_idx = grid_indices(req)
            times = np.asarray(req.times)
            if times.ndim == 0:
                by_date[int(self._date_indices(times[None])[0])].append((ri, slice(None), grid_idx))
                continue
            date_idx = self._date_indices(times)
            order = np.argsort(date_idx, kind="stable")
            uniq, starts = np.unique(date_idx[order], return_index=True)
            for d, rows in zip(uniq, np.split(order, starts[1:]), strict=True):
                by_date[int(d)].append((ri, rows, grid_idx[rows]))

        def fill_date(d: int, arr: NDArray) -> None:
            # Rows of different (request, date) pairs never overlap, so threads
            # can write into the shared target arrays without locking.
            for ri, rows, grid_idx in by_date[d]:
                targets[ri][rows] = arr[grid_idx]

        needed = sorted(by_date)
        to_read = []
        for d in needed:
            cached = self._cache.get((cache_ns, d))
            if cached is None:
                to_read.append(d)
            else:
                fill_date(d, cached)

        n_runs = 0
        if to_read:
            ds, var_idx = self._dataset_for(variables)
            n_ens = ds.shape[2] if len(ds.shape) == 4 else 1
            bytes_per_date = len(self._all_variables) * n_ens * self._grid.n_grid * 4
            runs, n_threads = self._read_plan(np.array(to_read, dtype=np.int64), bytes_per_date)
            n_runs = len(runs)

            def read_and_fill(run: NDArray) -> None:
                for d, arr in self._read_run(ds, var_idx, out_cols, len(channels), run).items():
                    fill_date(d, arr)
                    self._cache.put((cache_ns, d), arr)

            if n_threads <= 1:
                for run in runs:
                    read_and_fill(run)
            else:
                with ThreadPoolExecutor(max_workers=n_threads) as pool:
                    list(pool.map(read_and_fill, runs))

        _logger.info(
            f"Anemoi targets: {len(requests)} targets from {len(needed)} unique dates "
            f"({len(to_read)} read in {n_runs} slice(s), {len(needed) - len(to_read)} cached)."
        )
        return targets
