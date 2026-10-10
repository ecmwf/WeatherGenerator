# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""External target sources for the zarr I/O path.

With an :class:`AnemoiTargetSource` the I/O workers skip the zarr ``target`` group and
the source fills the targets afterwards in the parent process.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import anemoi.datasets
import numpy as np
from numpy.typing import NDArray
from omegaconf import ListConfig

_logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TargetRequest:
    """Target ``results[result_idx][1][target_idx]`` to fill.

    Its ``n_rows`` rows are the anemoi grid points in order, repeated once per
    valid time; ``times`` is one time per row or a single shared time.
    """

    result_idx: int
    target_idx: int
    n_rows: int
    times: NDArray


def _fill_blocks(targets: list[NDArray], blocks: list[tuple[int, int]], arr: NDArray) -> None:
    """
    Copy one date's anemoi grid into every target block valid at that date.

    Parameters
    ----------
    targets : list[NDArray]
        Target arrays of shape ``(n_rows, n_channels)``, modified in place.
    blocks : list[tuple[int, int]]
        ``(request index, block index)`` pairs; block ``b`` of a target holds
        rows ``b * n_grid`` to ``(b + 1) * n_grid``.
    arr : NDArray
        ``(n_grid, n_channels)`` values of the date.

    Returns
    -------
    None
    """
    for ri, b in blocks:
        targets[ri].reshape(-1, *arr.shape)[b] = arr


class AnemoiTargetSource:
    """Reads targets from an anemoi dataset.

    Each date needed by a call is read once, consecutive dates in one slice and
    slices in parallel on up to ``n_threads`` threads.  A date read decompresses
    every variable, so ``max_read_bytes`` bounds the data in flight;
    ``cache_bytes`` bounds the LRU of dates kept across calls.
    """

    def __init__(
        self,
        filenames: list[str],
        ensemble_member: int = 0,
        n_threads: int | None = None,
        max_read_bytes: int = 2 * 1024**3,
        cache_bytes: int = 1024**3,
    ) -> None:
        """
        Set up a target source for one or more anemoi datasets.

        Parameters
        ----------
        filenames : list[str]
            Anemoi dataset paths; several files are combined by
            ``anemoi.datasets.open_dataset``.
        ensemble_member : int
            Member to read when the dataset has an ensemble dimension.
        n_threads : int | None
            Maximum number of threads reading dates in parallel (default:
            ``min(16, cpu_count)``).
        max_read_bytes : int
            Upper bound on data decompressed by in-flight reads.
        cache_bytes : int
            Size of the LRU of already-read dates, kept across calls.

        Returns
        -------
        None
        """
        self.filenames = [str(f) for f in filenames]
        self.ensemble_member = ensemble_member
        self.n_threads = n_threads or min(16, os.cpu_count() or 4)
        self.max_read_bytes = max_read_bytes
        self.cache_bytes = cache_bytes
        self._cache: OrderedDict[tuple, NDArray] = OrderedDict()
        self._cache_used = 0
        self._ds = None
        self._warned: set[str] = set()

    @classmethod
    def from_inference_config(
        cls, inference_cfg: dict, stream: str, overrides: dict | None = None
    ) -> AnemoiTargetSource | None:
        """
        Build a target source for a stream from the inference config.

        Parameters
        ----------
        inference_cfg : dict
            Inference run config with ``streams`` and ``data_path_anemoi``.
        stream : str
            Stream name.
        overrides : dict | None
            Keyword arguments passed on to the constructor (e.g. ``n_threads``).

        Returns
        -------
        AnemoiTargetSource | None
            The source, or ``None`` if the stream is not an anemoi stream or
            has no filenames.
        """
        streams = inference_cfg.get("streams", {})
        if isinstance(streams, list | ListConfig):
            info = next((s for s in streams if s.get("name") == stream), {})
        else:
            info = streams.get(stream, {})
        # All anemoi stream types (anemoi, anemoi_operan, ...) read the same datasets.
        if not str(info.get("type", "")).startswith("anemoi") or not info.get("filenames"):
            return None
        data_path = Path(inference_cfg.get("data_path_anemoi", ""))
        filenames = [str(data_path / f) for f in info["filenames"]]
        _logger.info(f"Anemoi target source for stream {stream}: {filenames}")
        return cls(filenames, **dict(overrides or {}))

    def _open(self):
        """
        Open the anemoi dataset.

        Parameters
        ----------
        None

        Returns
        -------
        anemoi.datasets.data.dataset.Dataset
            The dataset over ``self.filenames``.
        """
        return anemoi.datasets.open_dataset(
            self.filenames[0] if len(self.filenames) == 1 else self.filenames
        )

    def _dataset(self):
        """
        Return the anemoi dataset, opening it and caching its metadata on first use.

        Parameters
        ----------
        None

        Returns
        -------
        anemoi.datasets.data.dataset.Dataset
            The dataset; its variables and dates are stored in ``self._variables``
            and ``self._dates``.
        """
        if self._ds is None:
            self._ds = ds = self._open()
            self._variables = list(ds.variables)
            self._dates = np.asarray(ds.dates).astype("datetime64[s]")
        return self._ds

    def _date_indices(self, times: NDArray) -> NDArray:
        """
        Find the dataset date index of each valid time.

        Parameters
        ----------
        times : NDArray
            Valid times (datetime64).

        Returns
        -------
        NDArray
            Index into the dataset's date axis for each time.

        Raises
        ------
        ValueError
            If a valid time is not a date of the dataset.
        """
        times = np.asarray(times).astype("datetime64[s]")
        idx = np.searchsorted(self._dates, times)
        missing = self._dates[np.minimum(idx, len(self._dates) - 1)] != times
        if np.any(missing):
            raise ValueError(
                f"Valid time(s) {np.unique(times[missing])[:5]} not found in anemoi dataset "
                f"{self.filenames} (available {self._dates[0]} .. {self._dates[-1]})."
            )
        return idx

    def fill_targets(self, requests: list[TargetRequest], channels: list[str]) -> list[NDArray]:
        """
        Read the targets of a batch of requests from the anemoi dataset.

        Each date needed by the batch is read once, consecutive dates in one
        slice and slices in parallel; dates read in earlier calls come from the
        cache.  Channels missing from the dataset are left as NaN.

        Parameters
        ----------
        requests : list[TargetRequest]
            Targets to fill; their rows are the anemoi grid in order, repeated
            once per valid time.
        channels : list[str]
            Channel names, in the column order of the targets.

        Returns
        -------
        list[NDArray]
            One ``(n_rows, len(channels))`` float32 array per request.

        Raises
        ------
        ValueError
            If a request's rows do not match the anemoi grid, valid times change
            within a grid block, or a valid time is not in the dataset.
        """
        ds = self._dataset()
        present = [ch for ch in channels if ch in self._variables]
        for ch in sorted(set(channels) - set(present) - self._warned):
            _logger.warning(
                f"Channel '{ch}' is not in anemoi dataset {self.filenames}; target is NaN."
            )
            self._warned.add(ch)
        targets = [np.full((r.n_rows, len(channels)), np.nan, dtype=np.float32) for r in requests]
        if not present:
            return targets
        cols = [channels.index(ch) for ch in present]
        var_idx = [self._variables.index(ch) for ch in present]

        # Every n_grid-row block of a target is the anemoi grid at one valid time:
        # group the blocks by anemoi date, date -> [(request, block)].
        n_grid = ds.shape[-1]
        by_date: dict[int, list] = defaultdict(list)
        for ri, req in enumerate(requests):
            if req.n_rows % n_grid:
                raise ValueError(
                    f"{req.n_rows} prediction rows do not match the {n_grid}-point grid "
                    f"of anemoi dataset {self.filenames}."
                )
            if req.n_rows == 0:
                continue
            times = np.asarray(req.times)
            if times.ndim == 0:
                block_times = np.broadcast_to(times, req.n_rows // n_grid)
            else:
                blocks = times.reshape(-1, n_grid)
                if np.any(blocks != blocks[:, :1]):
                    raise ValueError(
                        f"Valid times change within a {n_grid}-row block: predictions are not "
                        f"stored grid by grid per valid time (e.g. interleaved by time), so "
                        f"they cannot be matched to anemoi dataset {self.filenames}."
                    )
                block_times = blocks[:, 0]
            for b, d in enumerate(self._date_indices(block_times)):
                by_date[int(d)].append((ri, b))

        to_read = []
        for d in sorted(by_date):
            if (key := (tuple(channels), d)) in self._cache:
                self._cache.move_to_end(key)
                _fill_blocks(targets, by_date[d], self._cache[key])
            else:
                to_read.append(d)
        if not to_read:
            return targets

        # Split into runs of consecutive dates, keeping <= max_read_bytes in flight.
        bytes_per_date = len(self._variables) * ds.shape[2] * n_grid * 4
        in_flight = max(1, self.max_read_bytes // bytes_per_date)
        n_threads = max(1, min(self.n_threads, in_flight))
        run_len = max(1, in_flight // n_threads)
        to_read = np.array(to_read)
        runs = [
            run[i : i + run_len]
            for run in np.split(to_read, np.flatnonzero(np.diff(to_read) != 1) + 1)
            for i in range(0, len(run), run_len)
        ]
        read_run = partial(self._read_run, ds, cols, var_idx, len(channels))
        with ThreadPoolExecutor(max_workers=min(n_threads, len(runs))) as pool:
            for out in pool.map(read_run, runs):
                for d, arr in out.items():
                    _fill_blocks(targets, by_date[d], arr)
                    self._cache_put((tuple(channels), d), arr)

        _logger.info(
            f"Anemoi targets: {len(requests)} targets from {len(by_date)} dates "
            f"({len(to_read)} read in {len(runs)} slice(s))."
        )
        return targets

    def _read_run(
        self, ds, cols: list[int], var_idx: list[int], n_channels: int, run: NDArray
    ) -> dict[int, NDArray]:
        """
        Read a run of consecutive dates in one slice.

        Parameters
        ----------
        ds : anemoi.datasets.data.dataset.Dataset
            The anemoi dataset.
        cols : list[int]
            Target columns of the channels present in the dataset.
        var_idx : list[int]
            Dataset variable index of each of those channels.
        n_channels : int
            Number of requested channels (target columns).
        run : NDArray
            Consecutive date indices to read.

        Returns
        -------
        dict[int, NDArray]
            ``{date index: (n_grid, n_channels)}`` float32 arrays, NaN for
            channels missing from the dataset.
        """
        lo = int(run[0])
        block = np.asarray(ds[lo : int(run[-1]) + 1])  # (n_dates, n_vars, n_ens, n_grid)
        member = min(self.ensemble_member, block.shape[2] - 1)
        out = {}
        for i in range(len(block)):
            arr = np.full((block.shape[3], n_channels), np.nan, dtype=np.float32)
            arr[:, cols] = block[i, var_idx, member].T
            out[lo + i] = arr
        return out

    def _cache_put(self, key: tuple, arr: NDArray) -> None:
        """
        Add a date array to the LRU cache, evicting the oldest entries if full.

        Parameters
        ----------
        key : tuple
            ``(channels, date index)`` cache key.
        arr : NDArray
            ``(n_grid, n_channels)`` values of the date; not cached if larger
            than the whole cache.

        Returns
        -------
        None
        """
        if arr.nbytes > self.cache_bytes:
            return
        self._cache[key] = arr
        self._cache_used += arr.nbytes
        while self._cache_used > self.cache_bytes:
            self._cache_used -= self._cache.popitem(last=False)[1].nbytes
