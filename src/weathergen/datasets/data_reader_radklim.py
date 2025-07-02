# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import json
import logging
from pathlib import Path
from typing import override
from dataclasses import dataclass

import fsspec
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    DType,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)

_logger = logging.getLogger(__name__)

@dataclass
class _LazyInfo:
    ds: xr.Dataset
    nx: int
    ny: int
    points_per_slice: int
    latitudes: NDArray[np.float32]
    longitudes: NDArray[np.float32]
    base_coords: NDArray[np.float32]



class RadklimKerchunkReader(DataReaderTimestep):
    """
    Reader for RADKLIM data accessed via a Kerchunk reference.

    The reader handles temporal subsetting, lazy loading, and normalization.
    """

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
    ) -> None:
        """
        Construct RADKLIM data reader from Kerchunk reference and stats JSON.
        NOTE: This __init__ method is designed to be fork-safe.
        Parameters
        ----------
        tw_handler : TimeWindowHandler
            Time window configuration
        stream_info : dict
            Dictionary with stream metadata including paths to reference and stats
        chunks : dict, optional
            Chunking options to apply to the xarray dataset

        Returns
        -------
        None

        """
        self._empty: bool = False
        # The dataset will be process-local, initialized on first access
        self._info: _LazyInfo | None = None

        # Read Kerchunk reference and normalization stats paths from stream_info
        self.ref_path = Path(stream_info.get("reference", filename))
        self.norm_path = Path(stream_info.get("stats_path"))
        # Load Kerchunk reference
        if not self.ref_path.exists():
            raise FileNotFoundError(f"Kerchunk reference JSON not found: {self.ref_path}")

        # Load normalization stats
        if not self.norm_path.exists():
            raise FileNotFoundError(f"Normalisation JSON not found: {self.norm_path}")

        self.stream_info = stream_info

        # Load source and target channels from config
        self.source_channels = ["RR"]  # Hrad coded due to unkown error in config
        self.target_channels = ["RR"]  # Hrad coded due to unkown error in config
        self.geoinfo_channels = []  # Hrad coded due to unkown error in config

        _logger.info("source_channels: %s", self.source_channels)
        _logger.info("target_channels: %s", self.target_channels)

        # TODO: is it the right index?
        self.source_idx = list(range(len(self.source_channels)))
        self.target_idx = list(range(len(self.target_channels)))
        self.geoinfo_idx = list(range(len(self.geoinfo_channels)))

        _logger.info("Reading time metadata from: %s", self.ref_path)

        times_full: NDArray[np.datetime64]

        # with fsspec.open(self.ref_path, "rt") as f:
        #     kerchunk_ref = json.load(f)

        # fs_meta = fsspec.filesystem("reference", fo=kerchunk_ref)
        # mapper_meta = fs_meta.get_mapper("")

        ds = _open_dataset(self.ref_path)
        times_full = ds["time"].values

        # # TODO: move try/except in _open_dataset
        # # MODIFICATION: Changed consolidated=True to consolidated=False
        # # This is more robust if consolidated metadata is not available.
        # try:
        #     with xr.open_dataset(
        #         mapper_meta, engine="zarr", consolidated=False, chunks={}
        #     ) as ds_meta:
        #         times_full = ds_meta["time"].values
        # except Exception:
        #     _logger.error(
        #         "Failed to open Kerchunk reference even with consolidated=False. "
        #         "The reference file may be corrupt. "
        #     )
        #     raise

        # Ensure times_full is a datetime64 array
        if not np.issubdtype(times_full.dtype, np.datetime64):
            raise TypeError(
                f"Time coordinate from {self.ref_path} was not decoded to datetime64. "
                f"Actual dtype: {times_full.dtype}. Check 'units' and 'calendar' attributes."
            )

        # Check if times_full is empty
        if times_full.size == 0:
            super().__init__(tw_handler, stream_info, None, None, None)
            self.init_empty()
            return

        # Check time step regularity and define period
        deltas_sec = np.diff(times_full.astype("datetime64[s]"))
        unique_deltas = np.unique(deltas_sec)
        if unique_deltas.size != 1:
            raise ValueError("RADKLIM Kerchunk reference has irregular time steps")
        period = unique_deltas[0]

        data_start = times_full[0]
        data_end = times_full[-1]
        super().__init__(tw_handler, stream_info, data_start, data_end, period)

        # If there is no overlap with the time window, exit early
        if tw_handler.t_start >= data_end or tw_handler.t_end <= data_start:
            self.init_empty()
            return

        # Determine time index window for slicing
        self.start_idx = int(np.searchsorted(times_full, tw_handler.t_start, side="left"))
        self.end_idx = int(np.searchsorted(times_full, tw_handler.t_end, side="right"))
        self.num_steps_per_window = int(tw_handler.t_window_len / period)

        stats = json.loads(self.norm_path.read_text())
        self.mean = np.asarray(stats.get("mean", []), dtype=np.float32)
        self.stdev = np.asarray(stats.get("std", []), dtype=np.float32)
        self.mean_geoinfo = np.asarray(stats.get("mean_geoinfo", []), dtype=np.float32)
        self.stdev_geoinfo = np.asarray(stats.get("std_geoinfo", []), dtype=np.float32)

        if len(self.mean) != len(self.source_channels):
            raise ValueError("Normalisation stats length does not match number of variables")

    def _dataset(self) -> _LazyInfo:
        if self._info is not None:
            return self._info
        
        # Lazy loading of the dataset
        _logger.info("Lazy loading RADKLIM Kerchunk dataset inside worker...")
        ds_full = _open_dataset(self.ref_path)
        subset = ds_full.isel(time=slice(self.start_idx, self.end_idx))[self.source_channels]

        if "chunks" in self.stream_info:
            ds = subset.chunk(self.stream_info.get("chunks", {}))
        else:
            ds = subset

        # Extract dimensions and coordinates
        y1d = ds["y"].values.astype(np.float32)
        x1d = ds["x"].values.astype(np.float32)

        lat_var = ds["lat"]
        lon_var = ds["lon"]
        raw_lat = lat_var.isel(time=0).values if "time" in lat_var.dims else lat_var.values
        raw_lon = lon_var.isel(time=0).values if "time" in lon_var.dims else lon_var.values
        latitudes = _clip_lat(raw_lat)
        longitudes = _clip_lon(raw_lon)

        self._info = _LazyInfo(
            ds=ds,
            nx=len(x1d),
            ny = len(y1d),
            points_per_slice=len(x1d)*len(y1d),
            latitudes = latitudes,
            longitudes = longitudes,
            base_coords=np.column_stack(
            (latitudes.reshape(-1), longitudes.reshape(-1))
        ).astype(DType)
        )
        return self._info

    @override
    def init_empty(self) -> None:
        """
        Initialize empty dataset state.
        """
        self._empty = True
        super().init_empty()

    @override
    def length(self) -> int:
        """
        Total number of valid windows available for sampling.

        Returns
        -------
        int
            Number of available time windows
        """
        if self._empty:
            return 0
        # The number of available time steps is determined by start_idx and end_idx
        nt = self.end_idx - self.start_idx
        return max(0, nt - self.num_steps_per_window + 1)

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """
        Fetch data for a specific time index and channels.

        Parameters
        ----------
        idx : TIndex
            Time index to fetch data for, can be a single integer or a slice.
        channels_idx : list[int]
            List of channel indices to fetch data for.
        Returns
        -------
        ReaderData
            ReaderData object containing the requested data, coordinates, and metadata.
        Raises
        ------
        ValueError
            If channels_idx is empty or contains invalid indices.
        IndexError
            If channels_idx contains indices out of bounds for the dataset.
        """
        # Crucial step: ensure dataset is open in the current process
        info = self._dataset()
        ds = info.ds

        # Safety check for empty channels_idx
        if not channels_idx:
            raise ValueError("channels_idx cannot be empty")

        t_idxs_abs, dtr = self._get_dataset_idxs(idx)

        # If the dataset is empty or t_idxs_abs is empty, return empty ReaderData
        if self._empty or t_idxs_abs.size == 0 or ds is None:  # Added self.ds check
            return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

        # Shift from global to local time indices
        t_idxs_rel = t_idxs_abs - self.start_idx
        if np.any(t_idxs_rel < 0) or np.any(t_idxs_rel >= ds.sizes["time"]):
            return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

        # Slice the dataset to get the relevant time window
        start, stop = int(t_idxs_rel[0]), int(t_idxs_rel[-1]) + 1
        ds_win = ds.isel(time=slice(start, stop))

        # Stack the data into a 4D array (time, y, x, var)
        da = ds_win.to_array(dim="var").transpose("time", "y", "x", "var")
        nt, ny, nx, nvars = da.shape

        # Validate channels_idx
        if min(channels_idx) < 0 or max(channels_idx) >= nvars:
            raise IndexError("channels_idx out of bounds")
        if len(set(channels_idx)) != len(channels_idx):
            raise ValueError("channels_idx must be unique")

        # Flatten spatial/temporal and select channels
        flat_data = da.values.astype(np.float32, copy=False).reshape(-1, nvars)[:, channels_idx]

        # Expand coordinates and time axis
        coords = np.tile(info.base_coords, (nt, 1))
        times = np.repeat(ds_win["time"].values.astype("datetime64[ns]"), info.points_per_slice)

        rdata = ReaderData(
            coords=coords,
            geoinfos=np.zeros((coords.shape[0], 0), dtype=DType),
            data=flat_data,
            datetimes=times,
        )
        check_reader_data(rdata, dtr)
        _logger.info(f"fetch data: {rdata}")
        return rdata


def _clip_lat(lats: NDArray[np.floating]) -> NDArray[np.float32]:
    """
    Clip latitudes to the range [-90, 90] and ensure periodicity.
    """
    return (2 * np.clip(lats, -90.0, 90.0) - lats).astype(np.float32)


def _clip_lon(lons: NDArray[np.floating]) -> NDArray[np.float32]:
    """
    Clip longitudes to the range [-180, 180] and ensure periodicity.
    """
    return ((lons + 180.0) % 360.0 - 180.0).astype(np.float32)

def _open_dataset(ref_path: Path) -> xr.Dataset:
    kerchunk_ref = json.loads(ref_path.read_text())
    fs = fsspec.filesystem("reference", fo=kerchunk_ref)
    mapper = fs.get_mapper("")

    # Set up consolidated=False as the default for robustness.
    # can be changed to True if metadata is consolidated.
    ds_full = xr.open_dataset(mapper, engine="zarr", consolidated=False)
    return ds_full


