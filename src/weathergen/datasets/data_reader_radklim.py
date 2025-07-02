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
        self.ds: xr.Dataset | None = None

        # Read Kerchunk reference and normalization stats paths from stream_info
        self.ref_path = Path(stream_info.get("reference", filename))
        self.norm_path = Path(stream_info.get("stats_path"))
        self.stream_info = stream_info

        # Load source and target channels from config
        self.source_channels = ["RR"]  # Hrad coded due to unkown error in config
        self.target_channels = ["RR"]  # Hrad coded due to unkown error in config
        self.geoinfo_channels = []  # Hrad coded due to unkown error in config

        _logger.info("source_channels: %s", self.source_channels)
        _logger.info("target_channels: %s", self.target_channels)

        self.source_idx = list(range(len(self.source_channels)))
        self.target_idx = list(range(len(self.target_channels)))
        self.geoinfo_idx = list(range(len(self.geoinfo_channels)))

        # Load Kerchunk reference
        if not self.ref_path.exists():
            raise FileNotFoundError(f"Kerchunk reference JSON not found: {self.ref_path}")

        # Load normalization stats
        if not self.norm_path.exists():
            raise FileNotFoundError(f"Normalisation JSON not found: {self.norm_path}")

        _logger.info("Reading time metadata from: %s", self.ref_path)

        times_full: NDArray[np.datetime64]

        with fsspec.open(self.ref_path, "rt") as f:
            kerchunk_ref = json.load(f)

        fs_meta = fsspec.filesystem("reference", fo=kerchunk_ref)
        mapper_meta = fs_meta.get_mapper("")

        # MODIFICATION: Changed consolidated=True to consolidated=False
        # This is more robust if consolidated metadata is not available.
        try:
            with xr.open_dataset(
                mapper_meta, engine="zarr", consolidated=False, chunks={}
            ) as ds_meta:
                times_full = ds_meta["time"].values
                _logger.info(f"🧭 RADKLIM timeline: start={times_full[0]}, end={times_full[-1]}, len={len(times_full)}")
        except Exception:
            _logger.error(
                "Failed to open Kerchunk reference even with consolidated=False. "
                "The reference file may be corrupt. "
            )
            raise

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
        _logger.info(f"🕒 RADKLIM period detected: {period} seconds between steps")


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
        _logger.info(f"📊 Normalization loaded: mean={self.mean.tolist()}, std={self.stdev.tolist()}")
        self.mean_geoinfo = np.asarray(stats.get("mean_geoinfo", []), dtype=np.float32)
        self.stdev_geoinfo = np.asarray(stats.get("std_geoinfo", []), dtype=np.float32)

        if len(self.mean) != len(self.source_channels):
            raise ValueError("Normalisation stats length does not match number of variables")

    def _lazy_open(self):
        """
        Opens the Kerchunk reference and initializes the xarray.Dataset.
        This method is called on the first data access within a DataLoader worker.
        """
        if self.ds is not None:
            return

        _logger.info("Lazy loading RADKLIM Kerchunk dataset inside worker...")

        kerchunk_ref = json.loads(self.ref_path.read_text())
        fs = fsspec.filesystem("reference", fo=kerchunk_ref)
        mapper = fs.get_mapper("")

        # Set up consolidated=False as the default for robustness.
        # can be changed to True if metadata is consolidated.
        ds_full = xr.open_dataset(mapper, engine="zarr", consolidated=False)

        subset = ds_full[self.source_channels].isel(time=slice(self.start_idx, self.end_idx))

        if "chunks" in self.stream_info:
            self.ds = subset.chunk(self.stream_info.get("chunks", {}))
        else:
            self.ds = subset

        # Extract dimensions and coordinates
        y1d = self.ds["y"].values.astype(np.float32)
        x1d = self.ds["x"].values.astype(np.float32)
        self.ny = len(y1d)
        self.nx = len(x1d)
        self.points_per_slice = self.ny * self.nx

        lat_var = self.ds["lat"]
        lon_var = self.ds["lon"]
        raw_lat = lat_var.isel(time=0).values if "time" in lat_var.dims else lat_var.values
        raw_lon = lon_var.isel(time=0).values if "time" in lon_var.dims else lon_var.values

        self.latitudes = raw_lat.astype(np.float32)
        self.longitudes = raw_lon.astype(np.float32)

        self._base_coords = np.column_stack(
            (self.latitudes.reshape(-1), self.longitudes.reshape(-1))
        ).astype(DType)

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
        self._lazy_open()

        # Safety check for empty channels_idx
        if not channels_idx:
            raise ValueError("channels_idx cannot be empty")

        t_idxs_abs, dtr = self._get_dataset_idxs(idx)

        # If the dataset is empty or t_idxs_abs is empty, return empty ReaderData
        if self._empty or t_idxs_abs.size == 0 or self.ds is None:  # Added self.ds check
            return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

        # Shift from global to local time indices
        t_idxs_rel = t_idxs_abs - self.start_idx
        if np.any(t_idxs_rel < 0) or np.any(t_idxs_rel >= self.ds.sizes["time"]):
            return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

        start, stop = int(t_idxs_rel[0]), int(t_idxs_rel[-1]) + 1
        ds_win = self.ds.isel(time=slice(start, stop))

        # DATA & COORD PREP
        nt = ds_win.sizes["time"]
        points_per_slice = ds_win.sizes["y"] * ds_win.sizes["x"]

        # Flatten data array
        da = ds_win.to_array(dim="var").transpose("time", "y", "x", "var")
        raw_shape = da.shape  # (time, y, x, var)
        flat_data = (
            da.values.reshape(-1, raw_shape[-1])[:, channels_idx]
            .astype(np.float32)
        )

        _logger.debug("[Radklim] raw.shape (time, y, x, var) = %s", raw_shape)
        _logger.debug(
            "[Radklim] after channel‐select shape = (%d, %d, %d, %d)",
            raw_shape[0], raw_shape[1], raw_shape[2], len(channels_idx)
        )
        _logger.debug("[Radklim] flattened data.shape = %s", flat_data.shape)

        # 2-D lat/lon grid (first time slice if necessary)
        # raw_lat = ds_win["lat"].isel(time=0).values
        # raw_lon = ds_win["lon"].isel(time=0).values
        lat_var = ds_win["lat"]
        lon_var = ds_win["lon"]

        if "time" in lat_var.dims:
            raw_lat = lat_var.isel(time=0).values
        else:
            raw_lat = lat_var.values

        if "time" in lon_var.dims:
            raw_lon = lon_var.isel(time=0).values
        else:
            raw_lon = lon_var.values

        print("raw_lat shape:", raw_lat.shape, "raw_lat min/max:", np.min(raw_lat), np.max(raw_lat))
        print("raw_lon shape:", raw_lon.shape, "raw_lon min/max:", np.min(raw_lon), np.max(raw_lon))
        assert np.max(raw_lat) > 45 and np.min(raw_lat) > 45, "latitudes look wrong!"
        assert np.max(raw_lon) > 5 and np.min(raw_lon) > 5, "longitudes look wrong!"

        # Ensure valid ranges
        lat2d = np.clip(raw_lat, -90.0, 90.0).astype(np.float32)
        lon2d = ((raw_lon + 180.0) % 360.0 - 180.0).astype(np.float32)

        # STACK IN [lon, lat] ORDER!
        flat_coords = np.stack([lat2d.ravel(), lon2d.ravel()], axis=1).astype(DType)
        full_coords = np.tile(flat_coords, (nt, 1))

        # Times
        full_times = np.repeat(
            ds_win["time"].values.astype("datetime64[ns]"), points_per_slice
        )

        assert full_coords.shape[0] == flat_data.shape[0]
        assert full_coords.shape[0] == full_times.shape[0]

        # More diagnostics
        if full_coords.shape[0] > 0:
            _logger.debug(
                "[Radklim] final coords: min(lon)=%.3f, max(lon)=%.3f, min(lat)=%.3f, max(lat)=%.3f, any NaN=%d",
                full_coords[:,0].min(), full_coords[:,0].max(),
                full_coords[:,1].min(), full_coords[:,1].max(),
                np.isnan(full_coords).any()
            )
        else:
            _logger.warning("[Radklim] All coordinates filtered out—no valid data for this window!")

        _logger.debug("[Radklim] coords.shape   = %s", full_coords.shape)
        _logger.debug(
            "[Radklim] geoinfos.shape = (%d, %d)",
            full_coords.shape[0], len(self.geoinfo_idx)
        )
        _logger.debug("[Radklim] datetimes.shape = %s", full_times.shape)
        if full_times.shape[0] > 0:
            _logger.debug(
                "[Radklim] first/last datetime = %s, %s",
                str(full_times[0]), str(full_times[-1])
            )

        # Package ReaderData
        L = flat_data.shape[0]
        assert (
            full_coords.shape[0] == L and full_times.shape[0] == L
        ), "Shape mismatch after flattening"

        rdata = ReaderData(
            coords=full_coords,
            geoinfos=np.zeros((L, len(self.geoinfo_idx)), dtype=DType),
            data=flat_data,
            datetimes=full_times,
        )
        check_reader_data(rdata, dtr)

        _logger.debug("[Radklim] ReaderData OK")
        return rdata
