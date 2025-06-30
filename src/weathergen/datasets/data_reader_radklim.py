# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import contextlib
import json
from pathlib import Path
from typing import override

import fsspec
import numpy as np
import xarray as xr
import zarr
from numpy.typing import NDArray
import logging
_logger = logging.getLogger(__name__)

from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    DType,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)


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

        # Load source and target channels from config
        self.source_channels = ["RR"] # Hrad coded due to unkown error in config
        self.target_channels = ["RR"] # Hrad coded due to unkown error in config
        self.geoinfo_channels = [] # Hrad coded due to unkown error in config

        
        _logger.info("source_channels: %s", self.source_channels)
        _logger.info("target_channels: %s", self.target_channels)

        self.source_idx = list(range(len(self.source_channels)))
        self.target_idx = list(range(len(self.target_channels)))
        self.geoinfo_idx = list(range(len(self.geoinfo_channels)))
        
        #ref_path_str = "/p/scratch/weatherai/data/npp-atms-unpacked/temp_radklim/radklim_output_kerchunk/radklim_full_dataset.json"
        ref_path = Path(stream_info.get("reference", filename))
        # Load Kerchunk reference
        if not ref_path.exists():
            raise FileNotFoundError(f"Kerchunk reference JSON not found: {ref_path}")

        kerchunk_ref = json.loads(ref_path.read_text())
        fs = fsspec.filesystem("reference", fo=kerchunk_ref)
        mapper = fs.get_mapper("")

        # Consolidate metadata (may be a no-op depending on upstream)
        with contextlib.suppress(Exception):
            zarr.consolidate_metadata(mapper)
        assert all(isinstance(c, str) for c in self.source_channels), f"source_channels malformed: {self.source_channels}"

        ds_full = xr.open_dataset(mapper, engine="zarr", consolidated=True)
        times_full: NDArray[np.datetime64] = ds_full["time"].values
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

        # Subset and optionally rechunk the dataset
        if len(self.source_channels) == 1 and isinstance(self.source_channels[0], list):
            self.source_channels = self.source_channels[0]
        subset = ds_full.isel(time=slice(self.start_idx, self.end_idx))[self.source_channels]
        self.ds = subset.chunk(stream_info.get("chunks", {})) if "chunks" in stream_info else subset
        #norm_path_str = "/p/scratch/weatherai/data/npp-atms-unpacked/temp_radklim/radklim_output_kerchunk/dummy_radklim_stats.json"
        norm_path = Path(stream_info.get("stats_path"))
        if not norm_path.exists():
            raise FileNotFoundError(f"normalisation JSON not found: {norm_path}")

        stats = json.loads(norm_path.read_text())
        self.mean = np.asarray(stats.get("mean", []), dtype=np.float32)
        self.stdev = np.asarray(stats.get("std", []), dtype=np.float32)
        self.mean_geoinfo = np.asarray(stats.get("mean_geoinfo", []), dtype=np.float32)
        self.stdev_geoinfo = np.asarray(stats.get("std_geoinfo", []), dtype=np.float32)

        if len(self.mean) != len(self.source_channels):
            raise ValueError("normalisation stats length does not match number of variables")

        # Load and flatten coordinate grid
        y1d = self.ds["y"].values.astype(np.float32)
        x1d = self.ds["x"].values.astype(np.float32)
        self.ny = len(y1d)
        self.nx = len(x1d)
        self.points_per_slice = self.ny * self.nx

        lat_var = self.ds["lat"]
        lon_var = self.ds["lon"]
        raw_lat = lat_var.isel(time=0).values if "time" in lat_var.dims else lat_var.values
        raw_lon = lon_var.isel(time=0).values if "time" in lon_var.dims else lon_var.values

        self.latitudes = _clip_lat(raw_lat)
        self.longitudes = _clip_lon(raw_lon)
        self._base_coords = np.column_stack(
            (self.latitudes.reshape(-1), self.longitudes.reshape(-1))
        ).astype(DType)

        self.num_steps_per_window = int(tw_handler.t_window_len / period)
        
    # developed only for tackling the fork issue/ stil not working    
    def _lazy_open(self):
        if hasattr(self, "ds"):
            return  # Already opened

        _logger.info("Lazy loading RADKLIM Kerchunk dataset inside worker...")

        kerchunk_ref = json.loads(Path(self.stream_info["reference"]).read_text())
        fs = fsspec.filesystem("reference", fo=kerchunk_ref)
        mapper = fs.get_mapper("")

        # Optional: consolidate
        with contextlib.suppress(Exception):
            zarr.consolidate_metadata(mapper)

        ds_full = xr.open_dataset(mapper, engine="zarr", consolidated=True)

        # Store dataset
        self.ds = ds_full.isel(time=slice(self.start_idx, self.end_idx))[self.source_channels]

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
        nt = int(self.ds.sizes["time"])
        return max(0, nt - self.num_steps_per_window + 1)

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """
        Extract a window of data, filtered by channels and validity.

        Parameters
        ----------
        idx : int
            Window index
        channels_idx : list of int
            Channel indices to extract

        Returns
        -------
        ReaderData
            Data window including coordinates, values, and timestamps
        """
        self._lazy_open()
        if not channels_idx:
            raise ValueError("channels_idx cannot be empty")

        t_idxs_abs, dtr = self._get_dataset_idxs(idx)

        # Early return for empty or invalid request
        if self._empty or t_idxs_abs.size == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx),
                num_geo_fields=len(self.geoinfo_idx),
            )

        # Shift from global to local time indices
        t_idxs_rel = t_idxs_abs - self.start_idx
        if np.any(t_idxs_rel < 0) or np.any(t_idxs_rel >= self.ds.sizes["time"]):
            return ReaderData.empty(
                num_data_fields=len(channels_idx),
                num_geo_fields=len(self.geoinfo_idx),
            )

        # 2. Slice time window
        start, stop = int(t_idxs_rel[0]), int(t_idxs_rel[-1]) + 1
        ds_win = self.ds.isel(time=slice(start, stop))  # (time, y, x, var)

        # 3. Stack variables into one array: (t, y, x, var)
        da = ds_win.to_array(dim="var").transpose("time", "y", "x", "var")
        nt, ny, nx, nvars = da.shape

        # 4. Validate channel indices
        if min(channels_idx) < 0 or max(channels_idx) >= nvars:
            raise IndexError("channels_idx out of bounds")
        if len(set(channels_idx)) != len(channels_idx):
            raise ValueError("channels_idx must be unique")

        # 5. Flatten spatial/temporal and select channels
        flat_data = (
            da.values.astype(np.float32, copy=False).reshape(-1, nvars)[  # (t, y, x, var)
                :, channels_idx
            ]  # → (nt * ny * nx, len(channels_idx))
        )

        # 6. Expand coordinates and time axis
        coords = np.tile(self._base_coords, (nt, 1))
        times = np.repeat(
            ds_win["time"].values.astype("datetime64[ns]"),
            self.points_per_slice,
        )

        rdata = ReaderData(
            coords=coords,
            geoinfos=np.zeros((coords.shape[0], 0), dtype=DType),  # RADKLIM has no geoinfo
            data=flat_data,
            datetimes=times,
        )
        check_reader_data(rdata, dtr)
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
