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
        _logger.info(f"📂 RADKLIM dataset opened with shape: {self.ds[self.source_channels[0]].shape}")
        #_logger.info(f"🗺️ Dataset spatial dims: lat={self.latitudes.shape}, lon={self.longitudes.shape}")
        #_logger.debug(f"type(latitudes): {type(self.latitudes)}, example: {str(self.latitudes)[:100]}")
        
        #_logger.debug(f"type(longitudes): {type(self.longitudes)}, example: {str(self.longitudes)[:100]}")

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

    # @override
    # def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
    #     """
    #     Fetch data for a specific time index and channels.

    #     Parameters
    #     ----------
    #     idx : TIndex
    #         Time index to fetch data for, can be a single integer or a slice.
    #     channels_idx : list[int]
    #         List of channel indices to fetch data for.
    #     Returns
    #     -------
    #     ReaderData
    #         ReaderData object containing the requested data, coordinates, and metadata.
    #     Raises
    #     ------
    #     ValueError
    #         If channels_idx is empty or contains invalid indices.
    #     IndexError
    #         If channels_idx contains indices out of bounds for the dataset.
    #     """
    #     # Crucial step: ensure dataset is open in the current process
    #     self._lazy_open()
    #     _logger.debug(f"🔎 RADKLIM _get(idx={idx}, channels={channels_idx})")

    #     # Safety check for empty channels_idx
    #     if not channels_idx:
    #         raise ValueError("channels_idx cannot be empty")

    #     t_idxs_abs, dtr = self._get_dataset_idxs(idx)
    #     _logger.debug(f"📅 Time indices: absolute={t_idxs_abs}, relative={t_idxs_abs - self.start_idx}")
    #     _logger.warning(
    #         f"⚠️ Returning empty ReaderData: empty={self._empty}, "
    #         f"idx={idx}, t_idxs_abs={t_idxs_abs}, "
    #         f"start_idx={self.start_idx}, end_idx={self.end_idx}, "
    #         f"ds=None? {self.ds is None}"
    #         )

    #     # If the dataset is empty or t_idxs_abs is empty, return empty ReaderData
    #     if self._empty or t_idxs_abs.size == 0 or self.ds is None:  # Added self.ds check
    #         _logger.warning(f"⚠️ Returning empty ReaderData: empty={self._empty}, t_idxs_abs={t_idxs_abs}, ds=None? {self.ds is None}")
    #         return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

    #     # Shift from global to local time indices
    #     t_idxs_rel = t_idxs_abs - self.start_idx
    #     if np.any(t_idxs_rel < 0) or np.any(t_idxs_rel >= self.ds.sizes["time"]):
    #         return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

    #     # Slice the dataset to get the relevant time window
    #     start, stop = int(t_idxs_rel[0]), int(t_idxs_rel[-1]) + 1
    #     ds_win = self.ds.isel(time=slice(start, stop))
    #     _logger.debug(f"📦 Windowed dataset dims: {ds_win.dims}")

    #     # Stack the data into a 4D array (time, y, x, var)
    #     da = ds_win.to_array(dim="var").transpose("time", "y", "x", "var")
    #     _logger.debug(f"📐 Merged array shape (t,y,x,v): {da.shape}")
    #     _logger.debug(f"[RADKLIM] data_vars: {list(self.ds.data_vars)}")
    #     _logger.debug(f"[RADKLIM] da.shape: {da.shape}")
    #     _logger.debug(f"[RADKLIM] channels_idx: {channels_idx}")
        

    #     nt, ny, nx, nvars = da.shape

    #     # Validate channels_idx
    #     if min(channels_idx) < 0 or max(channels_idx) >= nvars:
    #         raise IndexError("channels_idx out of bounds")
    #     if len(set(channels_idx)) != len(channels_idx):
    #         raise ValueError("channels_idx must be unique")

    #     # Flatten spatial/temporal and select channels
    #     flat_data = da.values.astype(np.float32, copy=False).reshape(-1, nvars)[:, channels_idx]
    #     _logger.debug(f"🧪 Flattened data shape: {flat_data.shape}")
        

    #     # Expand coordinates and time axis
    #     coords = np.tile(self._base_coords, (nt, 1))
        
    #     times = np.repeat(ds_win["time"].values.astype("datetime64[ns]"), self.points_per_slice)
        
    #     if np.isnan(flat_data).any():
    #         _logger.warning(f"⚠️ NaNs detected in RADKLIM flat_data at index {idx}")
    #     if coords.shape[0] != flat_data.shape[0]:
    #         _logger.error(f"❌ Mismatch: coords={coords.shape}, data={flat_data.shape}")

    #     # === TEMPORARY PATCH: Downsample tokens to reduce input size ===
    #     max_tokens = 80640  # Match ERA5 sample size (e.g., 112x720 grid * 1 time window * 1 channel)
    #     if flat_data.shape[0] > max_tokens:
    #         selected_indices = np.random.choice(flat_data.shape[0], size=max_tokens, replace=False)
    #         flat_data = flat_data[selected_indices]
    #         coords = coords[selected_indices]
    #         times = times[selected_indices]


    #     rdata = ReaderData(
    #         coords=coords,
    #         geoinfos=np.zeros((coords.shape[0], 0), dtype=DType),
    #         data=flat_data,
    #         datetimes=times,
    #     )
    #     check_reader_data(rdata, dtr)
    #     return rdata
# In class RadklimKerchunkReader

    # @override
    # def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
    #     """
    #     Fetch data for a specific time index and channels.
    #     Includes logic to coarsen high-resolution data to a target size if needed,
    #     preserving the spatial grid structure and correctly handling NaNs.
    #     """
    #     self._lazy_open()
    #     _logger.debug(f"🔎 RADKLIM _get(idx={idx}, channels={channels_idx})")

    #     if not channels_idx:
    #         raise ValueError("channels_idx cannot be empty")

    #     t_idxs_abs, dtr = self._get_dataset_idxs(idx)
    #     _logger.debug(f"📅 Time indices: absolute={t_idxs_abs}, relative={t_idxs_abs - self.start_idx}")

    #     if self._empty or t_idxs_abs.size == 0 or self.ds is None:
    #         _logger.warning(f"⚠️ Returning empty ReaderData: empty={self._empty}, t_idxs_abs={t_idxs_abs}, ds=None? {self.ds is None}")
    #         return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

    #     t_idxs_rel = t_idxs_abs - self.start_idx
    #     if np.any(t_idxs_rel < 0) or np.any(t_idxs_rel >= self.ds.sizes["time"]):
    #         return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

    #     start, stop = int(t_idxs_rel[0]), int(t_idxs_rel[-1]) + 1
    #     ds_win = self.ds.isel(time=slice(start, stop))
    #     _logger.debug(f"📦 Initial windowed dataset dims: {ds_win.dims}")

    #     # --- CORRECT SPATIAL DOWNSAMPLING LOGIC ---
    #     target_spatial_points = 80640  # Example: 112 * 720 grid
    #     current_spatial_points = ds_win.sizes["y"] * ds_win.sizes["x"]

    #     base_coords = self._base_coords
    #     points_per_slice = self.points_per_slice

    #     if current_spatial_points > target_spatial_points * 1.1:
    #         # # ratio = np.sqrt(current_spatial_points / target_spatial_points)
    #         # # y_factor = max(1, int(round(ds_win.sizes["y"] / (ds_win.sizes["y"] / ratio))))
    #         # # x_factor = max(1, int(round(ds_win.sizes["x"] / (ds_win.sizes["x"] / ratio))))
    #         # _logger.info(f"Coarsening RADKLIM data by factors: y={y_factor}, x={x_factor} to match target size.")
    #         # # Use coarsen and mean. .mean() will skip NaNs by default.
    #         # ds_win = ds_win.coarsen(y=y_factor, x=x_factor, boundary="trim").mean()
    #         factor = max(1, int(round(np.sqrt(current_spatial_points / target_spatial_points))))
    #         ds_win = ds_win.coarsen(y=factor, x=factor, boundary="trim").mean()
    #         _logger.debug(f"📦 Coarsened dataset dims: {ds_win.dims}")
            
    #         # We must re-generate coordinates for the new, smaller grid
    #         latitudes = _clip_lat(ds_win["lat"].isel(time=0).values)
    #         longitudes = _clip_lon(ds_win["lon"].isel(time=0).values)
    #         base_coords = np.stack([latitudes.ravel(), longitudes.ravel()], axis=-1).astype(DType)
    #         points_per_slice = base_coords.shape[0]


    #     # da = ds_win.to_array(dim="var").transpose("time", "y", "x", "var")
    #     # nt, _, _, nvars = da.shape
    #     # _logger.debug(f"📐 Merged array shape (t,y,x,v): {da.shape}")

    #     # flat_data = da.values.astype(np.float32, copy=False).reshape(-1, nvars)[:, channels_idx]
    #     # _logger.debug(f"🧪 Flattened data shape: {flat_data.shape}")

    #     # coords = np.tile(base_coords, (nt, 1))
    #     # times = np.repeat(ds_win["time"].values.astype("datetime64[ns]"), points_per_slice)

    #     # # Check for NaNs *after* coarsening. There shouldn't be any, but it's a good check.
    #     # if np.isnan(flat_data).any():
    #     #     _logger.warning(f"⚠️ NaNs still detected in RADKLIM data at index {idx} AFTER coarsening.")

    #     # rdata = ReaderData(
    #     #     coords=coords,
    #     #     geoinfos=np.zeros((coords.shape[0], 0), dtype=DType),
    #     #     data=flat_data,
    #     #     datetimes=times,
    #     # )
    #     # check_reader_data(rdata, dtr)
    #     # return rdata
    #     da = ds_win.to_array(dim="var").transpose("time", "y", "x", "var")
    #     nt, _, _, nvars = da.shape
    #     flat_data = da.values.astype(np.float32, copy=False).reshape(-1, nvars)[:, channels_idx]
    #     _logger.debug(f"🧪 Flattened data shape: {flat_data.shape}")

    #     # --- EXPERIMENTAL FIX: Replace all NaNs with 0.0 ---
    #     nan_count = np.count_nonzero(np.isnan(flat_data))
    #     if nan_count > 0:
    #         _logger.warning(
    #             f"⚠️ Found and replaced {nan_count} NaN values with 0.0 in RADKLIM data "
    #             f"at index {idx} (total points: {flat_data.size})."
    #         )
    #         # Use np.nan_to_num to perform the replacement in-place for efficiency.
    #         np.nan_to_num(flat_data, copy=False, nan=0.0)

    #     # The rest of the function remains the same...
    #     coords = np.tile(base_coords, (nt, 1))
    #     times = np.repeat(ds_win["time"].values.astype("datetime64[ns]"), points_per_slice)

    #     rdata = ReaderData(
    #         coords=coords,
    #         geoinfos=np.zeros((coords.shape[0], 0), dtype=DType),
    #         data=flat_data,
    #         datetimes=times,
    #     )
    #     check_reader_data(rdata, dtr)
    #     return rdata

    # @override
    # def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
    #     # 1) Ensure the dataset is opened in this worker
    #     self._lazy_open()
    #     _logger.debug(f"🔎 RADKLIM _get(idx={idx}, channels={channels_idx})")

    #     if not channels_idx:
    #         raise ValueError("channels_idx cannot be empty")

    #     # 2) Compute absolute & relative time indices
    #     t_idxs_abs, dtr = self._get_dataset_idxs(idx)
    #     _logger.debug(
    #         f"[RADKLIM] _get(idx={idx}) → "
    #         f"absolute time indices={t_idxs_abs.tolist()}, "
    #         f"start_idx={self.start_idx}, end_idx={self.end_idx}"
    #     )

    #     if self._empty or t_idxs_abs.size == 0 or self.ds is None:
    #         return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

    #     t_idxs_rel = t_idxs_abs - self.start_idx
    #     _logger.debug(f"[RADKLIM]          → relative indices={t_idxs_rel.tolist()}  (ds.sizes['time']={self.ds.sizes['time']})")
    #     if np.any(t_idxs_rel < 0) or np.any(t_idxs_rel >= self.ds.sizes["time"]):
    #         return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

    #     start, stop = int(t_idxs_rel[0]), int(t_idxs_rel[-1]) + 1
    #     ds_win = self.ds.isel(time=slice(start, stop))
    #     _logger.debug(f"📦 Window dims: {ds_win.sizes}")

    #     # 3) Optional coarsen to target ~80k points
    #     cur_pts = ds_win.sizes["y"] * ds_win.sizes["x"]
    #     target_pts = 80640
    #     if cur_pts > target_pts:
    #         factor = max(1, int(round(np.sqrt(cur_pts / target_pts))))
    #         _logger.info(f"Coarsening RADKLIM by factor={factor}")
    #         ds_win = ds_win.coarsen(y=factor, x=factor, boundary="trim").mean()
    #         ny2, nx2 = ds_win.sizes["y"], ds_win.sizes["x"]
    #         points_per_slice2 = ny2 * nx2
    #         _logger.info(f"[Coarsen] new grid: y={ny2}, x={nx2}, pts={points_per_slice2}")


    #     # 4) Rebuild base_coords & points_per_slice AFTER coarsening
    #     # lat0 = ds_win["lat"].isel(time=0).values.astype(np.float32)
    #     # lon0 = ds_win["lon"].isel(time=0).values.astype(np.float32)
    #     lat2d = ds_win["lat"].isel(time=0).values.astype(np.float32)
    #     lon2d = ds_win["lon"].isel(time=0).values.astype(np.float32)
    #     # lat_clipped = _clip_lat(lat0)
    #     # lon_clipped = _clip_lon(lon0)
    #     # base_coords = np.column_stack((lat_clipped.ravel(), lon_clipped.ravel())).astype(DType)
    #     flat_coords = np.stack([lat2d.reshape(-1), lon2d.reshape(-1)], axis=1)
    #     points_per_slice = flat_coords.shape[0]
    #     nt = ds_win.sizes["time"]

    #     # 5) Flatten data to shape (nt*pts, nvars), select channels
    #     da = ds_win.to_array(dim="var").transpose("time", "y", "x", "var")
    #     nt2, y2, x2, nvars = da.shape
    #     assert nt2 == nt and y2 * x2 == points_per_slice
    #     flat = da.values.reshape(-1, nvars)[:, channels_idx].astype(np.float32)
    #     _logger.debug(f"🧪 flat_data shape: {flat.shape}")

    #     # 6) Zero‐fill any remaining NaNs
    #     nan_cnt = int(np.isnan(flat).sum())
    #     if nan_cnt:
    #         _logger.warning(f"Replacing {nan_cnt} NaNs with 0.0 at idx={idx}")
    #         np.nan_to_num(flat, copy=False, nan=0.0)

    #     # 7) Tile coords & repeat times
    #     coords = np.tile(flat_coords, (nt, 1))
    #     # _logger.debug(f"[Coord check] base_coords head={base_coords[:5]} tail={base_coords[-5:]}")
    #     _logger.debug(f"[Coord check] flat_coords head={flat_coords[:5]} tail={flat_coords[-5:]}")
    #     times = np.repeat(ds_win["time"].values.astype("datetime64[ns]"), points_per_slice)

    #     # 8) Final sanity check
    #     L = flat.shape[0]
    #     assert coords.shape[0] == L and times.shape[0] == L, \
    #         f"Shape mismatch after flattening: coords={coords.shape[0]}, data={L}, times={times.shape[0]}"

    #     # 9) Package into ReaderData
    #     rdata = ReaderData(
    #         coords=coords,
    #         geoinfos=np.zeros((L, 0), dtype=DType),
    #         data=flat,
    #         datetimes=times,
    #     )
    #     check_reader_data(rdata, dtr)
    #     return rdata

    # @override
    # def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
    #     # 1) Ensure the dataset is opened in this worker
    #     self._lazy_open()
    #     _logger.debug(f"🔎 RADKLIM _get(idx={idx}, channels={channels_idx})")

    #     if not channels_idx:
    #         raise ValueError("channels_idx cannot be empty")

    #     # 2) Compute absolute & relative time indices
    #     t_idxs_abs, dtr = self._get_dataset_idxs(idx)
    #     _logger.debug(
    #         f"[RADKLIM] _get(idx={idx}) → "
    #         f"absolute time indices={t_idxs_abs.tolist()}, "
    #         f"start_idx={self.start_idx}, end_idx={self.end_idx}"
    #     )

    #     # 2b) early exit if outside
    #     if self._empty or t_idxs_abs.size == 0 or self.ds is None:
    #         return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

    #     t_idxs_rel = t_idxs_abs - self.start_idx
    #     if np.any(t_idxs_rel < 0) or np.any(t_idxs_rel >= self.ds.sizes["time"]):
    #         return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

    #     start, stop = int(t_idxs_rel[0]), int(t_idxs_rel[-1]) + 1
    #     ds_win = self.ds.isel(time=slice(start, stop))
    #     nt = ds_win.sizes["time"]
    #     _logger.debug(f"📦 Window dims: {ds_win.sizes}")

    #     # 3) Coarsen if needed
    #     cur_pts = ds_win.sizes["y"] * ds_win.sizes["x"]
    #     target_pts = 80640
    #     if cur_pts > target_pts:
    #         factor = max(1, int(round(np.sqrt(cur_pts / target_pts))))
    #         _logger.info(f"Coarsening RADKLIM by factor={factor}")
    #         ds_win = ds_win.coarsen(y=factor, x=factor, boundary="trim").mean()
    #     ny, nx = ds_win.sizes["y"], ds_win.sizes["x"]
    #     pts = ny * nx
    #     _logger.info(f"[Coarsen] new grid: y={ny}, x={nx}, pts={pts}")

    #     # 4) Build (and cache) flat lat/lon coords once per grid shape
    #     if not hasattr(self, "_flat_coords") or self._flat_coords.shape[0] != pts:
    #         lat2d = ds_win["lat"].isel(time=0).values.astype(np.float32)
    #         lon2d = ds_win["lon"].isel(time=0).values.astype(np.float32)
    #         flat = np.stack([lat2d.reshape(-1), lon2d.reshape(-1)], axis=1)
    #         self._flat_coords = flat  # shape (pts, 2)
    #     flat_coords = self._flat_coords

    #     # 5) Flatten data: since only one var ("RR"), pull it directly
    #     arr = ds_win["RR"].values  # shape (nt, ny, nx)
    #     flat_data = arr.reshape(nt * pts, 1)[:, channels_idx].astype(np.float32)

    #     # 6) Zero-fill NaNs
    #     nan_cnt = int(np.isnan(flat_data).sum())
    #     if nan_cnt:
    #         #_logger.warning(f"Replacing {nan_cnt} NaNs with 0.0 at idx={idx}")
    #         np.nan_to_num(flat_data, copy=False, nan=0.0)

    #     # 7) Tile coords & times
    #     coords = np.tile(flat_coords, (nt, 1))
    #     times  = np.repeat(ds_win["time"].values.astype("datetime64[ns]"), pts)

    #     # 8) Final sanity checks
    #     L = nt * pts
    #     assert flat_data.shape == (L, len(channels_idx))
    #     assert coords.shape     == (L, 2)
    #     assert times.shape      == (L,)

    #     # 9) Package into ReaderData
    #     rdata = ReaderData(
        
        
    #         coords=coords,
    #         geoinfos=np.zeros((L, 0), dtype=DType),
    #         data=flat_data,
    #         datetimes=times,
    #     )
    #     check_reader_data(rdata, dtr)
    #     return rdata
    # @override
    # def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
    #     # ——————————————————————————————————————————————————————————————
    #     # 1) Lazy‐open & time‐indexing (unchanged)
    #     self._lazy_open()
    #     if not channels_idx:
    #         raise ValueError("channels_idx cannot be empty")
    #     t_idxs_abs, dtr = self._get_dataset_idxs(idx)
    #     if self._empty or t_idxs_abs.size == 0 or self.ds is None:
    #         return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))
    #     t_idxs_rel = t_idxs_abs - self.start_idx
    #     if np.any(t_idxs_rel < 0) or np.any(t_idxs_rel >= self.ds.sizes["time"]):
    #         return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))
    #     start, stop = int(t_idxs_rel[0]), int(t_idxs_rel[-1]) + 1
    #     ds_win = self.ds.isel(time=slice(start, stop))

    #     # ——————————————————————————————————————————————————————————————
    #     # 2) Subsample instead of coarsen (preserves raw lat/lon)
    #     cur_pts = ds_win.sizes["y"] * ds_win.sizes["x"]
    #     target_pts = 80640
    #     if cur_pts > target_pts:
    #         factor = max(1, int(round(np.sqrt(cur_pts / target_pts))))
    #         _logger.info(f"Subsampling RADKLIM by factor={factor}")
    #         ds_win = ds_win.isel(
    #             y=slice(None, None, factor),
    #             x=slice(None, None, factor),
    #         )
    #         ny2, nx2 = ds_win.sizes["y"], ds_win.sizes["x"]
    #         _logger.info(f"[Subsample] new grid: y={ny2}, x={nx2}, pts={ny2*nx2}")

    #     # ——————————————————————————————————————————————————————————————
    #     # 3) Rebuild coords from the original lat/lon array
    #     lat2d_raw = ds_win["lat"].isel(time=0).values
    #     lon2d_raw = ds_win["lon"].isel(time=0).values

    #     # clip/warp into valid ranges (–90→+90, –180→+180)
    #     lat2d = np.clip(lat2d_raw, -90.0, 90.0)
    #     lon2d = ((lon2d_raw + 180.0) % 360.0) - 180.0

    #     # flatten spatial dims
    #     flat_coords_full = np.stack(
    #         [lat2d.reshape(-1), lon2d.reshape(-1)], axis=1
    #     ).astype(np.float32)   # shape=(pts,2)
    #     orig_pts = flat_coords_full.shape[0]
    #     nt = ds_win.sizes["time"]

    #     # ——————————————————————————————————————————————————————————————
    #     # 4) Flatten the data & select channels
    #     da = ds_win.to_array(dim="var").transpose("time", "y", "x", "var")
    #     nt2, y2, x2, nvars = da.shape
    #     assert nt2 == nt and y2*x2 == orig_pts
    #     flat_data_full = da.values.reshape(-1, nvars).astype(np.float32)
    #     flat = flat_data_full[:, channels_idx]  # shape=(nt*orig_pts, nch)

    #     # ——————————————————————————————————————————————————————————————
    #     # 5) zero‐fill any NaNs in the data
    #     nan_cnt = int(np.isnan(flat).sum())
    #     if nan_cnt:
    #         # _logger.warning(f"Replacing {nan_cnt} NaNs with 0.0 at idx={idx}")
    #         np.nan_to_num(flat, copy=False, nan=0.0)

    #     # ——————————————————————————————————————————————————————————————
    #     # 6) Build the full coords & times arrays, then mask out any invalid coords
    #     coords_full = np.tile(flat_coords_full, (nt, 1))  # shape=(nt*orig_pts,2)
    #     times_full  = np.repeat(
    #         ds_win["time"].values.astype("datetime64[ns]"), orig_pts
    #     )  # shape=(nt*orig_pts,)

    #     # mask to keep only rows where BOTH lat & lon are finite
    #     valid_cells = np.isfinite(flat_coords_full).all(axis=1)  # length=orig_pts
    #     if not valid_cells.all():
    #         drop = np.count_nonzero(~valid_cells)
    #         _logger.warning(f"Dropping {drop} invalid coordinate cells at idx={idx}")
    #     mask_full = np.tile(valid_cells, nt)  # shape=(nt*orig_pts,)

    #     coords = coords_full[mask_full]
    #     data   = flat      [mask_full]
    #     times  = times_full[mask_full]

    #     # ——————————————————————————————————————————————————————————————
    #     # 7) Final sanity check
    #     L = data.shape[0]
    #     assert coords.shape[0] == L and times.shape[0] == L, (
    #         f"Shape mismatch: coords={coords.shape[0]}, data={L}, times={times.shape[0]}"
    #     )

    #     # ——————————————————————————————————————————————————————————————
    #     # 8) If after masking there’s nothing left, return empty
    #     if L == 0:
    #         _logger.warning(f"All cells invalid at idx={idx} → treating as empty")
    #         return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

    #     # ——————————————————————————————————————————————————————————————
    #     # 9) Package and return
    #     rdata = ReaderData(
    #         coords=coords,
    #         geoinfos=np.zeros((L, 0), dtype=DType),
    #         data=data,
    #         datetimes=times,
    #     )
    #     check_reader_data(rdata, dtr)
    #     return rdata

    # @override
    # def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
    #     # 1) Ensure the dataset is opened in this worker
    #     self._lazy_open()
    #     _logger.debug(f"🔎 RADKLIM _get(idx={idx}, channels={channels_idx})")

    #     if not channels_idx:
    #         raise ValueError("channels_idx cannot be empty")

    #     # 2) Compute absolute & relative time indices
    #     t_idxs_abs, dtr = self._get_dataset_idxs(idx)
    #     _logger.debug(
    #         f"[RADKLIM] _get(idx={idx}) → "
    #         f"absolute time indices={t_idxs_abs.tolist()}, "
    #         f"start_idx={self.start_idx}, end_idx={self.end_idx}"
    #     )

    #     if self._empty or t_idxs_abs.size == 0 or self.ds is None:
    #         return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

    #     t_idxs_rel = t_idxs_abs - self.start_idx
    #     if np.any(t_idxs_rel < 0) or np.any(t_idxs_rel >= self.ds.sizes["time"]):
    #         return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

    #     start, stop = int(t_idxs_rel[0]), int(t_idxs_rel[-1]) + 1
    #     ds_win = self.ds.isel(time=slice(start, stop))
    #     _logger.debug(f"📦 Window dims: {ds_win.sizes}")

    #     # # 3) Optional coarsen to target ~80k points
    #     # cur_pts = ds_win.sizes["y"] * ds_win.sizes["x"]
    #     # target_pts = 80640
    #     # if cur_pts > target_pts:
    #     #     factor = max(1, int(round(np.sqrt(cur_pts / target_pts))))
    #     #     _logger.info(f"Coarsening RADKLIM by factor={factor}")
    #     #     ds_win = ds_win.coarsen(y=factor, x=factor, boundary="trim").mean()
    #     #     _logger.info(f"[Coarsen] new grid: y={ds_win.sizes['y']}, x={ds_win.sizes['x']}, pts={ds_win.sizes['y']*ds_win.sizes['x']}")

    #     # 4) Build coords AFTER coarsening
    #     lat2d = ds_win["lat"].isel(time=0).values.astype(np.float32)   # may contain NaNs
    #     lon2d = ds_win["lon"].isel(time=0).values.astype(np.float32)   # may contain NaNs
    #     flat_coords = np.stack([lat2d.ravel(), lon2d.ravel()], axis=1)
    #     points_per_slice = flat_coords.shape[0]
    #     nt = ds_win.sizes["time"]

    #     # 5) Flatten data to (nt*pts, nvars) and select channels
    #     da = ds_win.to_array(dim="var").transpose("time", "y", "x", "var")
    #     nt2, y2, x2, nvars = da.shape
    #     assert nt2 == nt and y2 * x2 == points_per_slice
    #     flat = da.values.reshape(-1, nvars)[:, channels_idx].astype(np.float32)

    #     # 6) **NO MORE ZERO-FILL** — leave NaNs for the sampler to handle
    #     # nan_cnt = int(np.isnan(flat).sum())
    #     # if nan_cnt:
    #     #     _logger.warning(f"Replacing {nan_cnt} NaNs with 0.0 at idx={idx}")
    #     #     np.nan_to_num(flat, copy=False, nan=0.0)

    #     # 7) Tile coords & repeat times
    #     coords = np.tile(flat_coords, (nt, 1))
    #     times = np.repeat(ds_win["time"].values.astype("datetime64[ns]"), points_per_slice)

    #     # 8) Sanity check
    #     L = flat.shape[0]
    #     assert coords.shape[0] == L and times.shape[0] == L, \
    #         f"Shape mismatch after flattening: coords={coords.shape[0]}, data={L}, times={times.shape[0]}"

    #     # 9) Package into ReaderData — with NaNs intact
    #     rdata = ReaderData(
    #         coords=coords,
    #         geoinfos=np.zeros((L, 0), dtype=np.float32),
    #         data=flat,
    #         datetimes=times,
    #     )
    #     check_reader_data(rdata, dtr)
    #     return rdata

    # @override
    # def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
    #     """
    #     Correctly fetches and prepares RADKLIM data, with overflow‐safe
    #     coordinate clipping and repair.
    #     """
    #     # 1) Lazy open and initial checks
    #     self._lazy_open()
    #     if not channels_idx:
    #         raise ValueError("channels_idx cannot be empty")

    #     # 2) Compute time indices and slice the dataset
    #     t_idxs_abs, dtr = self._get_dataset_idxs(idx)
    #     if self._empty or t_idxs_abs.size == 0 or self.ds is None:
    #         return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

    #     t_idxs_rel = t_idxs_abs - self.start_idx
    #     if np.any(t_idxs_rel < 0) or np.any(t_idxs_rel >= self.ds.sizes["time"]):
    #         return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

    #     start, stop = int(t_idxs_rel[0]), int(t_idxs_rel[-1]) + 1
    #     ds_win = self.ds.isel(time=slice(start, stop))

    #     # 3) --- COORDINATE AND DATA PREPARATION ---
    #     nt = ds_win.sizes["time"]
    #     points_per_slice = ds_win.sizes["y"] * ds_win.sizes["x"]

    #     # 3a) Flatten data array
    #     da = ds_win.to_array(dim="var").transpose("time", "y", "x", "var")
    #     flat_data = da.values.reshape(-1, da.shape[-1])[:, channels_idx].astype(np.float32)

    #     # 3b) Read the raw 2D lat/lon grid at the first time slice
    #     raw_lat = ds_win["lat"].isel(time=0).values if "time" in ds_win["lat"].dims else ds_win["lat"].values
    #     raw_lon = ds_win["lon"].isel(time=0).values if "time" in ds_win["lon"].dims else ds_win["lon"].values

    #     # 3c) Clip into valid geographic ranges (and cast to float32)
    #     lat2d = _clip_lat(raw_lat)
    #     lon2d = _clip_lon(raw_lon)

    #     # 3d) Repair any remaining non-finite values
    #     invalid_mask = ~np.isfinite(lat2d) | ~np.isfinite(lon2d)
    #     if np.any(invalid_mask):
    #         # _logger.warning(
    #         #     f"Found and repaired {np.sum(invalid_mask)} invalid coordinates at idx={idx}."
    #         # )
    #         lat2d[invalid_mask] = 0.0
    #         lon2d[invalid_mask] = 0.0

    #     # 3e) Build the final flattened coordinate array
    #     flat_coords = np.stack([lat2d.ravel(), lon2d.ravel()], axis=1).astype(DType)
    #     full_coords = np.tile(flat_coords, (nt, 1))

    #     # 4) Times
    #     full_times = np.repeat(ds_win["time"].values.astype("datetime64[ns]"), points_per_slice)

    #     # 5) Sanity check and package
    #     L = flat_data.shape[0]
    #     assert full_coords.shape[0] == L and full_times.shape[0] == L, \
    #         "Shape mismatch after flattening"

    #     rdata = ReaderData(
    #         coords=full_coords,
    #         geoinfos=np.zeros((L, len(self.geoinfo_idx)), dtype=DType),
    #         data=flat_data,
    #         datetimes=full_times,
    #     )
    #     check_reader_data(rdata, dtr)
    #     return rdata

    # @override
    # def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
    #     """
    #     Fetch and prepare RADKLIM data with the **same granular DEBUG
    #     instrumentation** we added for the Anemoi reader.

    #     Log lines follow the exact pattern:

    #         [Radklim] raw.shape (time, y, x, var) = (...)
    #         [Radklim] after channel‐select shape = (...)
    #         [Radklim] flattened data.shape = (...)
    #         [Radklim] coords.shape = (...)
    #         [Radklim] geoinfos.shape = (...)
    #         [Radklim] datetimes.shape = (...)
    #         [Radklim] first/last datetime = <ts0>, <ts1>
    #         [Radklim] ReaderData OK
    #     """
    #     # 1) ――― Lazy open & guard clauses ―――
    #     self._lazy_open()
    #     if not channels_idx:
    #         raise ValueError("channels_idx cannot be empty")

    #     # 2) ――― Time-window slicing ―――
    #     t_idxs_abs, dtr = self._get_dataset_idxs(idx)
    #     if self._empty or t_idxs_abs.size == 0 or self.ds is None:
    #         return ReaderData.empty(
    #             num_data_fields=len(channels_idx),
    #             num_geo_field = len(self.geoinfo_idx))

    #     t_idxs_rel = t_idxs_abs - self.start_idx
    #     if np.any(t_idxs_rel < 0) or np.any(t_idxs_rel >= self.ds.sizes["time"]):
    #         return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

    #     start, stop = int(t_idxs_rel[0]), int(t_idxs_rel[-1]) + 1
    #     ds_win = self.ds.isel(time=slice(start, stop))

    #     # 3) ――― DATA & COORD PREP ―――
    #     nt = ds_win.sizes["time"]
    #     points_per_slice = ds_win.sizes["y"] * ds_win.sizes["x"]

    #     # 3a) Flatten data array
    #     da = ds_win.to_array(dim="var").transpose("time", "y", "x", "var")
    #     raw_shape = da.shape  # (time, y, x, var)
    #     flat_data = (
    #         da.values.reshape(-1, raw_shape[-1])[:, channels_idx]
    #         .astype(np.float32)
    #     )

    #     # ── diagnostics ────────────────────────────────────────────────
    #     _logger.debug("[Radklim] raw.shape (time, y, x, var) = %s", raw_shape)
    #     _logger.debug(
    #         "[Radklim] after channel‐select shape = (%d, %d, %d, %d)",
    #         raw_shape[0], raw_shape[1], raw_shape[2], len(channels_idx)
    #     )
    #     _logger.debug("[Radklim] flattened data.shape = %s", flat_data.shape)

    #     # 3b) 2-D lat/lon grid (first time slice if necessary)
    #     raw_lat = ds_win["lat"].isel(time=0).values
    #     raw_lon = ds_win["lon"].isel(time=0).values
    #     # First clip to valid ranges *before* cast:
    #     lat2d = np.clip(raw_lat, -90.0, 90.0).astype(np.float32)
    #     lon2d = ((raw_lon + 180.0) % 360.0 - 180.0).astype(np.float32)

    #     # # 3d) Repair any remaining NaNs/Infs
    #     # invalid_mask = ~np.isfinite(lat2d) | ~np.isfinite(lon2d)
    #     # if np.any(invalid_mask):
    #     #     lat2d[invalid_mask] = 0.0
    #     #     lon2d[invalid_mask] = 0.0

    #     # 3e) Flatten coordinates
    #     flat_coords = np.stack([lon2d.ravel(), lat2d.ravel()], axis=1).astype(DType)
    #     full_coords = np.tile(flat_coords, (nt, 1))

    #     # 4) ――― Times ―――
    #     full_times = np.repeat(
    #         ds_win["time"].values.astype("datetime64[ns]"), points_per_slice
    #     )

    #     # ── more diagnostics ───────────────────────────────────────────
    #     _logger.debug("[Radklim] coords.shape   = %s", full_coords.shape)
    #     _logger.debug(
    #         "[Radklim] geoinfos.shape = (%d, %d)",
    #         full_coords.shape[0], len(self.geoinfo_idx)
    #     )
    #     _logger.debug("[Radklim] datetimes.shape = %s", full_times.shape)
    #     _logger.debug(
    #         "[Radklim] first/last datetime = %s, %s",
    #         str(full_times[0]), str(full_times[-1])
    #     )

    #     # 5) ――― Package ReaderData ―――
    #     L = flat_data.shape[0]
    #     assert (
    #         full_coords.shape[0] == L and full_times.shape[0] == L
    #     ), "Shape mismatch after flattening"

    #     rdata = ReaderData(
    #         coords=full_coords,
    #         geoinfos=np.zeros((L, len(self.geoinfo_idx)), dtype=DType),
    #         data=flat_data,
    #         datetimes=full_times,
    #     )
    #     check_reader_data(rdata, dtr)

    #     _logger.debug("[Radklim] ReaderData OK")
    #     return rdata

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """
        Fetch and prepare RADKLIM data with robust coordinate handling and detailed DEBUG.
        """
        self._lazy_open()
        if not channels_idx:
            raise ValueError("channels_idx cannot be empty")

        # Time-window slicing
        t_idxs_abs, dtr = self._get_dataset_idxs(idx)
        if self._empty or t_idxs_abs.size == 0 or self.ds is None:
            return ReaderData.empty(
                num_data_fields=len(channels_idx),
                num_geo_field=len(self.geoinfo_idx)
            )

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
