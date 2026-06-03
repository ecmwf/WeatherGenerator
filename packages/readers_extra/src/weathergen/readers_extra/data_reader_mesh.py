import json
import logging
from pathlib import Path
from typing import override

import dask
import fsspec
import numpy as np
import torch
import xarray as xr
from numpy.typing import NDArray

from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    DTRange,
    ReaderData,
    TimeWindowHandler,
    TIndex,
)
from weathergen.train.utils import Stage

logging.getLogger("fsspec").setLevel(logging.WARNING)
logging.getLogger("fsspec.implementations.reference").setLevel(logging.WARNING)
_logger = logging.getLogger(__name__)

# Small epsilon to handle time boundary exclusivity
t_epsilon = np.timedelta64(1, "ms")
MIN_PATCH_POINTS = 512


class DataReaderMesh(DataReaderTimestep):
    """
    A data reader for unstructured mesh data accessed via Virtual Zarr.
    Features:
    - Separate Source and Target files.
    - Persistence of State time indexing (forward fill).
    - Robust Multi-Node/Worker support (Fork-safe, Dask-safe).
    - Dynamic Patching (local) OR Global Sparse Sampling.
    """

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
        stage: Stage | None = None,
    ) -> None:
        self.filename_source = Path(filename)
        if "target_file" in stream_info:
            self.filename_target = Path(stream_info["target_file"])
        else:
            self.filename_target = self.filename_source

        self._stream_info = stream_info
        self.roi = stream_info.get("roi")
        self.patch_size_deg = stream_info.get("patch_size_deg")
        self.sample_points = stream_info.get("sample_points")
        self._len_cached = 0

        self._dask_arrays_src = {}
        self._dask_arrays_trg = {}

        self.sampling_mode = stream_info.get("sampling_mode", "patch")
        self.sampling_step = stream_info.get("sampling_step", 1)
        self.patch_stability_window = stream_info.get("patch_stability_window", 1)
        self.filler_values = stream_info.get("filler_values", [])

        # Auto-enable staircase mode if window is defined and we are in patch mode
        auto_use_counter = self.sampling_mode == "patch" and "patch_stability_window" in stream_info
        self.patch_use_counter = stream_info.get("patch_use_counter", auto_use_counter)

        if self.filename_source != self.filename_target and self.sampling_mode != "patch":
            _logger.error(
                f"[Stream {stream_info.get('name')}] DIFFERENT Source and Target files detected! "
                "Forcing sampling_mode to 'patch'."
            )
            self.sampling_mode = "patch"

        self._initialized = False
        self.ds_source = None
        self.ds_target = None
        self.mapper_src = None
        self.mapper_trg = None

        if not self.filename_source.exists():
            _logger.warning(f"Source file {self.filename_source} not found. Stream skipped.")
            self.init_empty()
            super().__init__(tw_handler, stream_info, None, None, None)
            return

        self.col_map = {}
        self.stats_means = {}
        self.stats_vars = {}
        self.patch_counter = 0

        # 1. Probe Source
        meta_src = self._probe_file(self.filename_source, is_source=True)
        if not meta_src:
            return

        self.lats_src = meta_src["lats"]
        self.lons_src = meta_src["lons"]
        self.spatial_indices_src = meta_src["indices"]
        self.coords_src = meta_src["coords"]
        self.grid_dims_src = meta_src["grid_dims"]

        # 2. Probe Target
        if self.filename_target != self.filename_source:
            meta_trg = self._probe_file(self.filename_target, is_source=False)
            if not meta_trg:
                return
            self.lats_trg = meta_trg["lats"]
            self.lons_trg = meta_trg["lons"]
            self.spatial_indices_trg = meta_trg["indices"]
            self.coords_trg = meta_trg["coords"]
            self.grid_dims_trg = meta_trg["grid_dims"]
        else:
            self.lats_trg = self.lats_src
            self.lons_trg = self.lons_src
            self.spatial_indices_trg = self.spatial_indices_src
            self.coords_trg = self.coords_src
            self.grid_dims_trg = self.grid_dims_src

        ds_time_values = meta_src["time"]
        self._len_cached = len(ds_time_values)
        self._time_values_cached = ds_time_values

        data_start_time = np.datetime64(ds_time_values[0], "ns")
        if len(ds_time_values) > 1:
            native_period = np.datetime64(ds_time_values[1], "ns") - data_start_time
        else:
            native_period = np.timedelta64(24, "h")

        self.native_period = native_period

        if "frequency" in stream_info:
            from weathergen.readers_extra.data_reader_grep import _str_to_timedelta

            period = _str_to_timedelta(stream_info["frequency"])
        else:
            period = native_period

        data_end_time = np.datetime64(ds_time_values[-1], "ns")

        if self.roi:
            self.roi_min_lon, self.roi_min_lat, self.roi_max_lon, self.roi_max_lat = self.roi
        else:
            self.roi_min_lon, self.roi_min_lat, self.roi_max_lon, self.roi_max_lat = (
                -180.0,
                -90.0,
                180.0,
                90.0,
            )

        self.available_channels = list(self.col_map.keys())

        super().__init__(tw_handler, stream_info, data_start_time, data_end_time, period)

        self.source_idx = self._select_channels("source")
        self.target_idx = self._select_channels("target")
        self.geoinfo_idx = []
        self.geoinfo_channels = []

        self.source_channels = [self.available_channels[i] for i in self.source_idx]
        self.target_channels = [self.available_channels[i] for i in self.target_idx]

        self._init_stats_arrays()

    def _probe_file(self, filepath, is_source=True):
        mapper = fsspec.get_mapper("reference://", fo=str(filepath), remote_protocol="file")
        try:
            with xr.open_dataset(mapper, engine="zarr", chunks={}, consolidated=False) as ds:
                if "time" not in ds.coords:
                    all_vars = list(ds.coords) + list(ds.data_vars)
                    time_candidates = [v for v in all_vars if "time" in v.lower()]
                    if time_candidates:
                        target = time_candidates[0]
                        if target in ds.data_vars:
                            ds = ds.set_coords(target)
                        if target != "time":
                            ds = ds.rename({target: "time"})
                        if "time" in ds.dims and "time" not in ds.indexes:
                            ds = ds.assign_coords(time=ds["time"].values)

                if "time" not in ds.coords:
                    _logger.error(f"No time coordinate in {filepath}.")
                    return None

                meta = {
                    "time": ds.time.values,
                    "col_map": self._parse_attr(ds.attrs, "weathergen_col_map"),
                    "means": self._parse_attr(ds.attrs, "weathergen_means"),
                    "vars": self._parse_attr(ds.attrs, "weathergen_vars"),
                }

                self.col_map.update(meta["col_map"])
                self.stats_means.update(meta["means"])
                self.stats_vars.update(meta["vars"])

                lats = ds["lat"].values if "lat" in ds else ds["lat_c"].values
                lons = ds["lon"].values if "lon" in ds else ds["lon_c"].values

                lats = np.nan_to_num(lats, nan=0.0).astype(np.float32)
                lons = np.nan_to_num(lons, nan=0.0).astype(np.float32)
                if np.any(lats > 90.0):
                    lats = lats - 90.0
                lats = np.clip(lats, -90.0, 90.0)
                lons = ((lons + 180.0) % 360.0) - 180.0

                if self.roi:
                    min_lon, min_lat, max_lon, max_lat = self.roi
                    if min_lon > max_lon:
                        mask = (lons >= min_lon) | (lons <= max_lon)
                    else:
                        mask = (lons >= min_lon) & (lons <= max_lon)
                    mask &= (lats >= min_lat) & (lats <= max_lat)
                    spatial_indices = np.where(mask)[0]
                    lats = lats[spatial_indices]
                    lons = lons[spatial_indices]
                else:
                    spatial_indices = np.arange(len(lats))

                meta["lats"] = lats
                meta["lons"] = lons
                meta["indices"] = spatial_indices
                meta["coords"] = np.stack([lats, lons], axis=1)

                # Detect grid structure for 2D regular sampling
                meta["grid_dims"] = None
                lat_dims = [d for d in ds.sizes if d.lower() in ["lat", "latitude"]]
                lon_dims = [d for d in ds.sizes if d.lower() in ["lon", "longitude"]]
                if lat_dims and lon_dims:
                    meta["grid_dims"] = (ds.sizes[lat_dims[0]], ds.sizes[lon_dims[0]])

                return meta
        except Exception as e:
            _logger.error(f"Failed to probe {filepath}: {e}")
            return None

    def _lazy_init(self):
        if self._initialized:
            return

        self.mapper_src = fsspec.get_mapper(
            "reference://", fo=str(self.filename_source), remote_protocol="file"
        )
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*separate the stored chunks.*")
            self.ds_source = xr.open_dataset(
                self.mapper_src, engine="zarr", chunks={}, decode_times=True, consolidated=False
            )

        if self.filename_target != self.filename_source:
            self.mapper_trg = fsspec.get_mapper(
                "reference://", fo=str(self.filename_target), remote_protocol="file"
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*separate the stored chunks.*")
                self.ds_target = xr.open_dataset(
                    self.mapper_trg, engine="zarr", chunks={}, decode_times=True, consolidated=False
                )
        else:
            self.ds_target = self.ds_source

        for ch in self.source_channels:
            var = self.col_map[ch]["var"]
            if var in self.ds_source:
                self._dask_arrays_src[ch] = self.ds_source[var].data

        for ch in self.target_channels:
            var = self.col_map[ch]["var"]
            if var in self.ds_target:
                self._dask_arrays_trg[ch] = self.ds_target[var].data

        self._initialized = True

    def _get_persistent_time_idxs(self, idx: TIndex) -> tuple[NDArray, DTRange]:
        dtr = self.time_window_handler.window(idx)
        if dtr.end < self.data_start_time or dtr.start > self.data_end_time:
            return (np.array([], dtype=np.int64), dtr)

        start_idx = np.searchsorted(self._time_values_cached, dtr.start, side="left")
        end_idx = np.searchsorted(self._time_values_cached, dtr.end - t_epsilon, side="right") - 1

        stride = 1
        if self.period > self.native_period:
            stride = int(self.period / self.native_period)

        if start_idx > end_idx:
            # Persistent: find last before window
            last_before = start_idx - 1
            if last_before >= 0:
                return (np.array([last_before], dtype=np.int64), dtr)
            else:
                return (np.array([], dtype=np.int64), dtr)

        # Generate indices and then subsample by stride
        idxs = np.arange(start_idx, end_idx + 1, dtype=np.int64)[::stride]

        return (idxs, dtr)

    @override
    def get_source(self, idx: TIndex) -> ReaderData:
        return self._fetch_data(idx, self.source_channels, is_source=True)

    @override
    def get_target(self, idx: TIndex) -> ReaderData:
        return self._fetch_data(idx, self.target_channels, is_source=False)

    def _fetch_data(self, idx: TIndex, channels: list[str], is_source: bool) -> ReaderData:
        self._lazy_init()
        (t_idxs, dtr) = self._get_persistent_time_idxs(idx)

        if len(t_idxs) == 0 or not channels:
            return ReaderData.empty(len(channels), 0)

        channel_indices = [self.available_channels.index(c) for c in channels]
        start_t, end_t = t_idxs[0], t_idxs[-1] + 1
        n_steps = len(t_idxs)

        stride = 1
        if self.period > self.native_period:
            stride = int(self.period / self.native_period)
            # extend end_t to cover the final step when striding
            end_t = t_idxs[-1] + stride

        spatial_indices_ref = self.spatial_indices_src if is_source else self.spatial_indices_trg
        coords_ref = self.coords_src if is_source else self.coords_trg
        ds_ref = self.ds_source if is_source else self.ds_target
        arr_cache = self._dask_arrays_src if is_source else self._dask_arrays_trg

        # Patching Seed Logic:
        # Use internal counter for 'staircase' stability OR sample index for variety.
        if self.patch_use_counter:
            patch_idx = self.patch_counter // self.patch_stability_window
            local_seed = patch_idx + 12345
        else:
            # Fallback to time-based index (Warning: sampler often seeds this per-rank!)
            local_seed = int(idx) + 12345
            patch_idx = int(idx)

        patch_rng = np.random.default_rng(local_seed)

        # Increment counter for next fetch
        self.patch_counter += 1

        if self.sampling_mode == "global_sparse":
            total_points = len(spatial_indices_ref)
            target_n = self.sample_points if self.sample_points else 4096
            indices_local = patch_rng.choice(
                total_points, size=min(target_n, total_points), replace=False
            )
            patch_coords_base = coords_ref[indices_local]
            final_disk_indices = spatial_indices_ref[indices_local]
            use_contiguous_read = False

        elif self.sampling_mode == "regular":
            grid_dims = self.grid_dims_src if is_source else self.grid_dims_trg
            if grid_dims:
                h, w = grid_dims
                n = self.sampling_step
                rows = spatial_indices_ref // w
                cols = spatial_indices_ref % w
                mask = (rows % n == 0) & (cols % n == 0)
                indices_local = np.where(mask)[0]
            else:
                total_points = len(spatial_indices_ref)
                indices_local = np.arange(0, total_points, self.sampling_step)

            patch_coords_base = coords_ref[indices_local]
            final_disk_indices = spatial_indices_ref[indices_local]
            use_contiguous_read = False

        elif self.patch_size_deg:
            lat_range = max(0.0, (self.roi_max_lat - self.roi_min_lat) - self.patch_size_deg)
            lon_range = max(0.0, (self.roi_max_lon - self.roi_min_lon) - self.patch_size_deg)

            patch_indices_local = np.array([])

            lat_0 = self.roi_min_lat + patch_rng.random() * lat_range
            lon_0 = self.roi_min_lon + patch_rng.random() * lon_range

            mask_src = (
                (self.lats_src >= lat_0)
                & (self.lats_src < lat_0 + self.patch_size_deg)
                & (self.lons_src >= lon_0)
                & (self.lons_src < lon_0 + self.patch_size_deg)
            )
            mask_trg = (
                (self.lats_trg >= lat_0)
                & (self.lats_trg < lat_0 + self.patch_size_deg)
                & (self.lons_trg >= lon_0)
                & (self.lons_trg < lon_0 + self.patch_size_deg)
            )

            patch_indices_local = np.where(mask_src if is_source else mask_trg)[0]

            patch_coords_base = (
                self.coords_src[patch_indices_local]
                if is_source
                else (self.coords_trg[patch_indices_local])
            )
            final_disk_indices = (
                self.spatial_indices_src[patch_indices_local]
                if is_source
                else (self.spatial_indices_trg[patch_indices_local])
            )
            use_contiguous_read = True

        else:
            final_disk_indices = self.spatial_indices_src if is_source else self.spatial_indices_trg
            patch_coords_base = self.coords_src if is_source else self.coords_trg
            use_contiguous_read = True

        if len(final_disk_indices) == 0:
            _logger.warning(
                f"[Stream {self._stream_info.get('name')}] NO POINTS FOUND for patch! Skipping."
            )
            return ReaderData.empty(len(channels), n_steps)

        if use_contiguous_read:
            disk_start, disk_stop = np.min(final_disk_indices), np.max(final_disk_indices) + 1
            rel_indices = final_disk_indices - disk_start
            data_block = self._load_block_from_ds(
                ds_ref,
                arr_cache,
                channel_indices,
                start_t,
                end_t,
                stride,
                n_steps,
                slice(disk_start, disk_stop),
                rel_indices,
            )
        else:
            data_block = self._load_block_from_ds(
                ds_ref,
                arr_cache,
                channel_indices,
                start_t,
                end_t,
                stride,
                n_steps,
                final_disk_indices,
                None,
            )

        if data_block.size > 0:
            d_max = np.nanmax(np.abs(data_block))
            if d_max > 1e10:
                data_block[np.abs(data_block) > 1e10] = np.nan

        coords_flat = np.tile(patch_coords_base, (n_steps, 1))
        dt_values = self._time_values_cached[start_t:end_t:stride]
        dt_flat = np.repeat(dt_values, patch_coords_base.shape[0])

        if data_block.size > 0:
            # Check for NaNs across any channel
            valid_mask = ~np.isnan(data_block).any(axis=1)

            # Check for filler values across any channel
            if self.filler_values:
                valid_mask &= ~np.isin(data_block, self.filler_values).any(axis=1)

            data_block = data_block[valid_mask]
            coords_flat = coords_flat[valid_mask]
            dt_flat = dt_flat[valid_mask]

        if data_block.size == 0:
            _logger.warning(
                f"[Stream {self._stream_info.get('name')}] "
                "All points were filtered out (NaNs or filler values). Skipping."
            )
            return ReaderData.empty(len(channels), n_steps)

        rdata = ReaderData(
            coords=coords_flat,
            geoinfos=np.zeros((len(data_block), 0), dtype=np.float32),
            data=data_block,
            datetimes=dt_flat,
        )
        return rdata

    def _load_block_from_ds(
        self, ds, arr_cache, indices, start_t, end_t, stride, n_steps, disk_indices, rel_indices
    ) -> np.typing.NDArray:
        if rel_indices is not None:
            num_points = len(rel_indices)
        else:
            num_points = len(disk_indices)

        if not indices:
            return np.zeros((n_steps * num_points, 0), dtype=np.float32)

        output_block = np.zeros((n_steps * num_points, len(indices)), dtype=np.float32)

        with dask.config.set(scheduler="single-threaded"):
            for i, idx in enumerate(indices):
                ch_name = self.available_channels[idx]
                if ch_name not in arr_cache:
                    info = self.col_map[ch_name]
                    if info["var"] in ds:
                        arr_cache[ch_name] = ds[info["var"]].data
                    else:
                        continue

                info = self.col_map[ch_name]
                base_arr = arr_cache[ch_name]
                dims = ds[info["var"]].dims
                # 1. Apply Vertical Level Selection
                sliced = base_arr
                if info["sel"]:
                    sls = [slice(None)] * sliced.ndim
                    for d, val in info["sel"].items():
                        if d in dims:
                            sls[dims.index(d)] = val
                    sliced = sliced[tuple(sls)]

                # 2. Slice Time (keeps memory small before we flatten)
                if "time" in dims:
                    sliced = sliced[start_t:end_t:stride]

                # 3. Compute the block into memory
                chunk = sliced.compute().astype(np.float32)

                # 4. FLATTEN THE SPATIAL DIMENSIONS FIRST (Crucial for 2D Grids)
                if chunk.ndim > 1:
                    if "time" in dims:
                        # (time, lat, lon) -> (time, nodes)
                        chunk = chunk.reshape(chunk.shape[0], -1)
                    else:
                        # (lat, lon) -> (nodes)
                        chunk = chunk.reshape(-1)

                # 5. NOW apply the spatial indices (which are 1D flat indices)
                if rel_indices is not None:
                    if "time" in dims:
                        # Contiguous read: Apply raw disk bounds, then rel_indices
                        chunk = chunk[:, disk_indices]

                        # Safety check: if chunk is completely empty, fill with NaNs
                        if chunk.shape[1] == 0:
                            assert False, "Empty chunk after disk indexing with time dimension"

                        else:
                            chunk = chunk[:, rel_indices]
                    else:
                        chunk = chunk[disk_indices]
                        if chunk.size == 0:
                            assert False, "Empty chunk after disk indexing with rel_indices"
                        else:
                            chunk = chunk[rel_indices]
                        chunk = np.repeat(np.expand_dims(chunk, 0), n_steps, axis=0)
                else:
                    # Fancy Indexing (Sparse Global)
                    if "time" in dims:
                        chunk = chunk[:, disk_indices]
                    else:
                        chunk = chunk[disk_indices]
                        chunk = np.repeat(np.expand_dims(chunk, 0), n_steps, axis=0)

                # 6. Apply Land Masks
                chunk[(chunk <= -9000.0)] = np.nan
                chunk[~np.isfinite(chunk)] = np.nan
                output_block[:, i] = chunk.reshape(-1)

        return output_block

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        raise NotImplementedError("DataReaderMesh._get should not be called directly.")

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self._len_cached = 0

    @override
    def length(self) -> int:
        return self._len_cached

    def _parse_attr(self, attrs, key):
        val = attrs.get(key, {})
        return json.loads(val) if isinstance(val, str) else val

    def _select_channels(self, type_key: str) -> list[int]:
        select = self._stream_info.get(type_key)
        exclude = self._stream_info.get(f"{type_key}_exclude", [])
        return [
            i
            for i, ch in enumerate(self.available_channels)
            if (not select or any(s in ch for s in select)) and not any(e in ch for e in exclude)
        ]

    def _init_stats_arrays(self):
        self.mean = np.zeros(len(self.available_channels), dtype=np.float32)
        self.stdev = np.ones(len(self.available_channels), dtype=np.float32)
        for i, ch in enumerate(self.available_channels):
            mu = self.stats_means.get(ch, 0.0)
            var = self.stats_vars.get(ch, 1.0)
            if mu is None or np.isnan(mu) or np.isinf(mu):
                mu = 0.0
            if var is None or np.isnan(var) or np.isinf(var) or var < 1e-7:
                var = 1.0
            self.mean[i] = mu
            self.stdev[i] = np.sqrt(var)
        self.mean_geoinfo = np.zeros(0, dtype=np.float32)
        self.stdev_geoinfo = np.ones(0, dtype=np.float32)

    @override
    def normalize_source_channels(self, source: np.typing.NDArray) -> np.typing.NDArray:
        norm = (source - self.mean[self.source_idx]) / self.stdev[self.source_idx]
        return np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    @override
    def normalize_target_channels(self, target: np.typing.NDArray) -> np.typing.NDArray:
        norm = (target - self.mean[self.target_idx]) / self.stdev[self.target_idx]
        return np.nan_to_num(norm, nan=np.nan, posinf=np.nan, neginf=np.nan).astype(np.float32)

    @override
    def denormalize_source_channels(self, source):
        if isinstance(source, torch.Tensor):
            stdev = torch.tensor(
                self.stdev[self.source_idx], dtype=source.dtype, device=source.device
            )
            mean = torch.tensor(
                self.mean[self.source_idx], dtype=source.dtype, device=source.device
            )
            land_mask = source == 0.0
            denorm = (source * stdev) + mean
            denorm[land_mask] = torch.nan
            return denorm

        land_mask = source == 0.0
        denorm = (source * self.stdev[self.source_idx]) + self.mean[self.source_idx]
        denorm[land_mask] = np.nan
        return denorm

    @override
    def denormalize_target_channels(self, data):
        if isinstance(data, torch.Tensor):
            stdev = torch.tensor(self.stdev[self.target_idx], dtype=data.dtype, device=data.device)
            mean = torch.tensor(self.mean[self.target_idx], dtype=data.dtype, device=data.device)
            return (data * stdev) + mean
        return (data * self.stdev[self.target_idx]) + self.mean[self.target_idx]

    @override
    def normalize_geoinfos(self, geoinfos: np.typing.NDArray) -> np.typing.NDArray:
        norm = (geoinfos - self.mean_geoinfo) / self.stdev_geoinfo
        return np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
