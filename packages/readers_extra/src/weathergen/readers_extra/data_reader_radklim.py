# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import hashlib
import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import override

import netCDF4 as nc
import numpy as np
import yaml

from weathergen.datasets.data_reader_base import (
    NPDT64,
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)

_logger = logging.getLogger(__name__)

_INDEX_SCAN_WORKERS = 32
# bump when the index format or the file selection changes, to invalidate old caches
_INDEX_CACHE_VERSION = 2


def _scan_nc_file(path: Path) -> dict | None:
    """Read the time range of one RADKLIM file for parallelization."""

    with nc.Dataset(path, "r") as ds:
        if "RR" not in ds.variables:
            return None

        time_var = ds.variables["time"]
        if len(time_var) == 0:
            return None

        times = nc.num2date(time_var[:], time_var.units, time_var.calendar)

    start_time, end_time = times[0], times[-1]

    return {
        "path": path,
        "start": np.datetime64(start_time),
        "end": np.datetime64(end_time),
        "year": start_time.year,
        "month": start_time.month,
    }


class DataReaderRadklim(DataReaderTimestep):
    """
    Data reader for RADKLIM dataset in netCDF format.

    RADKLIM provides hourly precipitation estimates over Germany based on weather radar data.
    Data is organized in monthly netCDF files across multiple years.
    """

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
        **kwargs,
    ) -> None:
        """
        Construct data reader for RADKLIM dataset

        Parameters
        ----------
        tw_handler :
            time window handler
        filename :
            base directory containing year subdirectories with netCDF files
        stream_info :
            information about stream

        Returns
        -------
        None
        """

        self.base_path = Path(filename)
        if not self.base_path.exists():
            raise FileNotFoundError(f"RADKLIM base path not found: {self.base_path}")

        # Build index on the fly (takes ~2 seconds on JUPITER)
        _logger.info(f"Building file index for RADKLIM data in {self.base_path}")
        self.file_index = self._build_file_index()

        if not self.file_index:
            name = stream_info["name"]
            _logger.warning(f"No RADKLIM files found in {self.base_path}. Stream {name} is empty.")
            super().__init__(tw_handler, stream_info)
            self.init_empty()
            return

        # Determine data period from file index
        data_start_time = self.file_index[0]["start"]
        data_end_time = self.file_index[-1]["end"]

        # Frequency is hourly
        period = np.timedelta64(1, "h")

        # Load spatial grid from first file
        first_file = self.file_index[0]["path"]

        # Get spatial subsampling stride from stream_info (default: 1 = no subsampling)
        self.spatial_stride = stream_info.get("spatial_stride", 1)

        with nc.Dataset(first_file, "r") as ds:
            # Read lat/lon arrays (2D)
            lat_2d = ds.variables["lat"][:]
            lon_2d = ds.variables["lon"][:]

            # Apply spatial subsampling if stride > 1
            if self.spatial_stride > 1:
                lat_2d = lat_2d[::self.spatial_stride, ::self.spatial_stride]
                lon_2d = lon_2d[::self.spatial_stride, ::self.spatial_stride]
                _logger.info(f"Applied spatial stride {self.spatial_stride}: grid reduced to {lat_2d.shape}")

            # Flatten and create coordinate template
            lat_flat = lat_2d.flatten()
            lon_flat = lon_2d.flatten()
            self.coords_template = np.stack([lat_flat, lon_flat], axis=1).astype(np.float32)
            self.n_grid_points = len(self.coords_template)

            # Store grid dimensions
            self.grid_shape = lat_2d.shape  # (ny, nx)

        # Initialize parent class
        super().__init__(
            tw_handler,
            stream_info,
            data_start_time,
            data_end_time,
            period,
        )

        # If there is no overlap with the time range, the dataset will be empty
        if tw_handler.t_start >= data_end_time or tw_handler.t_end <= data_start_time:
            name = stream_info["name"]
            _logger.warning(f"{name} is not supported over data loader window. Stream is skipped.")
            self.init_empty()
            return

        # Calculate total length in hours
        total_hours = int((data_end_time - data_start_time) / np.timedelta64(1, "h")) + 1
        self.len = total_hours

        # Mean and standard deviation (approximate, if no statistics.yml exists)
        self.mean = np.array([0.1], dtype=np.float32)  # ~0.1 mm/hr average
        self.stdev = np.array([1.0], dtype=np.float32)  # ~1.0 mm/hr std dev

        stats_file = self.base_path / "statistics.yml"
        calc_stats = stream_info.get("calc_stats", False)

        if stats_file.exists():     # read saved mean and stdev
            _logger.info(f"Loading actual statistics from {stats_file}")
            with stats_file.open("r") as f:
                statistics = yaml.safe_load(f)
            
            if "mean" in statistics and statistics["mean"] is not None:
                self.mean = np.array([statistics["mean"]], dtype=np.float32)

            if "stdev" in statistics and statistics["stdev"] is not None:
                self.stdev = np.array([statistics["stdev"]], dtype=np.float32)
        elif calc_stats:        # calculate mean and stdev
            _logger.info("statistics.yml not found and calc_stats=True. Calculating statistics...")
            mean, stdev = self._calculate_and_save_statistics(stats_file)
            self.mean = np.array([mean], dtype=np.float32)
            self.stdev = np.array([stdev], dtype=np.float32)
            
        else:                   # no statistics.yml and calc_stats=False
            _logger.info(f"No statistics.yml found and calc_stats=False. Falling back to default mean (0.1) and stdev (1.0).")

        # Select channels (use netCDF variable name)
        available_channels = ["RR"]

        s_chs = stream_info.get("source")
        s_chs_exclude = stream_info.get("source_exclude", [])
        t_chs = stream_info.get("target")
        t_chs_exclude = stream_info.get("target_exclude", [])

        self.source_channels = self.select_channels(available_channels, s_chs, s_chs_exclude)
        self.source_idx = np.array(
            [i for i, ch in enumerate(available_channels) if ch in self.source_channels],
            dtype=np.int64,
        )

        self.target_channels = self.select_channels(available_channels, t_chs, t_chs_exclude)
        self.target_idx = np.array(
            [i for i, ch in enumerate(available_channels) if ch in self.target_channels],
            dtype=np.int64,
        )

        # No geoinfo for now
        self.geoinfo_channels = []
        self.geoinfo_idx = []
        self.mean_geoinfo = np.zeros(0)
        self.stdev_geoinfo = np.ones(0)

        # Get target channel weights
        self.target_channel_weights = self.parse_target_channel_weights()

        # File caching
        self.current_file = None
        self.current_filepath = None
        self.current_file_times = None

        ds_name = stream_info["name"]
        _logger.info(f"{ds_name}: source channels: {self.source_channels}")
        _logger.info(f"{ds_name}: target channels: {self.target_channels}")
        _logger.info(f"{ds_name}: data period: {data_start_time} to {data_end_time}")
        _logger.info(f"{ds_name}: frequency: {period}")
        _logger.info(f"{ds_name}: grid size: {self.grid_shape[0]} x {self.grid_shape[1]}")
        _logger.info(f"{ds_name}: number of files: {len(self.file_index)}")

    def _build_file_index(self) -> list[dict]:
        """
        Build index mapping time ranges to file paths.

        Files are scanned in parallel and the index is cached on disk. The cache is reused as
        long as the set of netCDF files under base_path is unchanged.

        Returns
        -------
        file_index :
            List of dicts with 'path', 'start', 'end', 'year', 'month'
        """
        # rglob recursively finds all .nc files
        nc_files = sorted(self.base_path.rglob("*.nc"))

        if not nc_files:
            _logger.warning(f"No .nc files found in {self.base_path}")
            return []

        cache_file = self._index_cache_path()
        file_index = self._load_index_cache(cache_file, nc_files)
        if file_index is not None:
            return file_index

        _logger.info(f"Building file index for {len(nc_files)} files...")

        file_index = []
        # processes instead of threads: netCDF4/HDF5 is not thread-safe
        n_workers = min(_INDEX_SCAN_WORKERS, os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_scan_nc_file, path): path for path in nc_files}
            for i, future in enumerate(as_completed(futures), start=1):
                if i % 1000 == 0:
                    _logger.info(f"Scanned {i}/{len(nc_files)} files...")

                try:
                    entry = future.result()
                except Exception as e:
                    _logger.warning(f"Could not read file {futures[future]} for indexing: {e}")
                    continue

                if entry is not None:
                    file_index.append(entry)

        n_skipped = len(nc_files) - len(file_index)
        if n_skipped > 0:
            _logger.info(f"Skipped {n_skipped} files without RR data or time steps.")

        file_index.sort(key=lambda x: x["start"])

        self._save_index_cache(cache_file, nc_files, file_index)

        return file_index

    def _index_cache_path(self) -> Path | None:
        """Return the cache file for this dataset directory, or None if no cache dir is usable."""

        xdg = os.environ.get("XDG_CACHE_HOME")
        cache_dir = (Path(xdg) if xdg else Path.home() / ".cache") / "weathergen"
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            _logger.warning(f"Could not create index cache dir {cache_dir}: {e}")
            return None

        # one cached file per dataset directory
        key = hashlib.sha1(str(self.base_path.resolve()).encode()).hexdigest()[:16]

        return cache_dir / f"radklim_index_{key}.pkl"

    @staticmethod
    def _load_index_cache(cache_file: Path | None, nc_files: list[Path]) -> list[dict] | None:
        """Return the cached index if it was built from exactly nc_files, otherwise None."""

        if cache_file is None or not cache_file.exists():
            return None

        try:
            with cache_file.open("rb") as f:
                cached = pickle.load(f)
        except Exception as e:
            _logger.warning(f"Failed to load index cache {cache_file} ({e}), rebuilding.")
            return None

        if (
            not isinstance(cached, dict)
            or cached.get("version") != _INDEX_CACHE_VERSION
            or cached.get("files") != nc_files
        ):
            _logger.info(f"Index cache {cache_file} is outdated, rebuilding.")

            return None

        _logger.info(f"Loaded cached file index from {cache_file}")
        
        return cached["index"]

    @staticmethod
    def _save_index_cache(
        cache_file: Path | None, nc_files: list[Path], file_index: list[dict]
    ) -> None:
        """Write the index cache atomically, so concurrent ranks never read a partial file."""

        if cache_file is None:
            return

        tmp_file = cache_file.with_suffix(f".{os.getpid()}.tmp")
        try:
            with tmp_file.open("wb") as f:
                pickle.dump(
                    {"version": _INDEX_CACHE_VERSION, "files": nc_files, "index": file_index}, f
                )
            tmp_file.replace(cache_file)
        except (OSError, pickle.PicklingError) as e:
            _logger.warning(f"Failed to save index cache {cache_file}: {e}")
            tmp_file.unlink(missing_ok=True)

            return

        _logger.info(f"Saved file index cache to {cache_file}")

    def _get_files_for_time_range(self, start: NPDT64, end: NPDT64) -> list[dict]:
        """
        Find all files that overlap with given time range

        Parameters
        ----------
        start :
            start of time range
        end :
            end of time range

        Returns
        -------
        files :
            List of file index entries that overlap with time range
        """
        files = []
        for entry in self.file_index:
            if entry["start"] >= end:
                break
            if entry["end"] >= start:
                files.append(entry)

        return files

    def _open_file(self, filepath: Path, cache: bool = True) -> tuple[nc.Dataset, np.ndarray]:
        """
        Open netCDF file and decode its timestamps, using cache if possible

        Parameters
        ----------
        filepath :
            path to netCDF file
        cache :
            whether to cache the opened file

        Returns
        -------
        dataset, file_times_dt64 :
            opened netCDF dataset and 1D array of np.datetime64 timestamps
        """
        # Use cached file if same path
        if (
            cache
            and getattr(self, "current_filepath", None) == filepath
            and getattr(self, "current_file", None) is not None
            and getattr(self, "current_file_times", None) is not None
        ):
            return self.current_file, self.current_file_times

        if getattr(self, "current_file", None) is not None:
            try:
                self.current_file.close()
            except Exception:
                pass  # Ignore errors when closing

        # Open new file and cache decoded timestamps
        ds = nc.Dataset(filepath, "r")
        time_var = ds.variables["time"]
        file_times = nc.num2date(time_var[:], time_var.units, time_var.calendar)
        file_times_dt64 = np.array([np.datetime64(t) for t in file_times])

        self.current_file = ds
        self.current_filepath = filepath
        self.current_file_times = file_times_dt64

        return self.current_file, self.current_file_times

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self.len = 0
        self.coords_template = np.zeros((0, 2), dtype=np.float32)
        self.n_grid_points = 0
        self.grid_shape = (0, 0)
        self.file_index = []

    @override
    def length(self) -> int:
        return self.len

    def select_channels(
        self, colnames: list[str], cols_select: list[str] | None, cols_exclude: list[str] | None
    ) -> list[str]:
        """
        Select channels based on include/exclude lists

        Parameters
        ----------
        colnames :
            available channel names
        cols_select :
            list of patterns to include (None = all)
        cols_exclude :
            list of patterns to exclude

        Returns
        -------
        selected_colnames :
            filtered list of channel names
        """
        return [
            c
            for c in colnames
            if (not cols_select or any(c_sel in c for c_sel in cols_select))
            and not (cols_exclude and any(c_excl in c for c_excl in cols_exclude))
        ]

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """
        Get data for window, handling multi-file scenarios

        Parameters
        ----------
        idx : int
            Index of temporal window
        channels_idx : np.array
            Selection of channels (0 for rainfall)

        Returns
        -------
        ReaderData providing coords, geoinfos, data, datetimes
        """

        # Get dataset time indices for the window
        (t_idxs, dtr) = self._get_dataset_idxs(idx)

        # Return empty if no valid data or no channels selected
        if self.len == 0 or len(t_idxs) == 0 or len(channels_idx) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx),
                num_geo_fields=len(self.geoinfo_idx),
            )

        # Find files that contain data for this time range
        files_needed = self._get_files_for_time_range(dtr.start, dtr.end)

        if not files_needed:
            _logger.warning(f"RADKLIM: No files found for time range {dtr.start} to {dtr.end}")
            return ReaderData.empty(
                num_data_fields=len(channels_idx),
                num_geo_fields=len(self.geoinfo_idx),
            )

        # Load data from each file
        data_chunks = []
        coords_chunks = []
        datetime_chunks = []

        for file_entry in files_needed:
            try:
                ds, file_times_dt64 = self._open_file(file_entry["path"])

                # Find timesteps within requested window
                mask = (file_times_dt64 >= dtr.start) & (file_times_dt64 < dtr.end)
                time_indices = np.where(mask)[0]

                if len(time_indices) > 0:
                    # Load rainfall data for these times
                    # Shape: (n_times, ny, nx)
                    rr_data = ds.variables["RR"][time_indices, :, :]

                    # Apply spatial subsampling if stride > 1
                    if self.spatial_stride > 1:
                        rr_data = rr_data[:, ::self.spatial_stride, ::self.spatial_stride]

                    # Convert masked array to regular array, replacing fill values with NaN
                    if np.ma.is_masked(rr_data):
                        rr_data = np.ma.filled(rr_data, np.nan)
                    else:
                        # Replace fill value (999.0) with NaN
                        rr_data = np.where(rr_data >= 900.0, np.nan, rr_data)

                    # Flatten spatial dimensions - (n_times, ny, nx) to (n_times * ny * nx, 1)
                    # base classes force this to be 2D
                    data_flat = rr_data.reshape(-1, 1)
                    data_chunks.append(data_flat.astype(np.float32))

                    # Repeat coords for each timestep
                    coords_chunk = (
                        self.coords_template
                        if len(time_indices) == 1
                        else np.tile(self.coords_template, (len(time_indices), 1))
                    )
                    coords_chunks.append(coords_chunk)

                    # Repeat datetimes for each grid point
                    datetimes_chunk = np.repeat(file_times_dt64[time_indices], self.n_grid_points)
                    datetime_chunks.append(datetimes_chunk)

            except Exception as e:
                _logger.error(f"Error reading file {file_entry['path']}: {e}")
                continue

        # Check if we got any data
        if not data_chunks:
            _logger.warning(f"RADKLIM: No valid data found for time range {dtr.start} to {dtr.end}")
            return ReaderData.empty(
                num_data_fields=len(channels_idx),
                num_geo_fields=len(self.geoinfo_idx),
            )

        # Concatenate all chunks
        data = np.concatenate(data_chunks, axis=0)
        coords = np.concatenate(coords_chunks, axis=0)
        datetimes = np.concatenate(datetime_chunks, axis=0)

        # Empty geoinfos for now
        geoinfos = np.zeros((len(data), 0), dtype=np.float32)

        # Create ReaderData
        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )

        check_reader_data(rd, dtr)

        return rd

    @staticmethod
    def _compute_file_stats(filepath: Path, spatial_stride: int = 1):
        """Standalone worker function for multiprocessing dataset statistics."""
        try:
            with nc.Dataset(filepath, "r") as ds:
                rr_data = ds.variables["RR"][:]

            if spatial_stride > 1:
                rr_data = rr_data[:, ::spatial_stride, ::spatial_stride]

            # Flatten and remove masked/invalid values
            if np.ma.is_masked(rr_data):
                valid_data = rr_data.compressed()
            else:
                valid_data = rr_data.flatten()

            # Filter out RADKLIM fill values and NaNs
            valid_data = valid_data[valid_data < 900.0]
            valid_data = valid_data[~np.isnan(valid_data)]

            # Accumulate sums (using float64 to prevent overflow)
            valid_data_64 = valid_data.astype(np.float64)

            return (
                float(np.sum(valid_data_64)),
                float(np.sum(valid_data_64**2)),
                int(len(valid_data_64)),
            )
        except Exception as e:
            return (filepath, str(e))

    def _calculate_and_save_statistics(self, stats_file: Path) -> tuple[float, float]:
        """
        Calculates the global mean and standard deviation across all valid RADKLIM
        data points and saves it to a YAML file using multiprocessing.
        """
        total_sum = 0.0
        total_sq_sum = 0.0
        count = 0
        _logger.info(
            f"Starting to process {len(self.file_index)} files for statistics using multiprocessing..."
        )
        filepaths = [entry["path"] for entry in self.file_index]
        n_workers = min(_INDEX_SCAN_WORKERS, os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(self._compute_file_stats, path, self.spatial_stride): path
                for path in filepaths
            }
            for i, future in enumerate(as_completed(futures), start=1):
                if i % 25 == 0:
                    _logger.info(f"Processed {i}/{len(filepaths)} files...")

                result = future.result()
                if len(result) == 3:
                    f_sum, f_sq_sum, f_count = result
                    total_sum += f_sum
                    total_sq_sum += f_sq_sum
                    count += f_count
                else:
                    path, err = result
                    _logger.warning(f"Could not read {path} during statistics calculation: {err}")

        if count == 0:
            raise ValueError("No valid data found to calculate statistics.")

        # Calculate population mean and standard deviation
        mean = float(total_sum / count)
        variance = (total_sq_sum / count) - (mean**2)
        stdev = float(np.sqrt(max(variance, 0.0)))

        stats = {
            "mean": mean,
            "stdev": stdev,
        }

        tmp_file = stats_file.with_suffix(f".{os.getpid()}.tmp")
        try:
            with tmp_file.open("w") as f:
                yaml.dump(stats, f, default_flow_style=False)
            tmp_file.replace(stats_file)
            _logger.info(f"Successfully calculated and saved statistics: {stats}")
        except OSError as e:
            _logger.warning(f"Failed to save statistics file {stats_file}: {e}")
            tmp_file.unlink(missing_ok=True)

        return mean, stdev
    
    def __del__(self):
        """Close any open files"""
        if hasattr(self, "current_file") and self.current_file is not None:
            try:
                self.current_file.close()
            except Exception:
                pass  # Ignore errors during cleanup