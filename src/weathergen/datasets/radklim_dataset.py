#!/usr/bin/env python

import re
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import xarray as xr

class RadklimDataset:
    """
    A Dataset class for loading and processing monthly RADKLIM NetCDF files.

    This dataloader is designed to work with the DWD RADKLIM radar-based
    precipitation climatology data, specifically the monthly aggregated files
    typically named like 'RW_YYYY.VVV_YYYYMM.nc' (e.g., 'RW_2017.002_201210.nc').
    It loads specified time windows of precipitation data ('RR' variable),
    normalizes it using provided statistics, and returns NumPy arrays.

    Attributes:
        data_dir (Path): The root directory containing year subdirectories
                         which in turn hold the monthly NetCDF files.
        stats_json_path (Path): Path to the JSON file containing normalization
                                statistics (mean and std for the 'RR' variable).
        start_datetime (datetime): The beginning of the overall period from which
                                   to draw samples.
        end_datetime (datetime): The end of the overall period (exclusive for the
                                 end of the last possible sample) from which
                                 to draw samples.
        sample_length_hours (int): The duration of each sample in hours.
        sample_step_hours (int): The time step between the start of consecutive
                                 samples, in hours.
        verbose (bool): If True, enables print statements for debugging.
        xr_parallel (bool): If True, enables parallel I/O in xarray.open_mfdataset
                            (requires a Dask cluster setup).
    """

    def __init__(
        self,
        data_dir: Path,
        stats_json: Path,
        start_datetime: datetime,
        end_datetime: datetime,
        sample_length_hours: int,
        sample_step_hours: Optional[int] = None,
        primary_file_pattern: str = "RW_*.nc", # Pattern to identify main Radklim files
        verbose: bool = False,
        xr_parallel: bool = False,
    ):
        """
        Initializes the RadklimDataset.

        Args:
            data_dir (Path): Root directory for Radklim data (e.g., containing year folders).
            stats_json (Path): Path to the JSON file with 'RR' mean/std statistics.
            start_datetime (datetime): Start datetime for the dataset period.
            end_datetime (datetime): End datetime for the dataset period.
            sample_length_hours (int): Length of each data sample in hours.
            sample_step_hours (Optional[int]): Step between samples in hours.
                                               Defaults to `sample_length_hours`.
            primary_file_pattern (str): Glob pattern to identify the primary Radklim files
                                        (e.g., "RW_*.nc" or "RW_2017.002_*.nc").
                                        This helps filter out other NetCDF files if present.
            verbose (bool): Enables verbose logging for debugging.
            xr_parallel (bool): Enables parallel file opening in xarray.
        """
        self.data_dir = Path(data_dir)
        self.stats_json_path = Path(stats_json)
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime # This is the exclusive end for data to be included
        self.sample_length_hours = sample_length_hours
        self.sample_step_hours = sample_step_hours or sample_length_hours
        self.primary_file_pattern = primary_file_pattern
        self.verbose = verbose
        self.xr_parallel = xr_parallel

        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"Data directory not found: {self.data_dir}")
        if not self.stats_json_path.is_file():
            raise FileNotFoundError(f"Statistics JSON file not found: {self.stats_json_path}")
        if self.sample_length_hours <= 0:
            raise ValueError("sample_length_hours must be positive.")
        if self.sample_step_hours <= 0:
            raise ValueError("sample_step_hours must be positive.")

        self._load_normalization_stats()
        self._select_and_load_files()
        self._prepare_time_axis_and_samples()

    def _load_normalization_stats(self) -> None:
        """Loads mean and standard deviation for 'RR' from the stats JSON file."""
        if self.verbose:
            print(f"[RadklimDataset] Loading stats from: {self.stats_json_path}")
        with open(self.stats_json_path, 'r') as f:
            stats_content = json.load(f)
        
        if 'RR' not in stats_content:
            raise ValueError("Statistics for 'RR' variable not found in stats JSON. Expected a top-level 'RR' key.")
        if 'mean' not in stats_content['RR'] or 'std' not in stats_content['RR']:
             raise ValueError("Stats JSON for 'RR' must contain 'mean' and 'std' sub-keys.")
        
        self.rr_mean = float(stats_content['RR']['mean'])
        self.rr_std = float(stats_content['RR']['std'])

        if self.rr_std == 0:
            if self.verbose:
                print("[RadklimDataset] Warning: Standard deviation from stats is 0. "
                      "Using std=1e-6 to prevent division by zero errors.")
            self.rr_std = 1e-6
        if self.verbose:
            print(f"[RadklimDataset] Stats loaded: RR mean={self.rr_mean}, RR std={self.rr_std}")

    def _collect_relevant_files(self) -> List[str]:
        """
        Collects paths to NetCDF files within the data directory that match
        the primary file pattern and whose monthly period overlaps with the
        requested [start_datetime, end_datetime] range.
        Assumes filenames contain YYYYMM (e.g., '..._201210.nc').
        """
        all_matching_files = list(self.data_dir.rglob(self.primary_file_pattern))
        if self.verbose:
            print(f"[RadklimDataset] Found {len(all_matching_files)} files matching pattern '{self.primary_file_pattern}' "
                  f"in {self.data_dir} and subdirectories.")

        valid_files_for_period = []
        # Regex to extract YYYYMM from typical Radklim filenames
        # e.g., RW_2017.002_201210.nc -> finds 2012 and 10
        date_pattern = re.compile(r'(\d{4})(\d{2})(?=(?:_|\.)nc)')

        for file_path in sorted(all_matching_files):
            file_name = file_path.name
            match = date_pattern.search(file_name)
            
            if not match:
                if self.verbose:
                    print(f"[RadklimDataset] Skipping '{file_name}': could not parse YYYYMM from filename.")
                continue

            year_str, month_str = match.group(1), match.group(2)
            try:
                file_year = int(year_str)
                file_month = int(month_str)
                
                file_month_start = datetime(file_year, file_month, 1)
                if file_month == 12:
                    file_month_end = datetime(file_year + 1, 1, 1) - timedelta(microseconds=1)
                else:
                    file_month_end = datetime(file_year, file_month + 1, 1) - timedelta(microseconds=1)
            except ValueError:
                if self.verbose:
                    print(f"[RadklimDataset] Invalid year/month '{year_str}{month_str}' in '{file_name}'.")
                continue

            # Check if the file's month overlaps with the requested datetime range
            if file_month_start <= self.end_datetime and file_month_end >= self.start_datetime:
                valid_files_for_period.append(str(file_path))
                if self.verbose:
                    print(f"[RadklimDataset] Including '{file_name}' (covers {file_month_start.date()} to {file_month_end.date()})")
            elif self.verbose:
                print(f"[RadklimDataset] Excluding '{file_name}': its month does not overlap with requested period.")
        
        if self.verbose and not valid_files_for_period and all_matching_files:
            print(f"[RadklimDataset] All {len(all_matching_files)} files matching pattern were outside the date range.")
        
        return valid_files_for_period

    def _select_and_load_files(self) -> None:
        """
        Selects relevant files and opens them as a single xarray Dataset,
        then slices it to the precise start_datetime and end_datetime.
        """
        self.selected_files = self._collect_relevant_files()

        if not self.selected_files:
            raise FileNotFoundError(
                f"No suitable NetCDF files found in '{self.data_dir}' (matching "
                f"'{self.primary_file_pattern}') for the period "
                f"[{self.start_datetime.strftime('%Y-%m-%d')} to "
                f"{self.end_datetime.strftime('%Y-%m-%d')}]."
            )
        if self.verbose:
            print(f"[RadklimDataset] Opening mfdataset with {len(self.selected_files)} selected files...")

        try:
            self.dataset = xr.open_mfdataset(
                self.selected_files,
                combine='by_coords',
                parallel=self.xr_parallel,
                engine='netcdf4'
            )
        except Exception as e:
            if self.verbose:
                print(f"[RadklimDataset] Error during xr.open_mfdataset with files: {self.selected_files[:3]}...")
            raise IOError(f"Failed to open NetCDF files with xarray: {e}") from e
        
        if self.verbose:
            print(f"[RadklimDataset] Raw mfdataset time range: {self.dataset.time.min().values} to {self.dataset.time.max().values}")

        # Slice to the exact requested period
        # xarray's slice is inclusive of start and end if exact matches are found in the coordinates.
        # If not, it behaves like standard Python slicing for sorted coordinates.
        self.dataset = self.dataset.sel(time=slice(self.start_datetime, self.end_datetime))
        self.dataset = self.dataset.sortby('time') # Crucial for consistent data

        if self.verbose:
            if self.dataset.time.size > 0:
                print(f"[RadklimDataset] Dataset sliced to time range: "
                      f"{self.dataset.time.min().values} to {self.dataset.time.max().values}")
            else:
                print(f"[RadklimDataset] Warning: No data remains after slicing to the precise requested time range. "
                      f"Dataset will be empty.")
                      
        if 'RR' not in self.dataset:
             raise ValueError("'RR' variable not found in the dataset after time selection and loading.")
        self.rr_data_array = self.dataset['RR']


    def _prepare_time_axis_and_samples(self) -> None:
        """
        Extracts the time axis and builds an index of valid sample start points.
        """
        if self.dataset.time.size == 0:
            if self.verbose:
                print("[RadklimDataset] No time steps in dataset; sample generation will be skipped.")
            self.available_times_np = np.array([], dtype='datetime64[ns]')
            self.sample_start_indices = []
            return

        self.available_times_np = self.dataset['time'].values # NumPy array of datetime64[ns]

        self.sample_start_indices = []
        current_window_start_dt = self.start_datetime
        
        # Iterate creating potential window start times
        while current_window_start_dt + timedelta(hours=self.sample_length_hours) <= self.end_datetime + timedelta(microseconds=1):
            # Find the index in self.available_times_np for the current_window_start_dt
            # This is the actual start of a window in our loaded data
            actual_data_start_idx = np.searchsorted(
                self.available_times_np, 
                np.datetime64(current_window_start_dt), 
                side='left'
            )
            
            # Ensure this found index is a valid start for a full sample
            if actual_data_start_idx + self.sample_length_hours <= self.available_times_np.size:
                # The last timestamp of this potential sample
                sample_actual_end_time = self.available_times_np[actual_data_start_idx + self.sample_length_hours - 1]
                
                # Ensure this sample's actual end time doesn't exceed the overall end_datetime
                if np.datetime64(sample_actual_end_time) <= np.datetime64(self.end_datetime):
                    self.sample_start_indices.append(actual_data_start_idx)
            
            current_window_start_dt += timedelta(hours=self.sample_step_hours)
        
        if self.verbose:
            print(f"[RadklimDataset] Generated {len(self.sample_start_indices)} sample start indices.")

    def __len__(self) -> int:
        """Returns the total number of available samples."""
        return len(self.sample_start_indices)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves, normalizes, and returns a data sample and its corresponding timestamps.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - data_sample (np.ndarray): Normalized precipitation data for the sample,
                  with shape (sample_length_hours, 1, height, width) and dtype float32.
                  The channel dimension is added.
                - time_stamps (np.ndarray): Timestamps for the sample,
                  with shape (sample_length_hours,) and dtype datetime64[ns].
        
        Raises:
            IndexError: If the index `idx` is out of bounds.
        """
        if not 0 <= idx < len(self.sample_start_indices):
            raise IndexError(f"Sample index {idx} is out of range for {len(self.sample_start_indices)} samples.")

        start_idx_in_times_axis = self.sample_start_indices[idx]
        end_idx_in_times_axis = start_idx_in_times_axis + self.sample_length_hours

        # Extract data for the window using integer-based slicing
        # .load().data ensures that if rr_data_array is a Dask array, it's computed and converted to NumPy
        data_slice_np = self.rr_data_array[start_idx_in_times_axis:end_idx_in_times_axis].load().data.astype(np.float32)
        
        # Add a channel dimension for model compatibility: (T, Y, X) -> (T, 1, Y, X)
        data_slice_np = data_slice_np[:, np.newaxis, :, :]
        
        # Normalize the data
        normalized_data = (data_slice_np - self.rr_mean) / self.rr_std

        time_stamps_for_sample = self.available_times_np[start_idx_in_times_axis:end_idx_in_times_axis]
        
        return normalized_data, time_stamps_for_sample

    def denormalize(self, normalized_data: np.ndarray) -> np.ndarray:
        """
        Reverses the normalization process on a data sample.

        Args:
            normalized_data (np.ndarray): The normalized data sample, typically
                                          output from __getitem__.

        Returns:
            np.ndarray: The data sample restored to its original scale.
        """
        return (normalized_data * self.rr_std) + self.rr_mean

    def get_sample_time_window(self, idx: int) -> Tuple[np.datetime64, np.datetime64]:
        """
        Returns the actual start and end np.datetime64 timestamps for a given sample index.

        Args:
            idx (int): The index of the sample.

        Returns:
            Tuple[np.datetime64, np.datetime64]: A tuple containing the start and end
                                                 timestamps of the sample.
        
        Raises:
            IndexError: If the index `idx` is out of bounds.
        """
        if not 0 <= idx < len(self.sample_start_indices):
            raise IndexError(f"Sample index {idx} is out of range.")
        
        start_idx_in_times_axis = self.sample_start_indices[idx]
        actual_start_time = self.available_times_np[start_idx_in_times_axis]
        actual_end_time = self.available_times_np[start_idx_in_times_axis + self.sample_length_hours - 1]
        
        return actual_start_time, actual_end_time

    def close(self) -> None:
        """Closes the underlying xarray Dataset."""
        try:
            if hasattr(self, 'dataset') and self.dataset is not None:
                self.dataset.close()
                if self.verbose:
                    print("[RadklimDataset] Dataset resources closed.")
        except Exception as e:
            if self.verbose:
                print(f"[RadklimDataset] Error during dataset close: {e}")
            pass

    def __del__(self):
        """Ensures dataset is closed when the object is deleted."""
        self.close()