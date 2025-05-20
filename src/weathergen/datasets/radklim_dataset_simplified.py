import json
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

class RadklimDataset:
    """
    Robust loader for RADKLIM RW precipitation data stored in monthly NetCDF files.

    Provides point-cloud formatting of spatiotemporal windows:
      - latlon: (N_points, 2) - Latitude and longitude for each point.
      - geoinfos: (N_points, 0) - Placeholder for additional geographic information.
      - data: (N_points, N_channels) - Data values for each point and channel.
      - times: (N_points,) - Timestamp for each point.

    The class handles loading data within a specified time range,
    slicing it into windows, and normalizing the data.

    Parameters
    ----------
    start_time : int or str
        Inclusive start timestamp in YYYYMMDDHHMM format.
    end_time : int or str
        Inclusive end timestamp in YYYYMMDDHHMM format.
    len_hrs : int
        Length of each data window in hours.
    step_hrs : int
        Step (stride) between the start of consecutive windows in hours.
    data_path : str or Path
        Root directory where NetCDF files are stored.
        Files are expected to be organized in subdirectories by year,
        e.g., data_path/YYYY/RW_2017.002_YYYYMM.nc.
    normalization_path : str or Path
        Path to a JSON file containing normalization statistics ("mean" and "std" lists).
        These lists should correspond to the sorted order of float data variables.
    fname_patt : str, optional
        Filename prefix for the NetCDF files (default is "RW_2017.002_").
        The full filename is expected to be `fname_patt + YYYYMM + .nc`.
    """

    def __init__(
        self,
        start_time, # str or int: YYYYMMDDHHMM
        end_time,   # str or int: YYYYMMDDHHMM
        len_hrs,    # int: window length in hours
        step_hrs,   # int: step between windows in hours
        data_path,  # str or Path: root directory for NetCDF files
        normalization_path, # str or Path: path to normalization JSON file
        fname_patt: str = "RW_2017.002_",  # str: file prefix
    ):
        """
        Initializes the RadklimDataset.

        This involves:
        1. Parsing input time parameters.
        2. Discovering relevant NetCDF files within the specified data path and time range.
        3. Performing a metadata pass on these files to determine available variables and time coverage.
        4. Loading normalization statistics (mean and standard deviation) from a JSON file.
        5. Calculating time indexing parameters (start/end indices, time step, window/stride steps).
        6. Extracting spatial coordinates (latitude, longitude, x, y) from the first data file.
        7. Setting up an xarray Dataset for efficient, chunked data access for the specified variables and time range.
        """
        # --- TIME PARAMETERS ---
        # Parse start and end times from string/int to pandas datetime objects
        self.start_time: pd.Timestamp = pd.to_datetime(str(start_time), format="%Y%m%d%H%M")
        self.end_time: pd.Timestamp = pd.to_datetime(str(end_time), format="%Y%m%d%H%M")
        # Store window length and step in hours
        self.len_hrs: int = len_hrs
        self.step_hrs: int = step_hrs
        # Store path to data and filename pattern
        self.data_path: Path = Path(data_path)
        self.fname_patt: str = fname_patt

        # --- FILE LIST ---
        # Get a list of NetCDF files that fall within the specified time range
        self.file_list: list[Path] = self._get_file_list()
        if not self.file_list:
            raise FileNotFoundError(
                f"No files matching pattern '{self.fname_patt}' in {self.data_path} "
                f"for the period between {self.start_time} and {self.end_time}."
            )

        # --- METADATA PASS (no chunks, read metadata only) ---
        # Open all found files as a single multi-file dataset to read metadata
        # `combine='nested'` and `concat_dim='time'` ensure correct chronological order
        # `parallel=False` is often more stable for metadata operations or few files
        metadata_dataset: xr.Dataset = xr.open_mfdataset(
            self.file_list,
            combine="nested",
            concat_dim="time",
            engine="netcdf4", # Explicitly use netcdf4 engine
            parallel=False,   # Disable parallel processing for metadata loading
        )

        # Identify suitable data variables:
        # - Must be floating-point type (metadata_dataset[v].dtype.kind == 'f')
        # - Must not be in the exclude set (coordinates, crs, time)
        exclude_vars: set[str] = {"crs", "lat", "lon", "x", "y", "time"}
        self.variables: list[str] = sorted(
            [v for v in metadata_dataset.data_vars
             if metadata_dataset[v].dtype.kind == 'f' and v not in exclude_vars]
        )
        if not self.variables:
            metadata_dataset.close() # Ensure dataset is closed before raising error
            raise ValueError("No suitable float data variables found in the NetCDF files.")

        # --- NORMALIZATION STATS ---
        # Load mean and standard deviation for normalization from a JSON file
        with open(normalization_path) as f:
            norm_stats: dict = json.load(f)
        
        means_list: list[float] | None = norm_stats.get("mean")
        stds_list: list[float] | None = norm_stats.get("std")

        # Validate the structure and content of the normalization statistics
        if not (isinstance(means_list, list) and isinstance(stds_list, list)):
            metadata_dataset.close()
            raise ValueError("Normalization JSON must contain 'mean' and 'std' as lists.")
        if len(means_list) != len(self.variables) or len(stds_list) != len(self.variables):
            metadata_dataset.close()
            raise ValueError(
                f"Normalization stats length (mean: {len(means_list)}, std: {len(stds_list)}) "
                f"does not match the number of selected variables ({len(self.variables)})."
            )
        
        # Store normalization parameters as numpy arrays
        self.mean: np.ndarray = np.array(means_list, dtype=np.float32)
        self.std: np.ndarray = np.array(stds_list, dtype=np.float32)
        # Guard against zero or very small std to prevent division by zero or NaN/inf issues.
        # A common practice is to set a minimum standard deviation, e.g., 1e-6 or 1.0
        # depending on the data scale and problem. Original code had:
        # self.std[self.std <= 0] = 1.0 
        # This line is crucial if std can be zero for some variables. For now, kept as in original.

        # --- TIME INDEXING ---
        # Extract all time values from the metadata dataset
        all_times_in_files: pd.DatetimeIndex = pd.to_datetime(metadata_dataset["time"].values)
        
        # Find the indices corresponding to the requested start_time and end_time in all_times_in_files
        # 'left' for start_idx to include the start_time if it matches an existing timestamp
        # 'right' for end_idx to make the slice exclusive of end_time if it matches (standard Python slicing behavior)
        self.start_idx: int = int(np.searchsorted(all_times_in_files, self.start_time, side='left'))
        self.end_idx: int = int(np.searchsorted(all_times_in_files, self.end_time, side='right'))
        
        # Slice the `all_times_in_files` array to get the relevant time range for this dataset instance
        self.times: pd.DatetimeIndex = all_times_in_files[self.start_idx : self.end_idx]
        if len(self.times) == 0:
            metadata_dataset.close()
            raise ValueError(
                f"No timestamps found in the data within the requested range: "
                f"{self.start_time} to {self.end_time}."
            )

        # Compute native time step (resolution) of the data in seconds
        if len(self.times) > 1:
            # Calculate differences between consecutive timestamps in seconds
            time_diffs_seconds = np.diff(self.times).astype('timedelta64[s]').astype(int)
            self.dt_seconds: int = int(time_diffs_seconds[0]) # Assume constant time step based on the first difference
        else:
            # If only one timestamp, or if len_hrs is the only reference for step duration
            self.dt_seconds: int = self.len_hrs * 3600  # Default to window length in seconds if only one data point
        
        # Calculate number of native time steps per window and per stride (step between windows)
        self.steps_per_window: int = int(round(self.len_hrs * 3600 / self.dt_seconds))
        self.steps_per_stride: int = int(round(self.step_hrs * 3600 / self.dt_seconds))
        
        # Close the metadata dataset as it's no longer needed
        metadata_dataset.close()

        # --- COORDINATES ---
        # Open the first file in the list to extract coordinate information
        # This assumes all files share the same spatial grid and coordinate system
        with xr.open_dataset(self.file_list[0], engine='netcdf4') as first_dataset_for_coords:
            # 1D projection coordinates (e.g., y/x in meters for a specific CRS)
            y_coords_1d: np.ndarray = first_dataset_for_coords['y'].values.astype(np.float32)
            x_coords_1d: np.ndarray = first_dataset_for_coords['x'].values.astype(np.float32)
            # 2D geographic coordinates (latitude and longitude grids)
            latitude_grid_2d: np.ndarray  = first_dataset_for_coords['lat'].values.astype(np.float32)
            longitude_grid_2d: np.ndarray = first_dataset_for_coords['lon'].values.astype(np.float32)
            
        self.num_y_points: int = len(y_coords_1d)
        self.num_x_points: int = len(x_coords_1d)

        # Validate that the shape of lat/lon grids matches the dimensions of y/x coordinates
        if latitude_grid_2d.shape != (self.num_y_points, self.num_x_points) or \
           longitude_grid_2d.shape != (self.num_y_points, self.num_x_points):
            raise ValueError("Latitude/longitude grid shape mismatch with y/x coordinate dimensions.")
        
        # Store and normalize coordinates:
        # Clip latitude values to be within the valid geographic range [-90, 90] degrees
        self.latitude_2d: np.ndarray = np.clip(latitude_grid_2d, -90, 90)
        # Normalize longitude values to be within the common range [-180, 180] degrees
        self.longitude_2d: np.ndarray = (longitude_grid_2d + 180) % 360 - 180

        # --- DATA PASS (with chunking for efficient access) ---
        # Open the multi-file dataset again, this time for actual data access.
        # Chunking is applied for performance, especially for time-based slicing:
        #   - 'time': chunks are set to roughly twice the window size for efficient window extraction.
        #   - 'y', 'x': -1 means load the entire spatial dimension into each chunk (no chunking along y, x).
        # `parallel=False` for sequential processing; consider `True` if I/O is a bottleneck
        # and files are numerous/large, and a Dask cluster is available/desired.
        data_access_dataset: xr.Dataset = xr.open_mfdataset(
            self.file_list,
            combine="nested",
            concat_dim="time",
            engine="netcdf4",
            chunks={"time": self.steps_per_window * 2, "y": -1, "x": -1},
            parallel=False,
        )
        # Select only the desired variables and the relevant time slice (determined by start_idx, end_idx)
        # This `self.ds` will be the main dataset used for fetching data windows.
        self.ds: xr.Dataset = data_access_dataset[self.variables].isel(time=slice(self.start_idx, self.end_idx))

    def __len__(self) -> int:
        """
        Calculates the total number of windows that can be extracted from the dataset.

        Returns
        -------
        int
            The number of available windows.
        """
        # Total number of time steps available within the selected range (self.times)
        total_available_timesteps: int = len(self.times)
        
        # If the total available time steps are less than a single window's length, no full windows can be formed
        if total_available_timesteps < self.steps_per_window:
            return 0
        
        # Ensure stride is at least 1 to prevent infinite loops or division by zero in calculation
        actual_stride_in_steps: int = max(1, self.steps_per_stride)
        
        # Calculate the number of windows:
        # (total_steps - window_size) // stride_size gives the number of full strides possible after the first window
        # +1 accounts for the initial window itself
        return (total_available_timesteps - self.steps_per_window) // actual_stride_in_steps + 1

    def _get(self, window_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieves a single window of data, formatted as a point cloud.

        This is an internal method, typically called by `__getitem__`, `get_source`, or `get_target`.

        Parameters
        ----------
        window_index : int
            The index of the window to retrieve. Must be between 0 and len(self) - 1.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple containing:
            - point_latlon_coordinates (np.ndarray): Shape (N_points, 2). Latitude and longitude for each point.
                                   N_points = num_timesteps_in_window * num_y_grid_points * num_x_grid_points.
            - geoinfos (np.ndarray): Shape (N_points, 0). Placeholder for additional geographic info.
            - flattened_point_data (np.ndarray): Shape (N_points, N_channels). Data values for each point and channel.
            - flattened_times_array (np.ndarray): Shape (N_points,). Timestamp for each point.

        Raises
        ------
        IndexError
            If `window_index` is out of the valid range.
        """
        # Validate window_index
        if window_index < 0 or window_index >= len(self):
            raise IndexError(f"Window index {window_index} out of range (0 to {len(self)-1}).")
        
        # Calculate the start and end time slice indices within `self.ds` for the current window
        time_slice_start_index: int = window_index * self.steps_per_stride
        time_slice_end_index: int = time_slice_start_index + self.steps_per_window
        
        # Select the data for the current window using xarray's integer-based selection `isel`
        window_dataset: xr.Dataset = self.ds.isel(time=slice(time_slice_start_index, time_slice_end_index))
        
        # Convert the xarray Dataset into a NumPy array:
        # - `to_array(dim='channel')`: stacks all data variables along a new 'channel' dimension.
        # - `transpose('time','y','x','channel')`: reorders dimensions to a standard (T, Y, X, C) format.
        # - `.load()`: explicitly loads data into memory (important if data is Dask-backed).
        # - `.data`: accesses the underlying NumPy array from the xarray.DataArray.
        window_data_array: np.ndarray = window_dataset.to_array(dim='channel').transpose('time','y','x','channel').load().data
        
        # Get dimensions of the window: (num_timesteps, num_y_points, num_x_points, num_channels)
        # These are common abbreviations: nt for num_times, ny for num_y, nx for num_x, nc for num_channels.
        nt, ny, nx, nc = window_data_array.shape
        
        # Reshape the 4D window data array (T, Y, X, C) into a 2D point-cloud format (N_points, C)
        # N_points = T * Y * X. Each row is a spatiotemporal point, columns are channels.
        flattened_point_data: np.ndarray = window_data_array.reshape(-1, nc)
        
        # --- Create point-cloud coordinates and times ---
        # Latitude and Longitude:
        # `self.latitude_2d` and `self.longitude_2d` are (Y, X) grids.
        # `reshape(-1)` flattens them to 1D arrays of length (Y*X).
        # `np.tile` repeats these 1D arrays for each timestep (`nt`) in the window.
        # `np.stack` combines lat and lon into a (N_points, 2) array.
        point_latitudes: np.ndarray = np.tile(self.latitude_2d.reshape(-1), nt)
        point_longitudes: np.ndarray = np.tile(self.longitude_2d.reshape(-1), nt)
        point_latlon_coordinates: np.ndarray = np.stack([point_latitudes, point_longitudes], axis=1)
        
        # Times:
        # `window_dataset['time']` is a 1D array of timestamps for the window.
        # `expand_dims` adds new dimensions for y and x, making it (T, 1, 1).
        # This allows broadcasting so that each (y,x) spatial point gets associated with the time.
        # `.data` gets the NumPy array.
        # `reshape(-1)` flattens the resulting (T, Y, X) array of times to match N_points.
        flattened_times_array: np.ndarray = window_dataset['time'].expand_dims(
            {'y':ny,'x':nx}, # Dimensions to expand, ny and nx from window_data_array.shape
            axis=[1,2]       # Axes positions for new 'y' and 'x' dimensions
        ).data.reshape(-1)   # Flatten to 1D array of size N_points
        
        # Placeholder for geoinfos (e.g., elevation, land use type).
        # Currently an empty array with shape (N_points, 0), signifying no geo-info features.
        geoinfos: np.ndarray = np.zeros((flattened_point_data.shape[0],0),dtype=np.float32)
        
        return point_latlon_coordinates, geoinfos, flattened_point_data, flattened_times_array

    def get_source(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieves a data window, typically used as source/input for a model.

        This method is an alias for `_get(idx)`.

        Parameters
        ----------
        idx : int
            The index of the window to retrieve.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Data window formatted as (latlon, geoinfos, data, times).
            See `_get` method for details on the returned tuple structure.
        """
        return self._get(idx)

    def get_target(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieves a data window, typically used as target/label for a model.

        This method is an alias for `_get(idx)`. In many scenarios, source and target
        might be the same window or derived from the same underlying data structure.

        Parameters
        ----------
        idx : int
            The index of the window to retrieve.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Data window formatted as (latlon, geoinfos, data, times).
            See `_get` method for details on the returned tuple structure.
        """
        return self._get(idx)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Allows dictionary-like or list-like bracket indexing (e.g., `dataset[i]`) to fetch a window.

        This method calls `_get(idx)`.

        Parameters
        ----------
        idx : int
            Window index.

        Returns
        -------
        tuple
            (latlon, geoinfos, data, times) for the window at `idx`.
            See `_get` method for details on the returned tuple structure.
        """
        return self._get(idx)
    
    def get_source_num_channels(self) -> int:
        """
        Returns the number of data channels for source data.

        This is determined by the number of selected float variables from the NetCDF files.

        Returns
        -------
        int
            Number of source data channels.
        """
        return len(self.variables)

    def get_target_num_channels(self) -> int:
        """
        Returns the number of data channels for target data.

        In this class, it's assumed to be the same as the source channels.

        Returns
        -------
        int
            Number of target data channels.
        """
        return len(self.variables)

    def get_coords_size(self) -> int:
        """
        Returns the dimensionality of the spatial coordinates (e.g., 2 for lat/lon).

        Returns
        -------
        int
            Size of the coordinate vector for each point (typically 2 for latitude, longitude).
        """
        return 2 # Corresponds to (latitude, longitude)

    def get_geoinfo_size(self) -> int:
        """
        Returns the dimensionality of the geographic information vector.

        Currently, this is a placeholder and returns 0.

        Returns
        -------
        int
            Size of the geoinfo vector for each point (currently 0).
        """
        return 0 # Corresponds to the empty geoinfos array

    def normalize_source_channels(self, data: np.ndarray) -> np.ndarray:
        """
        Normalizes the source data channels using pre-loaded mean and standard deviation.

        Normalization formula: (data - mean) / std.
        Assumes `data` is (N_points, N_channels) and `self.mean`, `self.std` are (N_channels,).

        Parameters
        ----------
        data : np.ndarray
            Data array to normalize. Expected shape (N_points, N_channels).

        Returns
        -------
        np.ndarray
            Normalized data array.
        """
        # self.mean and self.std are 1D arrays of shape (N_channels,)
        # NumPy broadcasting applies them correctly across the N_points dimension of `data`.
        return (data - self.mean) / self.std

    def normalize_target_channels(self, data: np.ndarray) -> np.ndarray:
        """
        Normalizes the target data channels using pre-loaded mean and standard deviation.

        Normalization formula: (data - mean) / std.
        Assumes `data` is (N_points, N_channels) and `self.mean`, `self.std` are (N_channels,).

        Parameters
        ----------
        data : np.ndarray
            Data array to normalize. Expected shape (N_points, N_channels).

        Returns
        -------
        np.ndarray
            Normalized data array.
        """
        return (data - self.mean) / self.std

    def denormalize_source_channels(self, data: np.ndarray) -> np.ndarray:
        """
        Denormalizes the source data channels using pre-loaded mean and standard deviation.

        Denormalization formula: data * std + mean.
        Assumes `data` is (N_points, N_channels) and `self.mean`, `self.std` are (N_channels,).

        Parameters
        ----------
        data : np.ndarray
            Normalized data array to denormalize. Expected shape (N_points, N_channels).

        Returns
        -------
        np.ndarray
            Denormalized (original scale) data array.
        """
        return data * self.std + self.mean

    def denormalize_target_channels(self, data: np.ndarray) -> np.ndarray:
        """
        Denormalizes the target data channels using pre-loaded mean and standard deviation.

        Denormalization formula: data * std + mean.
        Assumes `data` is (N_points, N_channels) and `self.mean`, `self.std` are (N_channels,).

        Parameters
        ----------
        data : np.ndarray
            Normalized data array to denormalize. Expected shape (N_points, N_channels).

        Returns
        -------
        np.ndarray
            Denormalized (original scale) data array.
        """
        return data * self.std + self.mean

    def time_window(self, idx: int) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        Returns the start and end timestamps for a given window index.

        Parameters
        ----------
        idx : int
            The index of the window. Must be between 0 and len(self)-1.

        Returns
        -------
        tuple[pd.Timestamp, pd.Timestamp]
            A tuple containing the start and end pandas Timestamps of the window.

        Raises
        ------
        IndexError
            If `idx` is out of the valid range.
        """
        # Validate window index
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for time_window (0 to {len(self)-1}).")
        
        # Calculate start time index in the `self.times` array for the window
        # `self.times` contains all timestamps available to this dataset instance.
        start_time_index_in_self_times: int = idx * self.steps_per_stride
        # Calculate end time index in the `self.times` array for the window
        # The end timestamp is inclusive for the window, so it's `start_index + steps_in_window - 1`.
        end_time_index_in_self_times: int = start_time_index_in_self_times + self.steps_per_window - 1
        
        return (self.times[start_time_index_in_self_times], self.times[end_time_index_in_self_times])

    @staticmethod
    def _get_file_list(self) -> list[Path]:
        """
        Generates a list of NetCDF file paths relevant to the specified time range.

        Files are expected to be in `data_path/YYYY/fname_pattYYYYMM.nc`.
        This method identifies all months spanning from `self.start_time` to `self.end_time`
        and constructs file paths for each of these months, then checks if the files exist.

        Returns
        -------
        list[Path]
            A list of `pathlib.Path` objects for existing NetCDF files, sorted implicitly by month.
        """
        # Determine the first day of the month for start_time and end_time.
        # This defines the inclusive range of months to scan for data files.
        # `.replace` ensures we get the very start of the month for range generation.
        start_month_boundary: pd.Timestamp = self.start_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_month_boundary: pd.Timestamp = self.end_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Generate a sequence of all month start dates within the [start_month_boundary, end_month_boundary] range.
        # 'MS' frequency stands for Month Start, ensuring one entry per month in the range.
        relevant_months_range: pd.DatetimeIndex = pd.date_range(start_month_boundary, end_month_boundary, freq='MS')
        
        # Construct file paths for each month and filter for existing files
        found_files: list[Path] = []
        for month_date in relevant_months_range:
            # Extract year string (e.g., "2023") for directory path
            year_str: str = month_date.strftime('%Y')
            # Extract year-month string for filename (e.g., "202310")
            year_month_str: str = month_date.strftime('%Y%m')
            
            # Construct the full path to the potential NetCDF file
            # Structure: data_path / YYYY / prefixYYYYMM.nc
            potential_file: Path = self.data_path / year_str / f"{self.fname_patt}{year_month_str}.nc"
            
            # Add to list only if the file actually exists on the filesystem
            if potential_file.is_file():
                found_files.append(potential_file)
        
        return found_files

    def close(self) -> None:
        """
        Closes the underlying xarray multi-file dataset.

        It's good practice to call this when the RadklimDataset object is no longer needed,
        especially if `parallel=True` was used in `xr.open_mfdataset` when opening `self.ds`.
        This helps ensure proper cleanup of resources like Dask clusters or file handles.
        """
        try:
            # Check if self.ds exists (it's initialized late in __init__) and has a close method
            if hasattr(self, 'ds') and self.ds is not None:
                self.ds.close()
        except Exception:
            # Silently pass if closing fails (e.g., already closed, not initialised, or other error)
            pass

    def __del__(self) -> None:
        """
        Destructor for the RadklimDataset object.

        Ensures that the `close` method is called to release resources (like open file handles
        associated with `self.ds`) when the object is garbage collected by Python.
        """
        self.close()