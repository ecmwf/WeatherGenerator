
# (C) Copyright [Year] [Your Organization/Project contributors]
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, [Your Organization] does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
# (Note: Added a placeholder license header similar to the first example.
# Please replace with your actual license information or remove if not applicable.)

import json
from pathlib import Path
from typing import Tuple, Union

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
import zarr


class RadklimKerchunkDataset:
    """
    Lazily load Radklim data via a Kerchunk reference JSON.

    This dataset class avoids loading any large arrays (e.g., time‐varying
    lat/lon) all at once, providing an efficient way to access slices of
    Radklim data. It supports normalization and denormalization of the data.

    Attributes
    ----------
    variables : list[str]
        List of variables to be loaded from the dataset (e.g., ["RR"]).
    start_time : pd.Timestamp
        The start time of the data window.
    end_time : pd.Timestamp
        The end time of the data window.
    len_hrs : int
        Length of each sample window in hours.
    step_hrs : int
        Step in hours between consecutive sample windows. Must be equal to `len_hrs`.
    mean : np.ndarray
        Mean values for normalization, corresponding to `variables`.
    std : np.ndarray
        Standard deviation values for normalization, corresponding to `variables`.
    ds : xr.Dataset
        The lazily loaded xarray Dataset, subset to the specified time range and variables.
    times : np.ndarray
        Array of `datetime64` timestamps within the selected `start_time` and `end_time`.
    start_idx : int
        The starting index in the full dataset's time dimension.
    end_idx : int
        The ending index (exclusive) in the full dataset's time dimension.
    num_steps_per_window : int
        Number of time steps within a single `len_hrs` window.
    ny : int
        Number of grid points in the y-dimension.
    nx : int
        Number of grid points in the x-dimension.
    latitudes : np.ndarray
        2D array of processed latitude values.
    longitudes : np.ndarray
        2D array of processed longitude values.
    latlon_shape : tuple[int, int]
        Shape of the 2D latitude/longitude arrays (ny, nx).
    """

    def __init__(
        self,
        start_time: Union[int, str],
        end_time: Union[int, str],
        len_hrs: int,
        step_hrs: int,
        reference_json_path: Union[str, Path],
        normalization_path: Union[str, Path],
    ) -> None:
        """
        Initialize the RadklimKerchunkDataset.

        Parameters
        ----------
        start_time : Union[int, str]
            Start time for the dataset slice, formatted as "YYYYMMDDHHMM"
            (e.g., 202001010000) or an integer representation.
        end_time : Union[int, str]
            End time for the dataset slice, formatted as "YYYYMMDDHHMM"
            or an integer representation.
        len_hrs : int
            Length of each data sample window in hours.
        step_hrs : int
            Step in hours between the start of consecutive sample windows.
            Currently, this must be equal to `len_hrs`.
        reference_json_path : Union[str, Path]
            Path to the Kerchunk reference JSON file.
        normalization_path : Union[str, Path]
            Path to the JSON file containing normalization statistics (mean, std).

        Raises
        ------
        ValueError
            If time parameters are invalid, `step_hrs` does not equal `len_hrs`,
            or normalization stats are incompatible.
        FileNotFoundError
            If `reference_json_path` or `normalization_path` does not exist.
        """
        # 1) Parse and validate window parameters
        self._parse_and_validate_window(start_time, end_time, len_hrs, step_hrs)

        # 2) Which variables we care about
        self.variables: list[str] = ["RR"]  # Example, could be parameterized

        # 3) Load normalization stats
        self._load_normalization_stats(normalization_path)

        # 4) Open Kerchunk reference → xarray.Dataset (lazy)
        self.ds: xr.Dataset = self._open_reference_dataset(reference_json_path)

        # 5) Compute indices, time step, spatial coordinate arrays
        self._compute_indices_and_coords()

        # 6) Subset to only our variables and our time range
        self._subset_main_variable()

    def _parse_and_validate_window(
        self,
        start_time: Union[int, str],
        end_time: Union[int, str],
        len_hrs: int,
        step_hrs: int,
    ) -> None:
        """
        Parse and validate time window parameters.

        Sets `self.start_time`, `self.end_time`, `self.len_hrs`, `self.step_hrs`.

        Parameters
        ----------
        start_time : Union[int, str]
            Start time for the dataset slice.
        end_time : Union[int, str]
            End time for the dataset slice.
        len_hrs : int
            Length of each data sample window in hours.
        step_hrs : int
            Step in hours between consecutive sample windows.

        Raises
        ------
        ValueError
            If start/end times cannot be parsed or if `step_hrs` != `len_hrs`.
        """
        try:
            self.start_time: pd.Timestamp = pd.to_datetime(
                str(start_time), format="%Y%m%d%H%M"
            )
            self.end_time: pd.Timestamp = pd.to_datetime(
                str(end_time), format="%Y%m%d%H%M"
            )
        except ValueError as e:
            raise ValueError(f"Could not parse start/end time: {e}")

        self.len_hrs: int = int(len_hrs)
        self.step_hrs: int = int(step_hrs)
        if self.step_hrs != self.len_hrs:
            raise ValueError(
                f"step_hrs ({self.step_hrs}) must equal len_hrs ({self.len_hrs})."
            )

    def _load_normalization_stats(
        self, normalization_path: Union[str, Path]
    ) -> None:
        """
        Load normalization statistics (mean and std) from a JSON file.

        Sets `self.mean` and `self.std`.

        Parameters
        ----------
        normalization_path : Union[str, Path]
            Path to the JSON file containing "mean" and "std" keys.

        Raises
        ------
        FileNotFoundError
            If the normalization file is not found.
        ValueError
            If normalization stats length doesn't match `self.variables`.
        """
        path = Path(normalization_path)
        if not path.exists():
            raise FileNotFoundError(f"Normalization JSON not found: {path}")
        with open(path, "r") as f:
            stats = json.load(f)

        self.mean: np.ndarray = np.array(stats.get("mean", []), dtype=np.float32)
        self.std: np.ndarray = np.array(stats.get("std", []), dtype=np.float32)

        if len(self.mean) != len(self.variables) or len(self.std) != len(
            self.variables
        ):
            raise ValueError(
                f"Normalization stats length ({len(self.mean)}) must match "
                f"variables ({len(self.variables)})"
            )

    def _open_reference_dataset(
        self, reference_json_path: Union[str, Path]
    ) -> xr.Dataset:
        """
        Open the Kerchunk reference JSON as a lazily-loaded xarray Dataset.

        Parameters
        ----------
        reference_json_path : Union[str, Path]
            Path to the Kerchunk reference JSON file.

        Returns
        -------
        xr.Dataset
            The lazily loaded xarray Dataset.

        Raises
        ------
        FileNotFoundError
            If the reference JSON file is not found.
        """
        ref_path = Path(reference_json_path)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference JSON not found: {ref_path}")

        with open(ref_path, "r") as f:
            kerchunk_ref = json.load(f)

        fs = fsspec.filesystem("reference", fo=kerchunk_ref)
        mapper = fs.get_mapper("")
        try:
            zarr.consolidate_metadata(mapper)
        except Exception:
            pass

        ds = xr.open_dataset(
            mapper,
            engine="zarr",
            consolidated=True,  
            chunks={
                "time": self.len_hrs * 2, 
                "y": -1,  
                "x": -1, 
            },
        )
        return ds

    def _compute_indices_and_coords(self) -> None:
        """
        Compute time indices and prepare spatial coordinate arrays.

        Loads minimal data (1D time, 1D y/x coords, first slice of lat/lon if time-varying).
        Sets `self.start_idx`, `self.end_idx`, `self.times`,
        `self.num_steps_per_window`, `self.ny`, `self.nx`, `self.latitudes`,
        `self.longitudes`, `self.latlon_shape`.

        Raises
        ------
        ValueError
            If no time overlap is found or if time steps are inconsistent.
        """
        ds = self.ds

        # 1) Time indexing:
        times_all = ds["time"].values  # 1D array of datetime64
        self.start_idx: int = int(
            np.searchsorted(times_all, np.datetime64(self.start_time), side="left")
        )
        self.end_idx: int = int(
            np.searchsorted(times_all, np.datetime64(self.end_time), side="right")
        )

        if self.start_idx >= self.end_idx:
            raise ValueError(
                f"No time overlap found: start_idx ({self.start_idx}) >= end_idx ({self.end_idx})."
            )

        self.times: np.ndarray = times_all[self.start_idx : self.end_idx]

        dt_arr = np.unique(np.diff(self.times.astype("datetime64[s]")))
        if dt_arr.size != 1:
            raise ValueError(
                "Inconsistent time steps in the selected window. Found: {dt_arr}"
            )
        dt_seconds = int(dt_arr[0].item().total_seconds())
        self.num_steps_per_window: int = int((self.len_hrs * 3600) / dt_seconds)

        # 2) Spatial coords:
        y1d = ds["y"].values.astype(np.float32)
        x1d = ds["x"].values.astype(np.float32)
        self.ny: int = len(y1d)
        self.nx: int = len(x1d)

        # 3) Lat/lon:
        lat_var = ds["lat"]
        if "time" in lat_var.dims:
            lat2d = lat_var.isel(time=0).values.astype(np.float32)
        else:
            lat2d = lat_var.values.astype(np.float32)

        lon_var = ds["lon"]
        if "time" in lon_var.dims:
            lon2d = lon_var.isel(time=0).values.astype(np.float32)
        else:
            lon2d = lon_var.values.astype(np.float32)

        # 4) Transform lat/lon once:
        self.latitudes: np.ndarray = 2 * np.clip(lat2d, -90, 90) - lat2d
        self.longitudes: np.ndarray = (lon2d + 180) % 360 - 180
        self.latlon_shape: tuple[int, int] = lat2d.shape  # Should be (ny, nx)

    def _subset_main_variable(self) -> None:
        """
        Subset the main `self.ds` to the selected variables and time range.

        Modifies `self.ds` in-place.
        """
        self.ds = self.ds[self.variables].isel(
            time=slice(self.start_idx, self.end_idx)
        )

    def __len__(self) -> int:
        """
        Return the number of available sample windows in the dataset.
        """
        total_time_steps = len(self.ds["time"])
        if total_time_steps < self.num_steps_per_window:
            return 0
        return total_time_steps - self.num_steps_per_window + 1

    def _get(
        self, idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieve data for a specific sample window index.

        Parameters
        ----------
        idx : int
            Index of the sample window.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple containing:
            - latlon (np.ndarray): Coordinates (latitude, longitude) for valid data points,
              shape (N_valid, 2).
            - geoinfos (np.ndarray): Empty array for geographical information,
              shape (N_valid, 0).
            - data (np.ndarray): The actual variable data for valid points,
              shape (N_valid, n_variables).
            - datetimes (np.ndarray): Timestamps for each valid data point,
              shape (N_valid,).

        Raises
        ------
        IndexError
            If `idx` is out of range.
        """
        if not (0 <= idx < len(self)):
            raise IndexError(
                f"Index out of range (0 <= idx < {len(self)}). Got {idx}"
            )

        # 1) Select the needed time window lazily
        ds_window = self.ds.isel(
            time=slice(idx, idx + self.num_steps_per_window)
        )

        # 2) Convert to (time, y, x, var) and load into memory
        data_array = ds_window.to_array("var").transpose("time", "y", "x", "var")
        # .load() triggers actual data reading for this slice
        arr_4d = data_array.load().data
        nt = arr_4d.shape[0]

        # Reshape to (nt*ny*nx, nvars)
        arr_flat = arr_4d.reshape(-1, arr_4d.shape[-1])

        # 3) Build lat/lon per (time, y, x) by broadcasting
        lat_bcast = np.broadcast_to(self.latitudes, (nt, *self.latlon_shape))
        lon_bcast = np.broadcast_to(self.longitudes, (nt, *self.latlon_shape))
        lat_flat = lat_bcast.reshape(-1)
        lon_flat = lon_bcast.reshape(-1)
        latlon = np.column_stack((lat_flat, lon_flat))

        # 4) Build a time array for each point
        t_vals = ds_window["time"].values
        # Repeat each time nt_y_x times
        t_arr = np.repeat(t_vals, self.ny * self.nx)

        # 5) Mask out rows where any variable is NaN
        mask_nan = np.any(np.isnan(arr_flat), axis=1)
        valid_idx = ~mask_nan

        # Prepare an empty geoinfo array as per the class structure
        geo_info_empty = np.zeros((valid_idx.sum(), 0), dtype=np.float32)

        return (
            latlon[valid_idx],  
            geo_info_empty,  
            arr_flat[valid_idx],  
            t_arr[valid_idx],  
        )

    def __getitem__(
        self, idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get a data sample by index.

        Parameters
        ----------
        idx : int
            Index of the sample window.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Data tuple as returned by `_get`.
        """
        return self._get(idx)

    def get_source(
        self, idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get source data for the given index.
        (Currently an alias for `_get`).

        Parameters
        ----------
        idx : int
            Index of the temporal window.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Source data (coords, geoinfos, data, datetimes).
        """
        return self._get(idx)

    def get_target(
        self, idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get target data for the given index.
        (Currently an alias for `_get`).

        Parameters
        ----------
        idx : int
            Index of the temporal window.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Target data (coords, geoinfos, data, datetimes).
        """
        return self._get(idx)

    def get_source_num_channels(self) -> int:
        """
        Get the number of source channels.

        Returns
        -------
        int
            Number of source data channels (variables).
        """
        return len(self.variables)

    def get_target_num_channels(self) -> int:
        """
        Get the number of target channels.

        Returns
        -------
        int
            Number of target data channels (variables).
        """
        return len(self.variables)

    def get_coords_size(self) -> int:
        """
        Get the size of the coordinate vector (latitude, longitude).

        Returns
        -------
        int
            Size of coordinates (always 2).
        """
        return 2  # lat, lon

    def get_geoinfo_size(self) -> int:
        """
        Get the size of the geographical information vector.

        Returns
        -------
        int
            Size of geoinfos (always 0 for this class).
        """
        return 0

    def normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Normalize coordinates (latitude, longitude).

        Latitude is transformed using sine.
        Longitude is transformed using sine of half the angle.

        Parameters
        ----------
        coords : np.ndarray
            Coordinates array of shape (..., 2) where coords[..., 0] is latitude
            and coords[..., 1] is longitude.

        Returns
        -------
        np.ndarray
            Normalized coordinates.
        """
        coords_norm = coords.copy()
        coords_norm[..., 0] = np.sin(np.deg2rad(coords_norm[..., 0])) 
        coords_norm[..., 1] = np.sin(
            0.5 * np.deg2rad(coords_norm[..., 1])
        )  # Normalize lon
        return coords_norm

    def normalize_geoinfos(self, geoinfos: np.ndarray) -> np.ndarray:
        """
        Normalize geographical information features.
        (Currently a no-op as geoinfos are empty).

        Parameters
        ----------
        geoinfos : np.ndarray
            Geoinfos array. Expected to have shape (..., 0).

        Returns
        -------
        np.ndarray
            The unchanged geoinfos array.

        Raises
        ------
        ValueError
            If `geoinfos` does not have 0 channels.
        """
        if geoinfos.shape[-1] != 0:
            raise ValueError(
                f"Expected geoinfos with 0 channels, got {geoinfos.shape[-1]}"
            )
        return geoinfos

    def normalize_source_channels(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize source channel data using pre-loaded mean and std.

        Parameters
        ----------
        data : np.ndarray
            Source data array of shape (..., num_source_channels).

        Returns
        -------
        np.ndarray
            Normalized source data.
        """
        if data.shape[-1] != len(self.variables):
            raise ValueError(
                f"Data has {data.shape[-1]} channels, expected {len(self.variables)}"
            )
        return (data - self.mean) / self.std

    def normalize_target_channels(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize target channel data using pre-loaded mean and std.

        Parameters
        ----------
        data : np.ndarray
            Target data array of shape (..., num_target_channels).

        Returns
        -------
        np.ndarray
            Normalized target data.
        """
        if data.shape[-1] != len(self.variables):
            raise ValueError(
                f"Data has {data.shape[-1]} channels, expected {len(self.variables)}"
            )
        return (data - self.mean) / self.std

    def denormalize_source_channels(self, data: np.ndarray) -> np.ndarray:
        """
        Denormalize source channel data using pre-loaded mean and std.

        Parameters
        ----------
        data : np.ndarray
            Normalized source data array of shape (..., num_source_channels).

        Returns
        -------
        np.ndarray
            Denormalized source data.
        """
        if data.shape[-1] != len(self.variables):
            raise ValueError(
                f"Data has {data.shape[-1]} channels, expected {len(self.variables)}"
            )
        return data * self.std + self.mean

    def denormalize_target_channels(self, data: np.ndarray) -> np.ndarray:
        """
        Denormalize target channel data using pre-loaded mean and std.

        Parameters
        ----------
        data : np.ndarray
            Normalized target data array of shape (..., num_target_channels).

        Returns
        -------
        np.ndarray
            Denormalized target data.
        """
        if data.shape[-1] != len(self.variables):
            raise ValueError(
                f"Data has {data.shape[-1]} channels, expected {len(self.variables)}"
            )
        return data * self.std + self.mean

    def time_window(self, idx: int) -> Tuple[np.datetime64, np.datetime64]:
        """
        Return the start and end datetimes for the sample window at `idx`.

        Parameters
        ----------
        idx : int
            Index of the sample window.

        Returns
        -------
        tuple[np.datetime64, np.datetime64]
            A tuple containing the start and end `np.datetime64` of the window.
            The end time is exclusive of the actual data points included if it
            aligns with the start of the next window.

        Raises
        ------
        IndexError
            If `idx` is out of range.
        """
        if not (0 <= idx < len(self)):
            max_time_idx = len(self.ds["time"]) - self.num_steps_per_window
            if not (0 <= idx <= max_time_idx): # idx can be max_time_idx for the last window start
                 raise IndexError(
                    f"Index out of range for time_window. Valid: 0 to {max_time_idx}. Got: {idx}"
                )


        t_start = self.ds["time"].isel(time=idx).values.item()
        # The end of the window is `num_steps_per_window` after the start.
        # If idx + num_steps_per_window equals len(self.ds["time"]), it's the timestamp
        # *after* the last actual data point in the window.
        t_end_idx = idx + self.num_steps_per_window
        if t_end_idx < len(self.ds["time"]):
            t_end = self.ds["time"].isel(time=t_end_idx).values.item()
        else:
            # Calculate end time if it goes beyond available discrete time points
            # This can happen if the last window is partial or if we want the conceptual end
            last_time_in_window = self.ds["time"].isel(time=t_end_idx -1).values.item()
            time_step_duration = np.timedelta64(self.times[1] - self.times[0]) # Assuming regular steps
            t_end = last_time_in_window + time_step_duration


        return np.datetime64(t_start), np.datetime64(t_end)

    def close(self) -> None:
        """
        Close the underlying xarray Dataset.
        """
        if hasattr(self, "ds") and self.ds is not None:
            self.ds.close()
            self.ds = None
