import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr


class RadklimDataset:
    """
    Data loader to read the monthly netCDF-files of the RADKLIM dataset.

    Parameters
    ----------
    start_time : int or str
        Inclusive start timestamp in YYYYMMDDHHMM format.
    end_time : int or str
        Inclusive end timestamp in YYYYMMDDHHMM format.
    len_hrs : int
        Window length in hours.
    step_hrs : int
        Step between windows in hours (not used!).
    filename : str or Path
        Root directory with NetCDF files organized by year/month.
    normalization_file : str or Path
        JSON file with "mean" and "std" lists matching sorted variables.
    fname_patt : str
        Filename prefix (default "RW_2017.002_").
    """

    def __init__(
        self,
        start_time: int | str,
        end_time: int | str,
        len_hrs: int,
        step_hrs: int,
        filename: str | Path,
        normalization_file: str | Path,
        fname_patt: str = "RW_2017.002_",  # file prefix
    ):
        # Parse parameters
        self.start_time = pd.to_datetime(str(start_time), format="%Y%m%d%H%M")
        self.end_time = pd.to_datetime(str(end_time), format="%Y%m%d%H%M")
        self.len_hrs = len_hrs
        self.step_hrs = step_hrs
        if not self.step_hrs == self.len_hrs:
            raise ValueError(
                f"Parameters step_hrs {step_hrs} and len_hrs {len_hrs} must be the same."
            )
        self.data_path = Path(
            filename
        )  # rename to path (filename is ekept for consistency with other dataloaders)
        self.fname_patt = fname_patt

        # get list of required files
        self.file_list = self._get_file_list()
        if not self.file_list:
            raise FileNotFoundError(
                f"No files matching pattern '{self.fname_patt}' in {self.data_path}"
                f" between {self.start_time} and {self.end_time}."
            )

        # Retrieve metadata
        self.variables = ["RR"]  # TODO: allow support for YW product

        # Read normalization data
        # If normaliation file does not exist, look for it under the data path
        normalization_file = Path(normalization_file)
        if not normalization_file.is_file():
            normalization_file = self.data_path / normalization_file

        if not normalization_file.is_file():
            raise FileNotFoundError(f"Normalization file '{normalization_file}' not found.")
        with open(normalization_file) as f:
            stats = json.load(f)

        means = stats.get("mean")
        stds = stats.get("std")
        if not (isinstance(means, list) and isinstance(stds, list)):
            raise ValueError("Normalization JSON must have 'mean' and 'std' lists.")

        if len(means) != len(self.variables) or len(stds) != len(self.variables):
            raise ValueError(
                f"Stats length {len(means)}/{len(stds)} != num variables {len(self.variables)}"
            )
        self.mean = np.array(means, dtype=np.float32)
        self.std = np.array(stds, dtype=np.float32)

        ds = xr.open_mfdataset(
            self.file_list,
            combine="nested",
            concat_dim="time",
            engine="netcdf4",
            parallel=False,
            data_vars="minimal",
            coords="minimal",
            compat="override",
        )

        # get relevant metadata from dataset
        # time-coordinate
        times_all = ds["time"].load()

        # get start and end indices for slicing of dataset
        self.start_idx = int(np.searchsorted(times_all, np.datetime64(self.start_time)))
        self.end_idx = int(np.searchsorted(times_all, np.datetime64(self.end_time), side="right"))

        times_req = times_all.isel({"time": slice(self.start_idx, self.end_idx)})
        dt = times_req.diff(dim="time")
        assert len(np.unique(dt)) == 1, "Inconsistent time step length in dataset."
        self.times = times_req.values

        self.num_steps_per_window = int((len_hrs * 3600) / (dt[0] / np.timedelta64(1, "s")))

        # handle spatial coordinates
        y1d = ds["y"].values.astype(np.float32)
        x1d = ds["x"].values.astype(np.float32)
        # 2D geographic coords
        lat2d = ds["lat"].values.astype(np.float32)
        lon2d = ds["lon"].values.astype(np.float32)

        self.ny, self.nx = len(y1d), len(x1d)
        if lat2d.shape != (self.ny, self.nx) or lon2d.shape != (self.ny, self.nx):
            raise ValueError("lat/lon shape mismatch with y/x dims.")
        # clip/wrap
        self.latitudes = 2 * np.clip(lat2d, -90, 90) - lat2d
        self.longitudes = (lon2d + 180) % 360 - 180
        self.latlon_sh = np.shape(self.latitudes)

        ds = ds[self.variables].isel({"time": slice(self.start_idx, self.end_idx)})

        # re-chunk the data for efficient data retrieval
        self.ds = ds.chunk({"time": self.num_steps_per_window * 2, "y": -1, "x": -1})

    def __len__(self) -> int:
        """
        Length of dataset

        Parameters
        ----------
        None

        Returns
        -------
        length of dataset
        """
        return len(self.ds["time"]) - self.num_steps_per_window

    def _get(self, idx: int) -> tuple:
        """
        Get data for window

        Parameters
        ----------
        idx : int
            Index of temporal window

        Returns
        -------
        data (coords, geoinfos, data, datetimes)
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range (0..{len(self) - 1}).")
        # slice required data
        ds_now = self.ds.isel(
            {"time": slice(idx, idx + self.num_steps_per_window)}
        )  # shape (len_hrs, nvar, y, x)
        arr = ds_now.to_array("var").load()
        arr = arr.transpose("time", "y", "x", "var")
        nt, ny, nx, nc = arr.shape
        arr = arr.values.reshape(-1, nc)

        # get coords
        lats_data, lons_data = (
            np.broadcast_to(self.latitudes, (nt, *self.latlon_sh)),
            np.broadcast_to(self.longitudes, (nt, *self.latlon_sh)),
        )
        lats_data, lons_data = lats_data.reshape(-1), lons_data.reshape(-1)
        latlon = np.column_stack((lats_data, lons_data))

        tda = ds_now["time"].expand_dims({"y": ny, "x": nx}, axis=[1, 2]).data.reshape(-1)

        # filter for Nan
        mask = np.any(np.isnan(arr), axis=1)
        idx_valid = np.where(~mask)[0]

        latlon = latlon[idx_valid, :]
        arr = arr[idx_valid, :]
        tda = tda[idx_valid]

        # placeholder for geoinfos
        geoinfos = np.zeros((arr.shape[0], 0), dtype=np.float32)

        return latlon, geoinfos, arr, tda

    def get_source(self, idx: int) -> tuple:
        """
        Get source data for idx

        Parameters
        ----------
        idx : int
            Index of temporal window

        Returns
        -------
        source data (coords, geoinfos, data, datetimes)
        """
        return self._get(idx)

    def get_target(self, idx: int) -> tuple:
        """
        Get target data for idx

        Parameters
        ----------
        idx : int
            Index of temporal window

        Returns
        -------
        target data (coords, geoinfos, data, datetimes)
        """
        return self._get(idx)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get data for idx

        Parameters
        ----------
        idx : int
            Index of temporal window

        Returns
        -------
        data (coords, geoinfos, data, datetimes)
        """
        return self._get(idx)

    def get_source_num_channels(self) -> int:
        """
        Get number of source channels

        Parameters
        ----------
        None

        Returns
        -------
        int
        number of source channels
        """
        return len(self.variables)

    def get_target_num_channels(self) -> int:
        """
        Get number of target channels

        Parameters
        ----------
        None

        Returns
        -------
        int
        number of target channels
        """
        return len(self.variables)

    def get_coords_size(self) -> int:
        """
        Get size of coords

        Parameters
        ----------
        None

        Returns
        -------
        size of coords
        """
        return 2

    def get_geoinfo_size(self) -> int:
        """
        Get size of geoinfos

        Parameters
        ----------
        None

        Returns
        -------
        size of geoinfos
        """
        return 0

    def normalize_coords(self, coords: torch.tensor) -> torch.tensor:
        """
        Normalize coordinates

        Parameters
        ----------
        coords :
            coordinates to be normalized

        Returns
        -------
        Normalized coordinates
        """
        coords[..., 0] = np.sin(np.deg2rad(coords[..., 0]))
        coords[..., 1] = np.sin(0.5 * np.deg2rad(coords[..., 1]))

        return coords

    def normalize_geoinfos(self, geoinfos: torch.tensor) -> torch.tensor:
        """
        Normalize geoinfos

        Parameters
        ----------
        geoinfos :
            geoinfos to be normalized

        Returns
        -------
        Normalized geoinfo
        """

        assert geoinfos.shape[-1] == 0, "incorrect number of geoinfo channels"
        return geoinfos

    def normalize_source_channels(self, data: torch.tensor) -> torch.tensor:
        """
        Normalize source channels

        Parameters
        ----------
        data :
            data to be normalized

        Returns
        -------
        Normalized data
        """
        return (data - self.mean) / self.std

    def normalize_target_channels(self, data: torch.tensor) -> torch.tensor:
        """
        Normalize target channels

        Parameters
        ----------
        data :
            data to be normalized

        Returns
        -------
        Normalized data
        """
        return (data - self.mean) / self.std

    def denormalize_source_channels(self, data: torch.tensor) -> torch.tensor:
        """
        Denormalize source channels

        Parameters
        ----------
        data :
            data to be denormalized

        Returns
        -------
        Denormalized data
        """
        return data * self.std + self.mean

    def denormalize_target_channels(self, data: torch.tensor) -> torch.tensor:
        """
        Denormalize target channels

        Parameters
        ----------
        data :
            data to be denormalized (target or pred)

        Returns
        -------
        Denormalized data
        """
        return data * self.std + self.mean

    def time_window(self, idx: int):
        """
        Temporal window corresponding to index

        Parameters
        ----------
        idx :
            index of temporal window

        Returns
        -------
            start and end data of temporal window
        """
        if not self.ds:
            return (np.array([], dtype=np.datetime64), np.array([], dtype=np.datetime64))

        return (self.ds.dates[idx], self.ds.dates[idx + self.num_steps_per_window])

    def _get_file_list(self):
        """
        Get list of files matching the pattern in the data path
        between start and end time

        Parameters
        ----------
        None

        Returns
        -------
        list of files matching the pattern
        """
        # Generate months start sequence
        end = self._last_day_of_month(self.end_time)
        months = pd.date_range(self.start_time.strftime("%Y-%m-01"), end, freq="ME")
        files = [
            self.data_path / m.strftime("%Y") / f"{self.fname_patt}{m.strftime('%Y%m')}.nc"
            for m in months
            if (
                self.data_path / m.strftime("%Y") / f"{self.fname_patt}{m.strftime('%Y%m')}.nc"
            ).is_file()
        ]
        return files

    def close(self):
        """
        Close the dataset.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.ds.close()

    @staticmethod
    def _last_day_of_month(any_day: pd.Timestamp) -> pd.Timestamp:
        """
        TODO: Move to utils
        Returns the last day of a month

        Parameters
        ----------
        any_day : datetime object with any day of the month
            any day of the month

        Returns
        -------
            datetime object of lat day of month
        """
        next_month = any_day.replace(day=28) + pd.Timedelta(days=4)  # this will never fail
        return next_month - pd.Timedelta(days=next_month.day)
