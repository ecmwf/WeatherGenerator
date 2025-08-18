import json
import logging
from pathlib import Path
from typing import override

import numpy as np
import xarray as xr

from weathergen.datasets.data_reader_anemoi import _clip_lat, _clip_lon

from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)

_logger = logging.getLogger(__name__)


class DataReaderCams(DataReaderTimestep):
    "Wrapper for CAMs data variables"

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
    ) -> None:
        """
        Parameters
        ----------
        tw_handler : TimeWindowHandler
            Handles temporal slicing and mapping from time indices to datetime
        filename :
            filename (and path) of dataset
        stream_info : dict
            Stream metadata
        """

        # ======= Reading the Dataset ================

        # Open the dataset using Xarray with Zarr engine
        self.ds = xr.open_dataset(filename, engine="zarr")

        # Column (variable) names and indices
        self.colnames = stream_info["variables"] # list(self.ds)
        self.cols_idx = np.array(list(np.arange(len(self.colnames))))

        # Load associated statistics file for normalization
        stats_filename = Path(filename).with_name(Path(filename).stem + "_stats.json")
        with open(stats_filename) as stats_file:
            self.stats = json.load(stats_file)

        # Variables included in the stats
        self.stats_vars = list(self.stats)

        # Load mean and standard deviation per variable
        self.mean = np.array([self.stats[var]["mean"] for var in self.stats_vars], dtype=np.float64)
        self.stdev = np.array([self.stats[var]["std"] for var in self.stats_vars], dtype=np.float64)

        # Extract coordinates and pressure level
        self.lat =  _clip_lat(self.ds["latitude"].values)
        self.lon =  _clip_lon(self.ds["longitude"].values)

        # Time range in the dataset
        self.time = self.ds["time"].values
        start_ds = np.datetime64(self.time[0])
        end_ds = np.datetime64(self.time[-1])
        self.temporal_frequency = self.time[1] - self.time[0]

        # # Skip stream if it doesn't intersect with time window
        # print(f"start_ds = {start_ds}")
        # print(f"tw_handler.t_end  = {tw_handler.t_end}")
        # print(f"end_ds = {end_ds}")
        # print(f"tw_handler.t_start = {tw_handler.t_start}")
        # """
        # 0: start_ds = 2017-10-01T00:00:00.000000000
        # 0: tw_handler.t_end  = 2017-01-03T00:00:00.000000
        # 0: end_ds = 2022-05-31T21:00:00.000000000
        # 0: tw_handler.t_start = 2017-01-01T00:00:00.000000

        # """

        if start_ds > tw_handler.t_end or end_ds < tw_handler.t_start:
            # print("inside skipping stream")
            name = "plop" # stream_info["name"]
            _logger.warning(f"{name} is not supported over data loader window. Stream is skipped.")
            super().__init__(tw_handler, stream_info)
            self.init_empty()
            return

        # Initialize parent class with resolved time window
        super().__init__(
            tw_handler,
            stream_info,
            start_ds,
            end_ds,
            self.temporal_frequency,
        )

        # Compute absolute start/end indices in the dataset based on time window
        self.start_idx = (tw_handler.t_start - start_ds).astype("timedelta64[ns]").astype(int)
        self.end_idx = (tw_handler.t_end - start_ds).astype("timedelta64[ns]").astype(int) + 1

        # Asma TODO check if self.len checks out
        # Number of time steps in selected range
        self.len = self.end_idx - self.start_idx + 1

        # Placeholder; currently unused
        self.step_hrs = 1

        # Stream metadata
        self.properties = {
            "stream_id": 0,
        }

        # === Normalization statistics ===

        # Ensure stats match dataset columns
        assert self.stats_vars == self.colnames, (
            f"Variables in normalization file {self.stats_vars} do not match "
            f"dataset columns {self.colnames}"
        )

        # === Channel selection ===

        # Source channels and levels
        source_channels = stream_info.get("source")
        if source_channels:
            self.source_channels, self.source_idx = self.select(source_channels)
        else:
            self.source_channels = self.colnames
            self.source_idx = self.cols_idx
        # self.source_levels = self.get_levels(self.source_channels)

        # Target channels and levels
        target_channels = stream_info.get("target")
        if target_channels:
            self.target_channels, self.target_idx = self.select(target_channels)
        else:
            self.target_channels = self.colnames
            self.target_idx = self.cols_idx
        # self.target_levels = self.get_levels(self.target_channels)

        

        # Ensure all selected channels have valid standard deviations
        selected_channel_indices = list(set(self.source_idx).union(set(self.target_idx)))
        non_positive_stds = np.where(self.stdev[selected_channel_indices] <= 0)[0]
        assert len(non_positive_stds) == 0, (
            f"Abort: Encountered non-positive standard deviations for selected columns "
            f"{[self.colnames[selected_channel_indices][i] for i in non_positive_stds]}."
        )

        # === Geo-info channels (currently unused) ===
        self.geoinfo_channels = []
        self.geoinfo_idx = []


    def select(self, ch_filters: list[str]) -> (np.array, list[str]):
        """
        Allow user to specify which columns they want to access.
        Get functions only returned for these specified columns.
        Parameters
        ----------
        ch_filters: list[str]
            list of patterns to access
        Returns
        -------
        selected_colnames: np.array,
            Selected columns according to the patterns specified in ch_filters
        selected_cols_idx
            respective index of these patterns in the data array
        """
        mask = [np.array([f in c for f in ch_filters]).any() for c in self.colnames]

        selected_cols_idx = self.cols_idx[np.where(mask)[0]]
        selected_colnames = [self.colnames[int(i)] for i in np.where(mask)[0]]

        return selected_colnames, selected_cols_idx

    # # Asma TODO test it once kerchunk is ready
    # def get_levels(self, channels: list[str]) -> list:

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self.len = 0

    @override
    def length(self) -> int:
        """
        Length of dataset
        Parameters
        ----------
        None
        Returns
        -------
        length of dataset
        """
        return self.len


    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """
        Get data for temporal window
        Parameters
        ----------
        idx : int
            Index of temporal window
        channels_idx : np.array
            Selection of channels
        Returns
        -------
        data (coords, geoinfos, data, datetimes)
        """

        (t_idxs, dtr) = self._get_dataset_idxs(idx)

        if self.ds is None or self.len == 0 or len(t_idxs) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        # TODO: handle sub-sampling
        # print(f"#2.5.1 inside DataReaderCams._get()", flush=True)
        t_idxs_start = t_idxs[0]
        t_idxs_end = t_idxs[-1] + 1

        # datetime
        datetimes = self.time[t_idxs_start:t_idxs_end]

        # =========== lat/lon coordinates + tiling to match time steps ==========

        # making lon, lat like a mesh for easier flattening later
        lon2d, lat2d = np.meshgrid(self.lon, self.lat)

        # Flatten to match (lat, lon) storage order in your array
        lat = lat2d.flatten()[:, np.newaxis]   # shape (241*480,)
        lon = lon2d.flatten()[:, np.newaxis]

        lat = np.tile(lat, len(datetimes))
        lon = np.tile(lon, len(datetimes))

        coords = np.concatenate([lat, lon], axis=0)

        # data
        channels = np.array(self.colnames)[channels_idx]
        # print(f"#2.5.2 inside DataReaderCams._get() before data", flush=True)
        # for ch_ in channels:
        #     print(f"self.ds[{ch_}] = {self.ds[ch_].shape}", flush=True)
        data_reshaped = [
            np.asarray(self.ds[ch_][t_idxs_start:t_idxs_end, :, :]).reshape(-1, 1) for ch_ in channels
        ]
        # print(f"#2.5.3 inside DataReaderCams._get() after data", flush=True)
        data = np.concatenate(data_reshaped, axis=1)

        # time coordinate repeated to match grid points
        datetimes = np.repeat(datetimes, len(data) // len(t_idxs))

        # empty geoinfos
        geoinfos = np.zeros((data.shape[0], 0), dtype=data.dtype)
        # print(f"self.lon.shape = {self.lon.shape} self.lat.shape = {self.lat.shape}",flush=True)
        # print(f"lon.shape = {lon.shape} lat.shape = {lat.shape}",flush=True)
        # print(f"datetimes.shape = {datetimes.shape}",flush=True)
        # print(f"data.shape = {data.shape}",flush=True)
        # print(f"coords.shape = {coords.shape}",flush=True)


        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )
        check_reader_data(rd, dtr)

        return rd