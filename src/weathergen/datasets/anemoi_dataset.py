# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging

import numpy as np
import torch
from anemoi.datasets import open_dataset
import pandas as pd
_logger = logging.getLogger(__name__)


class AnemoiDataset:
    "Wrapper for Anemoi dataset"

    def __init__(
        self,
        start: int,
        end: int,
        len_hrs: int,
        step_hrs: int,
        filename: str,
        stream_info: dict,
    ) -> None:
        """
        Construct dataset based on anemoi dataset

        Parameters
        ----------
        start : int
            Start time
        end : int
            End time
        len_hrs : int
            length of data window
        step_hrs :
            delta hours between start times of windows
        filename :
            filename (and path) of dataset
        stream_info :
            information about stream

        Returns
        -------
        None
        """

        # TODO: add support for different normalization modes

        assert len_hrs == step_hrs, "Currently only step_hrs=len_hrs is supported"

        # open  dataset to peak that it is compatible with requested parameters
        ds = open_dataset(filename)

        # check that start and end time are within the dataset time range

        ds_dt_start = ds.dates[0]
        ds_dt_end = ds.dates[-1]

        format_str = "%Y%m%d%H%M%S"
        dt_start = datetime.datetime.strptime(str(start), format_str)
        dt_end = datetime.datetime.strptime(str(end), format_str)

        # TODO, TODO, TODO: we need proper alignment for the case where self.ds.frequency
        # is not a multile of len_hrs
        self.num_steps_per_window = int((len_hrs * 3600) / ds.frequency.seconds)

        # open dataset

        # caches lats and lons
        self.latitudes = ds.latitudes.astype(np.float32)
        self.longitudes = ds.longitudes.astype(np.float32)

        # TODO: define in base class
        self.geoinfo_idx = []

        # Determine source and target channels, filtering out forcings etc and using
        # specified source and target channels if specified
        source_channels = stream_info["source"] if "source" in stream_info else None
        self.source_idx = np.sort(
            [
                ds.name_to_index[k]
                for i, (k, v) in enumerate(ds.typed_variables.items())
                if (
                    not v.is_computed_forcing
                    and not v.is_constant_in_time
                    and (
                        np.array([f in k for f in source_channels]).any()
                        if source_channels
                        else True
                    )
                )
            ]
        )
        target_channels = stream_info["target"] if "target" in stream_info else None
        self.target_idx = np.sort(
            [
                ds.name_to_index[k]
                for (k, v) in ds.typed_variables.items()
                if (
                    not v.is_computed_forcing
                    and not v.is_constant_in_time
                    and (
                        np.array([f in k for f in target_channels]).any()
                        if target_channels
                        else True
                    )
                )
            ]
        )
        self.source_channels = [ds.variables[i] for i in self.source_idx]
        self.target_channels = [ds.variables[i] for i in self.target_idx]

        self.properties = {
            "stream_id": 0,
        }

        # sigma_t = pd.read_csv("./grouped_standard_deviations.csv")
        # wg_sigma_t = np.array(
        #     [
        #         sigma_t.loc[sigma_t.group == var].SF_temporal.values[0]
        #         if var in sigma_t.group.values
        #         else 1.0
        #         for i, var in enumerate(ds.variables)
        #     ]
        # )

        # self.tendency_mean = ds.statistics_tendencies()["mean"]
        # self.tendency_stdev = ds.statistics_tendencies()["stdev"]
        # self.tendency_stdev = self.tendency_stdev #/self.tendency_stdev.mean()
        # self.tendency_stdev = np.where(self.tendency_stdev < 5e-4, 5e-4, self.tendency_stdev) #cap very small values
        # breakpoint()
        # self.sigma_t = ds.statistics_tendencies()["stdev"]
        # self.sigma_t[ self.sigma_t  == 0.0] = 1

        self.tendency_mean  = ds.statistics_tendencies()["mean"]
        self.tendency_stdev = ds.statistics_tendencies()["stdev"]

        for i, var in enumerate(ds.variables):
            if var not in self.source_channels+ self.target_channels:
                continue
            # print(f"{var}: WG {wg_sigma_t[i]}, ANEMOI mean: {tendency_anemoi_mean[i]}, ANEMOI std: {self.sigma_t[i]}")
            print(f"{var}: ANEMOI mean: {self.tendency_mean[i]}, ANEMOI std: {self.tendency_stdev[i]}")


        self.mean = ds.statistics["mean"]
        self.stdev = ds.statistics["stdev"]

        # set dataset to None when no overlap with time range
        if dt_start >= ds_dt_end or dt_end <= ds_dt_start:
            self.ds = None
        else:
            self.ds = open_dataset(ds, frequency=str(step_hrs) + "h", start=dt_start, end=dt_end)

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
        if not self.ds:
            return 0

        return len(self.ds)

    def get_source(self, idx: int) -> tuple[np.array, np.array, np.array, np.array]:
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
        return self._get(idx, self.source_idx)

    def get_target(self, idx: int) -> tuple[np.array, np.array, np.array, np.array]:
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
        return self._get(idx, self.target_idx)

    def _get(
        self, idx: int, channels_idx: np.array
    ) -> tuple[np.array, np.array, np.array, np.array]:
        """
        Get data for window

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

        if not self.ds:
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
            )

        # extract number of time steps and collapse ensemble dimension

        data = self.ds[idx : idx + self.num_steps_per_window][:, :, 0]

        # # extract channels
        data = (
            data[:, channels_idx].transpose([0, 2, 1]).reshape((data.shape[0] * data.shape[2], -1))
        )

        # construct lat/lon coords
        latlon = np.concatenate(
            [
                np.expand_dims(self.latitudes, 0),
                np.expand_dims(self.longitudes, 0),
            ],
            0,
        ).transpose()
        latlon = np.repeat(latlon, self.num_steps_per_window, axis=0).reshape((-1, latlon.shape[1]))

        # empty geoinfos for anemoi
        geoinfos = np.zeros((data.shape[0], 0), dtype=data.dtype)

        # date time matching #data points of data
        datetimes = np.repeat(
            np.expand_dims(self.ds.dates[idx : idx + self.num_steps_per_window], 0),
            data.shape[0],
            axis=0,
        ).flatten()

        return (latlon, geoinfos, data, datetimes)

    def get_source_num_channels(self) -> int:
        """
        Get number of source channels

        Parameters
        ----------
        None

        Returns
        -------
        number of source channels
        """
        return len(self.source_idx)

    def get_target_num_channels(self) -> int:
        """
        Get number of target channels

        Parameters
        ----------
        None

        Returns
        -------
        number of target channels
        """
        return len(self.target_idx)

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
        return len(self.geoinfo_idx)

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

    def normalize_source_channels(self, source: torch.tensor) -> torch.tensor:
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
        assert source.shape[-1] == len(self.source_idx), "incorrect number of channels"
        hasnan = False
        for i, ch in enumerate(self.source_idx):
            source[..., i] = (source[..., i] - self.mean[ch]) / (self.stdev[ch])
            hasnan = source[..., i].isnan().any()
            if hasnan:
                print(f"source channel {i} has nan values")
                print(source[..., i])

        return source

    def normalize_target_channels(self, target: torch.tensor) -> torch.tensor:
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
        assert target.shape[-1] == len(self.target_idx), "incorrect number of channels"
        for i, ch in enumerate(self.target_idx):
            target[..., i] = (target[..., i] - self.mean[ch]) / (self.stdev[ch])

        hasnan = target[..., i].isnan().any()
        if hasnan:
            print(f"target channel {i} has nan values")
            print(target[..., i])

        return target

    def denormalize_source_channels(self, source: torch.tensor) -> torch.tensor:
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
        assert source.shape[-1] == len(self.source_idx), "incorrect number of channels"
        for i, ch in enumerate(self.source_idx):
            source[..., i] = (source[..., i] * self.stdev[ch]) + self.mean[ch] 

        return source

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
        assert data.shape[-1] == len(self.target_idx), "incorrect number of channels"
        for i, ch in enumerate(self.target_idx):
            data[..., i] = (data[..., i] * self.stdev[ch]) + self.mean[ch]

        return data

    def time_window(self, idx: int) -> tuple[np.datetime64, np.datetime64]:
        """
        Temporal window corresponding to index

        Parameters
        ----------
        idx :
            index of temporal window

        Returns
        -------
            start and end of temporal window
        """
        if not self.ds:
            return (np.array([], dtype=np.datetime64), np.array([], dtype=np.datetime64))

        return (self.ds.dates[idx], self.ds.dates[idx])
