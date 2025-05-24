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
from pathlib import Path

import numpy as np
import torch
import anemoi
import anemoi.datasets as anemoi_datasets

from weathergen.datasets.data_reader_base import DataReaderBase
from weathergen.datasets.data_reader_base import ReaderData

_logger = logging.getLogger(__name__)


class DataReaderAnemoi( DataReaderBase):
    "Wrapper for Anemoi dataset"

    def __init__(
        self,
        start: int,
        end: int,
        len_hrs: int,
        step_hrs: int,
        filename: Path,
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

        super().__init__( start, end, len_hrs, step_hrs, filename, stream_info)

        # TODO: add support for different normalization modes

        assert len_hrs == step_hrs, "Currently only step_hrs=len_hrs is supported"

        # open  dataset to peak that it is compatible with requested parameters
        ds = anemoi_datasets.open_dataset(filename)

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

        # Ensures that coordinates remain into the interval [-90,90] for latitudes
        # and [-180, 180] for longitudes. Ensures that periodicity has been taken
        # into consideration for the specific intervals.
        self.latitudes = 2 * np.clip(self.latitudes, -90, 90) - self.latitudes

        self.longitudes = (self.longitudes + 180) % 360 - 180

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
        self.mean = ds.statistics["mean"]
        self.stdev = ds.statistics["stdev"]

        # set dataset to None when no overlap with time range
        if dt_start >= ds_dt_end or dt_end <= ds_dt_start:
            self.ds = None
        else:
            self.ds = anemoi_datasets.open_dataset(ds, frequency=str(step_hrs) + "h", start=dt_start, end=dt_end)
            self.len = len(self.ds)

    def _get( self, idx: int, channels_idx: np.array) -> ReaderData :
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

        return (self.ds.dates[idx], self.ds.dates[idx] + np.timedelta64(self.len_hrs, "h"))
