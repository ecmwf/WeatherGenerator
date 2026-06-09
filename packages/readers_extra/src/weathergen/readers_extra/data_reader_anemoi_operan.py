# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from pathlib import Path
from typing import override

import numpy as np
from anemoi.datasets.data import MissingDateError

from weathergen.datasets.data_reader_anemoi import DataReaderAnemoi
from weathergen.datasets.data_reader_base import (
    ReaderData,
    TimeWindowHandler,
    TIndex,
)
from weathergen.train.utils import Stage

_logger = logging.getLogger(__name__)


def dt2cal(dt):
    """
    Convert array of datetime64 to a calendar array of year, month, day, hour,
    minute, seconds, microsecond with these quantites indexed on the last axis.

    Parameters
    ----------
    dt : datetime64 array (...)
        numpy.ndarray of datetimes of arbitrary shape

    Returns
    -------
    cal : uint32 array (..., 7)
        calendar array with last axis representing year, month, day, hour,
        minute, second, microsecond
    """

    # allocate output
    out = np.empty(dt.shape + (7,), dtype="u4")
    # decompose calendar floors
    year, month, day, hour, min, sec = [dt.astype(f"M8[{x}]") for x in "YMDhms"]
    out[..., 0] = year + 1970  # Gregorian Year
    out[..., 1] = (month - year) + 1  # month
    out[..., 2] = (day - month) + 1  # dat
    out[..., 3] = (dt - day).astype("m8[h]")  # hour
    out[..., 4] = (dt - hour).astype("m8[m]")  # minute
    out[..., 5] = (dt - min).astype("m8[s]")  # second
    out[..., 6] = (dt - sec).astype("m8[us]")  # microsecond
    return out


class DataReaderAnemoiOperan(DataReaderAnemoi):
    "Wrapper for Anemoi datasets"

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
        stage: Stage,
    ) -> None:
        """
        Construct data reader for anemoi dataset

        Parameters
        ----------
        filename :
            filename (and path) of dataset
        stream_info :
            information about stream

        Returns
        -------
        None
        """

        super().__init__(tw_handler, filename, stream_info, stage)

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """
        Get data for window (for either source or target, through public interface)

        Parameters
        ----------
        idx : int
            Index of temporal window
        channels_idx : np.array
            Selection of channels

        Returns
        -------
        ReaderData providing coords, geoinfos, data, datetimes
        """

        t_idxs, dtr = self._get_dataset_idxs(idx)
        if self.ds is None or self.len == 0 or len(t_idxs) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        # get additional timestep to ensure we have one valid timestep
        t_idxs = np.insert(t_idxs, 0, t_idxs[0] - 1)

        didx_start = t_idxs[0]
        didx_end = t_idxs[-1] + 1
        datetimes = self.ds.dates[didx_start:didx_end]
        datetimes_split = dt2cal(datetimes)

        # compute corrected datetimes that account for actual availability
        nts = self.stream_info["nominal_time_mapping"]
        deltas = [int(nts[str(hour)]) - int(hour) for hour in datetimes_split[:, 3]]
        datetimes_offset = [
            dt + np.timedelta64(delta, "h") for dt, delta in zip(datetimes, deltas, strict=False)
        ]

        # use latest available sample that is valid w.r.t the input data window
        datetimes_mask = [dt < dtr.end for dt in datetimes_offset]
        if np.array(datetimes_mask).sum() == 0:
            t_idxs = []
        else:
            t_idxs = [t_idxs[datetimes_mask][-1].item()]

        # _get from DataReaderAnemoi

        if self.ds is None or self.len == 0 or len(t_idxs) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        assert t_idxs[0] >= 0, "index must be non-negative"
        didx_start = t_idxs[0]
        # End is inclusive
        didx_end = t_idxs[-1] + 1

        # extract number of time steps and collapse ensemble dimension
        # ds is a wrapper around zarr with get_coordinate_selection not being exposed since
        # subsetting is pushed to the ctor via frequency argument; this also ensures that no sub-
        # sampling is required here
        try:
            data = self.ds[didx_start:didx_end][:, :, 0].astype(np.float32)
        except MissingDateError as e:
            _logger.debug(f"Date not present in anemoi dataset: {str(e)}. Skipping.")
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        # coords-first representation and collapse multiple steps
        data = data.transpose([0, 2, 1]).reshape((data.shape[0] * data.shape[2], -1))

        # extract geoinfo channels (can be time-varying, so read from dataset)
        geoinfos = data[:, list(self.geoinfo_idx)]
        # extract channels
        data = data[:, list(channels_idx)]

        # construct lat/lon coords
        latlon = np.concatenate(
            [
                np.expand_dims(self.latitudes, 0),
                np.expand_dims(self.longitudes, 0),
            ],
            axis=0,
        ).transpose()
        # repeat latlon len(t_idxs) times
        coords = np.vstack((latlon,) * len(t_idxs))

        # date time matching #data points of data
        # Assuming a fixed frequency for the dataset
        datetimes = np.repeat(self.ds.dates[didx_start:didx_end], len(data) // len(t_idxs))

        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )
        # check_reader_data(rd, dtr)

        return rd
