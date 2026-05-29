# (C) Copyright 2026 WeatherGenerator contributors.
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
from numpy.typing import NDArray

from weathergen.common.config import parse_timedelta
from weathergen.datasets.data_reader_base import (
    DataReaderBase,
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)

_logger = logging.getLogger(__name__)


class DataReaderOffgrid(DataReaderTimestep):
    """Offgrid datareader.

    1) loads a template grid from a numpy file,
    2) uses a configured frequency,
    3) generates coords + datetimes for inference

    Expected template file format:
    - .npy file with shape (N, 2), columns [lat, lon] in degrees

    Required stream_info entries:
    - name
    - frequency (for example "6h")
    """

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        frequency: str | int | float | np.timedelta64,
        stream_info: dict,
        ref_reader: DataReaderBase | None = None,
    ) -> None:
        """
        Construct data reader for offgrid inference

        Parameters
        ----------
        tw_handler :
            TimeWindowHandler defining the temporal window for inference
        filename :
            filename (and path) of dataset
        frequency :
            temporal spacing of offgrid samples
        stream_info :
            information about stream
        ref_reader :
            optional reference reader to inherit metadata from (e.g. source/target channels, normalization)

        Returns
        -------
        None
        """

        # parse frequency into numpy timedelta64
        period = parse_timedelta(frequency)

        # initialize base class with time window and frequency info
        super().__init__(
            tw_handler,
            stream_info,
            data_start_time=tw_handler.t_start,
            data_end_time=tw_handler.t_end,
            period=period,
        )

        # load grid template
        grid = np.load(filename)
        if grid.ndim != 2 or grid.shape[1] != 2:
            raise ValueError(
                f"Template must be .npy with shape (N, 2) [lat, lon], got {grid.shape}"
            )

        # caches lats and lons
        self.latitudes = _clip_lat(grid[:, 0].astype(np.float32))
        self.longitudes = _clip_lon(grid[:, 1].astype(np.float32))
        self.n_points = len(self.latitudes)

        # number of time steps that fit in the requested window
        self.len = max(0, int((tw_handler.t_end - tw_handler.t_start) / period))

        # Optionally inherit stream/channel metadata from a reference reader
        if ref_reader is not None:
            # select/filter requested source channels
            self.source_idx = np.asarray(ref_reader.source_idx, dtype=np.int64)
            self.source_channels = list(ref_reader.source_channels)

            # select/filter requested target channels
            self.target_idx = np.asarray(ref_reader.target_idx, dtype=np.int64)
            self.target_channels = list(ref_reader.target_channels)

            # set target channel weights
            self.target_channel_weights = list(ref_reader.target_channel_weights)

            # set normalization parameters
            self.mean = np.array(ref_reader.mean, copy=True)
            self.stdev = np.array(ref_reader.stdev, copy=True)

        # if not provided, initialize with empty metadata and neutral normalization
        else:
            # empty source channels (needed from base class)
            self.source_idx: NDArray[np.int64] = np.array([], dtype=np.int64)
            self.source_channels: list[str] = []

            # empty target channels (needed from base class)
            self.target_idx: NDArray[np.int64] = np.array([], dtype=np.int64)
            self.target_channels: list[str] = []

            # empty target channel weights
            self.target_channel_weights: list[float] = []

            # neutral normalization
            self.mean = np.zeros(0, dtype=np.float32)
            self.stdev = np.ones(0, dtype=np.float32)

        # TODO add support for geoinfos
        self.geoinfo_channels: list[str] = []
        self.geoinfo_idx = np.array([], dtype=np.int64)
        self.mean_geoinfo = np.zeros(0, dtype=np.float32)
        self.stdev_geoinfo = np.ones(0, dtype=np.float32)

        ds_name = stream_info["name"]
        _logger.info(
            f"{ds_name}: offgrid reader active (source={len(self.source_channels)}, "
            f"target={len(self.target_channels)}, geoinfo={len(self.geoinfo_channels)})."
        )

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self.len = 0
        self.latitudes = np.zeros(0, dtype=np.float32)
        self.longitudes = np.zeros(0, dtype=np.float32)
        self.n_points = 0

    @override
    def length(self) -> int:
        return self.len

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

        (t_idxs, dtr) = self._get_dataset_idxs(idx)

        if self.len == 0 or len(t_idxs) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=0
            )
        assert t_idxs[0] >= 0, "index must be non-negative"

        n_steps = len(t_idxs)
        n_total = self.n_points * n_steps

        # tile lat/lon pair for each time step
        latlon = np.stack([self.latitudes, self.longitudes], axis=1)
        coords = np.tile(latlon, (n_steps, 1))

        # no atmospheric data fields, using zeros as placeholder
        data = np.zeros((n_total, len(channels_idx)), dtype=np.float32)

        # TODO add support for geoinfos
        geoinfos = np.zeros((n_total, 0), dtype=np.float32)

        # compute absolute times for each step, then repeat per grid point
        step_times = self.data_start_time + self.period * t_idxs
        datetimes = np.repeat(step_times, self.n_points)

        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes
        )
        check_reader_data(rd, dtr)

        return rd

def _clip_lat(lats: NDArray) -> NDArray[np.float32]:
    """Clip latitudes to the range [-90, 90] and ensure periodicity."""
    return (2 * np.clip(lats, -90.0, 90.0) - lats).astype(np.float32)

def _clip_lon(lons: NDArray) -> NDArray[np.float32]:
    """Clip longitudes to the range [-180, 180] and ensure periodicity."""
    return ((lons + 180.0) % 360.0 - 180.0).astype(np.float32)