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

    Loads a coordinate template from .npy and provides coordinates for
    each timestep in the requested window.

    Expected coordinate template format:
    - .npy with shape (N, 2) and columns [lat, lon] in degrees.

    Optional geoinfo format:
    - .npy with shape (N, G) where G == len(geoinfo_channels).
    """

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        frequency: str | int | float | np.timedelta64,
        stream_info: dict,
        ref_reader: DataReaderBase,
        geoinfos_filename: Path | None = None,
    ) -> None:
        """Construct data reader for offgrid inference.

        Parameters
        ----------
        tw_handler :
            TimeWindowHandler defining the temporal window for inference.
        filename :
            Path to coords template .npy with shape (N, 2).
        frequency :
            Temporal spacing between consecutive offgrid samples.
        stream_info :
            Stream configuration dictionary.
        ref_reader :
            Reference reader used to inherit source/target channel metadata and
            data normalization statistics.
        geoinfos_filename :
            Optional path to geoinfo template .npy with shape (N, G).
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

        # load and validate coordinate template
        coords_arr = np.load(filename)
        if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
            raise ValueError(
                f"Template must be .npy with shape (N, 2) [lat, lon], got {coords_arr.shape}"
            )

        # caches lats and lons
        self.latitudes = _clip_lat(coords_arr[:, 0])
        self.longitudes = _clip_lon(coords_arr[:, 1])

        # number of template points and available temporal samples
        self.n_points = len(self.latitudes)
        self.len = max(0, int((tw_handler.t_end - tw_handler.t_start) / period))

        # inherit stream/channel metadata from the reference reader
        self.source_idx = np.asarray(ref_reader.source_idx, dtype=np.int64)
        self.source_channels = list(ref_reader.source_channels)
        self.target_idx = np.asarray(ref_reader.target_idx, dtype=np.int64)
        self.target_channels = list(ref_reader.target_channels)
        self.target_channel_weights = list(ref_reader.target_channel_weights)

        # inherit data normalization statistics
        self.mean = np.array(ref_reader.mean, copy=True)
        self.stdev = np.array(ref_reader.stdev, copy=True)

        # resolve geoinfo channels: stream config > reference reader > empty
        geoinfo_channels_cfg = stream_info.get("geoinfo_channels")
        if geoinfo_channels_cfg is None and ref_reader is not None:
            geoinfo_channels_cfg = list(ref_reader.geoinfo_channels)
        geoinfo_channels_cfg = list(geoinfo_channels_cfg or [])

        # load optional geoinfos and validate row/column alignment
        if geoinfos_filename is not None:
            geoinfo_arr = np.load(geoinfos_filename)
            if geoinfo_arr.ndim != 2 or geoinfo_arr.shape[0] != self.n_points:
                raise ValueError(
                    f"Geoinfos file {geoinfos_filename} has shape {geoinfo_arr.shape}, "
                    f"expected ({self.n_points}, G)."
                )
            if geoinfo_arr.shape[1] != len(geoinfo_channels_cfg):
                raise ValueError(
                    f"Geoinfos file has {geoinfo_arr.shape[1]} columns but "
                    f"geoinfo_channels has {len(geoinfo_channels_cfg)} entries."
                )
            self.geoinfo_arr = geoinfo_arr.astype(np.float32)
        else:
            if len(geoinfo_channels_cfg) > 0:
                _logger.warning(
                    "geoinfo_channels configured but no geoinfo .npy file was provided; "
                    "offgrid reader will use empty geoinfos."
                )
            geoinfo_channels_cfg = []
            self.geoinfo_arr = np.zeros((self.n_points, 0), dtype=np.float32)

        # geoinfo metadata and normalization statistics
        self.geoinfo_channels = geoinfo_channels_cfg
        self.geoinfo_idx = np.arange(len(self.geoinfo_channels), dtype=np.int64)

        if len(self.geoinfo_channels) > 0 and geoinfos_filename is not None:
            # derive normalization directly from offgrid geoinfo values
            self.mean_geoinfo = np.mean(self.geoinfo_arr, axis=0).astype(np.float32)
            self.stdev_geoinfo = np.std(self.geoinfo_arr, axis=0).astype(np.float32)
            # avoid division-by-zero during normalization
            near_zero = self.stdev_geoinfo < 1e-6
            if np.any(near_zero):
                self.stdev_geoinfo[near_zero] = 1.0
        else:
            # neutral defaults for empty geoinfos
            self.mean_geoinfo = np.zeros(len(self.geoinfo_channels), dtype=np.float32)
            self.stdev_geoinfo = np.ones(len(self.geoinfo_channels), dtype=np.float32)

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
        self.geoinfo_arr = np.zeros((0, 0), dtype=np.float32)
        self.n_points = 0

    @override
    def length(self) -> int:
        return self.len

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """Get offgrid samples for one time window.

        Parameters
        ----------
        idx : int
            Index of temporal window
        channels_idx : np.array
            Selection of channels

        Returns
        -------
        ReaderData
            Window data with tiled coords/geoinfos, placeholder data fields,
            and datetimes.
        """

        (t_idxs, dtr) = self._get_dataset_idxs(idx)

        if self.len == 0 or len(t_idxs) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )
        assert t_idxs[0] >= 0, "index must be non-negative"

        n_steps = len(t_idxs)
        n_total = self.n_points * n_steps

        # repeat the template coordinates for each timestep in the window
        latlon = np.stack([self.latitudes, self.longitudes], axis=1)
        coords = np.tile(latlon, (n_steps, 1))

        # offgrid reader provides no atmospheric values; keep placeholder zeros
        data = np.zeros((n_total, len(channels_idx)), dtype=np.float32)

        # repeat geoinfos for each timestep
        geoinfos = np.tile(self.geoinfo_arr, (n_steps, 1))

        # compute absolute times for each step, then repeat per grid point
        step_times = self.data_start_time + self.period * t_idxs
        datetimes = np.repeat(step_times, self.n_points)

        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )
        check_reader_data(rd, dtr)

        return rd


def _clip_lat(lats: NDArray) -> NDArray[np.float32]:
    """Clip latitudes to the range [-90, 90] and ensure periodicity."""
    return (2 * np.clip(lats, -90.0, 90.0) - lats).astype(np.float32)


def _clip_lon(lons: NDArray) -> NDArray[np.float32]:
    """Clip longitudes to the range [-180, 180] and ensure periodicity."""
    return ((lons + 180.0) % 360.0 - 180.0).astype(np.float32)
