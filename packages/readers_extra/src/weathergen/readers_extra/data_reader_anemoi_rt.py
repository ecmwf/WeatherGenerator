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
from typing import override

import anemoi.datasets as _anemoi_datasets
import numpy as np
from anemoi.datasets.data.dataset import Dataset
from earthkit.data.utils.dates import to_datetime
from numpy.typing import NDArray
from omegaconf import OmegaConf

from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
)
from weathergen.train.utils import Stage
from weathergen.utils.distributed import is_root

_logger = logging.getLogger(__name__)


class DataReaderAnemoiRT(DataReaderTimestep):
    """
    Real-time version of DataReaderAnemoi for inference from a pretrained model. This data reader is
    for use for the target stream

    The filename is expected to be a template file containing a grid and static geoinfo such
    as orography and land-sea mask. Dynamic forcings are computed on the fly.
    """

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

        # use _anemoi_config if it's defined; ignore filename in this case
        data_paths = stream_info.get("data_paths", [])
        _anemoi_config = stream_info.get("_anemoi_config")
        if _anemoi_config:
            # convert OmegaConf DictConfig to a plain dict for anemoi.open_dataset.
            filename = OmegaConf.to_container(_anemoi_config, resolve=True)
            # add additional data paths
            for path in data_paths:
                _anemoi_datasets.add_dataset_path(path)
            # provide some visibility since we ignore filename
            if is_root():
                _logger.info("Ignoring filename and using _anemoi_config option.")

        assert stream_info.get("frequency") is not None, "stream_info must contain 'frequency' key"
        # self.frequency = np.timedelta64( timedelta_to_str(stream_info["frequency"]))
        # TODO, TODO, TODO
        self.frequency = np.timedelta64(
            1, "h"
        )  # np.timedelta64( timedelta_to_str(stream_info["frequency"]))

        # open  dataset to peak that it is compatible with requested parameters
        ds: Dataset = _anemoi_datasets.open_dataset(filename)
        self.ds = ds

        super().__init__(
            tw_handler,
            stream_info,
            tw_handler.t_start,
            tw_handler.t_end,
            self.frequency,
        )

        # caches lats and lons
        self.latitudes = _clip_lat(ds.latitudes)
        self.longitudes = _clip_lon(ds.longitudes)

        # channels and geoinfos are taken from stream_info

        # select/filter requested source channels
        assert stream_info.get(str(stage) + "_source_channels") is not None, (
            "pretrained model expected."
        )
        self.source_channels = stream_info.get(str(stage) + "_source_channels")
        self.source_idx = [ds.variables.index(ch) for ch in self.source_channels]

        # select/filter requested target channels
        assert stream_info.get(str(stage) + "_target_channels") is not None, (
            "pretrained model expected."
        )
        self.target_channels = stream_info.get(str(stage) + "_target_channels")
        self.target_idx = [ds.variables.index(ch) for ch in self.target_channels]

        # select/filter requested geoinfo channels (can be any variable, not just constant-in-time)
        assert stream_info.get("geoinfo_channels") is not None, "pretrained model expected."
        self.geoinfo_channels = stream_info.get("geoinfo_channels")
        self.geoinfo_idx = [ds.variables.index(ch) for ch in self.geoinfo_channels]
        self.geoinfo_channels_static, self.geoinfo_channels_dynamic = [], []
        self.geoinfo_idx_static, self.geoinfo_idx_dynamic = [], []
        (
            self.geoinfo_idx_static_lin,
            self.geoinfo_idx_dynamic_lin,
        ) = [], []
        idx = 0
        for _, (k, v) in enumerate(ds.typed_variables.items()):
            if k not in self.geoinfo_channels:
                continue

            if v.is_constant_in_time:
                self.geoinfo_channels_static += [k]
                self.geoinfo_idx_static += [ds.variables.index(k)]
                self.geoinfo_idx_static_lin += [idx]
            else:
                assert k in _anemoi_dynamic_forcings(), (
                    f"Dynamic forcing {k} not implemented in DataReaderAnemoiRT"
                )
                self.geoinfo_channels_dynamic += [k]
                self.geoinfo_idx_dynamic += [ds.variables.index(k)]
                self.geoinfo_idx_dynamic_lin += [idx]
            idx += 1

        # set geoinfo normalization statistics
        if len(self.geoinfo_idx) > 0:
            self.mean_geoinfo = ds.statistics["mean"][self.geoinfo_idx]
            self.stdev_geoinfo = ds.statistics["stdev"][self.geoinfo_idx]
        else:
            self.mean_geoinfo = np.zeros(0)
            self.stdev_geoinfo = np.ones(0)

        self.mean = ds.statistics["mean"]
        self.stdev = ds.statistics["stdev"]

    @override
    def length(self) -> int:
        return 1

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
        ReaderData providing coords, geoinfos, -, datetimes
        """

        (t_idxs, dtr) = self._get_dataset_idxs(idx)

        if self.ds is None or len(t_idxs) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

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

        # use time_window and frequency to compute required time information
        datetimes = []
        t_cur = dtr.start
        while t_cur < dtr.end:
            t_cur += self.frequency
            datetimes += [t_cur]

        # extract geoinfo channels (can be time-varying, so read from dataset)
        geoinfos_static = np.repeat(
            self.ds[0, list(self.geoinfo_idx_static)][0], len(t_idxs), axis=1
        )
        geoinfos_dynamic = _anemoi_get_dynamic_forcings(
            datetimes, self.latitudes, self.longitudes, self.geoinfo_channels_dynamic
        )
        geoinfos = np.empty((coords.shape[0], len(self.geoinfo_idx)))
        for i, data in zip(self.geoinfo_idx_static_lin, geoinfos_static, strict=False):
            geoinfos[:, i] = data
        for i, ch in zip(self.geoinfo_idx_dynamic_lin, self.geoinfo_channels_dynamic, strict=False):
            geoinfos[:, i] = geoinfos_dynamic[ch]

        # extract channels
        data = np.empty((0, len(channels_idx)))

        # date time matching #data points of data
        # Assuming a fixed frequency for the dataset
        datetimes = np.repeat(np.array(datetimes, dtype=np.datetime64), (1, len(self.latitudes)))
        datetimes = datetimes.flatten()

        import code

        code.interact(local=locals())

        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )

        return rd


def _clip_lat(lats: NDArray) -> NDArray[np.float32]:
    """
    Clip latitudes to the range [-90, 90] and ensure periodicity.
    """
    return (2 * np.clip(lats, -90.0, 90.0) - lats).astype(np.float32)


def _clip_lon(lons: NDArray) -> NDArray[np.float32]:
    """
    Clip longitudes to the range [-180, 180] and ensure periodicity.
    """
    return ((lons + 180.0) % 360.0 - 180.0).astype(np.float32)


# Code taken from https://github.com/metno/bris-inference/blob/main/bris/forcings.py
# based on EarthKit's implementation as used in Anemoi datasets


# This is copied from earthkit, probably need to declare this somewhere
def _anemoi_julian_day(date):
    date = to_datetime(date)
    delta = date - datetime.datetime(date.year, 1, 1)
    julian_day = delta.days + delta.seconds / 86400.0
    return julian_day


def _anemoi_cos_julian_day(date):
    radians = _anemoi_julian_day(date) / 365.25 * np.pi * 2
    return np.cos(radians)


def _anemoi_sin_julian_day(date):
    radians = _anemoi_julian_day(date) / 365.25 * np.pi * 2
    return np.sin(radians)


def _anemoi_local_time(date, lon):
    date = to_datetime(date)
    delta = date - datetime.datetime(date.year, date.month, date.day)
    hours_since_midnight = (delta.days + delta.seconds / 86400.0) * 24
    return (lon / 360.0 * 24.0 + hours_since_midnight) % 24


def _anemoi_cos_local_time(date, lon):
    radians = _anemoi_local_time(date, lon) / 24 * np.pi * 2
    return np.cos(radians)


def _anemoi_sin_local_time(date, lon):
    radians = _anemoi_local_time(date, lon) / 24 * np.pi * 2
    return np.sin(radians)


def _anemoi_insolation(date, lat, lon):
    return _anemoi_cos_solar_zenith_angle(date, lat, lon)


def _anemoi_toa_incident_solar_radiation(date, lat, lon):
    from earthkit.meteo.solar import toa_incident_solar_radiation

    date = to_datetime(date)
    result = toa_incident_solar_radiation(
        date - datetime.timedelta(minutes=30),
        date + datetime.timedelta(minutes=30),
        lat,
        lon,
        intervals_per_hour=2,
    )
    return result.flatten()


def _anemoi_cos_solar_zenith_angle(date, lat, lon):
    from earthkit.meteo.solar import cos_solar_zenith_angle

    date = to_datetime(date)
    result = cos_solar_zenith_angle(
        date,
        lat,
        lon,
    )
    return result.flatten()


def _anemoi_dynamic_forcings():
    """
    Returns list of dynamic forcings calculated by anemoi datasets.
    If this list is updated the forcing should also be implemented in get_dynamic_forcings
    """
    return [
        "cos_julian_day",
        "sin_julian_day",
        "cos_local_time",
        "sin_local_time",
        "insolation",
    ]


def _anemoi_get_dynamic_forcings(times, lats, lons, selection):
    forcings = {}
    if selection is None:
        return forcings

    forcings = {k: [] for k in selection}

    for time in times:
        if "cos_julian_day" in selection:
            forcings["cos_julian_day"] += [np.full(lats.shape, _anemoi_cos_julian_day(time))]
        if "sin_julian_day" in selection:
            forcings["sin_julian_day"] += [np.full(lats.shape, _anemoi_sin_julian_day(time))]
        if "cos_local_time" in selection:
            forcings["cos_local_time"] += [_anemoi_cos_local_time(time, lons)]
        if "sin_local_time" in selection:
            forcings["sin_local_time"] += [_anemoi_sin_local_time(time, lons)]
        if "insolation" in selection:
            forcings["insolation"] += [_anemoi_insolation(time, lats, lons)]

    for k in selection:
        forcings[k] = np.concatenate(forcings[k], axis=0)

    return forcings
