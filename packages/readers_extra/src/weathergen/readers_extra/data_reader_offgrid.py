# (C) Copyright 2026 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Data reader for offgrid inference.

Runs a trained model at an arbitrary set of points instead of the points of the dataset it was
trained on. The stream is taken from the config of the trained model (``--from-run-id``) and
an overwrite config switches its reader to this one::

    streams:
      ERA5:
        type: offgrid
        base_type: anemoi
        offgrid:
          coords: /path/to/coords.npy
          geoinfos:
            file: /path/to/geoinfos.npy
            channels: [z, lsm]
          frequency: 6h

``base_type`` is the reader type the stream was trained with, i.e. its ``type`` in the config of
the trained model. It has to be repeated because ``type`` now selects this reader.

``coords`` is a file with the points or the name of a default grid, e.g. ``o96`` or ``N320``
(see ``_grid_coords``). All points are returned: ``max_num_targets`` of the trained model is not
applied; set ``offgrid.max_num_targets`` to limit them.

``geoinfos`` provides the static geoinfo channels of the stream at the points: ``file`` has one
column per entry of ``channels``, and each static geoinfo channel takes the column of its name.
Geoinfo channels that depend on time, e.g. ``insolation`` or ``cos_local_time``, are computed
for each timestep (see ``_anemoi_dynamic_forcings``) and need no column.

A relative path of a file is resolved from the working directory.

The model input (source) of the stream stays what it is: the data read by the reader of
``base_type``. Only the points the model predicts at (target) change: their coordinates and
geoinfos are those of the offgrid points, and their values are NaN since there is no data there.
Channels and normalization are taken from the reader of ``base_type`` as well, so they are the
ones the model was trained with, whatever the format of the dataset. For a stream with several
files the offgrid points are returned once, by the reader of the first file.
"""

import datetime
import logging
from pathlib import Path
from typing import override

import numpy as np
from earthkit.data.utils.dates import to_datetime
from numpy.typing import NDArray

from weathergen.common.config import parse_timedelta
from weathergen.datasets.data_reader_anemoi import DataReaderAnemoi
from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)
from weathergen.datasets.data_reader_obs import DataReaderObs
from weathergen.readers_extra.registry import get_extra_reader
from weathergen.train.utils import TRAIN, Stage

_logger = logging.getLogger(__name__)


class DataReaderOffgrid(DataReaderTimestep):
    "Reader for inference at arbitrary points with a stream of a trained model"

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
        stage: Stage,
    ) -> None:
        """
        Construct data reader for offgrid inference

        Parameters
        ----------
        filename :
            filename (and path) of the dataset of the stream
        stream_info :
            information about stream. Keys beyond the common ones: ``base_type`` (reader type
            the stream was trained with) and ``offgrid`` with ``coords`` (file with an (N, 2)
            array of [lat, lon] in degrees, or the name of a default grid such as ``o96`` or
            ``N320``), ``geoinfos`` (``file`` with an (N, G) array and ``channels`` with the names
            of its G columns, which have to include the static geoinfo channels of the stream;
            needed if the stream has static geoinfo channels), ``frequency`` (spacing of the
            offgrid samples; default is the time window step) and ``max_num_targets`` (default
            is -1, i.e. all points).

        Returns
        -------
        None
        """

        ds_name = stream_info["name"]
        offgrid = stream_info["offgrid"]

        # there is no data at the offgrid points, hence nothing to train on
        assert stage != TRAIN, f"{ds_name}: offgrid streams are for inference only."
        # one reader is built per file of the stream; only the reader of the first file returns
        # the offgrid points, they would otherwise be returned once per file
        filenames = stream_info.get("filenames", [])
        self.is_first_file = len(filenames) == 0 or Path(filename).name == Path(filenames[0]).name

        # the offgrid samples cover the data loader window at a fixed frequency
        period = parse_timedelta(offgrid.get("frequency", tw_handler.t_window_step))
        super().__init__(
            tw_handler,
            stream_info,
            tw_handler.t_start,
            tw_handler.t_end,
            period,
        )
        self.len = int((tw_handler.t_end - tw_handler.t_start) / period)

        # reader the stream was trained with, selected as in MultiStreamDataSampler; it provides
        # the channels, the normalization and the source data
        match stream_info["base_type"]:
            case "obs":
                dataset = DataReaderObs
            case "anemoi":
                dataset = DataReaderAnemoi
            case type_name:
                dataset = get_extra_reader(type_name)
        assert dataset not in (None, DataReaderOffgrid), (
            f"{ds_name}: unsupported base_type '{stream_info['base_type']}'."
        )
        self.reader = dataset(
            tw_handler=tw_handler, filename=filename, stream_info=stream_info, stage=stage
        )
        # If that reader is empty, its dataset has no data in the data loader window and there
        # are no channels to predict; the stream is skipped, as the reader does itself
        if len(self.reader.source_channels) + len(self.reader.target_channels) == 0:
            _logger.warning(
                f"{ds_name} is not supported over data loader window. Stream is skipped."
            )
            self.init_empty()
            return

        # caches lats and lons; coords is a file with the points or names a default grid
        if str(offgrid["coords"]).endswith(".npy"):
            coords = np.load(offgrid["coords"])
        else:
            coords = _grid_coords(str(offgrid["coords"]))
        assert coords.ndim == 2 and coords.shape[1] == 2, (
            f"{ds_name}: offgrid.coords must have shape (N, 2) for [lat, lon], got {coords.shape}."
        )
        self.latitudes = _clip_lat(coords[:, 0])
        self.longitudes = _clip_lon(coords[:, 1])
        self.n_points = len(self.latitudes)

        # all offgrid points are returned by default; max_num_targets of the trained model is a
        # training setting and would drop a random part of them
        stream_info["max_num_targets"] = offgrid.get("max_num_targets", -1)

        # source channels
        self.source_channels = self.reader.source_channels
        self.source_idx = self.reader.source_idx

        # target channels
        self.target_channels = self.reader.target_channels
        self.target_idx = self.reader.target_idx

        # target channel weights
        self.target_channel_weights = self.reader.target_channel_weights

        # geoinfo channels
        self.geoinfo_channels = self.reader.geoinfo_channels
        self.geoinfo_idx = self.reader.geoinfo_idx

        # geoinfo channels that depend on time are computed on the fly, all others are static
        self.geoinfo_channels_static, self.geoinfo_channels_dynamic = [], []
        self.geoinfo_idx_static_lin, self.geoinfo_idx_dynamic_lin = [], []
        for idx, ch in enumerate(self.geoinfo_channels):
            if ch in _anemoi_dynamic_forcings():
                self.geoinfo_channels_dynamic += [ch]
                self.geoinfo_idx_dynamic_lin += [idx]
            else:
                self.geoinfo_channels_static += [ch]
                self.geoinfo_idx_static_lin += [idx]

        # static geoinfos at the offgrid points; the columns of the file are named by
        # offgrid.geoinfos.channels and each static geoinfo channel takes the column of its name
        geoinfos, geoinfos_channels = np.zeros((self.n_points, 0), dtype=np.float32), []
        if offgrid.get("geoinfos") is not None:
            geoinfos = np.load(offgrid["geoinfos"]["file"]).astype(np.float32)
            geoinfos_channels = list(offgrid["geoinfos"]["channels"])
        assert geoinfos.shape == (self.n_points, len(geoinfos_channels)), (
            f"{ds_name}: offgrid.geoinfos.file must have shape "
            f"({self.n_points}, {len(geoinfos_channels)}) to match the offgrid points and "
            f"offgrid.geoinfos.channels, got {geoinfos.shape}."
        )
        missing = [ch for ch in self.geoinfo_channels_static if ch not in geoinfos_channels]
        assert len(missing) == 0, (
            f"{ds_name}: static geoinfo channels {missing} are not in offgrid.geoinfos.channels "
            f"{geoinfos_channels}."
        )
        self.geoinfos_static = geoinfos[
            :, [geoinfos_channels.index(ch) for ch in self.geoinfo_channels_static]
        ]

        _logger.info(f"{ds_name}: offgrid points: {len(self.latitudes)}")
        _logger.info(f"{ds_name}: source channels: {self.source_channels}")
        _logger.info(f"{ds_name}: target channels: {self.target_channels}")
        _logger.info(f"{ds_name}: geoinfo channels: {self.geoinfo_channels}")

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self.len = 0

    @override
    def length(self) -> int:
        return self.len

    @override
    def get_source(self, idx: TIndex) -> ReaderData:
        """
        Get source data for idx

        The model input is the data of the stream itself; only the targets are offgrid.

        Parameters
        ----------
        idx : int
            Index of temporal window

        Returns
        -------
        source data (coords, geoinfos, data, datetimes)
        """
        return self.reader.get_source(idx)

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """
        Get offgrid samples for one time window.

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

        if not self.is_first_file or self.len == 0 or len(t_idxs) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        assert t_idxs[0] >= 0, "index must be non-negative"

        n_steps = len(t_idxs)
        n_total = self.n_points * n_steps

        # repeat the offgrid coordinates for each timestep in the window
        latlon = np.stack([self.latitudes, self.longitudes], axis=1)
        coords = np.tile(latlon, (n_steps, 1))

        # there is no data at the offgrid points
        data = np.full((n_total, len(channels_idx)), np.nan, dtype=np.float32)

        # compute absolute times for each step, then repeat per point
        step_times = self.data_start_time + self.period * t_idxs
        datetimes = np.repeat(step_times, self.n_points)

        # repeat static geoinfos for each timestep and compute the dynamic ones for it
        geoinfos_static = np.tile(self.geoinfos_static, (n_steps, 1))
        geoinfos_dynamic = _anemoi_get_dynamic_forcings(
            step_times, self.latitudes, self.longitudes, self.geoinfo_channels_dynamic
        )
        # insert static and dynamic into common array
        geoinfos = np.empty((n_total, len(self.geoinfo_idx)), dtype=np.float32)
        geoinfos[:, self.geoinfo_idx_static_lin] = geoinfos_static
        for i, ch in zip(self.geoinfo_idx_dynamic_lin, self.geoinfo_channels_dynamic, strict=True):
            geoinfos[:, i] = geoinfos_dynamic[ch]

        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )
        check_reader_data(rd, dtr)

        return rd

    # normalization is that of the reader the stream was trained with; some readers implement
    # their own instead of the mean / stdev scaling of the base class

    @override
    def normalize_source_channels(self, source: NDArray) -> NDArray:
        return self.reader.normalize_source_channels(source)

    @override
    def normalize_target_channels(self, target: NDArray) -> NDArray:
        return self.reader.normalize_target_channels(target)

    @override
    def normalize_geoinfos(self, geoinfos: NDArray) -> NDArray:
        return self.reader.normalize_geoinfos(geoinfos)

    @override
    def denormalize_source_channels(self, source: NDArray) -> NDArray:
        return self.reader.denormalize_source_channels(source)

    @override
    def denormalize_target_channels(self, target: NDArray) -> NDArray:
        return self.reader.denormalize_target_channels(target)


def _grid_coords(grid: str) -> NDArray[np.float64]:
    """
    Coordinates of a reduced Gaussian grid, north to south and eastwards from 0 deg

    Parameters
    ----------
    grid :
        "o<N>" for an octahedral grid (e.g. "o96") or "N<N>" for a classic one (e.g. "N320")

    Returns
    -------
    array of shape (num_points, 2) with [lat, lon] in degrees
    """

    assert grid[:1].lower() in ("o", "n") and grid[1:].isdigit(), (
        f"unknown grid '{grid}': expected 'o<N>' or 'N<N>', e.g. 'o96' or 'N320'."
    )
    kind, num_lats = grid[0].lower(), int(grid[1:])

    # Gaussian latitudes, north to south
    x, _ = np.polynomial.legendre.leggauss(2 * num_lats)
    lats = np.rad2deg(np.arcsin(x))[::-1]

    # number of points on each latitude
    if kind == "o":
        half = 20 + 4 * np.arange(num_lats)
        lons_per_lat = np.concatenate([half, half[::-1]])
    else:
        # classic grids follow a table, not a formula; eccodes ships it
        import eccodes

        handle = eccodes.codes_grib_new_from_samples(f"reduced_gg_pl_{num_lats}_grib2")
        lons_per_lat = eccodes.codes_get_array(handle, "pl")
        eccodes.codes_release(handle)

    lat = np.repeat(lats, lons_per_lat)
    lon = np.concatenate([np.arange(n) * (360.0 / n) for n in lons_per_lat])

    return np.stack([lat, lon], axis=1)


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


# Dynamic forcings, same code as in data_reader_anemoi_rt.py:

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
