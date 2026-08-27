# (C) Copyright 2026 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Reader that provides a deterministic gridded target template for inference."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import override

import numpy as np
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
from weathergen.train.utils import Stage

_DYNAMIC_GEOINFO_CHANNELS = {
    "cos_julian_day",
    "sin_julian_day",
    "cos_local_time",
    "sin_local_time",
    "insolation",
}


class DataReaderOffgrid(DataReaderTimestep):
    """Generate mean-valued data over the grid of an Anemoi reference dataset.

    This reader is intended for inference only. It retains the reference stream's
    channel layout and normalization metadata while generating deterministic global
    coordinates at the requested times. Dynamic decoder geoinfos are reconstructed
    for each output timestamp; static geoinfos are read once from the reference data.
    """

    def __init__(
        self, tw_handler: TimeWindowHandler, filename: Path, stream_info: dict, stage: Stage
    ) -> None:
        reference_start = np.datetime64(stream_info["reference_start_date"])
        reference_end = np.datetime64(stream_info["reference_end_date"])
        reference_window = TimeWindowHandler(
            reference_start,
            reference_end,
            tw_handler.t_window_len,
            tw_handler.t_window_step,
        )
        reference_reader = DataReaderAnemoi(
            reference_window,
            filename,
            stream_info,
            stage,
        )
        if reference_reader.length() == 0:
            raise ValueError(
                f"{stream_info['name']}: offgrid reference window "
                f"[{reference_start}, {reference_end}) has no Anemoi data."
            )

        frequency = parse_timedelta(stream_info["frequency"])
        super().__init__(
            tw_handler,
            stream_info,
            data_start_time=tw_handler.t_start,
            data_end_time=tw_handler.t_end,
            period=frequency,
        )

        self.latitudes = np.array(reference_reader.latitudes, copy=True)
        self.longitudes = np.array(reference_reader.longitudes, copy=True)
        self.n_points = len(self.latitudes)
        self.len = int((tw_handler.t_end - tw_handler.t_start) // frequency)

        self.source_idx = np.array(reference_reader.source_idx, copy=True)
        self.source_channels = list(reference_reader.source_channels)
        self.target_idx = np.array(reference_reader.target_idx, copy=True)
        self.target_channels = list(reference_reader.target_channels)
        self.target_channel_weights = list(reference_reader.target_channel_weights)
        self.geoinfo_idx = np.array(reference_reader.geoinfo_idx, copy=True)
        self.geoinfo_channels = list(reference_reader.geoinfo_channels)
        self.mean = np.array(reference_reader.mean, copy=True)
        self.stdev = np.array(reference_reader.stdev, copy=True)
        self.mean_geoinfo = np.array(reference_reader.mean_geoinfo, copy=True)
        self.stdev_geoinfo = np.array(reference_reader.stdev_geoinfo, copy=True)
        self.properties = dict(reference_reader.properties)

        reference_data = reference_reader.get_target(0)
        if len(reference_data.data) < self.n_points:
            raise ValueError(
                f"{stream_info['name']}: offgrid reference did not return one complete grid."
            )
        self.static_geoinfos = np.array(reference_data.geoinfos[: self.n_points], copy=True)

    @override
    def length(self) -> int:
        return self.len

    @override
    def get_source(self, idx: TIndex) -> ReaderData:
        """Keep diagnostic inputs empty; this reader exists only for decoder targets."""
        del idx
        return ReaderData.empty(
            num_data_fields=len(self.source_idx), num_geo_fields=len(self.geoinfo_idx)
        )

    def _geoinfos_at(self, timestamp: np.datetime64) -> NDArray[np.float32]:
        """Return raw reference geoinfos, updating time-dependent fields."""
        geoinfos = np.array(self.static_geoinfos, copy=True)
        if not _DYNAMIC_GEOINFO_CHANNELS.intersection(self.geoinfo_channels):
            return geoinfos

        timestamp_text = np.datetime_as_string(timestamp, unit="s")
        current_time = dt.datetime.fromisoformat(timestamp_text).replace(tzinfo=dt.UTC)
        year_start = dt.datetime(current_time.year, 1, 1, tzinfo=dt.UTC)
        day_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        # The Anemoi forcing uses a one-based day-of-year phase: 1.0 at the
        # beginning of 1 January. This agrees with its stored
        # ``cos_julian_day``/``sin_julian_day`` fields.
        julian_day = (current_time - year_start).total_seconds() / 86400.0 + 1.0
        hour = (current_time - day_start).total_seconds() / 3600.0

        angle = julian_day / 365.25 * np.pi * 2.0
        local_time = (self.longitudes / 360.0 * 24.0 + hour) % 24.0
        local_angle = local_time / 24.0 * np.pi * 2.0

        declination = (
            0.396372
            - 22.91327 * np.cos(angle)
            + 4.025430 * np.sin(angle)
            - 0.387205 * np.cos(2.0 * angle)
            + 0.051967 * np.sin(2.0 * angle)
            - 0.154527 * np.cos(3.0 * angle)
            + 0.084798 * np.sin(3.0 * angle)
        )
        time_correction = (
            0.004297
            + 0.107029 * np.cos(angle)
            - 1.837877 * np.sin(angle)
            - 0.837378 * np.cos(2.0 * angle)
            - 2.340475 * np.sin(2.0 * angle)
        )
        solar_angle = np.deg2rad((hour - 12.0) * 15.0 + self.longitudes + time_correction)
        insolation = np.sin(np.deg2rad(declination)) * np.sin(np.deg2rad(self.latitudes))
        insolation += (
            np.cos(np.deg2rad(declination))
            * np.cos(np.deg2rad(self.latitudes))
            * np.cos(solar_angle)
        )

        values = {
            "cos_julian_day": np.full(self.n_points, np.cos(angle), dtype=np.float32),
            "sin_julian_day": np.full(self.n_points, np.sin(angle), dtype=np.float32),
            "cos_local_time": np.cos(local_angle).astype(np.float32),
            "sin_local_time": np.sin(local_angle).astype(np.float32),
            "insolation": np.clip(insolation, 0.0, None).astype(np.float32),
        }
        for column, name in enumerate(self.geoinfo_channels):
            if name in values:
                geoinfos[:, column] = values[name]
        return geoinfos

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """Return one deterministic full-grid template for every time in a window."""
        t_idxs, dtr = self._get_dataset_idxs(idx)
        if len(t_idxs) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        timestamps = self.data_start_time + self.period * t_idxs
        coords_template = np.stack([self.latitudes, self.longitudes], axis=1)
        coords = np.tile(coords_template, (len(timestamps), 1)).astype(np.float32)
        geoinfos = np.concatenate([self._geoinfos_at(time) for time in timestamps], axis=0)
        values = np.asarray(self.mean[np.asarray(channels_idx, dtype=np.int64)], dtype=np.float32)
        data = np.tile(values, (len(timestamps) * self.n_points, 1))
        datetimes = np.repeat(timestamps, self.n_points)

        rdata = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )
        check_reader_data(rdata, dtr)
        return rdata
