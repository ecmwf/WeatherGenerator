# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Synthetic data reader for scalar conditioning.

Generates synthetic scalar values like time_of_day, day_of_year, and noise_level
that can be used for model conditioning.
"""

import logging

import numpy as np

from weathergen.datasets.data_reader_base import (
    DataReaderBase,
    ReaderData,
    TimeWindowHandler,
    check_reader_data,
)
from weathergen.train.utils import Stage

_logger = logging.getLogger(__name__)


class DataReaderTimeConditioning(DataReaderBase):
    """
    DataReader that generates synthetic scalar conditioning values.

    Supports three types:
    - time_based: extracts hour/day from timestamp
    - constant: fixed value
    - random: random uniform value (per timestep)
    """

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename,
        stream_info: dict,
        stage: Stage,
    ) -> None:
        super().__init__(tw_handler, stream_info)

        self.source_idx = []
        self.target_idx = []
        self.geoinfo_idx = []

        self.conditioning = stream_info.get("conditioning", True)
        self.conditioning_type = stream_info.get("conditioning_type", "time_based")
        value_type = stream_info.get("value_type")
        if isinstance(value_type, list):
            self.value_types = value_type
        elif value_type is not None:
            self.value_types = [value_type]
        else:
            raise ValueError("value_type in time_conditioning must be specified in stream_info")

        self.source_channels = [f"time_conditioning_{vt}" for vt in self.value_types]
        self.target_channels = []
        self.geoinfo_channels = []
        self.target_channel_weights = []

        self.mean = np.zeros(len(self.source_channels), dtype=np.float32)
        self.stdev = np.ones(len(self.source_channels), dtype=np.float32)
        self.mean_geoinfo = np.zeros(0, dtype=np.float32)
        self.stdev_geoinfo = np.ones(0, dtype=np.float32)

    def length(self) -> int:
        return self.time_window_handler.get_index_range().end

    def _get(self, idx, channels_idx) -> ReaderData:
        dt_range = self.time_window_handler.window(idx)
        dt = dt_range.start

        values = []
        for vt in self.value_types:
            # The values for time conditioning below denote fractional values of the
            # day or year, normalized to [0, 1].
            if self.conditioning_type == "time_based":
                if vt == "hour":
                    hours = dt.astype("datetime64[h]").astype(int) % 24
                    minutes = dt.astype("datetime64[m]").astype(int) % 60
                    total_hours = hours + minutes / 60.0
                    value = total_hours / 24.0
                elif vt == "day":
                    days = dt.astype("datetime64[D]").astype(int) % 365
                    value = days / 365.0
                else:
                    raise ValueError(f"Unknown value_type: {vt}")
                values.append(value)
            else:
                raise ValueError(f"Unknown conditioning_type: {self.conditioning_type}")

        coords = np.zeros((1, 2), dtype=np.float32)
        geoinfos = np.zeros((1, 0), dtype=np.float32)
        datetimes = np.array([dt], dtype=np.datetime64)

        rdata = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=np.array(values, dtype=np.float32).reshape(1, -1),
            datetimes=datetimes,
        )
        check_reader_data(rdata, dt_range)

        return rdata

    def get_geoinfo_size(self) -> int:
        return 0
