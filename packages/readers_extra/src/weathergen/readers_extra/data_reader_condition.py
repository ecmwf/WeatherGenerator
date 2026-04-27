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

from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    DTRange
)

_logger = logging.getLogger(__name__)


class DataReaderCondition(DataReaderTimestep):
    "Wrapper for forecast condition variables derived from time window metadata"

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
    ) -> None:
        """
        Construct data reader for forecast condition variables.

        Parameters
        ----------
        tw_handler :
            time window handler
        filename :
            unused; kept for interface compatibility (filenames should be empty)
        stream_info :
            information about stream; must include 'transform' and 'variables'

        Returns
        -------
        None
        """
        self.source_idx = []
        self.source_channels = []
        self.target_channels = []
        self.geoinfo_channels = []
        self.target_idx = []
        self.geoinfo_idx = []
        self.target_channel_weights = []
        self.condition_idx = []

        self.transform: str = stream_info.get("transform", "absolute")
        self.variables: list[str] = list(
            stream_info.get("variables", ["start_day", "start_time", "end_day", "end_time"])
        )
        self.emb_dimension: int = stream_info.get("emb_dimension", 64)
        self.num_channels: int = self._compute_num_channels(stream_info)
        self.source_idx = []

        super().__init__(
            tw_handler,
            stream_info,
            tw_handler.t_start,
            tw_handler.t_end,
            tw_handler.t_window_step,
        )

        self.len = int((tw_handler.t_end - tw_handler.t_start) // tw_handler.t_window_step)

    def _compute_num_channels(self, stream_info: dict) -> int:
        if self.transform == "absolute":
            return len(self.variables)
        elif self.transform == "cos_sin":
            return 2 * len(self.variables)
        elif self.transform in ("fourier", "fourier_amplified"):
            assert "emb_dimension" in stream_info, "Fourier transform requires 'emb_dimension' in stream_info"
            return stream_info.get("emb_dimension") * len(self.variables)
        else:
            raise ValueError(f"Unknown transform: {self.transform!r}")

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self.len = 0

    @override
    def length(self) -> int:
        return self.len

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """
        Compute condition variables for a given time window.

        Parameters
        ----------
        idx : TIndex
            Index of temporal window
        channels_idx : list[int]
            Selection of channels

        Returns
        -------
        ReaderData providing coords, geoinfos, data, datetimes
        """

        dtr = self.time_window_handler.window(idx)
        encoded_condtions = self._encode(dtr, self.variables)
        return encoded_condtions 

    def _encode(self, dtr: DTRange, variables: list[str]) -> list[float]:
        """
        Encode start/end datetimes into condition variable values.

        Parameters
        ----------
        dtr :
            time window date range
        variables :
            list of variable names to encode

        Returns
        -------
        list of floats of length num_channels
        """
        _RAW: dict[str, float] = {
            "start_day":  _day_of_year(dtr.start),
            "start_time": _hour_of_day(dtr.start),
            "end_day":    _day_of_year(dtr.end),
            "end_time":   _hour_of_day(dtr.end),
        }

        values: list[float] = []
        for var in variables:
            raw = _RAW[var]
            if self.transform == "absolute":
                values.append(raw)
            elif self.transform == "cos_sin":
                values.extend(self._cos_sin_embed(raw))
            elif self.transform == "fourier":
                values.extend(self._fourier_embed(raw))
            elif self.transform == "fourier_amplified":
                values.extend(self._fourier_embed_amplified(raw))

        return values

    def _cos_sin_embed(self, value: float) -> list[float]:
        angle = 2.0 * np.pi * value
        return [float(np.sin(angle)), float(np.cos(angle))]

    def _fourier_embed(self, value: float) -> np.ndarray:
        """Sinusoidal Fourier embedding of a scalar into emb_dimension dims."""
        half = self.emb_dimension // 2
        freqs = np.arange(1, half + 1, dtype=np.float64)
        angles = 2.0 * np.pi * freqs * value
        return np.concatenate([np.sin(angles), np.cos(angles)])

    def _fourier_embed_amplified(self, value: float) -> np.ndarray:
        """Fourier embedding scaled by value as amplitude — larger inputs dominate."""
        half = self.emb_dimension // 2
        freqs = np.arange(1, half + 1, dtype=np.float64)
        angles = 2.0 * np.pi * freqs * value
        return value * np.concatenate([np.sin(angles), np.cos(angles)])


def _day_of_year(dt: np.datetime64) -> float:
    """Return 1-indexed day of year for a numpy datetime64."""
    jan1 = dt.astype("datetime64[Y]").astype("datetime64[D]")
    return float((dt.astype("datetime64[D]") - jan1).astype(int) + 1)


def _hour_of_day(dt: np.datetime64) -> float:
    """Return fractional hour of day [0, 24) for a numpy datetime64."""
    day = dt.astype("datetime64[D]")
    return float((dt - day) / np.timedelta64(1, "h"))
