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
from numpy.typing import NDArray

from weathergen.common.config import parse_timedelta
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


def latest_available_analysis_index(
    valid_times: NDArray[np.datetime64],
    source_window_end: np.datetime64,
    availability_mode: str,
    nominal_time_mapping: dict | None = None,
    available_until: np.datetime64 | None = None,
) -> tuple[int | None, NDArray[np.datetime64]]:
    """Return the newest analysis available at a source-window boundary.

    ``dataset`` is for a materialized real-time store: only analyses which
    have already been downloaded are present. ``available_until`` optionally
    records the cycle at which that snapshot was materialized and prevents a
    reused store from leaking later analyses into an earlier inference run.
    ``nominal_time_mapping`` preserves the historical archive behaviour.
    """
    if availability_mode == "dataset":
        availability_times = valid_times
        cutoff = source_window_end
        if available_until is not None:
            cutoff = min(cutoff, available_until)
        is_available = availability_times <= cutoff
    elif availability_mode == "nominal_time_mapping":
        if nominal_time_mapping is None:
            raise ValueError("nominal_time_mapping is required for mapped analysis availability.")
        hours = dt2cal(valid_times)[:, 3]
        try:
            deltas = np.array([int(nominal_time_mapping[str(hour)]) - int(hour) for hour in hours])
        except KeyError as exc:
            raise KeyError(
                f"nominal_time_mapping has no entry for analysis hour {exc.args[0]}."
            ) from exc
        availability_times = valid_times + deltas.astype("timedelta64[h]")
        # Keep the historical archive contract: an analysis becoming
        # available exactly at the end boundary is used by the next window.
        is_available = availability_times < source_window_end
    else:
        raise ValueError(f"Unsupported analysis_availability {availability_mode!r}.")

    available_idxs = np.flatnonzero(is_available)
    return (int(available_idxs[-1]) if len(available_idxs) else None, availability_times)


class DataReaderAnemoiOperan(DataReaderAnemoi):
    "Read the latest operational analysis available for a forecast window."

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

        # Operational input streams must keep their model schema when an
        # analysis is not available so the sampler can create a shape-correct
        # spoof fallback. Keep this local to the reader rather than changing
        # the generic ``anemoi`` reader's default behaviour.
        stream_info = dict(stream_info)
        stream_info.setdefault("retain_schema_when_empty", True)

        self.analysis_availability = stream_info.get(
            "analysis_availability", "nominal_time_mapping"
        )
        if self.analysis_availability not in {"dataset", "nominal_time_mapping"}:
            raise ValueError(
                f"{stream_info['name']}: unsupported analysis_availability "
                f"{self.analysis_availability!r}."
            )

        lookback = parse_timedelta(stream_info.get("analysis_lookback", "12h"))
        max_age = stream_info.get("max_analysis_age")
        self.max_analysis_age = parse_timedelta(max_age) if max_age is not None else None
        available_until = stream_info.get("analysis_available_until")
        self.analysis_available_until = (
            np.datetime64(available_until) if available_until is not None else None
        )

        # An analysis normally predates the source window. A sparse real-time
        # store therefore has to be opened with a short pre-window lookback;
        # the sampler's actual forecast windows remain defined by tw_handler.
        super().__init__(
            tw_handler,
            filename,
            stream_info,
            stage,
            data_load_start=tw_handler.t_start - lookback,
            data_load_end=tw_handler.t_end,
        )

    @override
    def get_target(self, idx: TIndex) -> ReaderData:
        """Avoid resolving unused targets for an input-only analysis stream."""
        if self.stream_info.get("forcing", False):
            return ReaderData.empty(
                num_data_fields=len(self.target_idx), num_geo_fields=len(self.geoinfo_idx)
            )
        return super().get_target(idx)

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

        dtr = self.time_window_handler.window(idx)
        if self.ds is None or self.len == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        datetimes_all = np.asarray(self.ds.dates)
        didx, availability_times = latest_available_analysis_index(
            datetimes_all,
            dtr.end,
            self.analysis_availability,
            self.stream_info.get("nominal_time_mapping"),
            self.analysis_available_until,
        )
        if didx is None:
            _logger.info(
                "%s: no analysis available for source window [%s, %s).",
                self.stream_info["name"],
                dtr.start,
                dtr.end,
            )
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        if self.max_analysis_age is not None:
            # Availability controls selection, but age refers to the physical
            # state represented by the analysis rather than its publication
            # timestamp. These differ in ``nominal_time_mapping`` mode.
            age = dtr.end - datetimes_all[didx]
            if age > self.max_analysis_age:
                _logger.warning(
                    "%s: latest analysis is too old for source window [%s, %s): age=%s, limit=%s.",
                    self.stream_info["name"],
                    dtr.start,
                    dtr.end,
                    age,
                    self.max_analysis_age,
                )
                return ReaderData.empty(
                    num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
                )

        # extract number of time steps and collapse ensemble dimension
        # ds is a wrapper around zarr with get_coordinate_selection not being exposed since
        # subsetting is pushed to the ctor via frequency argument; this also ensures that no sub-
        # sampling is required here
        try:
            data = self.ds[didx : didx + 1][:, :, 0].astype(np.float32)
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
        coords = latlon

        # Retain the analysis valid time rather than its availability time.
        # This matches historical operan training and lets time encoding express
        # the analysis age relative to the source window.
        datetimes = np.repeat(datetimes_all[didx], len(data))

        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )
        # check_reader_data(rd, dtr)

        return rd
