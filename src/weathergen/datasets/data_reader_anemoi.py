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

import anemoi.datasets as anemoi_datasets
import numpy as np
from anemoi.datasets.data import MissingDateError
from anemoi.datasets.data.dataset import Dataset
from numpy.typing import NDArray

from weathergen.common.config import parse_timedelta, timedelta_to_str
from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)
from weathergen.train.utils import Stage

_logger = logging.getLogger(__name__)


class DataReaderAnemoi(DataReaderTimestep):
    "Wrapper for Anemoi datasets"

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
        stage: Stage,
        data_load_start: np.datetime64 | None = None,
        data_load_end: np.datetime64 | None = None,
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

        load_start = data_load_start if data_load_start is not None else tw_handler.t_start
        load_end = data_load_end if data_load_end is not None else tw_handler.t_end

        # Open dataset to check that it is compatible with requested parameters.
        ds0: Dataset = anemoi_datasets.open_dataset(filename)
        # Dataset dates are point samples, so a sample exactly at ``load_start``
        # is still usable. The end of the requested range remains exclusive.
        if load_start > ds0.dates[-1] or load_end <= ds0.dates[0]:
            name = stream_info["name"]
            _logger.warning(f"{name} is not supported over requested data range.")
            self._init_missing_data(tw_handler, stream_info, stage, ds0)
            return

        kwargs = {}
        if "frequency" in stream_info:
            frequency = timedelta_to_str(stream_info["frequency"])
            requested_frequency = parse_timedelta(stream_info["frequency"])
            # Subsampling to `frequency` needs anemoi-datasets to infer the
            # window-restricted subset's own native frequency, which requires
            # >=2 dates inside [t_start, t_end) and raises otherwise. Skip the
            # (then redundant) kwarg when ds0 is already natively at that
            # frequency, so a sparse real-time stream with one date in the
            # window doesn't crash on what would be a no-op anyway.
            try:
                already_at_frequency = parse_timedelta(ds0.frequency) == requested_frequency
            except ValueError:
                already_at_frequency = False
            if not already_at_frequency:
                kwargs["frequency"] = frequency
        if "statistics" in stream_info:
            kwargs["statistics"] = stream_info["statistics"]
        if "subsampling_rate" in stream_info:
            name = stream_info["name"]
            _logger.warning(
                f"subsampling_rate specified for anemoi dataset for stream {name}. "
                + "Use frequency instead."
            )
        ds: Dataset = anemoi_datasets.open_dataset(ds0, **kwargs, start=load_start, end=load_end)

        if len(ds.dates) >= 2:
            period = parse_timedelta(ds.frequency)
        elif "frequency" in stream_info:
            period = requested_frequency
        else:
            # ds.frequency infers spacing from ds's own dates and needs >=2;
            # a sparse real-time stream can have just one date inside the
            # requested window even though the full dataset has more.
            period = parse_timedelta(ds0.frequency)
        data_start_time = ds.dates[0]
        data_end_time = ds.dates[-1]
        assert data_start_time is not None and data_end_time is not None, (
            data_start_time,
            data_end_time,
        )
        super().__init__(
            tw_handler,
            stream_info,
            data_start_time,
            data_end_time,
            period,
        )
        # If there is no overlap with the time range, no need to keep the dataset.
        if load_start > data_end_time or load_end <= data_start_time:
            self._init_missing_data(tw_handler, stream_info, stage, ds)
            return
        else:
            self.ds = ds
            self.len = len(ds)

        self._set_metadata(ds, stream_info, stage)

    def _init_missing_data(
        self,
        tw_handler: TimeWindowHandler,
        stream_info: dict,
        stage: Stage,
        ds: Dataset,
    ) -> None:
        """Initialize an unavailable stream, retaining schema only when requested."""
        if not stream_info.get("retain_schema_when_empty", False):
            super().__init__(tw_handler, stream_info)
            self.init_empty()
            return

        _logger.info(
            "%s: retaining channel and normalization schema for missing-data fallback.",
            stream_info["name"],
        )
        period = parse_timedelta(stream_info.get("frequency", ds.frequency))
        super().__init__(
            tw_handler,
            stream_info,
            ds.dates[0],
            ds.dates[-1],
            period,
        )
        self.ds = None
        self.len = 0
        self._set_metadata(ds, stream_info, stage)

    def _set_metadata(self, ds: Dataset, stream_info: dict, stage: Stage) -> None:
        """Populate channel, coordinate, and normalization metadata from an Anemoi dataset."""
        # caches lats and lons
        self.latitudes = _clip_lat(ds.latitudes)
        self.longitudes = _clip_lon(ds.longitudes)

        # select/filter requested source channels
        if stream_info.get(str(stage) + "_source_channels") is None:
            self.source_idx = self.select_channels(ds, "source")
            self.source_channels = [ds.variables[i] for i in self.source_idx]
        else:
            self.source_channels = stream_info.get(str(stage) + "_source_channels")
            self.source_idx = [ds.variables.index(ch) for ch in self.source_channels]

        # select/filter requested target channels
        if stream_info.get(str(stage) + "_target_channels") is None:
            self.target_idx = self.select_channels(ds, "target")
            self.target_channels = [ds.variables[i] for i in self.target_idx]
        else:
            self.target_channels = stream_info.get(str(stage) + "_target_channels")
            self.target_idx = [ds.variables.index(ch) for ch in self.target_channels]

        # get target channel weights from stream config
        if stream_info.get("target_channel_weights") is None:
            self.target_channel_weights = self.parse_target_channel_weights()
        else:
            self.target_channel_weights = stream_info.get("target_channel_weights")

        # select/filter requested geoinfo channels (can be any variable, not just constant-in-time)
        if stream_info.get("geoinfo_channels") is None:
            self.geoinfo_idx = self.select_geoinfo_channels(ds)
            self.geoinfo_channels = [ds.variables[i] for i in self.geoinfo_idx]
        else:
            self.geoinfo_idx = self.select_geoinfo_channels(ds)
            self.geoinfo_channels = [ds.variables[i] for i in self.geoinfo_idx]
            # self.geoinfo_channels = stream_info.get("geoinfo_channels")
            # self.geoinfo_idx = [ds.variables.index(ch) for ch in self.geoinfo_channels]

        # set geoinfo normalization statistics
        if len(self.geoinfo_idx) > 0:
            self.mean_geoinfo = ds.statistics["mean"][self.geoinfo_idx]
            self.stdev_geoinfo = ds.statistics["stdev"][self.geoinfo_idx]
        else:
            self.mean_geoinfo = np.zeros(0)
            self.stdev_geoinfo = np.ones(0)

        ds_name = stream_info["name"]
        _logger.info(f"{ds_name}: source channels: {self.source_channels}")
        _logger.info(f"{ds_name}: target channels: {self.target_channels}")
        _logger.info(f"{ds_name}: geoinfo channels: {self.geoinfo_channels}")

        self.properties = {
            "stream_id": 0,
        }
        self.mean = ds.statistics["mean"]
        self.stdev = ds.statistics["stdev"]

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self.ds = None
        self.len = 0

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
        check_reader_data(rd, dtr)

        return rd

    def select_channels(self, ds0: anemoi_datasets, ch_type: str) -> NDArray[np.int64]:
        """
        Select source or target channels

        Parameters
        ----------
        ds0 :
            raw anemoi dataset with available channels
        ch_type :
            "source" or "target", i.e channel type to select

        Returns
        -------
        ReaderData providing coords, geoinfos, data, datetimes

        """

        channels = self.stream_info.get(ch_type)
        channels_exclude = self.stream_info.get(ch_type + "_exclude", [])
        # sanity check
        is_empty = len(channels) == 0 if channels is not None else False
        if is_empty:
            stream_name = self.stream_info["name"]
            _logger.warning(f"No channel for {stream_name} for {ch_type}.")

        chs_idx = np.sort(
            [
                ds0.name_to_index[k]
                for (k, v) in ds0.typed_variables.items()
                if (
                    not v.is_computed_forcing
                    and not v.is_constant_in_time
                    and (
                        np.array([f == k for f in channels]).any() if channels is not None else True
                    )
                    and not np.array([f == k for f in channels_exclude]).any()
                )
            ]
        )

        return np.array(chs_idx, dtype=np.int64)

    def select_geoinfo_channels(self, ds0: anemoi_datasets) -> NDArray[np.int64]:
        """
        Select geoinfo channels (can be any variable, not just constant-in-time)

        Parameters
        ----------
        ds0 :
            raw anemoi dataset with available channels

        Returns
        -------
        NDArray of channel indices for geoinfo variables

        """

        geoinfo_channels = self.stream_info.get("geoinfo_channels", [])

        if len(geoinfo_channels) == 0:
            return np.array([], dtype=np.int64)

        # Select channels that match the geoinfo list (exact match required)
        chs_idx = np.sort(
            [ds0.name_to_index[k] for k in ds0.typed_variables.keys() if k in geoinfo_channels]
        )

        if len(chs_idx) == 0 and len(geoinfo_channels) > 0:
            stream_name = self.stream_info["name"]
            _logger.warning(
                f"No matching geoinfo channels found for {stream_name}. "
                f"Requested: {geoinfo_channels}"
            )

        return np.array(chs_idx, dtype=np.int64)


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
