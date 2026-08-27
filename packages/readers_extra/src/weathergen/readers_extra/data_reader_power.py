# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import json
import logging
from pathlib import Path
from typing import override

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from weathergen.datasets.data_reader_anemoi import _clip_lat, _clip_lon
from weathergen.datasets.data_reader_base import (
    DataReaderBase,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)
from weathergen.train.utils import Stage

_logger = logging.getLogger(__name__)

# Fixed parquet schema (``plant_id`` is for evaluation only and is not read here).
PARQUET_READ_COLUMNS: tuple[str, ...] = ("timestamp", "lat", "lon", "power")
PARQUET_COLUMN_SET: frozenset[str] = frozenset(PARQUET_READ_COLUMNS)

# Internal WeatherGenerator column names after normalization.
INTERNAL_DATETIME = "datetime"
POWER_CHANNEL = "obsvalue_power"
INTERNAL_COLUMNS: tuple[str, ...] = (INTERNAL_DATETIME, "lat", "lon", POWER_CHANNEL)

PARQUET_TO_INTERNAL_RENAME: dict[str, str] = {
    "timestamp": INTERNAL_DATETIME,
    "power": POWER_CHANNEL,
}

_MIN_STDDEV = 1e-5

# Lazy reader init sets df, stats, indices_*, etc. outside __init__ (same pattern as other readers).
# pylint: disable=attribute-defined-outside-init


class DataReaderPower(DataReaderBase):
    """
    Data reader for power production observations in Parquet format.

    Parquet schema (required)
    -------------------------
    - ``timestamp`` (UTC), ``lat``, ``lon``, ``power`` (capacity-normalized generation)

    Internally, ``timestamp`` -> ``datetime`` and ``power`` -> ``obsvalue_power``.
    ``plant_id`` and other non-training columns are not read.

    Statistics
    ----------
    Normalization statistics are loaded from a required JSON file at
    ``metadata/{stream_name}_stats.json`` (mean/std per variable under a top-level
    ``statistics`` key).
    """

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
        stage: Stage | None = None,
    ) -> None:
        # `stage` is accepted for API compatibility with the upstream reader
        # signature (added in PR #2336). This reader uses a single fixed
        # observation schema for both train and val, so `stage` is ignored.
        del stage
        super().__init__(tw_handler, stream_info)

        self._filename = Path(filename)
        self._tw_handler = tw_handler
        self._stream_info = stream_info

        self.init_empty()
        self._initialized = False

        # Prime channel lists eagerly (schema-only; safe pre-fork).
        self._prime_channels_from_schema()

    @staticmethod
    def _assert_parquet_columns(path: Path) -> None:
        """Raise if the parquet file does not expose the expected column set."""
        dataset = ds.dataset(path, format="parquet")
        missing = PARQUET_COLUMN_SET - set(dataset.schema.names)
        if missing:
            raise ValueError(
                f"Power parquet at {path} is missing required columns: {sorted(missing)}. "
                f"Expected exactly: {sorted(PARQUET_COLUMN_SET)}",
            )

    def _prime_channels_from_schema(self) -> None:
        """Populate channel lists before lazy init (schema check only; safe pre-fork)."""
        if self._filename.is_file() or self._filename.is_dir():
            self._assert_parquet_columns(self._filename)

        self._data_colnames = [POWER_CHANNEL]
        self._setup_channels(self._data_colnames, [])

        self.mean = np.zeros((len(self._data_colnames),), dtype=np.float32)
        self.stdev = np.ones((len(self._data_colnames),), dtype=np.float32)
        self.mean_geoinfo = np.zeros(0, dtype=np.float32)
        self.stdev_geoinfo = np.ones(0, dtype=np.float32)

    def _lazy_init(self) -> None:
        """Lazy initialization — called once per worker process."""
        if self._initialized:
            return

        if self._filename.is_dir():
            data_dir = self._filename
        else:
            data_dir = self._filename.parent

        stats_file = self._find_stats_file(data_dir)
        with open(stats_file, encoding="utf-8") as f:
            self.stats = json.load(f)
        _logger.info("Loaded stats from %s", stats_file)

        self.df = self._load_parquet_as_dataframe(self._filename, self._tw_handler)
        if self.df is None or len(self.df) == 0:
            _logger.warning("No data found in the requested time window")
            self.df = None
            self.len = 0
            self.indices_start = np.zeros((0,), dtype=np.int64)
            self.indices_end = np.zeros((0,), dtype=np.int64)
            self._data_colnames = [POWER_CHANNEL]
            self._setup_channels(self._data_colnames, [])
            self._load_statistics(self._data_colnames)
            self._initialized = True
            return

        self.df = self._normalize_schema(self.df)
        self.df = self.df.sort_values("datetime").reset_index(drop=True)
        self._validate_schema()

        self.df["lat"] = _clip_lat(self.df["lat"].to_numpy())
        self.df["lon"] = _clip_lon(self.df["lon"].to_numpy())

        self._data_colnames = [POWER_CHANNEL]
        self._setup_channels(self._data_colnames, [])
        self._load_statistics(self._data_colnames)

        self._setup_sample_index()
        self.len = int(self.indices_start.shape[0])

        ds_name = self._stream_info.get("name", "Power")
        _logger.info("%s: source channels: %s", ds_name, self.source_channels)
        _logger.info("%s: target channels: %s", ds_name, self.target_channels)
        _logger.info("%s: dataset length (num windows): %s", ds_name, self.len)

        self._initialized = True

    def _load_parquet_as_dataframe(
        self,
        path: Path,
        tw_handler: TimeWindowHandler,
    ) -> pd.DataFrame:
        """Load parquet rows in ``PARQUET_READ_COLUMNS`` for the loader time window."""
        if not path.is_file() and not path.is_dir():
            raise FileNotFoundError(f"Power parquet path not found: {path}")

        self._assert_parquet_columns(path)

        dataset = ds.dataset(path, format="parquet")
        t0 = pd.Timestamp(tw_handler.t_start).to_pydatetime()
        t1 = pd.Timestamp(tw_handler.t_end).to_pydatetime()
        filt = (ds.field("timestamp") >= t0) & (ds.field("timestamp") < t1)
        table = dataset.to_table(columns=list(PARQUET_READ_COLUMNS), filter=filt)
        return table.to_pandas()

    def _normalize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename parquet columns to the fixed internal schema."""
        df = df.rename(columns=PARQUET_TO_INTERNAL_RENAME)
        df[INTERNAL_DATETIME] = pd.to_datetime(df[INTERNAL_DATETIME], errors="raise")
        df["lat"] = df["lat"].astype(np.float32)
        df["lon"] = df["lon"].astype(np.float32)
        df[POWER_CHANNEL] = df[POWER_CHANNEL].astype(np.float32)
        return df[list(INTERNAL_COLUMNS)]

    def _find_stats_file(self, data_dir: Path) -> Path:
        """Return ``metadata/{stream_name}_stats.json``, or raise if missing."""
        dataset_name = self._stream_info.get("name", "power").lower()
        stats_file = data_dir.parent / "metadata" / f"{dataset_name}_stats.json"
        if not stats_file.exists():
            raise FileNotFoundError(f"Power stats file not found: {stats_file}")
        return stats_file

    def _validate_schema(self) -> None:
        """Validate the dataframe matches the fixed internal column order."""
        assert self.df is not None

        if list(self.df.columns) != list(INTERNAL_COLUMNS):
            raise ValueError(
                f"Expected internal columns {list(INTERNAL_COLUMNS)}, got {list(self.df.columns)}",
            )

    def _setup_channels(
        self,
        data_colnames: list[str],
        geo_colnames: list[str],
    ) -> None:
        """Setup source/target/geoinfo channels from stream config filters."""
        source_filter = self._stream_info.get("source")
        source_exclude = self._stream_info.get("source_exclude", [])
        self.source_channels = self._select_channels(data_colnames, source_filter, source_exclude)
        self.source_idx = np.array(
            [data_colnames.index(ch) for ch in self.source_channels], dtype=np.int64
        )

        target_filter = self._stream_info.get("target")
        target_exclude = self._stream_info.get("target_exclude", [])
        self.target_channels = self._select_channels(data_colnames, target_filter, target_exclude)
        self.target_idx = np.array(
            [data_colnames.index(ch) for ch in self.target_channels], dtype=np.int64
        )

        self.geoinfo_channels = geo_colnames
        self.geoinfo_idx = list(range(len(geo_colnames)))

        # Base class default is []; physical loss needs one weight per target channel.
        yaml_weights = self._stream_info.get("target_channel_weights")
        if yaml_weights:
            assert len(yaml_weights) == len(self.target_channels), (
                f"target_channel_weights has {len(yaml_weights)} entries but "
                f"stream has {len(self.target_channels)} target channels"
            )
            self.target_channel_weights = list(yaml_weights)
        else:
            self.target_channel_weights = [1.0] * len(self.target_channels)

    def _select_channels(
        self,
        available_vars: list[str],
        include_filters: list[str] | None,
        exclude_filters: list[str] | None = None,
    ) -> list[str]:
        """Select channels based on include/exclude patterns."""
        if exclude_filters is None:
            exclude_filters = []

        selected = []
        for var in available_vars:
            if include_filters is not None:
                if not any(f in var or f == var for f in include_filters):
                    continue
            if any(f in var for f in exclude_filters):
                continue
            selected.append(var)

        return selected

    def _parse_channel_stats(self, channel: str, entry: object) -> tuple[float, float]:
        """Parse and validate one channel entry from the stats JSON."""
        if not isinstance(entry, dict):
            raise ValueError(
                f"Stats entry for {channel!r} must be a mapping, got {type(entry).__name__}",
            )
        if "mean" not in entry or "std" not in entry:
            raise ValueError(f"Stats entry for {channel!r} must include 'mean' and 'std'")
        try:
            mean = float(entry["mean"])
            std = float(entry["std"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Stats entry for {channel!r} must have numeric 'mean' and 'std'",
            ) from exc
        if not np.isfinite(mean):
            raise ValueError(f"Stats mean for {channel!r} must be finite, got {mean!r}")
        if not np.isfinite(std) or std <= _MIN_STDDEV:
            raise ValueError(
                f"Stats std for {channel!r} must be finite and > {_MIN_STDDEV}, got {std!r}",
            )
        return mean, std

    def _load_statistics(self, channels: list[str]) -> None:
        """Load mean/std for each channel from the stats JSON."""
        if not self.stats or "statistics" not in self.stats:
            raise ValueError("Stats JSON must contain a top-level 'statistics' object")

        stats_dict = self.stats["statistics"]
        if not isinstance(stats_dict, dict):
            raise ValueError("Stats JSON 'statistics' must be a mapping")

        means: list[float] = []
        stdevs: list[float] = []

        for ch in channels:
            if ch not in stats_dict:
                raise KeyError(f"No statistics entry for channel {ch!r} in stats file")
            mean, std = self._parse_channel_stats(ch, stats_dict[ch])
            means.append(mean)
            stdevs.append(std)

        self.mean = np.array(means, dtype=np.float32)
        self.stdev = np.array(stdevs, dtype=np.float32)

        self.mean_geoinfo = np.zeros(0, dtype=np.float32)
        self.stdev_geoinfo = np.ones(0, dtype=np.float32)

    def _setup_sample_index(self) -> None:
        """Build vectorized (start, end) row offsets for every time window."""
        assert self.df is not None

        dt = self.df["datetime"].to_numpy(dtype="datetime64[ns]")
        idx_range = self._tw_handler.get_index_range()
        n_windows = int(idx_range.end - idx_range.start)
        if n_windows <= 0:
            self.indices_start = np.zeros((0,), dtype=np.int64)
            self.indices_end = np.zeros((0,), dtype=np.int64)
            return

        win_offsets = np.arange(0, n_windows, dtype=np.int64)
        win_starts = self._tw_handler.t_start + self._tw_handler.t_window_step * win_offsets
        win_ends = win_starts + self._tw_handler.t_window_len

        win_starts = win_starts.astype("datetime64[ns]")
        win_ends = win_ends.astype("datetime64[ns]")

        self.indices_start = np.searchsorted(dt, win_starts, side="left").astype(np.int64)
        self.indices_end = np.searchsorted(dt, win_ends, side="left").astype(np.int64)

    @override
    def init_empty(self) -> None:
        """Initialize an empty reader."""
        super().init_empty()
        self.df: pd.DataFrame | None = None
        self._data_colnames: list[str] = []
        self.len = 0
        self.stats = None

    @override
    def length(self) -> int:
        if not self._initialized:
            self._lazy_init()
        return self.len

    @override
    def get_source(self, idx: TIndex) -> ReaderData:
        if not self._initialized:
            self._lazy_init()
        return self._get(idx, self.source_idx)

    @override
    def get_target(self, idx: TIndex) -> ReaderData:
        if not self._initialized:
            self._lazy_init()
        return self._get(idx, self.target_idx)

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        if not self._initialized:
            self._lazy_init()

        if self.len == 0 or self.df is None or len(channels_idx) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx),
                num_geo_fields=len(self.geoinfo_idx),
            )

        idx_i = int(idx)
        if idx_i < 0 or idx_i >= len(self.indices_end):
            return ReaderData.empty(
                num_data_fields=len(channels_idx),
                num_geo_fields=len(self.geoinfo_idx),
            )

        start_row = int(self.indices_start[idx_i])
        end_row = int(self.indices_end[idx_i])
        df_window = self.df.iloc[start_row:end_row]

        if len(df_window) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx),
                num_geo_fields=len(self.geoinfo_idx),
            )

        coords = df_window[["lat", "lon"]].to_numpy(dtype=np.float32)
        geoinfos = np.zeros((len(df_window), 0), dtype=np.float32)

        selected_cols = [self._data_colnames[i] for i in channels_idx]
        data = df_window[selected_cols].to_numpy(dtype=np.float32)
        datetimes = df_window["datetime"].to_numpy(dtype="datetime64[ns]")

        t_win = self._tw_handler.window(np.int64(idx_i))
        t_mask = (datetimes >= t_win.start) & (datetimes < t_win.end)

        rdata = ReaderData(
            coords=coords[t_mask],
            geoinfos=geoinfos[t_mask],
            data=data[t_mask],
            datetimes=datetimes[t_mask],
        )

        check_reader_data(rdata, t_win)

        return rdata
