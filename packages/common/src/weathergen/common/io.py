# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import dataclasses
import functools
import itertools
import logging
import pathlib
import typing

import dask.array as da
import numpy as np
import xarray as xr
import zarr
from numpy import datetime64
from numpy.typing import NDArray
from zarr.storage import LocalStore

# experimental value, should be inferred more intelligently
CHUNK_N_SAMPLES = 16392
type DType = np.float32
type NPDT64 = datetime64
type ArrayType = zarr.Array | np.NDArray[DType]


_logger = logging.getLogger(__name__)


def is_ndarray(obj: typing.Any) -> bool:
    """Check if object is an ndarray (wraps the linter warning)."""
    return isinstance(obj, (np.ndarray))  # noqa: TID251


class TimeRange:
    """
    Holds information about a time interval used in forecasting.

    Time interval is left-closed, right-open. TimeRange can be instatiated from
    numpy datetime64 objects or strings as outputed by TimeRange.as_dict.
    Both will be converted to datetime64 with nanosecond precision.

    Attrs:
        start: Start of the time range in nanoseconds.
        end: End of the time range in nanoseconds
    """

    def __init__(self, start: NPDT64 | str, end: NPDT64 | str):
        # ensure consistent type => convert serialized strings
        self.start = np.datetime64(start, "ns")
        self.end = np.datetime64(end, "ns")

        assert self.start < self.end

    def as_dict(self) -> dict[str, str]:
        """
        Convert instance to a JSON-serializable dict.

        will convert datetime objects as "YYYY-MM-DDThh:mm:s.sssssssss"

        Returns:
            JSON-serializable dict, wher datetime objects were converted to strings.
        """
        return {
            "start": str(self.start),
            "end": str(self.end),
        }

    def forecast_interval(self, forecast_dt_hours: int, fstep: int) -> "TimeRange":
        """
        Infer the interval cosidered at forecast step `fstep`.

        Args:
            forecast_dt_hours: number of hours the source TimeRange is shifted per forecast step.
            fstep: current forecast step.

        Returns:
            New TimeRange shifted TimeRange.
        """
        assert forecast_dt_hours > 0 and fstep >= 0
        offset = np.timedelta64(forecast_dt_hours * fstep, "h")
        return TimeRange(self.start + offset, self.end + offset)


@dataclasses.dataclass
class IOReaderData:
    """
    Equivalent to data_reader_base.ReaderData

    This class needs to exist since otherwise the common package would
    have a dependecy on the core model. Ultimately a unified data model
    should be implemented in the common package.
    """

    coords: NDArray[DType]
    geoinfos: NDArray[DType]
    data: NDArray[DType]
    datetimes: NDArray[NPDT64]

    def is_empty(self):
        """
        Test if data object is empty
        """
        return len(self.data) == 0

    @classmethod
    def create(cls, other: typing.Any) -> "IOReaderData":
        """
        Create an instance from data_reader_base.ReaderData instance.

        other should be such an instance.
        """
        coords = np.asarray(other.coords)
        geoinfos = np.asarray(other.geoinfos)
        data = np.asarray(other.data)
        datetimes = np.asarray(other.datetimes)

        n_datapoints = len(data)

        assert coords.shape == (n_datapoints, 2), "number of datapoints do not match data"
        assert geoinfos.shape[0] == n_datapoints, "number of datapoints do not match data"
        assert datetimes.shape[0] == n_datapoints, "number of datapoints do not match data"

        return cls(**dataclasses.asdict(other))

    @classmethod
    def combine(cls, others: list["IOReaderData"]) -> "IOReaderData":
        """
        Create an instance from data_reader_base.ReaderData instance by combining mulitple ones.

        others is list of ReaderData instances.
        """
        assert len(others) > 0, len(others)

        other = others[0]
        coords = np.zeros((0, other.coords.shape[1]), dtype=other.coords.dtype)
        geoinfos = np.zeros((0, other.geoinfos.shape[1]), dtype=other.geoinfos.dtype)
        data = np.zeros((0, other.data.shape[1]), dtype=other.data.dtype)
        datetimes = np.array([], dtype=other.datetimes.dtype)

        for other in others:
            n_datapoints = len(other.data)
            assert other.coords.shape == (n_datapoints, 2), "number of datapoints do not match"
            assert other.geoinfos.shape[0] == n_datapoints, "number of datapoints do not match"
            assert other.datetimes.shape[0] == n_datapoints, "number of datapoints do not match"

            coords = np.concatenate([coords, other.coords])
            geoinfos = np.concatenate([geoinfos, other.geoinfos])
            data = np.concatenate([data, other.data])
            datetimes = np.concatenate([datetimes, other.datetimes])

        return cls(coords, geoinfos, data, datetimes)


@dataclasses.dataclass
class ItemKey:
    """Metadata to identify one output item."""

    sample: int
    forecast_step: int
    stream: str

    @property
    def path(self) -> str:
        """Unique path within a hierarchy for one output item."""
        return f"{self.sample}/{self.stream}/{self.forecast_step}"

    @property
    def with_source(self) -> bool:
        """Decide if output item should contain source dataset."""
        return self.forecast_step == 0

    def with_target(self, forecast_offset: typing.Literal[0, 1]) -> bool:
        """Decide if output item should contain target and predictions."""
        assert forecast_offset in (0, 1)
        return (not self.with_source) or (forecast_offset == 0)

    @staticmethod
    def _infer_forecast_offset(datasets: dict[str, typing.Any]) -> int:
        """
        Infer forecast offset by the (non)presence of targets at fstep 0.

        Args:
            datasets: Datasets found in a fstep 0 OutputItem.
        """
        # forecast offset=1 should produce no targets at fstep 0
        return 0 if "target" in datasets else 1


@dataclasses.dataclass
class OutputDataset:
    """Access source/target/prediction zarr data contained in one output item."""

    name: str
    item_key: ItemKey
    source_interval: TimeRange

    # (datapoints, channels, ens)
    data: ArrayType  # wrong type => array like

    # (datapoints,)
    times: ArrayType

    # (datapoints, 2)
    coords: ArrayType

    # (datapoints, geoinfos) geoinfos are stream dependent => 0 for most gridded data
    geoinfo: ArrayType

    channels: list[str]
    geoinfo_channels: list[str]

    @classmethod
    def create(
        cls, name: str, key: ItemKey, arrays: dict[str, ArrayType], attrs: dict[str, typing.Any]
    ):
        """
        Create Output dataset from dictonaries.

        Args:
            name: Name of dataset (target/prediction/source)
            item_key: ItemKey to associated with the parent OutputItem.
            arrays: Data and Coordinate arrays.
            attrs: Additional metadata.
        """
        assert "source_interval" in attrs, "missing expected attribute 'source_interval'"

        source_interval = TimeRange(**attrs.pop("source_interval"))
        return cls(name, key, source_interval, **arrays, **attrs)

    @functools.cached_property
    def arrays(self) -> dict[str, ArrayType]:
        """Iterate over the arrays and their names."""
        return {
            "data": self.data,
            "times": self.times,
            "coords": self.coords,
            "geoinfo": self.geoinfo,
        }

    @functools.cached_property
    def datapoints(self) -> NDArray[np.int_]:
        return np.arange(self.data.shape[0])

    def as_xarray(self, chunk_nsamples=CHUNK_N_SAMPLES) -> xr.DataArray:
        """Convert raw dask arrays into chunked dask-aware xarray dataset."""
        chunks = (chunk_nsamples, *self.data.shape[1:])

        # maybe do dask conversion earlier? => usefull for parallel writing?
        data = da.from_zarr(self.data, chunks=chunks)  # dont call compute to lazy load
        # include pseudo ens dim so all data arrays have same dimensionality
        # TODO: does it make sense for target and source to have ens dim?
        additional_dims = (0, 1, 2) if len(data.shape) == 3 else (0, 1, 2, 5)
        expanded_data = da.expand_dims(data, axis=additional_dims)
        coords = da.from_zarr(self.coords).compute()
        times = da.from_zarr(self.times).compute().astype("datetime64[ns]")
        geoinfo = da.from_zarr(self.geoinfo).compute()
        geoinfo = {name: ("ipoint", geoinfo[:, i]) for i, name in enumerate(self.geoinfo_channels)}
        # TODO: make sample, stream, forecast_step DataArray attribute, test how it
        # interacts with concatenating
        dims = ["sample", "stream", "forecast_step", "ipoint", "channel", "ens"]
        ds_coords = {
            "sample": [self.item_key.sample],
            "source_interval_start": ("sample", [self.source_interval.start]),
            "source_interval_end": ("sample", [self.source_interval.end]),
            "stream": [self.item_key.stream],
            "forecast_step": [self.item_key.forecast_step],
            "ipoint": self.datapoints,
            "channel": self.channels,  # TODO: make sure channel names align with data
            "valid_time": ("ipoint", times),
            "lat": ("ipoint", coords[..., 0]),
            "lon": ("ipoint", coords[..., 1]),
            **geoinfo,
        }
        return xr.DataArray(expanded_data, dims=dims, coords=ds_coords, name=self.name)


class OutputItem:
    def __init__(
        self,
        key: ItemKey,
        forecast_offset=int | None,
        target: OutputDataset | None = None,
        prediction: OutputDataset | None = None,
        source: OutputDataset | None = None,
    ):
        """Collection of possible datasets for one output item."""
        self.key = key
        self.target = target
        self.prediction = prediction
        self.source = source

        self.datasets = []

        if self.key.with_source:
            self._append_dataset(self.source, "source")

        if self.key.with_target(forecast_offset):
            self._append_dataset(self.target, "target")
            self._append_dataset(self.prediction, "prediction")

    def _append_dataset(self, dataset: OutputDataset | None, name: str) -> None:
        if dataset:
            self.datasets.append(dataset)
        else:
            msg = f"Missing {name} dataset for item: {self.key.path}"
            raise ValueError(msg)


class ZarrIO:
    """Manage zarr storage hierarchy."""

    def __init__(self, store_path: pathlib.Path):
        self._store_path = store_path
        self.data_root: zarr.Group | None = None
        self._store: LocalStore | None = None

    def __enter__(self) -> typing.Self:
        self._store = LocalStore(self._store_path)
        self.data_root = zarr.group(store=self._store)

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self._store is not None:
            self._store.close()

    def write_zarr(self, item: OutputItem):
        """Write one output item to the zarr store."""
        group = self._get_group(item.key, create=True)
        for dataset in item.datasets:
            if dataset is not None:
                self._write_dataset(group, dataset)

    def get_data(self, sample: int, stream: str, forecast_step: int) -> OutputItem:
        """Get datasets for the output item matching the arguments."""
        key = ItemKey(sample, forecast_step, stream)

        return self.load_zarr(key)

    def load_zarr(self, key: ItemKey) -> OutputItem:
        """Get datasets for a output item."""
        datasets = self._get_datasets(key)

        return OutputItem(key=key, forecast_offset=self.forecast_offset, **datasets)

    def _get_datasets(self, key: ItemKey):
        group = self._get_group(key)
        return {
            name: OutputDataset.create(
                name, key, dict(dataset.arrays()), dict(dataset.attrs).copy()
            )
            for name, dataset in group.groups()
        }

    def _get_group(self, item: ItemKey, create: bool = False) -> zarr.Array | zarr.Group:
        assert self.data_root is not None, "ZarrIO must be opened before accessing data."
        if create:
            group = self.data_root.create_group(item.path)
        else:
            try:
                group = self.data_root.get(item.path)
                assert group is not None, f"Zarr group: {item.path} does not exist."
            except KeyError as e:
                msg = f"Zarr group: {item.path} has not been created."
                raise FileNotFoundError(msg) from e

        assert group is not None, f"Zarr group: {item.path} does not exist."
        return group

    def _write_dataset(self, item_group: zarr.Group, dataset: OutputDataset):
        dataset_group = item_group.require_group(dataset.name)
        self._write_metadata(dataset_group, dataset)
        self._write_arrays(dataset_group, dataset)

    def _write_metadata(self, dataset_group: zarr.Group, dataset: OutputDataset):
        dataset_group.attrs["channels"] = dataset.channels
        dataset_group.attrs["geoinfo_channels"] = dataset.geoinfo_channels
        dataset_group.attrs["source_interval"] = dataset.source_interval.as_dict()

    def _write_arrays(self, dataset_group: zarr.Group, dataset: OutputDataset):
        for array_name, array in dataset.arrays.items():  # suffix is eg. data or coords
            self._create_dataset(dataset_group, array_name, array)

    def _create_dataset(self, group: zarr.Group, name: str, array: NDArray):
        assert is_ndarray(array), f"Expected ndarray but got: {type(array)}"
        if array.size == 0:  # sometimes for geoinfo
            chunks = "auto"
        else:
            chunks = (CHUNK_N_SAMPLES, *array.shape[1:])
        _logger.debug(
            f"writing array: {name} with shape: {array.shape},chunks: {chunks}"
            + "into group: {group}."
        )
        group.create_array(name, data=array, chunks=chunks)

    @functools.cached_property
    def forecast_offset(self) -> int:
        fstep0_datasets = self._get_datasets(self.example_key)
        return ItemKey._infer_forecast_offset(fstep0_datasets)

    @functools.cached_property
    def example_key(self) -> ItemKey:
        try:
            sample, example_sample = next(self.data_root.groups())
            stream, example_stream = next(example_sample.groups())
            fstep = 0
        except StopIteration as e:
            msg = f"Data store at: {self._store_path} is empty."
            raise FileNotFoundError(msg) from e

        return ItemKey(sample, fstep, stream)

    @functools.cached_property
    def samples(self) -> list[int]:
        """Query available samples in this zarr store."""
        return list(self.data_root.group_keys())

    @functools.cached_property
    def streams(self) -> list[str]:
        """Query available streams in this zarr store."""
        # assume stream/samples are orthogonal => use first sample
        _, example_sample = next(self.data_root.groups())
        return list(example_sample.group_keys())

    @functools.cached_property
    def forecast_steps(self) -> list[int]:
        """Query available forecast steps in this zarr store."""
        # assume stream/samples/forecast_steps are orthogonal
        _, example_sample = next(self.data_root.groups())
        _, example_stream = next(example_sample.groups())

        all_steps = sorted(list(example_stream.group_keys()))

        if self.forecast_offset == 1:
            return all_steps[1:]  # exclude fstep with no targets/preds
        else:
            return all_steps


@dataclasses.dataclass
class DataCoordinates:
    times: typing.Any
    coords: typing.Any
    geoinfo: typing.Any
    channels: typing.Any
    geoinfo_channels: typing.Any


@dataclasses.dataclass
class OutputBatchData:
    """Provide convenient access to adapt existing output data structures."""

    # sample, stream, tensor(datapoint, channel+coords)
    # => datapoints is accross all datasets per stream
    sources: list[list[IOReaderData]]

    # sample
    source_intervals: list[TimeRange]

    # fstep, stream, redundant dim (size 1), tensor(sample x datapoint, channel)
    targets: list[list[list]]

    # fstep, stream, redundant dim (size 1), tensor(ens, sample x datapoint, channel)
    predictions: list[list[list]]

    # fstep, stream, tensor(sample x datapoint, 2 + geoinfos)
    targets_coords: list[list]

    # fstep, stream, (sample x datapoint)
    targets_times: list[list[NDArray[DType]]]

    # fstep, stream, redundant dim (size 1)
    targets_lens: list[list[list[int]]]

    # stream name: index into data (only streams in streams_output)
    streams: dict[str, int]

    # stream, channel name
    target_channels: list[list[str]]
    source_channels: list[list[str]]
    geoinfo_channels: list[list[str]]

    sample_start: int
    forecast_offset: int

    @functools.cached_property
    def samples(self):
        """Continous indices of all samples accross all batches."""

        # TODO associate samples with the sampel idx used for the time window
        return np.arange(len(self.sources)) + self.sample_start

    @functools.cached_property
    def forecast_steps(self):
        """Indices of all forecast steps adjusted by the forecast offset"""
        # forecast offset should be either 1 for forecasting or 0 for MTM
        assert self.forecast_offset in (0, 1)
        return np.arange(len(self.targets) + self.forecast_offset)

    def items(self) -> typing.Generator[OutputItem, None, None]:
        """Iterate over possible output items"""
        # TODO: filter for empty items?
        for s, fo_s, fi_s in itertools.product(
            self.samples, self.forecast_steps, self.streams.keys()
        ):
            yield self.extract(ItemKey(int(s), int(fo_s), fi_s))

    def extract(self, key: ItemKey) -> OutputItem:
        """Extract datasets from lists for one output item."""
        _logger.debug(f"extracting subset: {key}")
        offset_key = self._offset_key(key)
        stream_idx = self.streams[key.stream]

        source_interval = self.source_intervals[offset_key.sample]
        _logger.debug(
            f"forecast_step: {key.forecast_step} = {offset_key.forecast_step} (rel_step) + "
            + f"{self.forecast_offset} (forecast_offset)"
        )
        _logger.debug(f"stream: {key.stream} with index: {stream_idx}")

        assert self.forecast_offset in (0, 1)
        if key.with_source:
            source_dataset = self._extract_sources(
                offset_key.sample, stream_idx, key, source_interval
            )
        else:
            source_dataset = None

        if key.with_target(self.forecast_offset):
            target_dataset, prediction_dataset = self._extract_targets_predictions(
                stream_idx, offset_key, key, source_interval
            )
        else:
            target_dataset, prediction_dataset = (None, None)

        return OutputItem(
            key=key,
            forecast_offset=self.forecast_offset,
            source=source_dataset,
            target=target_dataset,
            prediction=prediction_dataset,
        )

    def _offset_key(self, key: ItemKey):
        """
        Correct indices in key to be useable for data extraction.

        `key` contains indices that are adjusted to have better output semantics.
        To be useable in extraction these have to be adjusted to bridge the differences
        compared to the semantics of the data.
            - `sample` is adjusted from a global continous index to a per batch index
            - `forecast_step` is adjusted from including `forecast_offset` to indexing
               the data (always starts at 0)
        """
        return ItemKey(
            key.sample - self.sample_start, key.forecast_step - self.forecast_offset, key.stream
        )

    def _extract_targets_predictions(self, stream_idx, offset_key, key, source_interval):
        datapoints = self._get_datapoints_per_sample(offset_key, stream_idx)
        data_coords = self._extract_coordinates(stream_idx, offset_key, datapoints)

        if (datapoints.stop - datapoints.start) == 0:
            target_data = np.zeros((0, len(self.target_channels[stream_idx])), dtype=np.float32)
            preds_data = np.zeros((0, len(self.target_channels[stream_idx])), dtype=np.float32)
        else:
            target_data = self.targets[offset_key.forecast_step][stream_idx][0][datapoints]
            preds_data = self.predictions[offset_key.forecast_step][stream_idx][0].transpose(
                1, 2, 0
            )[datapoints]

        assert len(data_coords.channels) == target_data.shape[1], (
            "Number of channel names does not align with target data."
        )
        assert len(data_coords.channels) == preds_data.shape[1], (
            "Number of channel names does not align with prediction data."
        )

        target_dataset = OutputDataset(
            "target",
            key,
            source_interval,
            target_data,
            **dataclasses.asdict(data_coords),
        )
        prediction_dataset = OutputDataset(
            "prediction",
            key,
            source_interval,
            preds_data,
            **dataclasses.asdict(data_coords),
        )

        return target_dataset, prediction_dataset

    def _get_datapoints_per_sample(self, offset_key, stream_idx):
        lens = self.targets_lens[offset_key.forecast_step][stream_idx]

        # empty target/prediction
        if len(lens) == 0:
            start = 0
            n_samples = 0
        else:
            start = sum(lens[: offset_key.sample])
            n_samples = lens[offset_key.sample]

        _logger.debug(
            f"sample: start:{self.sample_start} rel_idx:{offset_key.sample}"
            + f"range:{start}-{start + n_samples}"
        )

        return slice(start, start + n_samples)

    def _extract_coordinates(self, stream_idx, offset_key, datapoints) -> DataCoordinates:
        _coords = self.targets_coords[offset_key.forecast_step][stream_idx][datapoints].numpy()

        # ensure _coords has size (?,2)
        if len(_coords) == 0:
            _coords = np.zeros((0, 2), dtype=np.float32)

        coords = _coords[..., :2]  # first two columns are lat,lon
        geoinfo = _coords[..., 2:]  # the rest is geoinfo => potentially empty
        if geoinfo.size > 0:  # TODO: set geoinfo to be empty for now
            geoinfo = np.empty((geoinfo.shape[0], 0))
            _logger.warning(
                "geoinformation channels are not implemented yet."
                + "will be truncated to be of size 0."
            )
        times = self.targets_times[offset_key.forecast_step][stream_idx][
            datapoints
        ]  # make conversion to datetime64[ns] here?
        channels = self.target_channels[stream_idx]
        geoinfo_channels = self.geoinfo_channels[stream_idx]

        return DataCoordinates(times, coords, geoinfo, channels, geoinfo_channels)

    def _extract_sources(
        self, sample: int, stream_idx: int, key: ItemKey, source_interval: TimeRange
    ) -> OutputDataset:
        channels = self.source_channels[stream_idx]
        geoinfo_channels = self.geoinfo_channels[stream_idx]

        source: IOReaderData = self.sources[sample][stream_idx]

        assert source.data.shape[1] == len(channels), (
            "Number of source channel names does not align with source data"
        )

        source_dataset = OutputDataset(
            "source",
            key,
            source_interval,
            np.asarray(source.data),
            np.asarray(source.datetimes),
            np.asarray(source.coords),
            np.asarray(source.geoinfos),
            channels,
            geoinfo_channels,
        )

        _logger.debug(f"source shape: {source_dataset.data.shape}")

        return source_dataset
