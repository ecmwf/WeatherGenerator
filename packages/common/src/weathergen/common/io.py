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

# experimental value, should be inferred more intelligently
CHUNK_N_SAMPLES = 16392
DType: typing.TypeAlias = np.float32
type NPDT64 = datetime64


_logger = logging.getLogger(__name__)


np.ndarray(3)


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

    @classmethod
    def create(cls, other: typing.Any) -> "IOReaderData":
        """
        create an instance from data_reader_base.ReaderData instance.

        other should be such an instance.
        """
        coords = other.coords
        geoinfos = other.geoinfos
        data = other.data
        datetimes = other.datetimes

        n_datapoints = len(data)

        assert coords.shape == (n_datapoints, 2), "number of datapoints do not match data"
        assert geoinfos.shape[0] == n_datapoints, "number of datapoints do not match data"
        assert datetimes.shape[0] == n_datapoints, "number of datapoints do not match data"

        return cls(**dataclasses.asdict(other))


@dataclasses.dataclass
class ItemKey:
    """Metadata to identify one output item."""

    sample: int
    forecast_step: int
    stream: str

    @property
    def path(self):
        """Unique path within a hierarchy for one output item."""
        return f"{self.sample}/{self.stream}/{self.forecast_step}"

    @property
    def with_source(self):
        """Decide if output item should contain source dataset."""
        # TODO: is this valid for the adjusted (offsetted) forecast steps?
        # => if config.forecast_offset > 0 source will be never written
        return self.forecast_step == 0


@dataclasses.dataclass
class OutputDataset:
    """Access source/target/prediction zarr data contained in one output item."""

    name: str
    item_key: ItemKey

    # (datapoints, channels, ens)
    data: zarr.Array  # wrong type => array like

    # (datapoints,)
    times: zarr.Array

    # (datapoints, 2)
    coords: zarr.Array

    # (datapoints, geoinfos) geoinfos are stream dependent => 0 for most gridded data
    geoinfo: zarr.Array

    channels: list[str]
    geoinfo_channels: list[str]

    @functools.cached_property
    def arrays(self) -> dict[str, zarr.Array]:
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

    def as_xarray(self, chunk_nsamples=CHUNK_N_SAMPLES) -> xr.Dataset:
        """Convert raw dask arrays into chunked dask-aware xarray dataset."""
        chunks = (chunk_nsamples, *self.data.shape[1:])

        # maybe do dask conversion earlier? => usefull for parallel writing?
        data = da.from_zarr(self.data, chunks=chunks)  # dont call compute to lazy load
        # include pseudo ens dim so all data arrays have same dimensionality
        # TODO: does it make sense for target and source to have ens dim?
        additional_dims = (0, 1, 2) if len(data.shape) == 3 else (0, 1, 2, 5)
        expanded_data = da.expand_dims(data, axis=additional_dims)
        coords = da.from_zarr(self.coords).compute()
        times = da.from_zarr(self.times).compute()
        geoinfo = da.from_zarr(self.geoinfo).compute()

        geoinfo = {name: ("ipoint", geoinfo[:, i]) for i, name in enumerate(self.geoinfo_channels)}
        # TODO: make sample, stream, forecast_step DataArray attribute, test how it
        # interacts with concatenating
        return xr.DataArray(
            expanded_data,
            dims=["sample", "stream", "forecast_step", "ipoint", "channel", "ens"],
            coords={
                "sample": [self.item_key.sample],
                "stream": [self.item_key.stream],
                "forecast_step": [self.item_key.forecast_step],
                "ipoint": self.datapoints,
                "channel": self.channels,  # TODO: make sure channel names align with data
                "valid_time": ("ipoint", times.astype("datetime64[ns]")),
                "lat": ("ipoint", coords[..., 0]),
                "lon": ("ipoint", coords[..., 1]),
                **geoinfo,
            },
            name=self.name,
        )


class OutputItem:
    def __init__(
        self,
        key: ItemKey,
        target: OutputDataset | None = None,
        prediction: OutputDataset | None = None,
        source: OutputDataset | None = None,
    ):
        """Collection of possible datasets for one output item."""
        self.key = key
        self.target = target
        self.prediction = prediction
        self.source = source

        self.datasets = [self.target, self.prediction]

        if self.key.with_source:
            if self.source:
                self.datasets.append(self.source)
            else:
                msg = f"Missing source dataset for item: {self.key.path}"
                raise ValueError(msg)


class ZarrIO:
    """Manage zarr storage hierarchy."""

    def __init__(self, store_path: pathlib.Path):
        self._store_path = store_path
        self.data_root = None

    def __enter__(self) -> typing.Self:
        self._store = zarr.storage.DirectoryStore(self._store_path)
        self.data_root = zarr.group(store=self._store)

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._store.close()

    def write_zarr(self, item: OutputItem):
        """Write one output item to the zarr store."""
        group = self._get_group(item.key, create=True)
        for dataset in item.datasets:
            self._write_dataset(group, dataset)

    def get_data(self, sample: int, stream: str, forecast_step: int) -> OutputItem:
        """Get datasets for the output item matching the arguments."""
        key = ItemKey(sample, forecast_step, stream)

        return self.load_zarr(key)

    def load_zarr(self, key: ItemKey) -> OutputItem:
        """Get datasets for a output item."""
        group = self._get_group(key)
        datasets = {
            name: OutputDataset(name, key, **dict(dataset.arrays()), **dataset.attrs)
            for name, dataset in group.groups()
        }
        datasets["key"] = key

        return OutputItem(**datasets)

    def _get_group(self, item: ItemKey, create: bool = False) -> zarr.Group:
        assert self.data_root is not None, "ZarrIO must be opened before accessing data."
        group: zarr.Group | None
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

    def _write_arrays(self, dataset_group: zarr.Group, dataset: OutputDataset):
        for array_name, array in dataset.arrays.items():  # suffix is eg. data or coords
            self._create_dataset(dataset_group, array_name, array)

    def _create_dataset(self, group: zarr.Group, name: str, array: NDArray):
        if array.size == 0:  # sometimes for geoinfo
            chunks = None
        else:
            chunks = (CHUNK_N_SAMPLES, *array.shape[1:])
        _logger.debug(
            f"writing array: {name} with shape: {array.shape},chunks: {chunks}"
            + "into group: {group}."
        )
        group.create_dataset(name, data=array, chunks=chunks)

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
        return list(example_stream.group_keys())


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

    # stream name: index into data (only streams in analysis_streams_output)
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
        return np.arange(len(self.sources)) + self.sample_start

    @functools.cached_property
    def forecast_steps(self):
        """Indices of all forecast steps adjusted by the forecast offset"""
        return np.arange(len(self.targets)) + self.forecast_offset

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
        datapoints = self._get_datapoints_per_sample(offset_key, stream_idx)

        _logger.debug(
            f"forecast_step: {key.forecast_step} = {offset_key.forecast_step} (rel_step) + "
            + f"{self.forecast_offset} (forecast_offset)"
        )
        _logger.debug(f"stream: {key.stream} with index: {stream_idx}")

        if (datapoints.stop - datapoints.start) == 0:
            target_data = np.zeros((0, len(self.target_channels[stream_idx])), dtype=np.float32)
            preds_data = np.zeros((0, len(self.target_channels[stream_idx])), dtype=np.float32)
        else:
            target_data = (
                self.targets[offset_key.forecast_step][stream_idx][0][datapoints]
                .cpu()
                .detach()
                .numpy()
            )
            preds_data = (
                self.predictions[offset_key.forecast_step][stream_idx][0]
                .transpose(1, 0)
                .transpose(1, 2)[datapoints]
                .cpu()
                .detach()
                .numpy()
            )

        data_coords = self._extract_coordinates(stream_idx, offset_key, datapoints)

        assert len(data_coords.channels) == target_data.shape[1], (
            "Number of channel names does not align with target data."
        )
        assert len(data_coords.channels) == preds_data.shape[1], (
            "Number of channel names does not align with prediction data."
        )

        if key.with_source:
            source_dataset = self._extract_sources(offset_key.sample, stream_idx, key)
        else:
            source_dataset = None

        return OutputItem(
            key=key,
            source=source_dataset,
            target=OutputDataset("target", key, target_data, **dataclasses.asdict(data_coords)),
            prediction=OutputDataset(
                "prediction", key, preds_data, **dataclasses.asdict(data_coords)
            ),
        )

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

    def _extract_sources(self, sample, stream_idx, key):
        channels = self.source_channels[stream_idx]
        geoinfo_channels = self.geoinfo_channels[stream_idx]

        source = self.sources[sample][stream_idx]

        assert source.data.shape[1] == len(channels), (
            "Number of source channel names does not align with source data"
        )

        source_dataset = OutputDataset(
            "source",
            key,
            source.data,
            source.datetimes,
            source.coords,
            source.geoinfos,
            channels,
            geoinfo_channels,
        )

        _logger.debug(f"source shape: {source_dataset.data.shape}")

        return source_dataset
