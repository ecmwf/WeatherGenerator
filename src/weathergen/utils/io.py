import dataclasses
import itertools
import pathlib
import typing

import dask as da
import numpy as np
import torch
import xarray as xr
import zarr
from numpy.typing import NDArray

# experimental value, should be inferred more intelligently
CHUNK_N_SAMPLES = 16392
DType: typing.TypeAlias = np.float32

# TODO: extract geoinfo metadata, include it in xarray


@dataclasses.dataclass
class ItemKey:
    """Metadata to identify one output item."""

    sample: str
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
        return self.forecast_step == 0


@dataclasses.dataclass
class OutputDataset:
    """Access source/target/prediction zarr data contained in one output item."""

    name: str
    item_key: ItemKey

    # (datapoints, channels, ens)
    data: zarr.Array

    # (datapoints,)
    times: zarr.Array

    # (datapoints, 2) => maybe more??
    coords: zarr.Array

    # (datapoints, ???)
    geoinfo: zarr.Array | None

    channels: list[str]

    @property
    def datapoints(self) -> NDArray[np.int_]:
        return np.arange(self.data.shape[0])

    def as_xarray(self, chunk_nsamples=CHUNK_N_SAMPLES) -> xr.Dataset:
        """Convert raw dask arrays into chunked dask-aware xarray dataset."""
        chunks = (chunk_nsamples, *self.data.shape[1:])

        # maybe do dask conversion earlier? => usefull for parallel writing?
        data = da.from_zarr(self.data, chunks=chunks)  # dont call compute to lazy load
        coords = da.from_zarr(self.coords).compute()
        times = da.from_zarr(self.times).compute()

        # TODO include geoinfo

        return xr.DataArray(
            da.expand_dims(data, axis=(0, 1, 2)),
            dims=["sample", "stream", "forecast_step", "ipoint", "channel", "ens"],
            coords={
                "sample": [self.item_key.sample],
                "stream": [self.item_key.stream],
                "forecast_step": [self.item_key.forecast_step],
                "ipoint": self.datapoints,
                "channel": self.channels,
                "valid_time": ("ipoint", times.astype("datetime64[ns]")),
                "lat": ("ipoint", coords[:, 0]),
                "lon": ("ipoint", coords[:, 1]),
            },
            name=self.name,
        )


class OutputItem:
    def __init__(
        self, target: OutputDataset, prediction: OutputDataset, source: OutputDataset | None = None
    ):
        """Collection of possible datasets for one output item."""
        self.target = target
        self.prediction = prediction
        self.source = source

        self._key = self.target.item_key

        self.datasets = [self.target, self.prediction]

        if self._key.with_source:
            if self.source:
                self.datasets += self.source
            else:
                msg = f"Missing source dataset for item: {self._key.path}"
                raise ValueError(msg)


class MockIO:
    pass


class ZarrIO:
    """Manage zarr storage hierarchy."""

    def __init__(self, store_path: pathlib.Path):
        self._store_path = store_path
        self.data_root = None

    def __enter__(self) -> typing.Self:
        self._store = zarr.DirectoryStore(self._store_path)
        self.data_root = zarr.group(store=self._store)

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._store.close()

    def write_zarr(self, item: OutputItem):
        """Write one output item to the zarr store."""
        group = self._get_group(item._key, create=True)
        for dataset in item.datasets:
            self._write_dataset(group, dataset)

    def get_data(self, sample: int, stream: str, forecast_step: int) -> OutputItem:
        """Get datasets for the output item matching the arguments."""
        meta = ItemKey(sample, forecast_step, stream)

        return self.load_zarr(meta, self.data_root)

    def load_zarr(self, meta: ItemKey) -> OutputItem:
        """Get datasets for a output item."""
        group = self._get_group(meta)
        datasets = {
            OutputDataset(key, meta, **dataset.arrays(), **dataset.args)
            for key, dataset in group.groups()
        }

        return OutputItem(**datasets)

    def _get_group(self, item: ItemKey, create: bool = False) -> zarr.Group:
        if create:
            group = self.data_root.create_group(item.path)
        else:
            try:
                group = self.data_root.get(item.path)
            except KeyError as e:
                msg = f"Zarr group: {item.path} has not been created."
                raise FileNotFoundError(msg) from e

        return group

    def _write_dataset(self, item_group, dataset):
        dataset_group = item_group.require_group(dataset.name)
        self._write_metadata(dataset_group, dataset)
        self._write_arrays(dataset_group, dataset)

    def _write_metadata(self, dataset_group: zarr.Group, dataset: OutputDataset):
        dataset_group.attrs["item"] = dataclasses.asdict(dataset.item_key)
        dataset_group.attrs["channels"] = dataset.channels

    def _write_arrays(self, dataset_group, dataset):
        for array_name, array in dataset:  # suffix is eg. data or coords
            self._create_dataset(dataset_group, array_name, array)

    def _create_dataset(self, group: zarr.Group, name: str, array: NDArray | None):
        if array:  # TODO guard against geoinfo = None
            chunks = (CHUNK_N_SAMPLES, array.shape[1:])
            group.create_dataset(name, data=array, chunks=chunks)

    @property
    def samples(self) -> list[int]:
        """Query available samples in this zarr store."""
        return list(self.data_root.groups_keys())

    @property
    def streams(self) -> list[str]:
        """Query available streams in this zarr store."""
        # assume stream/samples are orthogonal => use first sample
        _, example_sample = next(self.data_root.groups())
        return list(example_sample.group_keys())

    @property
    def forecast_steps(self) -> list[int]:
        """Query available forecast steps in this zarr store."""
        # assume stream/samples/forecast_steps are orthogonal
        _, example_stream = next(next(self.data_root.groups()).groups())
        return list(example_stream.group_keys())


@dataclasses.dataclass
class OutputBatchData:
    """Provide convenient access to adapt existing output data structures."""

    # sample, stream, (datapoint, channel) => datapoints is accross all datasets per stream
    sources: list[list[torch.Tensor]]

    # fstep, stream, redundant dim (size 1), (sample x datapoint, channel)
    targets: list[list[list[torch.Tensor]]]

    # fstep, stream, redundant dim (size 1), (ens, sample x datapoint, channel)
    predictions: list[list[list[torch.Tensor]]]

    # fstep, stream, (sample x datapoint, 105)
    # => 105 (documentation StreamDatat.add_target) ???
    targets_coords: list[list[torch.Tensor]]

    # fstep, stream, (sample x datapoint)
    targets_times: list[list[NDArray]]

    # fstep, stream, redundant dim (size 1)
    targets_lens: list[list[list[int]]]

    stream_names: list[str]

    # stream, channel name
    channels: list[list[str]]

    sample_start: int
    forecast_offset: int

    def __post_init__(self):
        self.samples = np.arange(len(self.sources)) + self.sample_start
        self.fsteps = np.arange(len(self.targets)) + self.forecast_offset

    def items(self) -> typing.Generator[ItemKey, None, None]:
        """Iterate over possible output items"""
        filtered_streams = (stream for stream in self.stream_names if stream != "")
        # TODO: filter for empty items?
        for args in itertools.product(self.samples, self.fsteps, filtered_streams):
            yield self.extract(ItemKey(*args))

    def extract(self, meta: ItemKey) -> OutputItem:
        """Extract datasets from lists for one output item."""
        # adjust shifted values in ItemMeta
        sample = meta.sample - self.sample_start
        forecast_step = meta.forecast_step - self.forecast_offset
        stream_idx = self.stream_names.index(meta.stream)  # TODO: assure this is correct

        start = sum(self.targets_lens[:sample])
        datapoints = slice(start, self.targets_lens[sample])

        target_data = self.targets[forecast_step][stream_idx][0][datapoints].cpu().detach().numpy()
        preds_data = (
            self.predictions[forecast_step][stream_idx][0]
            .transpose(1, 0)
            .transpose(1, 2)[datapoints]
            .cpu()
            .detach()
            .numpy()
        )

        coords = self.target_coords[forecast_step][stream_idx][datapoints].numpy()
        times = self.target_times[forecast_step][stream_idx][
            datapoints
        ]  # make conversion to datetime64[ns] here?

        if meta.with_source:
            source_data = self.sources[sample][stream_idx].cpu().detach.numpy()
            OutputDataset("source", meta, source_data, times, coords, None, self.channels)
        else:
            source_dataset = None

        return OutputItem(
            source_dataset,
            OutputDataset("target", meta, target_data, coords, times, None, self.channels),
            OutputDataset("prediction", meta, preds_data, coords, times, None, self.channels),
        )
