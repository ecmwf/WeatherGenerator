import dataclasses
import itertools
import typing

import dask as da
import numpy as np
import torch
import xarray as xr
import zarr

# experimental value, should be inferred more intelligently
CHUNK_N_SAMPLES = 16392

# TODO: extract geoinfo metadata, include it in xarray

@dataclasses.dataclass
class ItemMeta:
    sample: str
    forecast_step: int
    stream: str

    @property
    def key(self):
        return f"{self.sample}/{self.stream}/{self.forecast_step}"

    @property
    def with_source(self):
        return self.forecast_step == 0


@dataclasses.dataclass
class OutputDataset:
    name: str
    item: ItemMeta

    # (datapoints, channels, ens)
    data: np.ndarray

    # (datapoints,)
    times: np.ndarray

    # (datapoints, 2) => maybe more??
    coords: np.ndarray

    # (datapoints, ???)
    geoinfo: np.ndarray | None

    channels: list

    @property
    def datapoints(self) -> np.ndarray:
        return np.arange(self.data.shape[0])

    def as_xarray(self, chunk_nsamples=CHUNK_N_SAMPLES) -> xr.Dataset:  #
        """"""
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
                "sample": [self.item.sample],
                "stream": [self.item.stream],
                "forecast_step": [self.item.forecast_step],
                "ipoint": self.datapoints,
                "channel": self.channels,
                "valid_time": ("ipoint", times.astype("datetime64[ns]")),
                "lat": ("ipoint", coords[:, 0]),
                "lon": ("ipoint", coords[:, 1]),
            },
            name=self.name,
        )


@dataclasses.dataclass
class OutputItem:
    target: OutputDataset
    prediction: OutputDataset
    source: OutputDataset | None = None

    # pseudo-field (init-only variable) => kept out of most dataclasses logic
    _meta: dataclasses.InitVar[ItemMeta | None] = None

    def __post_init__(self):
        self._meta = self.target.meta
        if self._meta.with_source and not self.source:
            msg = f"Missing source dataset for item: {self._meta.key}"
            raise ValueError(msg)

    @property
    def datasets(self):
        datasets = (self.source, self.target, self.prediction)
        return datasets if self._meta.with_source else datasets[1:]

class MockIO:
    pass

class ZarrIO:
    def __init__(self, store_path: zarr):
        self._store_path = store_path
        self.data_root = None

    def __enter__(self) -> typing.Self:
        self._store = zarr.DirectoryStore(self._store_path)
        self.data_root = zarr.group(store=self._store)

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._store.close()

    def write_zarr(self, item: OutputItem):
        group = self._get_group(item._meta, create=True)
        for dataset in item.datasets:
            self._write_dataset(group, dataset)

    def get_data(self, sample: int, stream: str, forecast_step: int) -> OutputItem:
        meta = ItemMeta(sample, forecast_step, stream)

        return self.load_zarr(meta, self.data_root)

    def load_zarr(self, meta: ItemMeta) -> OutputItem:
        group = self._get_group(meta)
        datasets = {
            OutputDataset(key, meta, **dataset.arrays(), **dataset.args)
            for key, dataset in group.groups()
        }

        return OutputItem(**datasets)

    def _get_group(self, item: ItemMeta, create: bool = False) -> zarr.Group:
        if create:
            group = self.data_root.create_group(item.key)
        else:
            try:
                group = self.data_root.get(item.key)
            except KeyError as e:
                msg = f"Zarr group: {item.key} has not been created."
                raise FileNotFoundError(msg) from e

        return group

    def _write_dataset(self, item_group, dataset):
        dataset_group = item_group.require_group(dataset.name)
        self._write_metadata(dataset_group, dataset)
        self._write_arrays(dataset_group, dataset)

    def _write_metadata(self, dataset_group: zarr.Group, dataset: OutputDataset):
        dataset_group.attrs["item"] = dataclasses.asdict(dataset.item)
        dataset_group.attrs["channels"] = dataset.channels

    def _write_arrays(self, dataset_group, dataset):
        for array_name, array in dataset:  # suffix is eg. data or coords
            self._create_dataset(dataset_group, array_name, array)

    def _create_dataset(self, group: zarr.Group, name: str, array: np.ndarray | None):
        if array:  # TODO guard against geoinfo = None
            chunks = (CHUNK_N_SAMPLES, array.shape[1:])
            group.create_dataset(name, data=array, chunks=chunks)

    @property
    def samples(self) -> list[int]:
        return list(self.data_root.groups_keys())

    @property
    def streams(self) -> list[str]:
        # assume stream/samples are orthogonal => use first sample
        _, example_sample = next(self.data_root.groups())
        return list(example_sample.group_keys())

    @property
    def forecast_steps(self) -> list[int]:
        # assume stream/samples/forecast_steps are orthogonal
        _, example_stream = next(next(self.data_root.groups()).groups())
        return list(example_stream.group_keys())


@dataclasses.dataclass
class OutputBatchData:
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
    targets_times: list[list[np.ndarray]]

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

    def items(self) -> typing.Generator[OutputItem]:
        filtered_streams = (stream for stream in self.stream_names if stream != "")
        for args in itertools.product(self.samples, self.fsteps, filtered_streams):
            yield self.extract(ItemMeta(*args))

    def extract(self, meta: ItemMeta) -> OutputItem:
        # adjust shifted values in ItemMeta
        sample = meta.sample - self.sample_start
        forecast_step = meta.forecast_step - self.forecast_offset
        stream_idx = self.stream_names.index(meta.stream)  # TODO: assure this is correct
        
        start = sum(self.targets_lens[: sample])
        datapoints = slice(start, self.targets_lens[sample])

        target_data = (
            self.targets[forecast_step][stream_idx][0][datapoints].cpu().detach().numpy()
        )
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
