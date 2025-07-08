import dataclasses
import functools
import itertools
import logging
import pathlib
import pdb
import typing

import dask.array as da
import numpy as np
import xarray as xr
import zarr
from numpy.typing import NDArray

# experimental value, should be inferred more intelligently
CHUNK_N_SAMPLES = 16392
DType: typing.TypeAlias = np.float32

_logger = logging.getLogger(__name__)


np.ndarray(3)


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
    data: zarr.Array  # wrong type => arary like

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
                "lat": ("ipoint", coords[:, 0]),
                "lon": ("ipoint", coords[:, 1]),
                **geoinfo,
            },
            name=self.name,
        )


class OutputItem:
    def __init__(
        self,
        target: OutputDataset,
        prediction: OutputDataset,
        source: OutputDataset | None = None,
    ):
        """Collection of possible datasets for one output item."""
        self.target = target
        self.prediction = prediction
        self.source = source

        self.key = self.target.item_key

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
        self._store = zarr.DirectoryStore(self._store_path)
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
class OutputBatchData:
    """Provide convenient access to adapt existing output data structures."""

    # sample, stream, tensor(datapoint, channel) => datapoints is accross all datasets per stream
    sources: list[list]

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

    stream_names: list[str]

    # stream, channel name
    channels: list[list[str]]
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
        filtered_streams = (stream for stream in self.stream_names if stream != "")
        # TODO: filter for empty items?
        for s, fo_s, fi_s in itertools.product(self.samples, self.forecast_steps, filtered_streams):
            yield self.extract(ItemKey(int(s), int(fo_s), fi_s))

    def extract(self, key: ItemKey) -> OutputItem:
        """Extract datasets from lists for one output item."""
        # adjust shifted values in ItemMeta
        sample = key.sample - self.sample_start
        forecast_step = key.forecast_step - self.forecast_offset
        stream_idx = self.stream_names.index(key.stream)  # TODO: assure this is correct
        lens = self.targets_lens[forecast_step][stream_idx]
        start = sum(lens[:sample])
        n_samples = lens[sample]

        _logger.info("extracting subset")
        _logger.info(
            f"sample: start:{self.sample_start} rel_idx:{sample} range:{start}-{start + n_samples}"
        )
        _logger.info(
            f"forecast_step: {key.forecast_step} = {forecast_step} (rel_step) + "
            + f"{self.forecast_offset} (forecast_offset)"
        )
        _logger.info(f"stream: {key.stream} with index: {stream_idx}")

        datapoints = slice(start, start + n_samples)

        target_data = self.targets[forecast_step][stream_idx][0][datapoints].cpu().detach().numpy()
        preds_data = (
            self.predictions[forecast_step][stream_idx][0]
            .transpose(1, 0)
            .transpose(1, 2)[datapoints]
            .cpu()
            .detach()
            .numpy()
        )

        _coords = self.targets_coords[forecast_step][stream_idx][datapoints].numpy()
        coords = _coords[:, :2]  # first two columns are lat,lon
        geoinfo = _coords[:, 2:]  # the rest is geoinfo => potentially empty
        if geoinfo.size > 0:  # TODO: set geoinfo to be empty for now
            geoinfo = np.empty((geoinfo.shape[0], 0))
            _logger.warning(
                "geoinformation channels are not implemented yet."
                + "will be truncated to be of size 0."
            )
        times = self.targets_times[forecast_step][stream_idx][
            datapoints
        ]  # make conversion to datetime64[ns] here?
        pdb.set_trace()
        channels = self.channels[stream_idx]
        geoinfo_channels = self.geoinfo_channels[stream_idx]

        if key.with_source:
            source_data = self.sources[sample][stream_idx].cpu().detach().numpy()
            source_dataset = OutputDataset(
                "source",
                key,
                source_data,
                times,
                coords,
                geoinfo,
                channels,
                geoinfo_channels,
            )
        else:
            source_dataset = None

        return OutputItem(
            source=source_dataset,
            target=OutputDataset(
                "target",
                key,
                target_data,
                times,
                coords,
                geoinfo,
                channels,
                geoinfo_channels,
            ),
            prediction=OutputDataset(
                "prediction",
                key,
                preds_data,
                times,
                coords,
                geoinfo,
                channels,
                geoinfo_channels,
            ),
        )
