from __future__ import annotations  # allow forward references in typehints

import dataclasses
import itertools
import typing
from collections.abc import Callable

import numpy as np
from omegaconf.errors import ConfigAttributeError

from weathergen.common.config import Config, get_path_results
from weathergen.common.io import (
    ItemKey,
    OutputDataset,
    OutputItem,
    TimeRange,
    zarrio_writer,
)
from weathergen.datasets.batch import ModelBatch
from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.model.model import ModelOutput
from weathergen.train.target_and_aux_module_base import PhysicalTarget, TargetAuxOutput


class Writer:
    def __init__(
        self,
        config: Config,
        val_cfg: Config,
        streams: list[Config],
    ):
        streams = {stream.name: stream for stream in streams}
        # TODO: nice: dont store all config
        self._twh = TimeWindowHandler(
            val_cfg.start_date,
            val_cfg.end_date,
            val_cfg.time_window_len,
            val_cfg.time_window_step,
        )

        _all_streams = list(streams.keys())
        _output_streams = val_cfg.get("output", None).get("streams", None)
        _output_streams = val_cfg.output.streams if val_cfg.output.streams else _all_streams

        self._streams = {
            name: config for name, config in streams.items() if name in _output_streams
        }
        self._forecast_offset = val_cfg.forecast.offset
        self._cf: Config = config  # used for zarr output path lookup => improve

    def write_batch(
        self,
        mini_epoch: int,  # TODO: nice: use iterstep for better consistency?
        batch: ModelBatch,
        targets: TargetAuxOutput,
        predictions: ModelOutput,
        normalizer: typing.Callable,
        fsteps: range,
    ) -> None:
        data = _BatchOutputData(batch, targets, predictions, normalizer)
        # TODO: nice: get result path differently
        with zarrio_writer(get_path_results(self._cf, mini_epoch)) as zio:
            for subset in self.itemize(data, fsteps):
                zio.write_zarr(subset)

    def itemize(
        self, data: _BatchOutputData, fstep_range: range
    ) -> typing.Generator[OutputItem, None, None]:
        """Iterate over possible output items"""

        # TODO: check: filter for empty items?
        for key in self.keys(fstep_range, data.samples):
            yield self.extract(data, key)

    def keys(
        self, forecast_steps: range, samples: list[int]
    ) -> typing.Generator[ItemKey, None, None]:
        """Iterate over possible output items"""
        streams: list[str] = self._streams.keys()

        # The order of iteration is important here:
        # streams is the outermost and samples the innermost loop variable
        # This is important since normalization is best done at a per stream/fstep basis
        for stream, forecast_step, sample_idx in itertools.product(
            streams, forecast_steps, samples
        ):
            yield ItemKey(sample_idx, forecast_step, stream)

    def extract(self, data: _BatchOutputData, key: ItemKey) -> OutputItem:
        data_invariants = self._get_invariants(key)

        source, target, prediction = None, None, None
        if key.with_source:
            source = data.extract_source(key).as_dataset(key, data_invariants)
        if key.with_target(self._forecast_offset):
            target = data.extract_target(key).as_dataset(key, data_invariants)
            prediction = data.extract_prediction(key).as_dataset(key, data_invariants)

        return OutputItem(
            key,
            self._forecast_offset,  # TODO nice: maybe drop it?
            target,
            prediction,
            source
        )        

    def _get_invariants(self, key: ItemKey) -> _DataInvariants:
        # TODO unify DTRange and TimeRange classes
        window = self._twh.window(key.sample)
        return _DataInvariants(
            source_interval=TimeRange(window.start, window.end),
            # val_source_channels are ListConfig[str] objects -> convert to list[str]
            source_channels=list(self._streams[key.stream].val_source_channels),
            target_channels=list(self._streams[key.stream].val_target_channels),
            geoinfo_channels=list(self._streams[key.stream].geoinfo_channels),
        )


@dataclasses.dataclass
class _BatchOutputData:
    _batch: ModelBatch
    _targets: TargetAuxOutput
    _predictions: ModelOutput
    _normalizer: Callable

    def extract_source(self, key: ItemKey) -> _ExtractedData:
        # TODO check this?
        # breakpoint()
        READER_DATA_INDEX_MYSTERY = 0
        source = (
            self._batch.source_samples.samples[key.forecast_step]
            .streams_data[key.stream]
            .source_raw[READER_DATA_INDEX_MYSTERY]
        )

        return _ExtractedData(
            "prediction",
            np.asarray(source.data),
            np.asarray(source.datetimes),
            np.asarray(source.coords),
            np.asarray(source.geoinfos),
        )
    
    @property
    def samples(self):
        # TODO check: data._batch.source_samples.samples
        return self._batch.source_samples.sample_idxs

    def extract_target(self, key: ItemKey) -> _ExtractedData:
        target = self._target(key)
        coords = self._target_coordinates(target)

        return _ExtractedData("target", target.data, coords.times, coords.coords, coords.geoinfos)

    def extract_prediction(self, key: ItemKey) -> _ExtractedData:
        try:
            data = self._predictions.get_physical_prediction_normalized(key, self._normalizer)
        except Exception as e:
            # TODO: if preds are empty so create copy of target and add ensemble dimension
            # preds = [targets[0].clone().unsqueeze(0)]
            raise ValueError("not handled yet") from e
            data = self._target(key)
        target = self._target(key)
        coords = self._target_coordinates(target)

        return _ExtractedData("prediction", data, coords.times, coords.coords, coords.geoinfos)

    # TODO guarantee this method is only called once per OutputItem
    # TODO try getting targets from batch directly
    def _target(self, key: ItemKey) -> PhysicalTarget:
        return self._targets.get_physical_target_normalized(key, self._normalizer)

    def _target_coordinates(self, target: PhysicalTarget) -> _ExtractedData:
        coords = target.coords[..., :2]  # first two columns are lat,lon
        geoinfos = target.coords[..., 2:]  # the rest is geoinfo => potentially empty

        return _ExtractedData("", None, target.datetimes, coords, geoinfos)


@dataclasses.dataclass
class _DataInvariants:
    source_interval: TimeRange
    source_channels: list[str]
    target_channels: list[str]
    geoinfo_channels: list[str]


@dataclasses.dataclass
class _ExtractedData:
    name: str  # TODO make enum
    data: typing.Any
    times: typing.Any
    coords: typing.Any
    geoinfos: typing.Any

    def as_dataset(self, key: ItemKey, invariants: _DataInvariants) -> OutputDataset:
        if self.data is None or self.data.shape == (0, 0):
            return None
        else:
            return OutputDataset(
                name=self.name,
                item_key=key,
                data=self.data,
                times=self.times,
                coords=self.coords,
                geoinfo=self.geoinfos,
                source_interval=invariants.source_interval,
                channels=invariants.target_channels,
                geoinfo_channels=invariants.geoinfo_channels,
            )
