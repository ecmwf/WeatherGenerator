# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import dataclasses
import logging
import pathlib
from collections.abc import Sequence

import numpy as np
import torch
from omegaconf import OmegaConf

from weathergen.common.config import Config
from weathergen.common.io import IOReaderData
from weathergen.datasets.batch import ModelBatch
from weathergen.datasets.data_reader_anemoi import DataReaderAnemoi
from weathergen.datasets.data_reader_base import (
    DataReaderBase,
    TimeWindowHandler,
    TIndex,
)
from weathergen.datasets.data_reader_fesom import DataReaderFesom
from weathergen.datasets.data_reader_obs import DataReaderObs
from weathergen.datasets.masking import Masker
from weathergen.datasets.stream_data import StreamData, spoof
from weathergen.datasets.tokenizer_masking import TokenizerMasking
from weathergen.datasets.utils import (
    get_tokens_lens,
)
from weathergen.readers_extra.registry import get_extra_reader
from weathergen.train.utils import Stage, get_batch_size_from_config
from weathergen.utils.distributed import is_root

type AnyDataReader = DataReaderBase | DataReaderAnemoi | DataReaderObs
type StreamName = str

logger = logging.getLogger(__name__)

FORECAST_DEFAULTS = {
    "offset": 0,
    "time_step": np.timedelta64(0, "ms"),
    "policy": None,
    "num_steps": np.array([0], dtype=np.int32),
}


def collect_datasources(stream_datasets: list, idx: int, type: str, rng) -> IOReaderData:
    """
    Utility function to collect all sources / targets from streams list

    rng and num_subset are used to drop data
    """

    rdatas = []

    for ds in stream_datasets:
        # number of points to sub-sample
        num_subset = -1

        if type == "source":
            get_reader_data = ds.get_source
            normalize_channels = ds.normalize_source_channels
            shuffle = ds.stream_info.get("shuffle_source", False)
        elif type == "target":
            get_reader_data = ds.get_target
            normalize_channels = ds.normalize_target_channels
            num_subset = ds.stream_info.get("max_num_targets", -1)
            shuffle = ds.stream_info.get("shuffle_target", False)
        else:
            assert False, "invalid value for argument `type`"

        # get source (of potentially multi-step length)
        rdata = (
            get_reader_data(idx).shuffle(rng, shuffle, num_subset).remove_nan_coords_and_geoinfos()
        )
        rdata.data = normalize_channels(rdata.data)
        rdata.geoinfos = ds.normalize_geoinfos(rdata.geoinfos)
        rdatas += [rdata]

    return IOReaderData.combine(rdatas)


@dataclasses.dataclass
class _Stream:
    info: Config
    readers: list[DataReaderBase]


class MultiStreamDataSampler(torch.utils.data.IterableDataset):
    def __init__(self, cf: Config, mode_cfg: dict, stage: Stage):
        super(MultiStreamDataSampler, self).__init__()

        self.mode_cfg = mode_cfg
        self._stage = stage

        self.mini_epoch = 0
        self.mask_value = 0.0
        self.rank = cf.rank
        self.world_size = cf.world_size
        self.repeat_data = cf.data_loading.get("repeat_data_in_mini_epoch", False)

        # initialise healpic
        self.healpix_level = cf.healpix_level
        self.num_healpix_cells = 12 * 4**self.healpix_level
        self.masker = Masker(cf.healpix_level, stage, cf.streams, self.mode_cfg)
        self.tokenizer = TokenizerMasking(cf.healpix_level, self.masker)

        forecast_cfg = FORECAST_DEFAULTS | OmegaConf.to_object(mode_cfg.get("forecast", {}))
        self.output_offset = forecast_cfg["offset"]
        self.time_step = forecast_cfg["time_step"]
        self.forecast_policy = forecast_cfg["policy"]
        steps = np.array(forecast_cfg["num_steps"], dtype=np.int32).reshape(-1)
        self.list_num_forecast_steps = np.array(steps, dtype=np.int32)

        # initialise fsm, but can change for future mini_epochs
        self.batch_size = get_batch_size_from_config(mode_cfg)
        self.shuffle = mode_cfg.shuffle

        self.len_timedelta = mode_cfg.time_window_len
        self.step_timedelta = mode_cfg.time_window_step
        tw = TimeWindowHandler(
            self.mode_cfg.start_date,
            self.mode_cfg.end_date,
            self.len_timedelta,
            self.step_timedelta,
        )

        self.time_window_handler = tw
        if is_root():
            logger.info(self.time_window_handler)
        self.index_range = tw.get_index_range()

        # check samples per mini epoch
        self.samples_per_mini_epoch = mode_cfg.samples_per_mini_epoch
        self.check_samples(self._get_fsm())
        self.streams_datasets = self._init_stream_datasets(cf)

        # RNG seed setup
        rs = cf.data_loading.rng_seed
        nw = cf.data_loading.num_workers
        self.data_loader_rng_seed = rs if rs > nw else rs * 97

        self.rng = None

    def check_samples(self, fsm: int):
        """Check if samples_per_mini_epoch is suitable
        Repeated both to initialise the MultiStreamDataSampler and for each mini epoch"""

        max_index = self.index_range.end - (
            (  # max time units needed to make a forecast
                self.time_step * (fsm + self.output_offset)  # translation due to forecasting
                + self.len_timedelta  # length of forecasting window
            )
            // self.step_timedelta  # as number of indexs
        )

        available_samples = max_index * self.batch_size  # as number of samples

        assert available_samples > 0, (
            "There is an insufficient date range to \
accomodate any number of samples or forecast steps"
        )

        # choose correct num samples
        if not self.repeat_data and self.samples_per_mini_epoch:
            if self.samples_per_mini_epoch >= available_samples:
                logger.warning(
                    f"There are only {available_samples} available_samples, \
samples_per_mini_epoch reduced to {available_samples} to avoid repeating data. \
Set repeat_data_in_mini_epoch to True if this is undesired."
                )
                self.samples_per_mini_epoch = max(available_samples - 1, 1)
            else:
                logger.info("Sufficient available samples in the time range specified")
        else:
            logger.info("Samples will be repeated within the time range")

        # streamlined calculation of length
        epoch_len = self.samples_per_mini_epoch

        # ensure epoch_len is large enough to produce at least one batch per rank
        min_samples = self.world_size * self.batch_size
        if epoch_len < min_samples:
            logger.warning(
                f"samples_per_mini_epoch={epoch_len} is too small for "
                f"world_size={self.world_size} and batch_size={self.batch_size}. "
                f"samples_per_mini_epoch has to be equal to or larger than"
                f"world_size*batch_size to ensure that each rank can produce at least one sample. "
                f"Automatically increasing to {min_samples}."
            )
            epoch_len = min_samples
            self.samples_per_mini_epoch = min_samples

        # adjust len to split loading across all workers and ensure it is multiple of batch_size
        self.len = ((epoch_len // self.world_size) // self.batch_size) * self.batch_size

        n_duplicates = self.len * self.world_size - available_samples
        if not self.repeat_data:
            assert n_duplicates <= 0

    def _calc_baseperms(self, fsm: int) -> np.typing.NDArray:
        """This calculates the base permutation array and
        depends on fsm so must be repeated for __init__ and reset"""
        perms_len = int(self.index_range.end - self.index_range.start)
        perms_len -= (fsm + self.output_offset) * (self.time_step // self.step_timedelta)

        return np.arange(perms_len)

    def _init_stream_datasets(self, cf) -> dict[StreamName, _Stream]:
        """Load dataset readers for all streams from config."""
        streams_datasets: dict[StreamName, _Stream] = {}
        for stream_name, stream_info in cf.streams.items():
            # list of sources for current stream
            streams_datasets[stream_name] = _Stream(stream_info, [])

            kwargs = {
                "tw_handler": self.time_window_handler,
                "stream_info": stream_info,
                "stage": self._stage,
            }
            dataset: type[AnyDataReader] | None = None
            match stream_info["type"]:
                case "obs":
                    dataset = DataReaderObs
                case "anemoi":
                    dataset = DataReaderAnemoi
                case "fesom":
                    dataset = DataReaderFesom
                case type_name:
                    dataset = get_extra_reader(type_name)
                    if dataset is None:
                        msg = f"Unsupported stream type {stream_info['type']}"
                        f"for stream name '{stream_name}'."
                        raise ValueError(msg)

            for fname in stream_info["filenames"]:
                fname = pathlib.Path(fname)
                # dont check if file exists since zarr stores might be directories
                if fname.exists():
                    # check if fname is a valid path to allow for simple overwriting
                    filename = fname
                else:
                    filenames = [pathlib.Path(path) / fname for path in cf.data_paths]

                    if not any(filename.exists() for filename in filenames):  # see above
                        msg = (
                            f"Did not find input data for {stream_info['type']} "
                            f"stream '{stream_name}': {filenames}."
                        )
                        raise FileNotFoundError(msg)

                    # The same dataset can exist on different locations in the filesystem,
                    # so we need to choose here.
                    filename = filenames[0]

                ds_type = stream_info["type"]
                if is_root():
                    logger.info(
                        f"Opening dataset with type: {ds_type}"
                        + f" from stream config {stream_name}.",
                    )
                ds = dataset(filename=filename, **kwargs)

                streams_datasets[stream_name].readers += [ds]

            stream_info[str(self._stage) + "_source_channels"] = ds.source_channels
            stream_info[str(self._stage) + "_target_channels"] = ds.target_channels
            stream_info["target_channel_weights"] = (
                ds.target_channel_weights
                if ds.target_channel_weights is not None
                else [1.0 for _ in ds.target_channels]
            )

        return streams_datasets

    def reset(self) -> tuple[Sequence[int], Sequence[int]]:
        """
        Reset RNG, return shuffled perms adn forecast steps for this mini epoch.

        The permutation index size is proportional to self.samples_per_mini_epoch,
        wheras the forecast steps index length is proportional to len(self).

        Returns: permutation index, forecast steps index
        """
        self.rng = np.random.default_rng(self.data_loader_rng_seed)
        fsm = self._get_fsm()
        self.check_samples(fsm)
        perms = self._calc_baseperms(fsm)

        # rng changed, repeat if needed
        n_requested_idxs = self.samples_per_mini_epoch // self.batch_size
        if self.repeat_data and len(perms) < n_requested_idxs:
            perms = np.tile(perms, n_requested_idxs // len(perms))
            filler = self.rng.choice(
                perms,
                size=n_requested_idxs - len(perms),
                replace=False,
            )
            perms = np.concatenate([perms, filler])

        # shuffle
        if self.shuffle:
            perms = self.rng.permutation(perms)

        len_dt = len(self) // self.batch_size

        if self.forecast_policy is None:
            fs = np.zeros(len_dt, dtype=np.int64)

        elif self.forecast_policy in ("fixed", "sequential"):
            fs = fsm * np.ones(len_dt, dtype=np.int64)

        elif self.forecast_policy in ("random", "sequential_random"):
            fs = self.rng.integers(
                low=self.list_num_forecast_steps.min(),
                high=fsm + 1,
                size=len_dt,
                dtype=np.int64,
            )
        else:
            raise ValueError(f"Unknown forecast policy {self.forecast_policy}")

        # reset tokenizer RNG
        self.tokenizer.reset_rng(self.rng)
        return (perms, fs)

    def _get_fsm(self) -> int:
        """Obtain maximum number of forecast steps for current mini epoch."""
        # fixed number of forecast steps for this run
        if self.forecast_policy != "random":
            idx = min(self.mini_epoch, len(self.list_num_forecast_steps) - 1)
            fsm = self.list_num_forecast_steps[idx]
        else:
            fsm = self.list_num_forecast_steps.max()

        if fsm > 0:
            logger.info(f"forecast_steps : {fsm}")
        return fsm

    def advance(self):
        """
        Advance mini_epoch (this is applied to the template for the worker processes)
        """
        self.mini_epoch += 1

    def get_sources_size(self):
        return [
            0
            if ds.readers[0].get_source_num_channels() == 0
            else ds.readers[0].get_source_num_channels()
            + ds.readers[0].get_geoinfo_size()
            + ds.readers[0].get_coords_size()
            + self.tokenizer.get_size_time_embedding()
            for ds in self.streams_datasets.values()
        ]

    def get_sources_num_channels(self):
        return [ds.readers[0].get_source_num_channels() for ds in self.streams_datasets.values()]

    def get_targets_num_channels(self):
        return [ds.readers[0].get_target_num_channels() for ds in self.streams_datasets.values()]

    def get_targets_coords_size(self):
        # TODO: avoid hard coding magic values
        # +6 at the end for stream_id and time encoding
        return [
            (ds.readers[0].get_geoinfo_size() + (5 * (3 * 5)) + 3 * 8) + 6
            for ds in self.streams_datasets.values()
        ]

    def denormalize_source_channels(self, stream_name, data) -> torch.Tensor:
        # [0]: with multiple ds per stream we use the first one
        return self.streams_datasets[stream_name].readers[0].denormalize_source_channels(data)

    def denormalize_target_channels(self, stream_name, data) -> torch.Tensor:
        # [0]: with multiple ds per stream we use the first one
        return self.streams_datasets[stream_name].readers[0].denormalize_target_channels(data)

    def _build_stream_data_input(
        self,
        mode: str,
        stream_data: StreamData,
        base_idx: TIndex,
        stream_info: dict,
        num_steps_input: int,
        input_data: list,
        input_tokens: list,
        mask: torch.Tensor | None = None,
    ) -> tuple[StreamData, dict | None]:
        """
        Build model network input

        Args:
            stream_data :
            base_idx: Time index for this sample
            num_forecast_steps: Number of forecast steps
            view_meta: ViewMetadata describing spatial mask
            stream_info: Stream configuration dict
            stream_ds: List of dataset readers for this stream

        Returns:
            StreamData with source and targets masked according to view_meta
        """

        if "network_input" in mode:
            # iterate overall input steps
            for step, idx in enumerate(range(base_idx, base_idx - num_steps_input, -1)):
                # TODO: check that we are not out of bounds when we go back in time

                time_win_source = self.time_window_handler.window(idx)

                # collect all targets for current stream
                # do we want this to be ascending or descending in time?
                rdata = input_data[-(step + 1)]
                token_data = input_tokens[-(step + 1)]

                if token_data[0] is None and token_data[1] is None:
                    continue

                # preprocess data for model input
                (source_cells, source_cells_lens) = self.tokenizer.get_source(
                    stream_info,
                    rdata,
                    token_data,
                    (time_win_source.start, time_win_source.end),
                    mask,
                )

                stream_data.add_source(
                    self._stage, step, rdata, source_cells_lens, source_cells, rdata.is_spoof
                )

        return stream_data

    def _build_stream_data_output(
        self,
        mode: str,
        stream_data: StreamData,
        idx: TIndex,
        stream_info: dict,
        num_forecast_steps: int,
        output_data: list,
        output_tokens: list,
        target_mask,
    ) -> StreamData:
        """
        Generate stream data for output

        """

        # collect for all forecast steps
        num_output_steps = self._get_output_length(num_forecast_steps)
        for step, timestep_idx in enumerate(range(self.output_offset, num_output_steps)):
            step_forecast_dt = idx + (self.time_step * timestep_idx) // self.step_timedelta
            time_win_target = self.time_window_handler.window(step_forecast_dt)

            # collect all targets for current stream
            rdata = output_data[step]
            token_data = output_tokens[step]

            if token_data[0] is None and token_data[1] is None:
                continue

            if "target_coords" in mode:
                (tc, tc_l) = self.tokenizer.get_target_coords(
                    stream_info,
                    rdata,
                    token_data,
                    (time_win_target.start, time_win_target.end),
                    target_mask,
                )
                stream_data.add_target_coords(self._stage, timestep_idx, tc, tc_l, rdata.is_spoof)

            if "target_values" in mode:
                (tt_cells, tt_t, tt_c, idxs_inv) = self.tokenizer.get_target_values(
                    stream_info,
                    rdata,
                    token_data,
                    (time_win_target.start, time_win_target.end),
                    target_mask,
                )

                stream_data.add_target_values(
                    self._stage, timestep_idx, tt_cells, tt_c, tt_t, idxs_inv, rdata.is_spoof
                )

        return stream_data

    def _build_stream_data(
        self,
        modes: str,
        base_idx: TIndex,
        num_forecast_steps: int,
        stream_info: dict,
        num_steps_input: int,
        input_data: list,
        output_data: list,
        input_tokens: list,
        output_tokens: list,
        output_mask,
        input_mask,
    ) -> StreamData:
        """
        Return one batch of data
        Build a StreamData object for a single view (teacher or student).

        Args:
            modes :
            stream_data :
            base_idx: Time index for this sample
            num_forecast_steps: Number of forecast steps
            stream_info: Stream configuration dict
            stream_ds: List of dataset readers for this stream

            output_mask : mask for output/prediction/target
            input_mask : mask for network input (can be source or target)


        Returns:
            StreamData with source and targets masked according to view_meta
        """

        num_output_steps = self._get_output_length(num_forecast_steps)
        stream_data = StreamData(
            base_idx,
            num_steps_input,
            num_output_steps,
            self.num_healpix_cells,
        )

        stream_data = self._build_stream_data_input(
            modes,
            stream_data,
            base_idx,
            stream_info,
            num_steps_input,
            input_data,
            input_tokens,
            input_mask,
        )

        stream_data = self._build_stream_data_output(
            modes,
            stream_data,
            base_idx,
            stream_info,
            num_forecast_steps,
            output_data,
            output_tokens,
            output_mask,
        )

        return stream_data

    def _get_data_windows(self, base_idx, num_forecast_steps, num_steps_input_max, stream_ds):
        """
        Collect all data needed for current stream to potentially amortize costs by
        generating multiple samples

        """

        # source data: iterate overall input steps
        input_data = []
        for idx in range(base_idx - num_steps_input_max + 1, base_idx + 1):
            # TODO: check that we are not out of bounds when we go back in time

            rdata = collect_datasources(stream_ds, idx, "source", self.rng)

            if rdata.is_empty():
                # work around for https://github.com/pytorch/pytorch/issues/158719
                # create non-empty mean data instead of empty tensor
                time_win = self.time_window_handler.window(idx)
                rdata = spoof(
                    self.healpix_level,
                    time_win.start,
                    stream_ds[0].get_geoinfo_size(),
                    len(stream_ds[0].mean[stream_ds[0].source_idx]),
                )
                rdata.is_spoof = True

            input_data += [rdata]

        # target data: collect for all forecast steps
        output_data = []
        num_output_steps = self._get_output_length(num_forecast_steps)
        for timestep_idx in range(self.output_offset, num_output_steps):
            step_forecast_dt = base_idx + (self.time_step * timestep_idx) // self.step_timedelta

            rdata = collect_datasources(stream_ds, step_forecast_dt, "target", self.rng)

            if rdata.is_empty():
                # work around for https://github.com/pytorch/pytorch/issues/158719
                # create non-empty mean data instead of empty tensor
                time_win = self.time_window_handler.window(step_forecast_dt)
                rdata = spoof(
                    self.healpix_level,
                    time_win.start,
                    stream_ds[0].get_geoinfo_size(),
                    len(stream_ds[0].mean[stream_ds[0].target_idx]),
                )
                rdata.is_spoof = True

            output_data += [rdata]

        return (input_data, output_data)

    def _get_source_target_masks(self, training_mode):
        """
        Generate source and target masks for all streams.
        """
        masks = {}
        for stream_name, stream_data in self.streams_datasets.items():
            stream_info = stream_data.info
            # Build source and target sample masks
            masks[stream_name] = self.tokenizer.build_samples_for_stream(
                training_mode,
                self.num_healpix_cells,
                stream_info,
            )
            # identical for all streams
            num_target_samples = len(masks[stream_name][0])
            num_source_samples = len(masks[stream_name][1])

        return masks, num_source_samples, num_target_samples

    def _get_output_length(self, num_forecast_steps):
        # max(1, ...) : self.output_offset and num_forecast_steps are zero for pure masking
        return max(1, self.output_offset + num_forecast_steps)

    def _preprocess_model_batch(
        self, batch: ModelBatch, source_input_steps: int, target_input_steps: int
    ):
        """
        Perform necessary pre-processing of model batch
        """
        stream_names = list(self.streams_datasets.keys())
        batch.source_samples.tokens_lens = get_tokens_lens(
            stream_names, batch.source_samples, source_input_steps
        )
        batch.target_samples.tokens_lens = get_tokens_lens(
            stream_names, batch.target_samples, target_input_steps
        )

        return batch

    def _get_batch(self, idx: int, num_forecast_steps: int):
        """
        Assemble a batch using the sample corresponding to idx
        """

        mode = self.mode_cfg.get("training_mode")
        source_cfgs = self.mode_cfg.get("model_input")
        target_cfgs = self.mode_cfg.get("target_input", {})

        # get/coordinate masks
        masks_streams, num_source_samples, num_target_samples = self._get_source_target_masks(mode)

        source_select, target_select = [], []
        if "masking" in mode:
            source_select += ["network_input", "target_coords"]
            target_select += ["target_values"]
        if "student_teacher" in mode or "latent_loss" in mode:
            source_select += ["network_input"]
            target_select += ["network_input"]
        # remove duplicates
        source_select, target_select = list(set(source_select)), list(set(target_select))
        if len(source_select) == 0 or len(target_select) == 0:
            raise NotImplementedError(f"Unsupported training mode {mode}.")

        num_output_steps = self._get_output_length(num_forecast_steps)
        batch = ModelBatch(
            list(self.streams_datasets.keys()),
            num_source_samples,
            num_target_samples,
            self.output_offset,
            num_output_steps,
        )

        # for all streams
        for stream_name, stream_data in self.streams_datasets.items():
            stream_info, stream_ds = stream_data.info, stream_data.readers
            (target_masks, source_masks, source_to_target) = masks_streams[stream_name]

            # max number of input steps
            input_steps = np.array([sc.get("num_steps_input", 1) for _, sc in source_cfgs.items()])
            assert input_steps.min() == input_steps.max(), (
                "Number of input steps has to be constant across configs."
            )
            assert input_steps.min(), "Number of input steps has to be greater than zero."

            # input_data and output_data is conceptually consecutive but differs
            # in source and target channels; overlap in one window when self.output_offset=0
            i_max = input_steps.max().item()
            (input_data, output_data) = self._get_data_windows(
                idx, num_forecast_steps, i_max, stream_ds
            )

            # tokenize windows
            # *_tokens = [ (cells_idx, cells_idx_lens), ... ] with length = #time_steps
            input_tokens = self.tokenizer.get_tokens_windows(stream_info, input_data, True)
            output_tokens = self.tokenizer.get_tokens_windows(stream_info, output_data, False)

            for sidx, source_mask in enumerate(source_masks.masks):
                # Map each source to its target
                tidx = source_to_target[sidx].item()
                sdata = self._build_stream_data(
                    source_select,
                    idx,
                    num_forecast_steps,
                    stream_info,
                    source_masks.metadata[sidx].params.get("num_steps_input", 1),
                    input_data,
                    output_data,
                    input_tokens,
                    output_tokens,
                    output_mask=target_masks.masks[tidx],
                    input_mask=source_mask,
                )

                batch.add_source_stream(sidx, tidx, stream_name, sdata, source_masks.metadata[sidx])

            # for t_idx, mask in enumerate(source_masks):
            for tidx, target_mask in enumerate(target_masks.masks):
                # depending on the mode, the the streamdata obj to have the target mask applied to
                # the inputs. Hence the target mask is also the source mask here.
                sdata = self._build_stream_data(
                    target_select,
                    idx,
                    num_forecast_steps,
                    stream_info,
                    target_masks.metadata[tidx].params.get("num_steps_input", 1),
                    input_data,
                    output_data,
                    input_tokens,
                    output_tokens,
                    output_mask=target_mask,
                    input_mask=target_mask,
                )
                target_metadata = target_masks.metadata[tidx]
                # also want to add the mask to the metadata
                target_metadata.mask = target_mask
                # Map target to all source students
                student_indices = [
                    s_idx for s_idx, tid in enumerate(source_to_target) if tid == tidx
                ]
                batch.add_target_stream(tidx, student_indices, stream_name, sdata, target_metadata)

        source_in_steps = input_steps.max().item()
        target_in_steps = np.array([tc.get("num_steps_input", 1) for _, tc in target_cfgs.items()])
        target_in_steps = 1 if len(target_in_steps) == 0 else target_in_steps.max().item()
        batch = self._preprocess_model_batch(batch, source_in_steps, target_in_steps)

        return batch

    def __iter__(self) -> ModelBatch:
        """
        Return one batch of data

        Return :
            batch of data
        """
        iter_start, iter_end = self.worker_workset()
        logger.info(f"iter_start={iter_start}, iter_end={iter_end}, len={self.len}")

        # create new shuffeling
        perms, perms_num_forecast_steps = self.reset()

        # bidx is used to count the #batches that have been emitted
        # idx_raw is used to index into the dataset; the decoupling is needed
        # since there are empty batches
        idx_raw = iter_start
        for i, _bidx in enumerate(range(iter_start, iter_end, self.batch_size)):
            # num_forecast_steps needs to be constant per batch
            # (amortized through data parallel training)
            num_forecast_steps = perms_num_forecast_steps[i]

            # use while loop due to the scattered nature of the data in time and to
            # ensure batches are not empty
            while True:
                idx: TIndex = perms[idx_raw % perms.shape[0]]
                idx_raw += 1

                batch = self._get_batch(idx, num_forecast_steps)

                # ensure the batch is valid, i.e. not completely empty and no NaN values
                # student teacher has no classical targets
                mode = self.mode_cfg.get("training_mode")
                not_valid = batch.sources_empty() or batch.is_nan()
                not_valid = not_valid or (batch.targets_empty() if "masking" in mode else False)

                # skip completely empty batch item or when all targets are empty -> no grad
                if not_valid:
                    logger.warning(f"Skipping empty batch with idx={idx}.")
                else:
                    break

            yield batch

    def __len__(self):
        return self.len

    def worker_workset(self):
        local_start, local_end = self.rank * self.len, (self.rank + 1) * self.len

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            assert self.world_size == 1, self.world_size
            iter_start = 0
            iter_end = len(self)

        else:
            # ensure the rng seed is fully unique across workers and mini_epochs
            # the worker processes are generated as bit-wise copy of the "template" (the actual
            # instance of the present class that is created) whenever __iter__ is started. This
            # happens for each mini_epoch, for train and validation, and independently for each DDP
            # worker. After the bit-wise copy, the rng seed needs to be made unique for
            # DDP workers, loader process, mini_epoch.
            dist = torch.distributed
            self.data_loader_rng_seed *= (
                (((dist.get_rank() + 1) * 73) if dist.is_initialized() else 1)
                * ((worker_info.id + 1) * 37)
                * (self.mini_epoch + 13)
                * 7
            )
            # split workload
            per_worker = (local_end - local_start) // worker_info.num_workers
            iter_start = local_start + worker_info.id * per_worker
            iter_end = iter_start + per_worker
            if worker_info.id + 1 == worker_info.num_workers:
                iter_end = local_end
            logger.info(
                f"{self.rank}::{worker_info.id}"
                + f" : dataset [{local_start},{local_end}) : [{iter_start},{iter_end})"
            )

        return iter_start, iter_end
