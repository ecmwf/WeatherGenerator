# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import pathlib

import numpy as np
import torch

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
from weathergen.train.utils import get_batch_size_from_config
from weathergen.utils.distributed import is_root
from weathergen.utils.train_logger import TRAIN, Stage

type AnyDataReader = DataReaderBase | DataReaderAnemoi | DataReaderObs
type StreamName = str

logger = logging.getLogger(__name__)


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
        rdata = get_reader_data(idx).shuffle(rng, shuffle, num_subset).remove_nan_coords()
        rdata.data = normalize_channels(rdata.data)
        rdata.geoinfos = ds.normalize_geoinfos(rdata.geoinfos)
        rdatas += [rdata]

    return IOReaderData.combine(rdatas)


class MultiStreamDataSampler(torch.utils.data.IterableDataset):
    def __init__(
        self,
        cf: Config,
        mode_cfg: dict,
        stage: Stage,
    ):
        super(MultiStreamDataSampler, self).__init__()

        self.mask_value = 0.0
        self._stage = stage

        self.streams = cf.streams
        self.rank = cf.rank
        self.world_size = cf.world_size

        self.healpix_level: int = cf.healpix_level
        self.num_healpix_cells: int = 12 * 4**self.healpix_level

        self.mode_cfg = mode_cfg
        self.samples_per_mini_epoch = mode_cfg.samples_per_mini_epoch
        self.shuffle = mode_cfg.shuffle
        self.batch_size = get_batch_size_from_config(mode_cfg)
        self.len_timedelta: np.timedelta64 = mode_cfg.time_window_len
        self.step_timedelta: np.timedelta64 = mode_cfg.time_window_step
        self.time_window_handler = TimeWindowHandler(
            mode_cfg.start_date, mode_cfg.end_date, self.len_timedelta, self.step_timedelta
        )
        if is_root():
            logger.info(self.time_window_handler)

        index_range = self.time_window_handler.get_index_range()
        perms_len = int(index_range.end - index_range.start)

        # Handle forecast_delta_hrs which might be int (hours) or string (timedelta)
        self.forecast_cfg = mode_cfg.get("forecast", {})
        if len(self.forecast_cfg) > 0:
            self.output_offset = self.forecast_cfg.get("offset", 0)
            self.time_step = self.forecast_cfg.get("time_step", np.timedelta64(0, "ms"))
            self.forecast_policy = self.forecast_cfg.get("policy", None)

            # forecast step
            self.list_num_forecast_steps = np.array(
                [self.forecast_cfg.get("num_steps", 0)]
                if isinstance(self.forecast_cfg.num_steps, int)
                else self.forecast_cfg.num_steps,
                dtype=np.int32,
            )

        else:
            # no forecast policy specified so set neutral default for no forecasting
            self.list_num_forecast_steps = np.array([0], dtype=np.int32)
            self.output_offset = 0
            self.forecast_policy = None
            self.time_step = np.timedelta64(0, "ms")

        fsm = self.list_num_forecast_steps[0]
        forecast_len = (self.time_step * (fsm + 1)) // self.step_timedelta
        perms_len = perms_len - (forecast_len + self.output_offset)

        self.repeat_data = cf.data_loading.get("repeat_data_in_mini_epoch", False)

        self.streams_datasets: dict[StreamName, list[AnyDataReader]] = {}
        for _, stream_info in enumerate(cf.streams):
            # list of sources for current stream
            self.streams_datasets[stream_info["name"]] = []

            for fname in stream_info["filenames"]:
                kwargs = {
                    "tw_handler": self.time_window_handler,
                    "stream_info": stream_info,
                }
                dataset: type[AnyDataReader] | None = None
                match stream_info["type"]:
                    case "obs":
                        dataset = DataReaderObs
                        datapath = cf.data_path_obs
                        # kwargs["end"] = end_date_padded # TODO: implement the padding
                    case "anemoi":
                        dataset = DataReaderAnemoi
                        datapath = cf.data_path_anemoi
                    case "fesom":
                        dataset = DataReaderFesom
                        datapath = cf.data_path_fesom
                    case type_name:
                        reader_entry = get_extra_reader(type_name, cf)
                        if reader_entry is not None:
                            dataset = reader_entry.constructor
                            datapath = reader_entry.data_path
                        else:
                            msg = f"Unsupported stream type {stream_info['type']}"
                            f"for stream name '{stream_info['name']}'."
                            raise ValueError(msg)

                datapath = pathlib.Path(datapath)
                fname = pathlib.Path(fname)
                # dont check if file exists since zarr stores might be directories
                if fname.exists():
                    # check if fname is a valid path to allow for simple overwriting
                    filename = fname
                else:
                    filename = pathlib.Path(datapath) / fname

                    if not filename.exists():  # see above
                        msg = (
                            f"Did not find input data for {stream_info['type']} "
                            f"stream '{stream_info['name']}': {filename}."
                        )
                        raise FileNotFoundError(msg)

                ds_type = stream_info["type"]
                if is_root():
                    logger.info(
                        f"Opening dataset with type: {ds_type}"
                        + f" from stream config {stream_info['name']}.",
                    )
                ds = dataset(filename=filename, **kwargs)

                stream_info[str(self._stage) + "_source_channels"] = ds.source_channels
                stream_info[str(self._stage) + "_target_channels"] = ds.target_channels
                stream_info["target_channel_weights"] = (
                    ds.target_channel_weights
                    if ds.target_channel_weights is not None
                    else [1.0 for _ in ds.target_channels]
                )

                self.streams_datasets[stream_info["name"]] += [ds]

        # length of dataset; check the repeat data flag and adjust len accordingly
        self.len = int(index_range.end - index_range.start)
        if not self.repeat_data:
            if self.samples_per_mini_epoch:
                if self.samples_per_mini_epoch <= self.len:
                    self.len = self.samples_per_mini_epoch
                else:
                    msg = (
                        f"WARNING: Adjusted length of data sampler to {self.len} "
                        f"(<samples_per_mini_epoch={self.samples_per_mini_epoch}) "
                        "due to insufficient number of data samples. "
                        "Enable repeat_data_in_mini_epoch to instead duplicate samples "
                        "to fill samples_per_mini_epoch."
                    )
                    logger.warning(msg)
        else:
            self.len = self.samples_per_mini_epoch

        # adjust len to split loading across all workers and ensure it is multiple of batch_size
        len_chunk = ((self.len // cf.world_size) // self.batch_size) * self.batch_size
        self.len = min(self.len, len_chunk)

        n_duplicates = self.len - perms_len
        if n_duplicates > 0:
            # TODO fix this more permanently (#1085)
            msg = (
                "WARNING: Missmatch between length of permutation indexes and"
                "length of MultiStreamDataSampler,"
                f"{n_duplicates} duplicate samples will be sampled."
                "To avoid this increase the the length of the"
                f"global sampling window by {n_duplicates * self.step_timedelta} hours."
            )
            logger.warning(msg)
        logger.info(f"index_range={index_range}, len={self.len}, len_chunk={len_chunk}")

        # ensure data_loader_rng_seed is not smaller than loader_num_workers to avoid
        # issues in per loader rng seed computation
        self.data_loader_rng_seed = (
            cf.data_loading.rng_seed
            if cf.data_loading.rng_seed > cf.data_loading.num_workers
            else cf.data_loading.rng_seed * 97
        )

        self.tokenizer = TokenizerMasking(cf.healpix_level, Masker(cf.healpix_level, stage))

        self.mini_epoch = 0

        self.rng = None
        self.perms = None
        self.perms_num_forecast_steps = None

    def advance(self):
        """
        Advance mini_epoch (this is applied to the template for the worker processes)
        """
        self.mini_epoch += 1

    def get_sources_size(self):
        return [
            0
            if ds[0].get_source_num_channels() == 0
            else ds[0].get_source_num_channels()
            + ds[0].get_geoinfo_size()
            + ds[0].get_coords_size()
            + self.tokenizer.get_size_time_embedding()
            for _, ds in self.streams_datasets.items()
        ]

    def get_sources_num_channels(self):
        return [ds[0].get_source_num_channels() for _, ds in self.streams_datasets.items()]

    def get_targets_num_channels(self):
        return [ds[0].get_target_num_channels() for _, ds in self.streams_datasets.items()]

    def get_targets_coords_size(self):
        # TODO: avoid hard coding magic values
        # +6 at the end for stream_id and time encoding
        return [
            (ds[0].get_geoinfo_size() + (5 * (3 * 5)) + 3 * 8) + 6
            for _, ds in self.streams_datasets.items()
        ]

    def reset(self):
        # initialize the random number generator: self.data_loader_rng_seed is set to a DDP-unique
        # value in worker_workset()
        self.rng = np.random.default_rng(self.data_loader_rng_seed)

        fsm = (
            self.list_num_forecast_steps[
                min(self.mini_epoch, len(self.list_num_forecast_steps) - 1)
            ]
            if self.forecast_policy != "random"
            else self.list_num_forecast_steps.max()
        )
        if fsm > 0:
            logger.info(f"forecast_steps : {fsm}")

        # data
        forecast_offset = self.output_offset
        index_range = self.time_window_handler.get_index_range()
        idx_end = index_range.end
        # native length of datasets, independent of mini_epoch length that has potentially been
        # specified
        forecast_len = (self.time_step * (fsm + 1)) // self.step_timedelta
        adjusted_idx_end = idx_end - (forecast_len + forecast_offset)
        msg = (
            f"dataset size ({idx_end}) too small for forecast length plus offset "
            f"({forecast_len + forecast_offset}) – dataset size must be strictly bigger. "
            "to fix this, it usually suffices to increase the data range "
        )
        assert adjusted_idx_end > 0, msg
        self.perms = np.arange(index_range.start, adjusted_idx_end)

        # check repeat_data flag and fill up perms accordingly
        if self.repeat_data and len(self.perms) < self.samples_per_mini_epoch:
            self.perms = np.tile(self.perms, self.samples_per_mini_epoch // len(self.perms))
            random_filler = self.rng.choice(
                self.perms, size=self.samples_per_mini_epoch - len(self.perms), replace=False
            )
            self.perms = np.concatenate([self.perms, random_filler])

        if self.shuffle:
            self.perms = self.rng.permutation(self.perms)

        # forecast time steps
        len_dt_samples = len(self) // self.batch_size
        if self.forecast_policy is None:
            self.perms_num_forecast_steps = np.zeros(len_dt_samples, dtype=np.int64)
        elif self.forecast_policy == "fixed" or self.forecast_policy == "sequential":
            self.perms_num_forecast_steps = fsm * np.ones(len_dt_samples, dtype=np.int64)
        elif self.forecast_policy == "random" or self.forecast_policy == "sequential_random":
            # randint high=one-past
            self.perms_num_forecast_steps = self.rng.integers(
                low=self.list_num_forecast_steps.min(),
                high=fsm + 1,
                size=len_dt_samples,
                dtype=np.int64,
            )
        else:
            assert False

        self.tokenizer.reset_rng(self.rng)

    def denormalize_source_channels(self, stream_name, data) -> torch.Tensor:
        # [0]: with multiple ds per stream we use the first one
        return self.streams_datasets[stream_name][0].denormalize_source_channels(data)

    def denormalize_target_channels(self, stream_name, data) -> torch.Tensor:
        # [0]: with multiple ds per stream we use the first one
        return self.streams_datasets[stream_name][0].denormalize_target_channels(data)

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

                stream_data.source_is_spoof = rdata.is_spoof

                # preprocess data for model input
                (source_cells, source_cells_lens) = self.tokenizer.get_source(
                    stream_info,
                    rdata,
                    token_data,
                    (time_win_source.start, time_win_source.end),
                    mask,
                )

                # collect data for stream
                stream_data.add_source(step, rdata, source_cells_lens, source_cells)

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

            stream_data.target_is_spoof = rdata.is_spoof

            if "target_coords" in mode:
                (tc, tc_l) = self.tokenizer.get_target_coords(
                    stream_info,
                    rdata,
                    token_data,
                    (time_win_target.start, time_win_target.end),
                    target_mask,
                )
                stream_data.add_target_coords(timestep_idx, tc, tc_l)

            if "target_values" in mode:
                (tt_cells, tt_t, tt_c, idxs_inv) = self.tokenizer.get_target_values(
                    stream_info,
                    rdata,
                    token_data,
                    (time_win_target.start, time_win_target.end),
                    target_mask,
                )
                stream_data.add_target_values(timestep_idx, tt_cells, tt_c, tt_t, idxs_inv)

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
        for idx in range(base_idx - num_steps_input_max, base_idx + 1):
            # TODO: check that we are not out of bounds when we go back in time

            rdata = collect_datasources(stream_ds, idx, "source", self.rng)

            if rdata.is_empty() and self._stage == TRAIN:
                # work around for https://github.com/pytorch/pytorch/issues/158719
                # create non-empty mean data instead of empty tensor
                time_win = self.time_window_handler.window(idx)
                rdata = spoof(
                    self.healpix_level,
                    time_win.start,
                    stream_ds[0].get_geoinfo_size(),
                    stream_ds[0].mean[stream_ds[0].source_idx],
                )
                rdata.is_spoof = True

            input_data += [rdata]

        # target data: collect for all forecast steps
        output_data = []
        num_output_steps = self._get_output_length(num_forecast_steps)
        for timestep_idx in range(self.output_offset, num_output_steps):
            step_forecast_dt = base_idx + (self.time_step * timestep_idx) // self.step_timedelta

            rdata = collect_datasources(stream_ds, step_forecast_dt, "target", self.rng)

            if rdata.is_empty() and self._stage == TRAIN:
                # work around for https://github.com/pytorch/pytorch/issues/158719
                # create non-empty mean data instead of empty tensor
                time_win = self.time_window_handler.window(timestep_idx)
                rdata = spoof(
                    self.healpix_level,
                    time_win.start,
                    stream_ds[0].get_geoinfo_size(),
                    stream_ds[0].mean[stream_ds[0].source_idx],
                )
                rdata.is_spoof = True

            output_data += [rdata]

        return (input_data, output_data)

    def _get_source_target_masks(self, training_mode):
        """
        Generate source and target masks for all streams
        """

        masks = {}
        for stream_info in self.streams:
            # Build source and target sample masks
            masks[stream_info["name"]] = self.tokenizer.build_samples_for_stream(
                training_mode,
                self.num_healpix_cells,
                self.mode_cfg,
                stream_info,
            )
            # identical for all streams
            num_target_samples = len(masks[stream_info["name"]][0])
            num_source_samples = len(masks[stream_info["name"]][1])

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
        batch.source_samples.tokens_lens = get_tokens_lens(
            self.streams, batch.source_samples, source_input_steps
        )
        batch.target_samples.tokens_lens = get_tokens_lens(
            self.streams, batch.target_samples, target_input_steps
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
            self.streams,
            num_source_samples,
            num_target_samples,
            self.output_offset,
            num_output_steps,
        )

        # for all streams
        for stream_info, (stream_name, stream_ds) in zip(
            self.streams, self.streams_datasets.items(), strict=True
        ):
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
                    tidx,
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
                    tidx,
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
        self.reset()

        # bidx is used to count the #batches that have been emitted
        # idx_raw is used to index into the dataset; the decoupling is needed
        # since there are empty batches
        idx_raw = iter_start
        for i, _bidx in enumerate(range(iter_start, iter_end, self.batch_size)):
            # num_forecast_steps needs to be constant per batch
            # (amortized through data parallel training)
            num_forecast_steps = self.perms_num_forecast_steps[i]

            # use while loop due to the scattered nature of the data in time and to
            # ensure batches are not empty
            while True:
                idx: TIndex = self.perms[idx_raw % self.perms.shape[0]]
                idx_raw += 1

                batch = self._get_batch(idx, num_forecast_steps)

                # skip completely empty batch item or when all targets are empty -> no grad
                if not batch.is_empty():
                    break
                else:
                    logger.warning(f"Skipping empty batch with idx={idx}.")

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
