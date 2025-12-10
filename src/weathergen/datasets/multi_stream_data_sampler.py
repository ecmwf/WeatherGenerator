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

from weathergen.common.io import IOReaderData
from weathergen.datasets.batch import ModelBatch, Sample
from weathergen.datasets.data_reader_anemoi import DataReaderAnemoi
from weathergen.datasets.data_reader_base import (
    DataReaderBase,
    TimeWindowHandler,
    TIndex,
    str_to_datetime64,
)
from weathergen.datasets.data_reader_fesom import DataReaderFesom
from weathergen.datasets.data_reader_obs import DataReaderObs
from weathergen.datasets.masking import Masker
from weathergen.datasets.stream_data import StreamData, spoof
from weathergen.datasets.tokenizer_masking import TokenizerMasking
from weathergen.datasets.utils import (
    compute_idxs_predict,
    compute_offsets_scatter_embed,
    compute_source_cell_lens,
)
from weathergen.readers_extra.registry import get_extra_reader
from weathergen.utils.distributed import is_root
from weathergen.utils.train_logger import Stage

type AnyDataReader = DataReaderBase | DataReaderAnemoi | DataReaderObs
type StreamName = str

logger = logging.getLogger(__name__)


def collect_datasources(stream_datasets: list, idx: int, type: str) -> IOReaderData:
    """
    Utility function to collect all sources / targets from streams list
    """

    rdatas = []

    for ds in stream_datasets:
        if type == "source":
            get_reader_data = ds.get_source
            normalize_channels = ds.normalize_source_channels
        elif type == "target":
            get_reader_data = ds.get_target
            normalize_channels = ds.normalize_target_channels
        else:
            assert False, "invalid value for argument `type`"

        # get source (of potentially multi-step length)
        rdata = get_reader_data(idx).remove_nan_coords()
        rdata.data = normalize_channels(rdata.data)
        rdata.geoinfos = ds.normalize_geoinfos(rdata.geoinfos)
        rdatas += [rdata]

    return IOReaderData.combine(rdatas)


class MultiStreamDataSampler(torch.utils.data.IterableDataset):
    def __init__(
        self,
        cf,
        start_date_,
        end_date_,
        batch_size,
        samples_per_mini_epoch,
        stage: Stage,
        shuffle=True,
    ):
        super(MultiStreamDataSampler, self).__init__()

        start_date = str_to_datetime64(start_date_)
        end_date = str_to_datetime64(end_date_)

        assert end_date > start_date, (end_date, start_date)

        self.mask_value = 0.0
        self._stage = stage

        self.len_hrs: int = cf.len_hrs
        self.step_hrs: int = cf.step_hrs
        self.time_window_handler = TimeWindowHandler(start_date, end_date, cf.len_hrs, cf.step_hrs)
        if is_root():
            logger.info(
                f"Time window handler: start={start_date}, end={end_date},"
                f"len_hrs={cf.len_hrs}, step_hrs={cf.step_hrs}"
            )

        self.forecast_offset = cf.forecast_offset
        self.forecast_delta_hrs = (
            cf.forecast_delta_hrs if cf.forecast_delta_hrs > 0 else self.len_hrs
        )
        assert self.forecast_delta_hrs == self.len_hrs, "Only supported option at the moment"
        self.forecast_steps = np.array(
            [cf.forecast_steps] if isinstance(cf.forecast_steps, int) else cf.forecast_steps
        )
        if cf.forecast_policy is not None:
            if self.forecast_steps.max() == 0 and is_root():
                logger.warning("forecast policy is not None but number of forecast steps is 0.")
        self.forecast_policy = cf.forecast_policy

        self.len = 100000000

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

                fsm = self.forecast_steps[0]
                if len(ds) > 0:
                    self.len = min(self.len, len(ds) - (self.len_hrs * (fsm + 1)) // self.step_hrs)

                stream_info[str(self._stage) + "_source_channels"] = ds.source_channels
                stream_info[str(self._stage) + "_target_channels"] = ds.target_channels
                stream_info["target_channel_weights"] = (
                    ds.target_channel_weights
                    if ds.target_channel_weights is not None
                    else [1.0 for _ in ds.target_channels]
                )

                self.streams_datasets[stream_info["name"]] += [ds]

        index_range = self.time_window_handler.get_index_range()
        self.len = int(index_range.end - index_range.start)
        self.len = min(self.len, samples_per_mini_epoch if samples_per_mini_epoch else self.len)
        # adjust len to split loading across all workers and ensure it is multiple of batch_size
        len_chunk = ((self.len // cf.world_size) // batch_size) * batch_size
        self.len = min(self.len, len_chunk)
        logger.info(f"index_range={index_range}, len={self.len}, len_chunk={len_chunk}")

        self.rank = cf.rank
        self.world_size = cf.world_size

        self.streams = cf.streams
        self.shuffle = shuffle
        # TODO: remove options that are no longer supported
        self.input_window_steps = cf.input_window_steps
        # TODO, TODO, TODO: this needs to be stream specific and should not be an attribute
        # current implementation needs to be cleaned up when batch_size > 1 is enabled
        self.num_steps_input = -1

        self.batch_size = batch_size

        # ensure data_loader_rng_seed is not smaller than loader_num_workers to avoid
        # issues in per loader rng seed computation
        self.data_loader_rng_seed = (
            cf.data_loader_rng_seed
            if cf.data_loader_rng_seed > cf.loader_num_workers
            else cf.data_loader_rng_seed * 13
        )

        self.healpix_level: int = cf.healpix_level
        self.num_healpix_cells: int = 12 * 4**self.healpix_level

        self.training_cfg = cf.get("training_config", None)

        masker = Masker(cf)
        self.tokenizer = TokenizerMasking(cf.healpix_level, masker)

        self.mini_epoch = 0

        self.rng = None
        self.perms = None
        self.perms_forecast_dt = None

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
        # +6 at the end for stram_id and time encoding
        return [
            (ds[0].get_geoinfo_size() + (5 * (3 * 5)) + 3 * 8) + 6
            for _, ds in self.streams_datasets.items()
        ]

    def reset(self):
        # initialize the random number generator: self.data_loader_rng_seed is set to a DDP-unique
        # value in worker_workset()
        self.rng = np.random.default_rng(self.data_loader_rng_seed)

        fsm = (
            self.forecast_steps[min(self.mini_epoch, len(self.forecast_steps) - 1)]
            if self.forecast_policy != "random"
            else self.forecast_steps.max()
        )
        if fsm > 0:
            logger.info(f"forecast_steps at mini_epoch={self.mini_epoch} : {fsm}")

        # data
        index_range = self.time_window_handler.get_index_range()
        idx_end = index_range.end
        # native length of datasets, independent of mini_epoch length that has potentially been
        # specified
        forecast_len = (self.len_hrs * (fsm + 1)) // self.step_hrs
        idx_end -= forecast_len + self.forecast_offset
        assert idx_end > 0, "dataset size too small for forecast range"
        self.perms = np.arange(index_range.start, idx_end)
        if self.shuffle:
            self.perms = self.rng.permutation(self.perms)

        # forecast time steps
        len_dt_samples = len(self) // self.batch_size
        if self.forecast_policy is None:
            self.perms_forecast_dt = np.zeros(len_dt_samples, dtype=np.int64)
        elif self.forecast_policy == "fixed" or self.forecast_policy == "sequential":
            self.perms_forecast_dt = fsm * np.ones(len_dt_samples, dtype=np.int64)
        elif self.forecast_policy == "random" or self.forecast_policy == "sequential_random":
            # randint high=one-past
            self.perms_forecast_dt = self.rng.integers(
                low=self.forecast_steps.min(), high=fsm + 1, size=len_dt_samples, dtype=np.int64
            )
        else:
            assert False

        self.tokenizer.reset_rng(self.rng)

    def denormalize_source_channels(self, stream_name, data) -> torch.Tensor:
        # TODO: with multiple ds per stream we need to distinguish these here
        return self.streams_datasets[stream_name][0].denormalize_source_channels(data)

    def denormalize_target_channels(self, stream_name, data) -> torch.Tensor:
        # TODO: with multiple ds per stream we need to distinguish these here
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
            forecast_dt: Number of forecast steps
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
                (source_cells, source_cells_lens, mask_state) = self.tokenizer.get_source(
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
        forecast_dt: int,
        output_data: list,
        output_tokens: list,
        target_mask,
    ) -> StreamData:
        """
        Generate stream data for output

        """

        # collect for all forecast steps
        dt = self.forecast_offset + forecast_dt
        for step, fstep in enumerate(range(self.forecast_offset, dt + 1)):
            step_forecast_dt = idx + (self.forecast_delta_hrs * fstep) // self.step_hrs
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
                stream_data.add_target_coords(fstep, tc, tc_l)

            if "target_values" in mode:
                (tt_cells, tt_t, tt_c, idxs_inv) = self.tokenizer.get_target_values(
                    stream_info,
                    rdata,
                    token_data,
                    (time_win_target.start, time_win_target.end),
                    target_mask,
                )
                stream_data.add_target_values(fstep, tt_cells, tt_c, tt_t, idxs_inv)

        return stream_data

    def _build_stream_data(
        self,
        modes: str,
        base_idx: TIndex,
        forecast_dt: int,
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
            forecast_dt: Number of forecast steps
            stream_info: Stream configuration dict
            stream_ds: List of dataset readers for this stream

            output_mask : mask for output/prediction/target
            input_mask : mask for network input (can be source or target)


        Returns:
            StreamData with source and targets masked according to view_meta
        """

        dt = self.forecast_offset + forecast_dt
        stream_data = StreamData(base_idx, num_steps_input, dt, self.num_healpix_cells)

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
            forecast_dt,
            output_data,
            output_tokens,
            output_mask,
        )

        return stream_data

    def _get_data_windows(self, base_idx, forecast_dt, num_steps_input_max, stream_ds):
        """
        Collect all data needed for current stream to potentially amortize costs by
        generating multiple samples

        """

        # source data: iterate overall input steps
        input_data = []
        for idx in range(base_idx - num_steps_input_max, base_idx + 1):
            # TODO: check that we are not out of bounds when we go back in time

            rdata = collect_datasources(stream_ds, idx, "source")

            if rdata.is_empty():
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
        for fstep in range(self.forecast_offset, self.forecast_offset + forecast_dt + 1):
            step_forecast_dt = base_idx + (self.forecast_delta_hrs * fstep) // self.step_hrs

            rdata = collect_datasources(stream_ds, step_forecast_dt, "target")

            if rdata.is_empty():
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

            output_data += [rdata]

        return (input_data, output_data)

    def _get_batch(self, idx: int, forecast_dt: int):
        """

        modes :
          ('student', 'teacher')
          ('physical_input', 'physical_target')
        idx :
        forecast_dt :
        TODO: these modes are not being used now.
        """

        mode = self.training_cfg.get("training_mode")
        source_cfgs = self.training_cfg.get("model_input")
        target_cfgs = self.training_cfg.get("target_input", source_cfgs)

        # get/coordinate masks
        # TODO: should also return number of views
        masks_streams, num_source_samples, num_target_samples = self._get_source_target_masks(mode)

        if mode == "masking":
            source_select = ["network_input", "target_coords"]
            target_select = ["target_values"]
        elif mode == "student_teacher" or mode == "latent_loss":
            source_select = ["network_input"]
            target_select = ["network_input"]
        else:
            raise NotImplementedError(f"Unsupported training mode {mode}.")

        batch = ModelBatch(self.streams, num_source_samples, num_target_samples)

        # for all streams
        for stream_info, (stream_name, stream_ds) in zip(
            self.streams, self.streams_datasets.items(), strict=True
        ):
            # TODO: data class for this or something similar
            (
                target_masks,
                source_masks,
                student_to_teacher,
                target_metadata_list,
                source_metadata_list,
            ) = masks_streams[stream_name]

            # input_data and output_data is conceptually consecutive but differs
            # in source and target channels; overlap in one window when self.forecast_offset=0
            # max number of input steps
            i_max = np.array([sc.get("num_steps_input", 1) for sc in source_cfgs]).max().item()
            self.num_steps_input = i_max
            (input_data, output_data) = self._get_data_windows(idx, forecast_dt, i_max, stream_ds)

            # tokenize windows
            # *_tokens = [ (cells_idx, cells_idx_lens), ... ] with length = #time_steps
            input_tokens = self.tokenizer.get_tokens_windows(stream_info, input_data, True)
            output_tokens = self.tokenizer.get_tokens_windows(stream_info, output_data, False)

            # collect source data for current stream
            # loop over student views
            for sidx, source_mask in enumerate(source_masks):
                sdata = self._build_stream_data(
                    source_select,
                    idx,
                    forecast_dt,
                    stream_info,
                    source_cfgs[sidx].get("num_steps_input", 1),
                    input_data,
                    output_data,
                    input_tokens,
                    output_tokens,
                    target_masks[student_to_teacher[sidx]],
                    source_mask,
                )

                # also want to add the mask to the metadata
                source_metadata = source_metadata_list[sidx]
                source_metadata.mask = source_mask

                # map each source to its target
                t_idx = student_to_teacher[sidx]
                batch.add_source_stream(sidx, t_idx, stream_name, sdata, source_metadata)

            for sidx, target_mask in enumerate(target_masks):
                sdata = self._build_stream_data(
                    target_select,
                    idx,
                    forecast_dt,
                    stream_info,
                    target_cfgs[sidx].get("num_steps_input", 1),
                    input_data,
                    output_data,
                    input_tokens,
                    output_tokens,
                    target_mask,
                    target_mask,
                )

                # get target config info
                target_metadata = target_metadata_list[sidx]
                target_metadata.mask = target_mask

                # find indices of all sources for current target
                student_indices = [
                    s_idx for s_idx, tid in enumerate(student_to_teacher) if tid == sidx
                ]
                batch.add_target_stream(sidx, student_indices, stream_name, sdata, target_metadata)

        return batch

    def _get_source_target_masks(self, training_mode):
        """
        Generate source and target masks for all streams
        """

        masks = {}
        for stream_info in self.streams:
            # Build source and target sample masks
            target_data, source_data, mapping = self.tokenizer.masker.build_samples_for_stream(
                training_mode, self.num_healpix_cells, self.training_cfg
            )

            # TODO: avoid the unpacking here
            masks[stream_info["name"]] = (
                target_data[0],
                source_data[0],
                mapping,
                target_data[1],
                source_data[1],
            )

        # Determine number of samples directly from config (teacher and student views)
        source_cfgs = self.training_cfg.get("model_input")
        target_cfgs = self.training_cfg.get("target_input", source_cfgs)
        target_cfgs = target_cfgs if target_cfgs is not None else source_cfgs
        num_source_samples = np.array([sc.get("num_samples", 1) for sc in source_cfgs]).sum().item()
        num_target_samples = np.array([tc.get("num_samples", 1) for tc in target_cfgs]).sum().item()

        return masks, num_source_samples, num_target_samples

    def _preprocess_model_data(self, batch, forecast_dt):
        """ """

        # TODO, TODO, TODO: cleanup
        num_steps_input = self.num_steps_input

        # aggregated lens of tokens per cell across input batch samples
        source_cell_lens = compute_source_cell_lens(batch, num_steps_input)

        # compute offsets for scatter computation after embedding
        batch = compute_offsets_scatter_embed(batch, num_steps_input)

        # compute offsets and auxiliary data needed for prediction computation
        # (info is not per stream so separate data structure)

        # TODO: only use when targets are predicted with decoders
        target_coords_idx = compute_idxs_predict(
            self.forecast_offset + forecast_dt, batch, self.streams
        )

        return batch, source_cell_lens, target_coords_idx

    def _preprocess_model_batch_sample(self, sample: Sample, forecast_dt: int):
        """ """
        streams = [sd for sd in sample.streams_data.values() if sd is not None]
        if not streams:
            sample.set_preprocessed([], {})
            return
        _, scl, tci = self._preprocess_model_data([streams], forecast_dt)
        sample.set_preprocessed(scl, tci)

    def _preprocess_model_batch(self, model_batch: ModelBatch, forecast_dt: int):
        """
        Perform necessary pre-processing of model batch
        """
        for sample in model_batch.source_samples:
            self._preprocess_model_batch_sample(sample, forecast_dt)
        for sample in model_batch.target_samples:
            self._preprocess_model_batch_sample(sample, forecast_dt)

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
            # forecast_dt needs to be constant per batch (amortized through data parallel training)
            forecast_dt = self.perms_forecast_dt[i]

            # use while loop due to the scattered nature of the data in time and to
            # ensure batches are not empty
            while True:
                idx: TIndex = self.perms[idx_raw % self.perms.shape[0]]
                idx_raw += 1

                batch = self._get_batch(idx, forecast_dt)

                # skip completely empty batch item or when all targets are empty -> no grad
                if not batch.is_empty():
                    break
                else:
                    logger.warning("Skipping empty batch.")

            self._preprocess_model_batch(batch, forecast_dt)

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
