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
from weathergen.datasets.data_reader_anemoi import DataReaderAnemoi
from weathergen.datasets.data_reader_base import (
    DataReaderBase,
    TimeWindowHandler,
    TIndex,
)
from weathergen.datasets.data_reader_fesom import DataReaderFesom
from weathergen.datasets.data_reader_obs import DataReaderObs
from weathergen.datasets.icon_dataset import IconDataset
from weathergen.datasets.masking import Masker
from weathergen.datasets.stream_data import StreamData, spoof
from weathergen.datasets.tokenizer_forecast import TokenizerForecast
from weathergen.datasets.tokenizer_masking import TokenizerMasking
from weathergen.datasets.utils import (
    compute_idxs_predict,
    compute_offsets_scatter_embed,
    compute_source_cell_lens,
)
from weathergen.utils.distributed import get_rank, get_world_size, is_root
from weathergen.utils.train_logger import Stage

type AnyDataReader = DataReaderBase | DataReaderAnemoi | DataReaderObs

logger = logging.getLogger(__name__)


def readerdata_to_torch(rdata: IOReaderData) -> IOReaderData:
    """
    Convert data, coords, and geoinfos to torch tensor
    """
    rdata.coords = torch.tensor(rdata.coords)
    rdata.geoinfos = torch.tensor(rdata.geoinfos)
    rdata.data = torch.tensor(rdata.data)

    return rdata


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
    ###################################################
    def __init__(
        self,
        cf,
        start_date,
        end_date,
        batch_size,
        samples_per_epoch,
        stage: Stage,
        shuffle=True,
    ):
        super(MultiStreamDataSampler, self).__init__()

        self.mask_value = 0.0
        self._stage = stage

        self.time_window_handler = TimeWindowHandler(start_date, end_date, cf.len_hrs, cf.step_hrs)
        if is_root():
            logger.info(f"Time window handler: {self.time_window_handler}")

        self.forecast_offset = cf.forecast_offset
        self.forecast_delta_hrs = cf.forecast_delta_hrs if cf.forecast_delta_hrs > 0 else cf.len_hrs
        assert self.forecast_delta_hrs == cf.len_hrs, "Only supported option at the moment"
        self.forecast_steps = np.array(
            [cf.forecast_steps] if isinstance(cf.forecast_steps, int) else cf.forecast_steps
        )
        self.forecast_policy = cf.forecast_policy

        self.streams_datasets: list[list[AnyDataReader]] = [
            create_datasets(stream_info, self.time_window_handler, cf) for stream_info in cf.streams
        ]

        # MODIFIES config !!!
        for stream_info, stream_datasets in zip(cf.streams, self.streams_datasets, strict=True):
            # assume all datasets within one stream have the same channels
            sample_ds = stream_datasets[0]
            stream_info[f"{self._stage}_source_channels"] = sample_ds.source_channels
            stream_info[f"{self._stage}_target_channels"] = sample_ds.target_channels
            # TODO no need to modify the stream config here
            stream_info["target_channel_weights"] = (
                sample_ds.target_channel_weights
                if sample_ds.target_channel_weights is not None
                else [1.0 for _ in sample_ds.target_channels]
            )

        self._len = self._get_len(self.time_window_handler, samples_per_epoch, batch_size)

        self.shuffle = shuffle

        self.sampling_rate_target = cf.sampling_rate_target

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

        if cf.training_mode == "forecast":
            self.tokenizer = TokenizerForecast(cf.healpix_level)
        elif cf.training_mode == "masking":
            masker = Masker(cf)
            self.tokenizer = TokenizerMasking(cf.healpix_level, masker)
            assert self.forecast_offset == 0, "masked token modeling requires auto-encoder training"
        else:
            assert False, f"Unsupported training mode: {cf.training_mode}"

        self.epoch = 0

    ###################################################
    def advance(self):
        """
        Advance epoch (this is applied to the template for the worker processes)
        """
        self.epoch += 1

    ###################################################
    def get_sources_size(self):
        return [
            0
            if ds[0].get_source_num_channels() == 0
            else ds[0].get_source_num_channels()
            + ds[0].get_geoinfo_size()
            + ds[0].get_coords_size()
            + self.tokenizer.get_size_time_embedding()
            for ds in self.streams_datasets
        ]

    ###################################################
    def get_sources_num_channels(self):
        return [ds[0].get_source_num_channels() for ds in self.streams_datasets]

    ###################################################
    def get_targets_num_channels(self):
        return [ds[0].get_target_num_channels() for ds in self.streams_datasets]

    ###################################################
    def get_targets_coords_size(self):
        # TODO: avoid hard coding magic values
        # +6 at the end for stram_id and time encoding
        return [
            (ds[0].get_geoinfo_size() + (5 * (3 * 5)) + 3 * 8) + 6 for ds in self.streams_datasets
        ]

    ###################################################
    def reset(self):
        # initialize the random number generator: self.data_loader_rng_seed is set to a DDP-unique
        # value in worker_workset()
        rng = np.random.default_rng(self.data_loader_rng_seed)

        fsm = (
            self.forecast_steps[min(self.epoch, len(self.forecast_steps) - 1)]
            if self.forecast_policy != "random"
            else self.forecast_steps.max()
        )
        if fsm > 0:
            logger.info(f"forecast_steps at epoch={self.epoch} : {fsm}")

        # data
        index_range = self.time_window_handler.get_index_range()
        # native length of datasets, independent of epoch length that has potentially been specified
        len_hrs = self.time_window_handler.t_window_len
        step_hrs = self.time_window_handler.t_window_step
        forecast_len = (len_hrs * (fsm + 1)) // step_hrs

        # maximum index to generate a forecast that falls within the datasets
        index_max = index_range.stop - forecast_len + self.forecast_offset
        assert index_max > 0, "dataset size too small for forecast range"
        self.perms = np.arange(index_range.start, index_max)
        if self.shuffle:
            self.perms = rng.permutation(self.perms)

        # forecast time steps
        len_dt_samples = len(self) // self.batch_size
        if self.forecast_policy is None:
            self.perms_forecast_dt = np.zeros(len_dt_samples, dtype=np.int64)
        elif self.forecast_policy == "fixed" or self.forecast_policy == "sequential":
            self.perms_forecast_dt = fsm * np.ones(len_dt_samples, dtype=np.int64)
        elif self.forecast_policy == "random" or self.forecast_policy == "sequential_random":
            # randint high=one-past
            self.perms_forecast_dt = rng.integers(
                low=self.forecast_steps.min(), high=fsm + 1, size=len_dt_samples, dtype=np.int64
            )
        else:
            assert False

        self.tokenizer.reset_rng(rng)

    ###################################################
    def denormalize_source_channels(self, stream_id, data) -> torch.Tensor:
        # TODO: with multiple ds per stream we need to distinguish these here
        return self.streams_datasets[stream_id][0].denormalize_source_channels(data)

    ###################################################
    def denormalize_target_channels(self, stream_id, data) -> torch.Tensor:
        # TODO: with multiple ds per stream we need to distinguish these here
        return self.streams_datasets[stream_id][0].denormalize_target_channels(data)

    ###################################################
    def __iter__(self):
        """
        Return one batch of data

        Return : list[list[StreamData]]
            len : number of batch items
            len[*] : number of streams
        """
        iter_start, iter_end = self.worker_workset()
        logger.info(f"iter_start={iter_start}, iter_end={iter_end}, len={len(self)}")

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
            batch = []
            while len(batch) < self.batch_size:
                idx: TIndex = self.perms[idx_raw % self.perms.shape[0]]
                idx_raw += 1

                time_win_source = self.time_window_handler.window(idx)

                # Sample masking strategy once per batch item
                if hasattr(self.tokenizer, "masker"):
                    self.tokenizer.masker.set_batch_strategy()

                streams_data: list[StreamData] = []

                # for all streams
                for stream_ds in self.streams_datasets:
                    stream_info = stream_ds[0].stream_info
                    stream_data = StreamData(
                        idx, forecast_dt + self.forecast_offset, self.num_healpix_cells
                    )

                    # collect all targets for current stream
                    rdata: IOReaderData = collect_datasources(stream_ds, idx, "source")

                    if rdata.is_empty():
                        # work around for https://github.com/pytorch/pytorch/issues/158719
                        # create non-empty mean data instead of empty tensor
                        rdata = spoof(
                            self.healpix_level,
                            time_win_source.start,
                            stream_ds[0].get_geoinfo_size(),
                            stream_ds[0].mean[stream_ds[0].source_idx],
                        )
                        stream_data.source_is_spoof = True

                    # preprocess data for model input
                    (ss_cells, ss_lens, ss_centroids) = self.tokenizer.batchify_source(
                        stream_info,
                        readerdata_to_torch(rdata),
                        (time_win_source.start, time_win_source.end),
                        stream_ds[0].normalize_coords,
                    )

                    # TODO: rdata only be collected in validation mode
                    stream_data.add_source(rdata, ss_lens, ss_cells, ss_centroids)

                    # target

                    # collect for all forecast steps
                    for fstep in range(
                        self.forecast_offset, self.forecast_offset + forecast_dt + 1
                    ):
                        step_forecast_dt = (
                            idx + (self.forecast_delta_hrs * fstep)
                            // self.time_window_handler.t_window_step
                        )
                        time_win_target = self.time_window_handler.window(step_forecast_dt)

                        # collect all targets for current stream
                        rdata: IOReaderData = collect_datasources(
                            stream_ds, step_forecast_dt, "target"
                        )

                        if rdata.is_empty():
                            # work around for https://github.com/pytorch/pytorch/issues/158719
                            # create non-empty mean data instead of empty tensor
                            rdata = spoof(
                                self.healpix_level,
                                time_win_target.start,
                                stream_ds[0].get_geoinfo_size(),
                                stream_ds[0].mean[stream_ds[0].target_idx],
                            )
                            stream_data.target_is_spoof = True

                        # preprocess data for model input
                        (tt_cells, tc, tt_c, tt_t) = self.tokenizer.batchify_target(
                            stream_info,
                            self.sampling_rate_target,
                            readerdata_to_torch(rdata),
                            (time_win_target.start, time_win_target.end),
                        )

                        stream_data.add_target(fstep, tt_cells, tc, tt_c, tt_t)

                    # merge inputs for sources and targets for current stream
                    streams_data += [stream_data]

                # Reset masking strategy for next batch item
                if hasattr(self.tokenizer, "masker"):
                    self.tokenizer.masker.reset_batch_strategy()

                # skip completely empty batch item or when all targets are empty -> no grad
                if not (all(s.empty() or s.target_empty() for s in streams_data)):
                    batch += [streams_data]

            # aggregated lens of tokens per cell
            source_cell_lens = compute_source_cell_lens(batch)

            # compute offsets for scatter computation after embedding
            batch = compute_offsets_scatter_embed(batch)

            # compute offsets and auxiliary data needed for prediction computation
            # (info is not per stream so separate data structure)
            target_coords_idx = compute_idxs_predict(self.forecast_offset + forecast_dt, batch)

            assert len(batch) == self.batch_size
            yield (batch, source_cell_lens, target_coords_idx, forecast_dt)

    ###################################################
    def __len__(self):
        return self._len

    @staticmethod
    def _get_len(twh: TimeWindowHandler, samples_per_epoch, batch_size) -> int:
        index_range = twh.get_index_range()
        _len = len(index_range)
        samples_per_epoch = samples_per_epoch if samples_per_epoch else _len
        _len = min(_len, samples_per_epoch)

        # adjust len to split loading across all workers and ensure it is multiple of batch_size
        len_chunk = ((_len // get_world_size()) // batch_size) * batch_size
        _len = min(_len, len_chunk)
        logger.info(f"index_range={index_range}, len={_len}, len_chunk={len(_len)}")
        return _len

    ###################################################
    def worker_workset(self):
        rank = get_rank()
        local_start, local_end = rank * len(self), (rank + 1) * len(self)

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            assert get_world_size() == 1, get_world_size()
            iter_start = 0
            iter_end = len(self)

        else:
            # ensure the rng seed is fully unique across workers and epochs
            # the worker processes are generated as bit-wise copy of the "template" (the actual
            # instance of the present class that is created) whenever __iter__ is started. This
            # happens for each epoch, for train and validation, and independently for each DDP
            # worker. After the bit-wise copy, the rng seed needs to be made unique for
            # DDP workers, loader process, epoch.
            dist = torch.distributed
            self.data_loader_rng_seed *= (
                (((dist.get_rank() + 1) * 73) if dist.is_initialized() else 1)
                * ((worker_info.id + 1) * 37)
                * (self.epoch + 13)
                * 7
            )
            # split workload
            per_worker = (local_end - local_start) // worker_info.num_workers
            iter_start = local_start + worker_info.id * per_worker
            iter_end = iter_start + per_worker
            if worker_info.id + 1 == worker_info.num_workers:
                iter_end = local_end
            logger.info(
                f"{rank}::{worker_info.id}"
                + f" : dataset [{local_start},{local_end}) : [{iter_start},{iter_end})"
            )

        return iter_start, iter_end


def create_datasets(stream_info, time_window_handler, cf) -> list[DataReaderBase]:
    datasets: list[DataReaderBase] = []

    for fname in stream_info["filenames"]:
        kwargs = {
            "tw_handler": time_window_handler,
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
            case "icon":
                dataset = IconDataset
                datapath = cf.data_path_icon
            case _:
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

        datasets += [ds]
    
    return datasets

