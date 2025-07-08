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

from weathergen.datasets.data_reader_anemoi import DataReaderAnemoi
from weathergen.datasets.data_reader_base import (
    DataReaderBase,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    str_to_datetime64,
)
from weathergen.datasets.data_reader_fesom import DataReaderFesom
from weathergen.datasets.data_reader_obs import DataReaderObs
from weathergen.datasets.icon_dataset import IconDataset
from weathergen.datasets.masking import Masker
from weathergen.datasets.stream_data import StreamData
from weathergen.datasets.tokenizer_forecast import TokenizerForecast
from weathergen.datasets.tokenizer_masking import TokenizerMasking
from weathergen.datasets.utils import (
    compute_idxs_predict,
    compute_offsets_scatter_embed,
    compute_source_cell_lens,
)
from weathergen.utils.logger import init_loggers, logger
from weathergen.utils.train_logger import Stage, TrainLogger

_logger = logging.getLogger(__name__)

type AnyDataReader = (
    DataReaderBase | DataReaderAnemoi | DataReaderObs
    # | FesomDataset | AtmorepDataset
)


class MultiStreamDataSampler(torch.utils.data.IterableDataset):
    ###################################################
    def __init__(
        self,
        cf,
        start_date_,
        end_date_,
        batch_size,
        samples_per_epoch,
        train_logger: TrainLogger,
        stage: Stage,
        shuffle=True,
    ):
        super(MultiStreamDataSampler, self).__init__()

        start_date = str_to_datetime64(start_date_)
        end_date = str_to_datetime64(end_date_)

        assert end_date > start_date, (end_date, start_date)

        self.mask_value = 0.0
        self._train_logger = train_logger
        self._stage = stage

        self.len_hrs: int = cf.len_hrs
        self.step_hrs: int = cf.step_hrs
        self.time_window_handler = TimeWindowHandler(start_date, end_date, cf.len_hrs, cf.step_hrs)
        _logger.info(
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
            if self.forecast_steps.max() == 0:
                logger.warning("forecast policy is not None but number of forecast steps is 0.")
        self.forecast_policy = cf.forecast_policy

        self.len = 100000000

        self.streams_datasets: list[list[AnyDataReader]] = []
        for _, stream_info in enumerate(cf.streams):
            self.streams_datasets.append([])

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
                logger.info(
                    f"Opening dataset with type: {ds_type}"
                    + f"from stream config {stream_info['name']}.",
                )
                ds = dataset(filename=filename, **kwargs)

                fsm = self.forecast_steps[0]
                if len(ds) > 0:
                    self.len = min(self.len, len(ds) - (self.len_hrs * (fsm + 1)) // self.step_hrs)

                # MODIFIES config !!!
                stream_info[str(self._stage) + "_source_channels"] = ds.source_channels
                stream_info[str(self._stage) + "_target_channels"] = ds.target_channels

                self.streams_datasets[-1] += [ds]

        index_range = self.time_window_handler.get_index_range()
        self.len = int(index_range.end - index_range.start)
        self.len = min(self.len, samples_per_epoch if samples_per_epoch else self.len)
        # adjust len to split loading across all workers and ensure it is multiple of batch_size
        len_chunk = ((self.len // cf.num_ranks) // batch_size) * batch_size
        self.len = min(self.len, len_chunk)
        _logger.info(f"index_range={index_range}, len={self.len}, len_chunk={len_chunk}")

        self.rank = cf.rank
        self.num_ranks = cf.num_ranks

        self.streams = cf.streams
        self.shuffle = shuffle
        # TODO: remove options that are no longer supported
        self.input_window_steps = cf.input_window_steps
        self.embed_local_coords = cf.embed_local_coords
        self.embed_centroids_local_coords = cf.embed_centroids_local_coords
        self.sampling_rate_target = cf.sampling_rate_target

        self.batch_size = batch_size
        self.rng = np.random.default_rng(cf.data_loader_rng_seed)

        self.healpix_level_source: int = cf.healpix_level
        self.healpix_level_target: int = cf.healpix_level
        self.num_healpix_cells_source: int = 12 * 4**self.healpix_level_source
        self.num_healpix_cells_target: int = 12 * 4**self.healpix_level_target

        if cf.training_mode == "forecast":
            self.tokenizer = TokenizerForecast(cf.healpix_level, cf.data_loader_rng_seed)
        elif cf.training_mode == "masking":
            masker = Masker(
                cf.masking_rate,
                cf.masking_strategy,
                cf.masking_rate_sampling,
                cf.get("masking_strategy_config", {}),
            )
            self.tokenizer = TokenizerMasking(cf.healpix_level, cf.data_loader_rng_seed, masker)
            assert self.forecast_offset == 0, "masked token modeling requires auto-encoder training"
            msg = "masked token modeling does not support self.input_window_steps > 1; "
            msg += "increase window length"
            assert self.input_window_steps == 1, msg
        else:
            assert False, f"Unsupported training mode: {cf.training_mode}"

        self.epoch = 0

    ###################################################
    def advance(self):
        """
        Advance epoch
        """
        self.epoch += 1
        # advance since only copies are used for actual loading with parallel loaders
        self.rng.random()

    ###################################################
    def get_sources_size(self):
        return [
            ds[0].get_source_num_channels()
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
        fsm = (
            self.forecast_steps[min(self.epoch, len(self.forecast_steps) - 1)]
            if self.forecast_policy != "random"
            else self.forecast_steps.max()
        )
        if fsm > 0:
            logger.info(f"forecast_steps at epoch={self.epoch} : {fsm}")

        # data
        index_range = self.time_window_handler.get_index_range()
        idx_end = index_range.end
        # native length of datasets, independent of epoch length that has potentially been specified
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
            self.perms_forecast_dt = np.random.randint(
                low=self.forecast_steps.min(), high=fsm + 1, size=len_dt_samples, dtype=np.int64
            )
        else:
            assert False

        self.tokenizer.reset()

    ###################################################
    def denormalize_source_channels(self, obs_id, data):
        # TODO: with multiple ds per stream we need to distinguish these here
        return self.streams_datasets[obs_id][0].denormalize_source_channels(data)

    ###################################################
    def denormalize_target_channels(self, obs_id, data):
        # TODO: with multiple ds per stream we need to distinguish these here
        return self.streams_datasets[obs_id][0].denormalize_target_channels(data)

    ###################################################
    def __iter__(self):
        """
        Return one batch of data

        Return : list[list[StreamData]]
            len : number of batch items
            len[*] : number of streams
        """
        init_loggers()
        iter_start, iter_end = self.worker_workset()
        _logger.info(f"iter_start={iter_start}, iter_end={iter_end}, len={self.len}")

        # create new shuffeling
        self.reset()

        nhc_target = self.num_healpix_cells_target
        nhc_source = self.num_healpix_cells_source

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

                time_win1 = self.time_window_handler.window(idx)

                streams_data: list[StreamData] = []

                # for all streams
                for stream_info, stream_ds in zip(self.streams, self.streams_datasets, strict=True):
                    stream_data = StreamData(
                        idx, forecast_dt + self.forecast_offset, nhc_source, nhc_target
                    )

                    # for all sources for current stream
                    for _, ds in enumerate(stream_ds):
                        # source window (of potentially multi-step length)
                        rdata: ReaderData = ds.get_source(idx)

                        if rdata.is_empty():
                            stream_data.add_empty_source()
                        else:
                            # TODO: handling of conversion from numpy to torch here and below
                            # TODO: this should only be collected in validation mode
                            source_raw = torch.from_numpy(
                                np.concatenate((rdata.coords, rdata.geoinfos, rdata.data), 1)
                            )

                            (ss_cells, ss_lens, ss_centroids) = self.tokenizer.batchify_source(
                                stream_info,
                                torch.from_numpy(rdata.coords),
                                torch.from_numpy(rdata.geoinfos),
                                torch.from_numpy(rdata.data),
                                rdata.datetimes,
                                (time_win1.start, time_win1.end),
                                ds,
                            )

                            stream_data.add_source(source_raw, ss_lens, ss_cells, ss_centroids)

                        # target

                        # collect for all forecast steps
                        for fstep in range(
                            self.forecast_offset, self.forecast_offset + forecast_dt + 1
                        ):
                            step_forecast_dt = (
                                idx + (self.forecast_delta_hrs * fstep) // self.step_hrs
                            )
                            time_win2 = self.time_window_handler.window(step_forecast_dt)

                            rdata = ds.get_target(step_forecast_dt)

                            if rdata.is_empty():
                                stream_data.add_empty_target(fstep)
                            else:
                                (tt_cells, tc, tt_c, tt_t) = self.tokenizer.batchify_target(
                                    stream_info,
                                    self.sampling_rate_target,
                                    torch.from_numpy(rdata.coords),
                                    torch.from_numpy(rdata.geoinfos),
                                    torch.from_numpy(rdata.data),
                                    rdata.datetimes,
                                    (time_win2.start, time_win2.end),
                                    ds,
                                )

                                stream_data.add_target(fstep, tt_cells, tc, tt_c, tt_t)

                    # merge inputs for sources and targets for current stream
                    stream_data.merge_inputs()
                    streams_data += [stream_data]

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
        return self.len

    ###################################################
    def worker_workset(self):
        # local_start, local_end = 0, len(self)
        local_start, local_end = self.rank * self.len, (self.rank + 1) * self.len

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            assert self.num_ranks == 1
            iter_start = 0
            iter_end = len(self)

        else:
            # split workload
            per_worker = (local_end - local_start) // worker_info.num_workers
            iter_start = local_start + worker_info.id * per_worker
            iter_end = iter_start + per_worker
            if worker_info.id + 1 == worker_info.num_workers:
                iter_end = local_end
            _logger.info(
                f"{self.rank}::{worker_info.id}"
                + f" : dataset [{local_start},{local_end}) : [{iter_start},{iter_end})"
            )
        # ensure the tokenizers use different seeds
        self.tokenizer.reset()

        return iter_start, iter_end
