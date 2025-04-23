# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging

import numpy as np
import torch

from weathergen.datasets.anemoi_dataset import AnemoiDataset
from weathergen.datasets.atmorep_dataset import AtmorepDataset
from weathergen.datasets.batchifyer import Batchifyer
from weathergen.datasets.fesom_dataset import FesomDataset
from weathergen.datasets.obs_dataset import ObsDataset
from weathergen.datasets.stream_data import StreamData
from weathergen.datasets.utils import (
    compute_idxs_predict,
    compute_offsets_scatter_embed,
    compute_source_cell_lens,
)
from weathergen.utils.logger import init_loggers, logger


class MultiStreamDataSampler(torch.utils.data.IterableDataset):
    ###################################################
    def __init__(self, cf, start_date, end_date, batch_size, samples_per_epoch, shuffle=True):
        super(MultiStreamDataSampler, self).__init__()

        assert end_date > start_date

        self.mask_value = 0.0

        self.len_hrs = cf.len_hrs
        self.step_hrs = cf.step_hrs

        self.forecast_delta_hrs = (
            cf.forecast_delta_hrs if cf.forecast_delta_hrs > 0 else self.len_hrs
        )
        assert self.forecast_delta_hrs == self.len_hrs, "Only supported option at the moment"
        self.forecast_steps = np.array(
            [cf.forecast_steps] if type(cf.forecast_steps) == int else cf.forecast_steps
        )
        if cf.forecast_policy is not None:
            if self.forecast_steps.max() == 0:
                logger.warning("forecast policy is not None but number of forecast steps is 0.")
        self.forecast_policy = cf.forecast_policy

        # end date needs to be adjusted to account for window length
        format_str = "%Y%m%d%H%M%S"
        end_dt = datetime.datetime.strptime(str(end_date), format_str)
        end_dt = end_dt + datetime.timedelta(hours=cf.len_hrs)
        end_date_padded = end_dt.strftime(format_str)

        self.len = 100000000

        self.streams_datasets = []
        for _, stream_info in enumerate(cf.streams):
            self.streams_datasets.append([])

            for fname in stream_info["filenames"]:
                # TODO: Should we translate the type to the class name and call based on this?
                if stream_info["type"] == "obs":
                    ds = ObsDataset(
                        start_date,
                        end_date_padded,
                        cf.len_hrs,
                        cf.step_hrs,
                        cf.data_path_obs + "/" + fname,
                        stream_info,
                    )

                elif stream_info["type"] == "anemoi":
                    ds = AnemoiDataset(
                        start_date,
                        end_date,
                        cf.len_hrs,
                        cf.step_hrs,
                        cf.data_path_anemoi + "/" + fname,
                        stream_info,
                    )

                elif stream_info["type"] == "fesom":
                    ds = FesomDataset(
                        start_date,
                        end_date,
                        cf.len_hrs,
                        cf.step_hrs,
                        cf.data_path_fesom + "/" + fname,
                        stream_info,
                    )

                elif stream_info["type"] == "atmorep":
                    ds = AtmorepDataset(
                        start_date,
                        end_date,
                        cf.len_hrs,
                        cf.step_hrs,
                        cf.data_path_anemoi + "/" + fname,
                        stream_info,
                    )

                else:
                    assert False, "Unsupported stream type {}.".format(stream_info["type"])

                fsm = self.forecast_steps[0]
                if len(ds) > 0:
                    self.len = min(self.len, len(ds) - (self.len_hrs * (fsm + 1)) // self.step_hrs)

                stream_info["source_channels"] = ds.source_channels
                stream_info["target_channels"] = ds.target_channels

                self.streams_datasets[-1] += [ds]

        # TODO: fix
        # determine start and end-time for all datasets, determine then the
        # by construction, this is identical for all datasets
        self.len_native = np.array(
            [len(ds) for dss in self.streams_datasets for ds in dss if len(ds) > 0]
        ).min()

        self.len = min(self.len, self.len if not samples_per_epoch else samples_per_epoch)
        # adjust len to split loading across all workers
        len_chunk = ((self.len_native // cf.num_ranks) // batch_size) * batch_size
        self.len = min(self.len, len_chunk)
        # ensure it is multiple of batch_size
        self.len = (self.len // batch_size) * batch_size

        # include auto-encodng of input data (forecast step = 0)
        self.auto_encode = cf.auto_encode

        self.rank = cf.rank
        self.num_ranks = cf.num_ranks

        self.streams = cf.streams
        self.shuffle = shuffle
        # TODO: remove options that are no longer supported
        self.input_window_steps = cf.input_window_steps
        self.embed_local_coords = cf.embed_local_coords
        self.embed_centroids_local_coords = cf.embed_centroids_local_coords
        self.target_coords_local = cf.target_coords_local
        self.sampling_rate_target = cf.sampling_rate_target

        self.masking_mode = cf.masking_mode
        self.masking_rate = cf.masking_rate
        self.masking_rate_sampling = cf.masking_rate_sampling

        self.batch_size = batch_size
        self.rng = np.random.default_rng(cf.data_loader_rng_seed)

        self.healpix_level_source = cf.healpix_level
        self.healpix_level_target = cf.healpix_level
        self.num_healpix_cells_source = 12 * 4**self.healpix_level_source
        self.num_healpix_cells_target = 12 * 4**self.healpix_level_target

        self.batchifyer = Batchifyer(cf.healpix_level)

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
            ds[0].get_source_num_channels() + ds[0].get_geoinfo_size() + ds[0].get_coords_size()
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
        if self.shuffle:
            # native length of datasets, independent of epoch length that has potentially been specified
            self.perms = self.rng.permutation(
                self.len_native - ((self.len_hrs * (fsm + 1)) // self.step_hrs)
            )
        else:
            self.perms = np.arange(self.len_native)

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

        # create new shuffeling
        self.reset()

        nhc_target = self.num_healpix_cells_target
        nhc_source = self.num_healpix_cells_source

        if self.auto_encode:
            fstep_start = 0
        else:
            fstep_start = 1
            if np.any(self.perms_forecast_dt == 0):
                logger.error("Deactivating auto-encoding requires all forecast_steps > 0. Check config.")
                raise ValueError("Deactivating auto-encoding requires all forecast_steps > 0.")
            
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
                idx = self.perms[idx_raw % self.perms.shape[0]]
                idx_raw += 1

                # TODO: this has to be independent of specific datasets
                time_win1, _ = (
                    self.streams_datasets[-1][0].time_window(idx),
                    self.streams_datasets[-1][0].time_window(idx + self.len_hrs // self.step_hrs),
                )

                streams_data = []

                # for all streams
                for _, (stream_info, stream_ds) in enumerate(
                    zip(self.streams, self.streams_datasets, strict=False)
                ):
                    stream_data = StreamData(forecast_dt, nhc_source, nhc_target)

                    # for all sources for current stream
                    for _, ds in enumerate(stream_ds):
                        # source window (of potentially multi-step length)
                        (coords, geoinfos, source, times) = ds.get_source(idx)
                        for it in range(1, self.input_window_steps):
                            (coords0, geoinfos0, source0, times0) = ds.get_source(
                                idx - it * self.len_hrs
                            )
                            coords = np.concatenate([coords0, coords], 0)
                            geoinfos = np.concatenate([geoinfos0, geoinfos], 0)
                            source = np.concatenate([source0, source], 0)
                            times = np.concatenate([times0, times], 0)

                        if source.shape[0] == 0:
                            stream_data.add_empty_source()
                        else:
                            # TODO: handling of conversion from numpy to torch here and below
                            # TODO: this should only be collected in validation mode
                            source_raw = torch.from_numpy(
                                np.concatenate((coords, geoinfos, source), 1)
                            )

                            (ss_cells, ss_lens, ss_centroids) = self.batchifyer.batchify_source(
                                stream_info,
                                self.masking_rate,
                                self.masking_rate_sampling,
                                self.rng,
                                torch.from_numpy(coords),
                                torch.from_numpy(geoinfos),
                                torch.from_numpy(source),
                                times,
                                time_win1,
                                ds,
                            )

                            stream_data.add_source(source_raw, ss_lens, ss_cells, ss_centroids)

                    # target

                    # collect for all forecast steps
                    for fstep in range(fstep_start, forecast_dt + 1):
                        # collect all targets
                        for _, ds in enumerate(stream_ds):
                            step_forecast_dt = (
                                idx + (self.forecast_delta_hrs * fstep) // self.step_hrs
                            )
                            _, time_win2 = (
                                self.streams_datasets[-1][0].time_window(idx),
                                self.streams_datasets[-1][0].time_window(step_forecast_dt),
                            )

                            (coords, geoinfos, target, times) = ds.get_target(step_forecast_dt)

                            if target.shape[0] == 0:
                                stream_data.add_empty_target(fstep)
                            else:
                                (tt_cells, tc, tt_c, tt_t) = self.batchifyer.batchify_target(
                                    stream_info,
                                    self.sampling_rate_target,
                                    self.rng,
                                    torch.from_numpy(coords),
                                    torch.from_numpy(geoinfos),
                                    torch.from_numpy(target),
                                    times,
                                    time_win2,
                                    ds,
                                )

                                stream_data.add_target(fstep, tt_cells, tc, tt_c, tt_t)

                    # merge inputs for sources and targets for current stream
                    stream_data.merge_inputs()
                    streams_data += [stream_data]

                # skip completely empty batch item
                if np.array([s.empty() for s in streams_data]).all():
                    continue

                batch += [streams_data]

            # aggregated lens of tokens per cell
            source_cell_lens = compute_source_cell_lens(batch)

            # compute offsets for scatter computation after embedding
            batch = compute_offsets_scatter_embed(batch)

            # compute offsets and auxiliary data needed for prediction computation
            # (info is not per stream so separate data structure)
            assert self.target_coords_local
            target_coords_idx = compute_idxs_predict(batch, forecast_dt)

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
            logging.getLogger("obslearn").info(
                f"{self.rank}::{worker_info.id}"
                + f" : dataset [{local_start},{local_end}) : [{iter_start},{iter_end})"
            )

        return iter_start, iter_end
