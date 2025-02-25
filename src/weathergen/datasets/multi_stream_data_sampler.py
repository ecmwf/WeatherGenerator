# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import torch
import math
import datetime
from copy import deepcopy
import logging
import time
import code
import os
import yaml

import pandas as pd

from weathergen.datasets.obs_dataset import ObsDataset
from weathergen.datasets.anemoi_dataset import AnemoiDataset
from weathergen.datasets.normalizer import DataNormalizer
from weathergen.datasets.batchifyer import Batchifyer
from weathergen.datasets.utils import merge_cells

from weathergen.utils.logger import logger


class MultiStreamDataSampler(torch.utils.data.IterableDataset):
    ###################################################
    def __init__(
        self,
        data_path,
        rank,
        num_ranks,
        streams,
        start_date,
        end_date,
        len_hrs,
        step_hrs,
        batch_size,
        masking_mode,
        masking_rate,
        masking_rate_sampling,
        shuffle=True,
        rng_seed=None,
        healpix_level=2,
        forecast_delta_hrs=0,
        forecast_steps=1,
        forecast_policy=None,
        samples_per_epoch=None,
        input_window_steps=1,
        embed_local_coords=False,
        embed_centroids_local_coords=False,
        target_coords_local=False,
        sampling_rate_target=1.0,
    ):
        super(MultiStreamDataSampler, self).__init__()

        assert end_date > start_date

        self.mask_value = 0.0
        # obs_id, year, day of year, minute of day
        self.geoinfo_offset = 6

        self.len_hrs = len_hrs
        self.step_hrs = step_hrs

        fc_policy_seq = "sequential" == forecast_policy or "sequential_random" == forecast_policy
        assert forecast_steps >= 0 if not fc_policy_seq else True
        self.forecast_delta_hrs = forecast_delta_hrs if forecast_delta_hrs > 0 else self.len_hrs
        self.forecast_steps = np.array(
            [forecast_steps] if type(forecast_steps) == int else forecast_steps
        )
        self.forecast_policy = forecast_policy

        # end date needs to be adjusted to account for window length
        format_str = "%Y%m%d%H%M%S"
        end_dt = datetime.datetime.strptime(str(end_date), format_str)
        end_dt = end_dt + datetime.timedelta(hours=len_hrs)
        end_date_padded = end_dt.strftime(format_str)

        self.len = 100000000

        self.obs_datasets_norm, self.obs_datasets_idxs = [], []
        for i, stream_info in enumerate(streams):
            self.obs_datasets_norm.append([])
            self.obs_datasets_idxs.append([])

            for fname in stream_info["filenames"]:
                ds = None
                if stream_info["type"] == "obs":
                    c_data_path = "/gpfs/scratch/ehpc01/dop/v1/"
                    ds = ObsDataset(
                        c_data_path + "/" + fname,
                        start_date,
                        end_date_padded,
                        len_hrs,
                        step_hrs,
                        False,
                    )

                    # skip pre-pended columns before lat,lon
                    do = 0
                    while ds.colnames[do] != "lat":
                        do += 1

                    # the processing here is not natural but a workaround to various inconsistencies in the
                    # current datasets
                    data_idxs = [
                        i for i, cn in enumerate(ds.selected_colnames[do:]) if "obsvalue_" == cn[:9]
                    ]
                    mask = np.ones(len(ds.selected_colnames[do:]), dtype=np.int32).astype(bool)
                    mask[data_idxs] = False
                    mask[-1] = False if "healpix" in ds.selected_colnames[-1] else mask[-1]
                    geoinfo_idx = (
                        np.arange(len(ds.selected_colnames[do:]), dtype=np.int64)[mask]
                    ).tolist()
                    logger.info(
                        "{} :: {} : {}".format(
                            stream_info["name"],
                            [ds.selected_colnames[do:][i] for i in geoinfo_idx],
                            [ds.selected_colnames[do:][i] for i in data_idxs],
                        )
                    )
                    stats_offset = 0

                elif stream_info["type"] == "anemoi":
                    c_data_path = data_path
                    if "CERRA" in stream_info["name"]:
                        c_data_path = "/gpfs/scratch/ehpc03/weathergen/"

                    ds = AnemoiDataset(
                        c_data_path + "/" + fname, start_date, end_date, len_hrs, step_hrs, False
                    )
                    do = 0
                    geoinfo_idx = [0, 1]
                    stats_offset = 2
                    data_idxs = list(ds.fields_idx + 2)

                else:
                    assert False, "Unsupported stream type {}.".format(stream_info["type"])

                fsm = self.forecast_steps[0]
                if len(ds) > 0:
                    self.len = min(self.len, len(ds) - (self.len_hrs * (fsm + 1)) // self.step_hrs)

                normalizer = DataNormalizer(
                    stream_info, self.geoinfo_offset, stats_offset, ds, geoinfo_idx, data_idxs, do
                )

                self.obs_datasets_norm[-1] += [(ds, normalizer, do)]
                self.obs_datasets_idxs[-1] += [(geoinfo_idx, data_idxs)]

        # by construction, this is identical for all datasets
        self.len_native = np.array(
            [len(ds[0]) for dss in self.obs_datasets_norm for ds in dss if len(ds[0]) > 0]
        ).min()

        self.len = min(self.len, self.len if not samples_per_epoch else samples_per_epoch)
        # adjust len to split loading across all workers
        len_chunk = ((self.len_native // num_ranks) // batch_size) * batch_size
        self.len = min(self.len, len_chunk)
        # ensure it is multiple of batch_size
        self.len = (self.len // batch_size) * batch_size

        self.rank = rank
        self.num_ranks = num_ranks

        self.streams = streams
        self.shuffle = shuffle
        self.input_window_steps = input_window_steps
        self.embed_local_coords = embed_local_coords
        self.embed_centroids_local_coords = embed_centroids_local_coords
        self.target_coords_local = target_coords_local
        self.sampling_rate_target = sampling_rate_target

        self.masking_mode = masking_mode
        self.masking_rate = masking_rate
        self.masking_rate_sampling = masking_rate_sampling

        self.batch_size = batch_size
        self.rng = np.random.default_rng(rng_seed)

        self.healpix_level_source = healpix_level
        self.healpix_level_target = healpix_level
        self.num_healpix_cells_source = 12 * 4**self.healpix_level_source
        self.num_healpix_cells_target = 12 * 4**self.healpix_level_target

        self.batchifyer = Batchifyer(healpix_level)

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
    def get_num_chs(self):
        gs = self.geoinfo_offset
        return [
            [len(idxs[0]) + gs + len(idxs[1]) for idxs in idxs_s]
            for idxs_s in self.obs_datasets_idxs
        ]

    ###################################################
    def reset(self):
        fsm = self.forecast_steps[min(self.epoch, len(self.forecast_steps) - 1)]
        if fsm > 0:
            logger.info(f"forecast_steps at epoch={self.epoch} : {fsm}")

        # data
        if self.shuffle:
            # native length of datasets, independent of epoch length that has potentially been specified
            self.perms = self.rng.permutation(
                self.len_native - ((self.len_hrs * (fsm + 1)) // self.step_hrs)
            )
            # self.perms = self.perms[:len(self)]
        else:
            self.perms = np.arange(self.len_native)
        # logging.getLogger('obslearn').info(  f'perms : {self.perms[:10]}')

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
    def denormalize_data(self, obs_id, data, with_offset=True):
        return self.obs_datasets_norm[obs_id][0][1].denormalize_data(data, with_offset)

    ###################################################
    def denormalize_coords(self, obs_id, coords):
        return self.obs_datasets_norm[obs_id][0][1].denormalize_coords(coords)

    ###################################################
    def get_geoinfo_size(self, obs_id, i_source):
        return len(self.obs_datasets_idxs[obs_id][i_source][0]) + self.geoinfo_offset

    ###################################################
    def get_geoinfo_sizes(self):
        return [self.get_geoinfo_size(i, 0) for i, _ in enumerate(self.obs_datasets_idxs)]

    ###################################################
    def __iter__(self):
        iter_start, iter_end = self.worker_workset()

        # create new shuffeling
        self.reset()

        nhc_target = self.num_healpix_cells_target
        nhc_source = self.num_healpix_cells_source

        # bidx is used to count the #batches that have been emitted
        # idx_raw is used to index into the dataset; the decoupling is needed
        # since there are empty batches
        idx_raw = iter_start
        for i, bidx in enumerate(range(iter_start, iter_end, self.batch_size)):
            # targets, targets_coords, targets_idxs = [], [], [],
            tcs, tcs_lens, target_tokens, source_tokens_cells, source_tokens_lens = (
                [],
                [],
                [],
                [],
                [],
            )
            target_tokens_lens, sources, source_centroids = [], [], []

            # forecast_dt needs to be constant per batch (amortized through data parallel training)
            forecast_dt = self.perms_forecast_dt[i]

            # use while loop due to the scattered nature of the data in time and to
            # ensure batches are not empty
            ib = 0
            while len(source_tokens_cells) < self.batch_size:
                idx = self.perms[idx_raw % self.perms.shape[0]]
                idx_raw += 1

                step_dt = self.len_hrs // self.step_hrs
                step_forecast_dt = (
                    step_dt + (self.forecast_delta_hrs * forecast_dt) // self.step_hrs
                )

                # TODO: this has to be independent of specific datasets
                time_win1, time_win2 = (
                    self.obs_datasets_norm[-1][0][0].time_window(idx),
                    self.obs_datasets_norm[-1][0][0].time_window(idx + step_forecast_dt),
                )

                c_tcs = [[] for _ in range(forecast_dt + 1)]
                c_tcs_lens = [[] for _ in range(forecast_dt + 1)]
                c_target_tokens = [[] for _ in range(forecast_dt + 1)]
                c_target_tokens_lens = [[] for _ in range(forecast_dt + 1)]
                c_source_tokens_cells = []
                c_source_tokens_lens = []
                c_source_centroids = []
                c_source_raw = []

                for obs_id, (stream_info, stream_dsn, stream_idxs) in enumerate(
                    zip(self.streams, self.obs_datasets_norm, self.obs_datasets_idxs)
                ):
                    s_tcs = []
                    s_tcs_lens = []
                    s_target_tokens = []
                    s_target_tokens_lens = []
                    s_source_tokens_cells = []
                    s_source_tokens_lens = []
                    s_source_centroids = []
                    s_source_raw = []

                    token_size = stream_info["token_size"]
                    grid = (
                        stream_info["gridded_output"] if "gridded_output" in stream_info else None
                    )
                    grid_info = (
                        stream_info["gridded_output_info"]
                        if "gridded_output_info" in stream_info
                        else None
                    )

                    for i_source, ((ds, normalizer, do), s_idxs) in enumerate(
                        zip(stream_dsn, stream_idxs)
                    ):
                        # source window (of potentially multi-step length)
                        (source1, times1) = ds[idx]
                        for it in range(1, self.input_window_steps):
                            (source0, times0) = ds[idx - it * step_dt]
                            source1 = np.concatenate([source0, source1], 0)
                            times1 = np.concatenate([times0, times1], 0)

                        if source1.shape[0] < token_size:
                            # skip if there are not enough data points
                            tt_cells, tt_lens = (
                                torch.tensor([]),
                                torch.zeros([nhc_target], dtype=torch.int32),
                            )
                            ss_cells, ss_lens = (
                                torch.tensor([]),
                                torch.zeros([nhc_source], dtype=torch.int32),
                            )
                            ss_centroids = torch.tensor([])
                            tc, tc_lens = (
                                torch.tensor([]),
                                torch.zeros([nhc_target], dtype=torch.int32),
                            )
                            source1_raw = torch.tensor([])
                        else:
                            oi = ds.properties["obs_id"]
                            source1 = self.prepare_window_source(
                                oi, do, normalizer, source1, times1, time_win1, s_idxs
                            )

                            # this should only be collected in validation mode
                            source1_raw = normalizer.denormalize_data(source1.clone())

                            (ss_cells, ss_lens, ss_centroids) = self.batchifyer.batchify_source(
                                stream_info,
                                self.geoinfo_offset,
                                self.get_geoinfo_size(obs_id, i_source),
                                self.masking_rate,
                                self.masking_rate_sampling,
                                self.rng,
                                source1,
                                times1,
                                normalizer.normalize_coords,
                            )

                        s_source_raw += [source1_raw]
                        s_source_tokens_lens += [ss_lens]
                        s_source_tokens_cells += [ss_cells]
                        s_source_centroids += (
                            [ss_centroids] if len(ss_centroids) > 0 else [torch.tensor([])]
                        )

                    # collect all sources in current stream and add to batch sample list when non-empty
                    if torch.tensor([len(s) for s in s_source_tokens_cells]).sum() > 0:
                        c_source_raw += [torch.cat(s_source_raw)]

                        # collect by merging entries per cells, preserving cell structure
                        c_source_tokens_cells += [merge_cells(s_source_tokens_cells, nhc_source)]
                        c_source_centroids += [merge_cells(s_source_centroids, nhc_source)]
                        # lens can be stacked and summed
                        c_source_tokens_lens += [torch.stack(s_source_tokens_lens).sum(0)]
                        # remove NaNs
                        c_source_tokens_cells[-1][torch.isnan(c_source_tokens_cells[-1])] = (
                            self.mask_value
                        )
                        c_source_centroids[-1][torch.isnan(c_source_centroids[-1])] = (
                            self.mask_value
                        )
                    else:
                        c_source_raw += [torch.tensor([])]
                        c_source_tokens_lens += [torch.zeros([nhc_source])]
                        c_source_tokens_cells += [torch.tensor([])]
                        c_source_centroids += [torch.tensor([])]

                    # target

                    # collect for all forecast steps
                    for fstep in range(forecast_dt + 1):
                        # collect all streams
                        for i_source, ((ds, normalizer, do), s_idxs) in enumerate(
                            zip(stream_dsn, stream_idxs)
                        ):
                            (source2, times2) = ds[idx + step_forecast_dt]

                            if source2.shape[0] < token_size:
                                # skip if there are not enough data points
                                tt_cells, tt_lens = (
                                    torch.tensor([]),
                                    torch.zeros([nhc_target], dtype=torch.int32),
                                )
                                ss_cells, ss_lens = (
                                    torch.tensor([]),
                                    torch.zeros([nhc_source], dtype=torch.int32),
                                )
                                ss_centroids = torch.tensor([])
                                tc, tc_lens = (
                                    torch.tensor([]),
                                    torch.zeros([nhc_target], dtype=torch.int32),
                                )
                                source1_raw = torch.tensor([])
                            else:
                                oi = ds.properties["obs_id"]
                                source2 = self.prepare_window_target(
                                    oi, do, normalizer, source2, times2, time_win2, s_idxs
                                )

                                (tt_cells, tt_lens, tc, tc_lens) = self.batchifyer.batchify_target(
                                    stream_info,
                                    self.geoinfo_offset,
                                    self.get_geoinfo_size(obs_id, i_source),
                                    self.sampling_rate_target,
                                    self.rng,
                                    source2,
                                    times2,
                                    normalizer.normalize_targets,
                                )

                            s_target_tokens_lens += (
                                [tt_lens] if len(tt_lens) > 0 else [torch.tensor([])]
                            )
                            s_target_tokens += (
                                [tt_cells] if len(tt_cells) > 0 else [torch.tensor([])]
                            )
                            s_tcs += [tc]
                            s_tcs_lens += [tc_lens]

                        # collect all sources in current stream and add to batch sample list when non-empty
                        if torch.tensor([len(s) for s in s_target_tokens]).sum() > 0:
                            c_tcs[fstep] += [merge_cells(s_tcs, nhc_target)]
                            c_target_tokens[fstep] += [merge_cells(s_target_tokens, nhc_target)]
                            # lens can be stacked and summed
                            c_target_tokens_lens[fstep] += [
                                torch.stack(s_target_tokens_lens).sum(0)
                            ]
                            c_tcs_lens[fstep] += [torch.stack(s_tcs_lens).sum(0)]
                            # remove NaNs
                            c_tcs[fstep][-1][torch.isnan(c_tcs[fstep][-1])] = self.mask_value

                        else:
                            c_tcs[fstep] += [torch.tensor([])]
                            c_tcs_lens[fstep] += [torch.tensor([])]
                            c_target_tokens[fstep] += [torch.tensor([])]
                            c_target_tokens_lens[fstep] += [torch.tensor([])]

                # add batch, ensure sample is not empty
                s1 = torch.tensor([stl.sum() for stl in c_source_tokens_lens]).sum()
                s2 = torch.tensor([len(t) for f_tcs in c_tcs for t in f_tcs]).sum()
                if s1 > 0 and s2 > 0:
                    # source
                    sources += [c_source_raw]
                    source_tokens_cells += [c_source_tokens_cells]
                    source_tokens_lens += [c_source_tokens_lens]
                    source_centroids += [c_source_centroids]
                    # target
                    tcs += [c_tcs]
                    tcs_lens += [c_tcs_lens]
                    target_tokens += [c_target_tokens]
                    target_tokens_lens += [c_target_tokens_lens]
                    ib += 1

            # skip if source is completely empty or nothing to predict (which causes errors in back prop)
            target_tokens_lens_total = torch.cat(
                [torch.cat(t) for tt in target_tokens_lens for t in tt]
            )
            if len(source_tokens_lens) == 0 or target_tokens_lens_total.sum() == 0:
                continue

            # precompute for processing in the model (with varlen flash attention)
            source_cell_lens = torch.stack(
                [
                    torch.stack(stl_b) if len(stl_b) > 0 else torch.tensor([])
                    for stl_b in source_tokens_lens
                ]
            )
            source_cell_lens = torch.sum(source_cell_lens, 1).flatten().to(torch.int32)
            source_cell_lens = torch.cat([torch.zeros(1, dtype=torch.int32), source_cell_lens])

            source_tokens_lens = torch.from_numpy(np.array(source_tokens_lens)).to(torch.int32)

            # precompute index sets for scatter operation after embed
            offsets_base = source_tokens_lens.sum(1).sum(0).cumsum(0)
            offsets = torch.cat([torch.zeros(1, dtype=torch.int32), offsets_base[:-1]])
            offsets_pe = torch.zeros_like(offsets)
            idxs_embed = []
            idxs_embed_pe = []
            for ib, sb in enumerate(source_tokens_cells):
                idxs_embed += [[]]
                idxs_embed_pe += [[]]
                for itype, s in enumerate(sb):
                    if s.shape[0] == 0:
                        idxs_embed[-1] += [torch.tensor([])]
                        idxs_embed_pe[-1] += [torch.tensor([])]
                        continue

                    idxs = torch.cat(
                        [
                            torch.arange(o, o + l, dtype=torch.int64)
                            for o, l in zip(offsets, source_tokens_lens[ib, itype])
                        ]
                    )
                    idxs_embed[-1] += [idxs.unsqueeze(1)]
                    idxs_embed_pe[-1] += [
                        torch.cat(
                            [
                                torch.arange(o, o + l, dtype=torch.int32)
                                for o, l in zip(offsets_pe, source_tokens_lens[ib][itype])
                            ]
                        )
                    ]
                    # advance offsets
                    offsets += source_tokens_lens[ib][itype]
                    offsets_pe += source_tokens_lens[ib][itype]

            target_coords_lens = tcs_lens

            # target coords idxs
            tcs_lens_merged = []
            tcs_idxs = []
            pad = torch.zeros(1, dtype=torch.int32)
            for ii in range(len(self.streams)):
                # generate len lists for varlen attention (per batch list for local, per-cell attention and
                # global, per-batch-item lists otherwise)
                if self.target_coords_local:
                    tcs_lens_merged += [
                        [
                            torch.cat(
                                [
                                    pad,
                                    torch.cat(
                                        [
                                            target_coords_lens[i_b][fstep][ii]
                                            for i_b in range(len(tcs))
                                        ]
                                    ),
                                ]
                            ).to(torch.int32)
                            for fstep in range(forecast_dt + 1)
                        ]
                    ]
                else:
                    tcs_lens_merged += [
                        torch.cat(
                            [pad, torch.tensor([len(tcs[i_b][ii]) for i_b in range(len(tcs))])]
                        ).to(torch.int32)
                    ]

                # lengths for varlen attention
                tcs_idxs += [
                    [torch.cat([torch.arange(l) for l in tlm]) for tlm in tcs_lens_merged[-1]]
                ]

            # reorder to have forecast step as first dimension, then batch items
            def list_transpose(clist):
                return [[l[i] for l in clist] for i in range(len(clist[0]))]

            target_tokens = list_transpose(target_tokens)
            target_tokens_lens = list_transpose(target_tokens_lens)
            tcs = list_transpose(tcs)
            target_coords_lens = list_transpose(target_coords_lens)
            tcs_lens_merged = list_transpose(tcs_lens_merged)
            tcs_idxs = list_transpose(tcs_idxs)

            assert len(source_tokens_cells) == self.batch_size
            yield (
                sources,
                source_tokens_cells,
                source_tokens_lens,
                source_centroids,
                source_cell_lens,
                [idxs_embed, idxs_embed_pe],
                target_tokens,
                target_tokens_lens,
                tcs,
                target_coords_lens,
                [tcs_lens_merged, tcs_idxs],
                forecast_dt,
            )

    ###################################################
    def prepare_window_source(
        self, obs_id, data_offset, normalizer, source, times, time_win, stream_idxs
    ):
        source = source[:, data_offset:]
        # select geoinfo and field channels (also ensure geoinfos is at the beginning)
        idxs = np.array(stream_idxs[0] + stream_idxs[1])
        source = source[:, idxs]

        # assemble tensor as fed to the network, combining geoinfo and data
        fp32 = torch.float32
        dt = pd.to_datetime(times)
        dt_win = pd.to_datetime(time_win)
        dt_delta = dt - dt_win[0]
        source = torch.cat(
            (
                torch.full([dt.shape[0], 1], obs_id, dtype=fp32),
                torch.tensor(dt.year, dtype=fp32).unsqueeze(1),
                torch.tensor(dt.dayofyear, dtype=fp32).unsqueeze(1),
                torch.tensor(dt.hour * 60 + dt.minute, dtype=fp32).unsqueeze(1),
                torch.tensor(dt_delta.seconds, dtype=fp32).unsqueeze(1),
                torch.tensor(dt_delta.seconds, dtype=fp32).unsqueeze(1),
                torch.from_numpy(source),
            ),
            1,
        )
        # normalize data (leave coords so that they can be utilized for task/masking)
        source = normalizer.normalize_data(source)

        return source

    ###################################################
    def prepare_window_target(
        self, obs_id, data_offset, normalizer, source, times, time_win, stream_idxs
    ):
        source = source[:, data_offset:]
        # select geoinfo and field channels (also ensure geoinfos is at the beginning)
        idxs = np.array(stream_idxs[0] + stream_idxs[1])
        source = source[:, idxs]

        # assemble tensor as fed to the network, combining geoinfo and data
        dt = pd.to_datetime(times)
        dt_win = pd.to_datetime(time_win)
        # for target only provide local time
        dt_delta = torch.tensor((dt - dt_win[0]).seconds, dtype=torch.float32).unsqueeze(1)
        source = torch.cat(
            (
                torch.full([dt.shape[0], 1], obs_id, dtype=torch.float32),
                dt_delta,
                dt_delta,
                dt_delta,
                dt_delta,
                dt_delta,
                torch.from_numpy(source),
            ),
            1,
        )
        # normalize data (leave coords so that they can be utilized for task/masking)
        source = normalizer.normalize_data(source)

        return source

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
