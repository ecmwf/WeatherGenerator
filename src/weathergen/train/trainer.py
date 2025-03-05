# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
import code
import time
import string
from pathlib import Path
import random
import functools

import math
import numpy as np
import torch
import logging
from functools import partial

import tqdm

import zarr

import torch.distributed as dist
import torch.utils.data.distributed

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    # default_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    StateDictType,
    FullStateDictConfig,
    FullOptimStateDictConfig,
)

from weathergen.train.trainer_base import Trainer_Base
from weathergen.train.lr_scheduler import LearningRateScheduler

from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler
from weathergen.model.model import Model, ModelParams
from weathergen.utils.config import Config
from weathergen.utils.logger import logger
from weathergen.utils.train_logger import TrainLogger
from weathergen.utils.validation_io import write_validation
from weathergen.train.utils import get_run_id

import weathergen.train.loss as losses


class Trainer(Trainer_Base):

    ###########################################
    def __init__(self, log_freq=20, checkpoint_freq=250, print_freq=10):
        Trainer_Base.__init__(self)

        assert print_freq < log_freq
        self.log_freq = log_freq
        self.checkpoint_freq = checkpoint_freq
        self.print_freq = print_freq

    ###########################################
    def init(
        self,
        cf,
        run_id_contd=None,
        epoch_contd=None,
        run_id_new=False,
        run_mode="training",
    ):

        self.cf = cf

        if isinstance(run_id_new, str):
            cf.run_id = run_id_new
        elif run_id_new or cf.run_id is None:
            cf.run_id = get_run_id()
        elif run_id_contd is not None and run_id_new == False:
            cf.run_id = run_id_contd
        assert cf.run_id is not None

        assert cf.samples_per_epoch % cf.batch_size == 0
        assert cf.samples_per_validation % cf.batch_size_validation == 0

        self.devices = self.init_torch()

        self.init_ddp(cf)

        # read configuration of data streams
        cf = self.init_streams(cf, run_id_contd)

        # create output directory
        path_out_base = Path("/p/home/jusers/langguth1/juwels/WeatherGenerator2/")
        path_run = str(path_out_base.joinpath("results/"))
        path_model = str(path_out_base.joinpath("models/"))

        # path_run = './results/' + cf.run_id + '/'
        # path_model = './models/' + cf.run_id + '/'
        if 0 == self.cf.rank:
            os.makedirs(path_run, exist_ok=True)
            os.makedirs(path_model, exist_ok=True)
            # save config
            cf.save()
            if run_mode == "training":
                cf.print()
        self.path_run = path_run

        self.init_perf_monitoring()

        self.train_logger = TrainLogger(cf, self.path_run)

    ###########################################
    def evaluate(self, cf, run_id_trained, epoch, run_id_new=False):

        # general initalization
        self.init(cf, run_id_trained, epoch, run_id_new, run_mode="evaluate")

        self.dataset_val = MultiStreamDataSampler(
            cf.data_path,
            cf.rank,
            cf.num_ranks,
            cf.streams,
            cf.start_date_val,
            cf.end_date_val,
            cf.len_hrs,
            cf.step_hrs,
            cf.batch_size_validation,
            cf.masking_mode,
            cf.masking_rate,
            cf.masking_rate_sampling,
            cf.shuffle,
            forecast_delta_hrs=cf.forecast_delta_hrs,
            forecast_steps=cf.forecast_steps,
            forecast_policy=cf.forecast_policy,
            healpix_level=cf.healpix_level,
            samples_per_epoch=cf.samples_per_validation,
            input_window_steps=cf.input_window_steps,
            embed_local_coords=cf.embed_local_coords,
            embed_centroids_local_coords=cf.embed_centroids_local_coords,
            target_coords_local=cf.target_coords_local,
            sampling_rate_target=cf.sampling_rate_target,
        )

        loader_params = {
            "batch_size": None,
            "batch_sampler": None,
            "shuffle": False,
            "num_workers": cf.loader_num_workers,
            "pin_memory": True,
        }
        self.data_loader_validation = torch.utils.data.DataLoader(
            self.dataset_val, **loader_params, sampler=None
        )

        if 0 == self.cf.rank:
            self.train_logger.initialize_file(
                self.dataset_val.stream_channels, val=True
            )

        num_channels = self.dataset_val.get_num_chs()
        self.geoinfo_sizes = self.dataset_val.get_geoinfo_sizes()
        self.num_selected_chas = [
            len(self.dataset_val.stream_channels[k])
            for k in self.dataset_val.stream_channels
        ]

        self.model = (
            Model(cf, num_channels, self.geoinfo_sizes).create().to(self.devices[0])
        )
        self.model.load(run_id_trained, epoch)
        print(f"Loaded model {run_id_trained} at epoch {epoch}.")
        self.ddp_model = self.model
        self.model_params = ModelParams().create(cf).to(self.devices[0])
        logging.getLogger("obslearn").info(
            f"Loaded model id={run_id_trained} at epoch={epoch}."
        )

        self.loss_fcts_val = []
        for name, w in cf.loss_fcts_val:
            self.loss_fcts_val += [[getattr(losses, name), w]]

        # evaluate validation set
        self.validate(epoch=0)
        print(f"Finished evaluation run with id: {cf.run_id}")

    ###########################################
    def evaluate_jac(
        self, cf, run_id, epoch, mode="row", date=None, obs_id=0, sample_id=0
    ):
        """Computes a row or column of the Jacobian as determined by mode ('row' or 'col'), i.e.
        determines sensitivities with respect to outputs or inputs
        """

        # general initalization
        self.init(cf, run_id, epoch, run_id_new=True, run_mode="offline")

        self.dataset = MultiStreamDataSampler(
            cf.streams,
            cf.start_date_val,
            cf.end_date_val,
            cf.delta_time,
            1,
            cf.masking_mode,
            cf.masking_rate_sampling,
            cf.t_win_hour,
            cf.loss_chs,
            shuffle=False,
            source_chs=cf.source_chs,
            forecast_steps=cf.forecast_steps,
            forecast_policy=cf.forecast_policy,
            healpix_level=cf.healpix_level,
        )

        num_channels = self.dataset.get_num_chs()

        self.model = Model(cf, num_channels).create().to(self.devices[0])
        self.model.load(run_id, epoch)
        print(f"Loaded model id={run_id}.")

        # TODO: support loading of specific data
        dataset_iter = iter(self.dataset)
        (sources, targets, targets_idxs, s_lens) = next(dataset_iter)

        dev = self.devices[0]
        sources = [source.to(dev, non_blocking=True) for source in sources]
        targets = [
            [toks.to(dev, non_blocking=True) for toks in target] for target in targets
        ]

        # evaluate model
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=cf.with_mixed_precision
        ):

            if "row" == mode:
                sources_in = [*sources, s_lens.to(torch.float32)]
                y = self.model(sources, s_lens)
                # vectors used to extract row from Jacobian
                vs_sources = [torch.zeros_like(y_obs) for y_obs in y[0]]
                vs_sources[obs_id][sample_id] = 1.0
                # evaluate
                out = torch.autograd.functional.vjp(
                    self.model.forward_jac, tuple(sources_in), tuple(vs_sources)
                )

            elif "col" == mode:
                # vectors used to extract col from Jacobian
                vs_sources = [torch.zeros_like(s_obs) for s_obs in sources]
                vs_sources[obs_id][sample_id] = 1.0
                vs_s_lens = torch.zeros_like(s_lens, dtype=torch.float32)
                # provide one tuple in the end
                sources_in = [*sources, s_lens.to(torch.float32)]
                vs_sources.append(vs_s_lens)
                # evaluate
                out = torch.autograd.functional.jvp(
                    self.model.forward_jac, tuple(sources_in), tuple(vs_sources)
                )
            else:
                assert False, "Unsupported mode."

        # extract and write output
        # TODO: refactor and try to combine with the code in compute_loss

        preds = out[0]
        jac = [j_obs.cpu().detach().numpy() for j_obs in out[1]]

        sources_all, preds_all = [[] for _ in cf.streams], [[] for _ in cf.streams]
        targets_all, targets_coords_all = [[] for _ in cf.streams], [
            [] for _ in cf.streams
        ]
        targets_idxs_all = [[] for _ in cf.streams]
        sources_lens = [toks.shape[0] for toks in sources]
        targets_lens = [[toks.shape[0] for toks in target] for target in targets]

        for i_obs, b_targets_idxs in enumerate(targets_idxs):
            for i_b, target_idxs_obs in enumerate(b_targets_idxs):  # 1 batch

                if len(targets[i_obs][i_b]) == 0:
                    continue

                gs = self.cf.geoinfo_size
                target_i_obs = torch.cat(
                    [t[:, gs:].unsqueeze(0) for t in targets[i_obs][i_b]], 0
                )
                preds_i_obs = preds[i_obs][target_idxs_obs]
                preds_i_obs = preds_i_obs.reshape(
                    [*preds_i_obs.shape[:2], *target_i_obs.shape[1:]]
                )

                if self.cf.loss_chs is not None:
                    if len(self.cf.loss_chs[i_obs]) == 0:
                        continue
                    target_i_obs = target_i_obs[..., self.cf.loss_chs[i_obs]]
                    preds_i_obs = preds_i_obs[..., self.cf.loss_chs[i_obs]]

                ds_val = self.dataset
                n = self.cf.geoinfo_size

                sources[i_obs][:, :, n:] = ds_val.denormalize_data(
                    i_obs, sources[i_obs][:, :, n:]
                )
                sources[i_obs][:, :, :n] = ds_val.denormalize_coords(
                    i_obs, sources[i_obs][:, :, :n]
                )
                sources_all[i_obs] += [sources[i_obs].detach().cpu()]

                preds_all[i_obs] += [
                    ds_val.denormalize_data(i_obs, preds_i_obs).detach().cpu()
                ]
                targets_all[i_obs] += [
                    ds_val.denormalize_data(i_obs, target_i_obs).detach().cpu()
                ]

                target_i_coords = (
                    torch.cat([t[:, :n].unsqueeze(0) for t in targets[i_obs][i_b]], 0)
                    .detach()
                    .cpu()
                )
                targets_coords_all[i_obs] += [
                    ds_val.denormalize_coords(i_obs, target_i_coords).detach().cpu()
                ]
                targets_idxs_all[i_obs] += [target_idxs_obs]

        cols = [ds[0][0].colnames for ds in dataset_val.obs_datasets_norm]
        write_validation(
            self.cf,
            self.path_run,
            self.cf.rank,
            epoch,
            cols,
            sources_all,
            preds_all,
            targets_all,
            targets_coords_all,
            targets_idxs_all,
            sources_lens,
            targets_lens,
            jac,
        )

    ###########################################
    def run(self, cf, run_id_contd=None, epoch_contd=None, run_id_new=False):

        # general initalization
        self.init(cf, run_id_contd, epoch_contd, run_id_new)

        self.dataset = MultiStreamDataSampler(
            cf.data_path,
            cf.rank,
            cf.num_ranks,
            cf.streams,
            cf.start_date,
            cf.end_date,
            cf.len_hrs,
            cf.step_hrs,
            cf.batch_size,
            cf.masking_mode,
            cf.masking_rate,
            cf.masking_rate_sampling,
            shuffle=True,
            rng_seed=cf.data_loader_rng_seed,
            forecast_delta_hrs=cf.forecast_delta_hrs,
            forecast_steps=cf.forecast_steps,
            forecast_policy=cf.forecast_policy,
            healpix_level=cf.healpix_level,
            samples_per_epoch=cf.samples_per_epoch,
            input_window_steps=cf.input_window_steps,
            embed_local_coords=cf.embed_local_coords,
            embed_centroids_local_coords=cf.embed_centroids_local_coords,
            target_coords_local=cf.target_coords_local,
            sampling_rate_target=cf.sampling_rate_target,
        )
        self.dataset_val = MultiStreamDataSampler(
            cf.data_path,
            cf.rank,
            cf.num_ranks,
            cf.streams,
            cf.start_date_val,
            cf.end_date_val,
            cf.len_hrs,
            cf.step_hrs,
            cf.batch_size_validation,
            cf.masking_mode,
            # validation mode is always full forecasting
            masking_rate=0.0,
            masking_rate_sampling=False,
            shuffle=True,
            rng_seed=cf.data_loader_rng_seed,
            forecast_delta_hrs=cf.forecast_delta_hrs,
            forecast_steps=cf.forecast_steps,
            forecast_policy=cf.forecast_policy,
            healpix_level=cf.healpix_level,
            samples_per_epoch=max(32, cf.samples_per_validation // cf.num_ranks),
            input_window_steps=cf.input_window_steps,
            embed_local_coords=cf.embed_local_coords,
            embed_centroids_local_coords=cf.embed_centroids_local_coords,
            target_coords_local=cf.target_coords_local,
            sampling_rate_target=cf.sampling_rate_target,
        )

        loader_params = {
            "batch_size": None,
            "batch_sampler": None,
            "shuffle": False,
            "num_workers": cf.loader_num_workers,
            "pin_memory": True,
        }
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset, **loader_params, sampler=None
        )
        self.data_loader_validation = torch.utils.data.DataLoader(
            self.dataset_val, **loader_params, sampler=None
        )

        if 0 == self.cf.rank:
            self.train_logger.initialize_file(
                self.dataset_val.stream_channels, train=True, val=True
            )

        num_channels = self.dataset.get_num_chs()
        self.num_selected_chas = [
            len(self.dataset.stream_channels[k]) for k in self.dataset.stream_channels
        ]
        self.geoinfo_sizes = self.dataset.get_geoinfo_sizes()

        self.model = Model(cf, num_channels, self.geoinfo_sizes).create()
        # load model if specified
        if run_id_contd is not None:
            self.model.load(run_id_contd, epoch_contd)
            print(f"Loaded model id={run_id_contd}.")

        if cf.forecast_freeze_model:
            self.model = self.model.freeze_weights_forecast()

        self.model = self.model.to(self.devices[0])

        if cf.compile_model:
            self.model = torch.compile(self.model, dynamic=True)

        self.ddp_model = self.model
        if cf.with_ddp and not cf.with_fsdp:
            self.ddp_model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                broadcast_buffers=True,
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
                bucket_cap_mb=512,
            )

        if cf.with_ddp and cf.with_fsdp:
            mp = (
                None
                if not cf.with_mixed_precision
                else MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True)
            )
            mp = None
            self.ddp_model = FSDP(
                self.model,
                auto_wrap_policy=size_based_auto_wrap_policy,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                cpu_offload=None,
                sync_module_states=(run_id_contd is not None),
                mixed_precision=mp,
            )

        self.model_params = ModelParams().create(cf).to("cuda")

        # if with_fsdp then parameter count is unreliable
        if (0 == self.cf.rank and not cf.with_fsdp) or not cf.with_ddp:
            self.model.print_num_parameters()

        # TODO: learning rate schedule
        # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
        kappa = cf.batch_size * cf.num_ranks
        beta1 = max(0.5, 1.0 - kappa * (1.0 - 0.9))
        beta2 = 1.0 - kappa * (1.0 - 0.999)
        eps = 1e-08 / np.sqrt(kappa)
        # beta1, beta2, eps = 0.125, 0.125, 1e-08
        self.optimizer = torch.optim.AdamW(
            self.ddp_model.parameters(),
            lr=cf.lr_start,
            weight_decay=cf.weight_decay,
            betas=(beta1, beta2),
            eps=eps,
        )
        self.grad_scaler = torch.amp.GradScaler("cuda")

        # lr is updated after each batch so account for this
        cf.lr_steps = int((len(self.dataset) * cf.num_epochs) / cf.batch_size)
        steps_decay = cf.lr_steps - cf.lr_steps_warmup - cf.lr_steps_cooldown
        # ensure that steps_decay has a reasonable value
        if steps_decay < int(0.2 * cf.lr_steps):
            cf.lr_steps_warmup = int(0.1 * cf.lr_steps)
            cf.lr_steps_cooldown = int(0.05 * cf.lr_steps)
            steps_decay = cf.lr_steps - cf.lr_steps_warmup - cf.lr_steps_cooldown
            str = f"cf.lr_steps_warmup and cf.lr_steps_cooldown were larger than cf.lr_steps={cf.lr_steps}"
            str += ". The value have been adjusted to cf.lr_steps_warmup={cf.lr_steps_warmup} and "
            str += " cf.lr_steps_cooldown={cf.lr_steps_cooldown} so that steps_decay={steps_decay}."
            logging.getLogger("obslearn").warning(f"")
        self.lr_scheduler = LearningRateScheduler(
            self.optimizer,
            cf.batch_size,
            cf.num_ranks,
            cf.lr_start,
            cf.lr_max,
            cf.lr_final_decay,
            cf.lr_final,
            cf.lr_steps_warmup,
            steps_decay,
            cf.lr_steps_cooldown,
            cf.lr_policy_warmup,
            cf.lr_policy_decay,
            cf.lr_policy_cooldown,
            cf.istep,
            cf.lr_scaling_policy,
        )

        if self.cf.istep > 0 and 0 == self.cf.rank:
            str = f"Continuing run with learning rate: {self.lr_scheduler.get_lr()}"
            logging.getLogger("obslearn").info(str)

        # get function handles for loss function terms
        self.loss_fcts = [[getattr(losses, name), w] for name, w in cf.loss_fcts]
        self.loss_fcts_val = [
            [getattr(losses, name), w] for name, w in cf.loss_fcts_val
        ]

        # recover epoch when continuing run
        epoch_base = int(self.cf.istep / len(self.data_loader))

        # torch.autograd.set_detect_anomaly(True)
        if cf.forecast_policy is not None:
            torch._dynamo.config.optimize_ddp = False

        # training loop

        # validate once at the beginning as reference
        if cf.val_initial:
            self.validate(-1)

        for epoch in range(epoch_base, cf.num_epochs):

            self.train(epoch)

            self.validate(epoch)

            self.save_model(epoch)

        # log final model
        self.save_model(cf.num_epochs)

    ###########################################
    def compute_loss(
        self,
        loss_fcts,
        sources,
        targets,
        targets_coords,
        targets_token_lens,
        preds,
        losses_all,
        stddev_all,
        losses_chn,
        preds_all=None,
        targets_all=None,
        targets_coords_all=None,
        targets_lens=None,
        mode="training",
    ):

        rng = np.random.default_rng()

        # merge across batch dimension (and keep streams and )
        targets_rt = [
            [
                torch.cat([t[i] for t in targets[fstep]])
                for i in range(len(targets[0][0]))
            ]
            for fstep in range(len(targets))
        ]
        targets_coords_rt = [
            [
                torch.cat([t[i] for t in targets_coords[fstep]])
                for i in range(len(targets_coords[0][0]))
            ]
            for fstep in range(len(targets_coords))
        ]
        targets_token_lens = [
            [
                torch.cat([t[i] for t in targets_token_lens[fstep]])
                for i in range(len(targets_token_lens[0][0]))
            ]
            for fstep in range(len(targets_token_lens))
        ]

        ctr = 0
        loss = torch.tensor(0.0, device=self.devices[0], requires_grad=True)
        # assert len(targets_rt) == len(preds) and len(preds) == len(self.cf.streams)
        for fstep in range(len(targets_rt)):
            for i_obs, (target, target_coords2, si) in enumerate(
                zip(targets_rt[fstep], targets_coords_rt[fstep], self.cf.streams)
            ):

                pred = preds[fstep][i_obs]

                gs = self.geoinfo_sizes[i_obs]
                num_channels = target[..., gs:].shape[-1]

                # set obs_loss_weight = 1. when not specified
                obs_loss_weight = si["loss_weight"] if "loss_weight" in si else 1.0
                channel_loss_weight = (
                    si["channel_weight"]
                    if "channel_weight" in si
                    else np.ones(num_channels)
                )
                # in validation mode, always unweighted loss is computed
                obs_loss_weight = 1.0 if mode == "validation" else obs_loss_weight
                channel_loss_weight = (
                    np.ones(num_channels)
                    if mode == "validation"
                    else channel_loss_weight
                )

                tok_spacetime = (
                    si["tokenize_spacetime"] if "tokenize_spacetime" in si else False
                )

                if target.shape[0] > 0 and pred.shape[0] > 0:

                    # extract content if tokens have been padded
                    if targets_token_lens[fstep][i_obs].shape[0] > 0:
                        sl = targets_token_lens[fstep][i_obs].to(
                            torch.int64
                        )  # TODO: why is it sometimes not torch.int
                        tro_type = (
                            si["target_readout"]["type"]
                            if "type" in si["target_readout"]
                            else "token"
                        )
                        if tro_type == "token":
                            pred = pred.reshape(
                                [
                                    *pred.shape[:2],
                                    target.shape[-2],
                                    target.shape[-1] - gs,
                                ]
                            )
                            pred = torch.cat(
                                [pred[:, i, :l] for i, l in enumerate(sl)], 1
                            )
                    else:
                        pred = pred.reshape([pred.shape[0], -1, target.shape[-1] - gs])
                    # extract data/coords and remove token dimension if it exists
                    target_coords = target[..., :gs].flatten(0, -2)
                    target_coords[:, 1:3] = target_coords2[..., 1:3]  # copy local time
                    target_data = target[..., gs:].flatten(0, -2)
                    pred = pred.reshape([pred.shape[0], *target_data.shape])

                    mask_nan = ~torch.isnan(target_data)

                    assert pred.shape[1] > 0
                    if pred[:, mask_nan].shape[1] == 0:
                        continue
                    ens = pred.shape[0] > 1

                    # accumulate loss from different loss functions and channels
                    for j, (loss_fct, w) in enumerate(loss_fcts):

                        # compute per channel loss
                        # val_uw is unweighted loss for logging
                        val, val_uw, ctr = (
                            torch.tensor(
                                0.0, device=self.devices[0], requires_grad=True
                            ),
                            0.0,
                            0.0,
                        )
                        for i in range(target_data.shape[-1]):

                            if tok_spacetime:
                                # iterate over time steps and compute loss separately for each
                                t_unique = torch.unique(target_coords[:, 1])
                                # tw = np.linspace( 1.0, 2.0, len(t_unique))
                                for jj, t in enumerate(t_unique):
                                    # if jj < len(t_unique)//2 and rng.uniform() < 0.5 and mode!='validation':
                                    #   continue
                                    mask_t = t == target_coords[:, 1]
                                    mask = torch.logical_and(mask_t, mask_nan[:, i])
                                    if mask.sum().item() > 0:
                                        temp = loss_fct(
                                            target_data[mask, i],
                                            pred[:, mask, i],
                                            pred[:, mask, i].mean(0),
                                            (
                                                pred[:, mask, i].std(0)
                                                if ens
                                                else torch.zeros(1)
                                            ),
                                        )
                                        val_uw += temp.item()
                                        # Add  temp loss to buffer
                                        losses_chn[j, i, i_obs] = temp
                                        val = (
                                            val + channel_loss_weight[i] * temp
                                        )  # * tw[jj]
                                        ctr += 1

                            else:
                                # only compute loss is there are non-NaN values
                                if mask_nan[:, i].sum().item() > 0:
                                    temp = loss_fct(
                                        target_data[mask_nan[:, i], i],
                                        pred[:, mask_nan[:, i], i],
                                        pred[:, mask_nan[:, i], i].mean(0),
                                        (
                                            pred[:, mask_nan[:, i], i].std(0)
                                            if ens
                                            else torch.zeros(1)
                                        ),
                                    )
                                    val_uw += temp.item()
                                    losses_chn[j, i, i_obs] = temp
                                    val = val + channel_loss_weight[i] * temp
                                    ctr += 1
                        val = val / ctr if (ctr > 0) else val
                        val_uw = val_uw / ctr if (ctr > 0) else val_uw

                        losses_all[j, i_obs] = val_uw
                        if (
                            self.cf.loss_fcts[j][0] == "stats"
                            or self.cf.loss_fcts[j][0] == "kcrps"
                        ):
                            stddev_all[i_obs] = pred[:, mask_nan].std(0).mean().item()
                        # ignore NaNs so that training can continue even if one pred-net diverges
                        loss = loss + (
                            (w * val * obs_loss_weight)
                            if not torch.isnan(val)
                            else torch.tensor(0.0, requires_grad=True)
                        )
                    ctr += 1

                    # log data for analysis
                    if preds_all is not None:

                        targets_lens[i_obs] += [target_data.shape[0]]
                        dn_data, dn_coords = (
                            self.dataset_val.denormalize_data,
                            self.dataset_val.denormalize_coords,
                        )

                        fp32 = torch.float32
                        preds_all[i_obs] += [
                            dn_data(i_obs, pred.to(fp32), False).detach().cpu()
                        ]
                        targets_all[i_obs] += [
                            dn_data(i_obs, target_data.to(fp32), False).detach().cpu()
                        ]
                        targets_coords_all[i_obs] += [
                            dn_coords(i_obs, target_coords.to(fp32)).detach().cpu()
                        ]

        return loss / ctr

    ###########################################
    def train(self, epoch):

        cf = self.cf
        self.ddp_model.train()

        dataset_iter = iter(self.data_loader)

        self.optimizer.zero_grad()
        self.losses_hist, self.stddev_hist, self.losses_hist_chn = [], [], []
        # training loop
        self.t_start = time.time()
        for bidx, data in enumerate(dataset_iter):

            data = self.input_to_device(data)
            (
                _,
                source_tokens_cells,
                source_tokens_lens,
                source_centroids,
                source_cell_lens,
                source_idxs_embed,
                target_tokens,
                target_token_lens,
                targets_coords,
                targets_coords_lens,
                targets_coords_idxs,
                forecast_dt,
            ) = data

            losses_all = (
                torch.ones((len(self.loss_fcts_val), len(cf.streams))) * torch.nan
            )
            losses_chn = (
                torch.ones(
                    (
                        len(self.loss_fcts_val),
                        max(self.num_selected_chas),
                        len(cf.streams),
                    )
                )
                * torch.nan
            )
            stddev_all = torch.zeros(len(cf.streams)) * torch.nan

            # evaluate model
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=cf.with_mixed_precision
            ):

                preds = self.ddp_model(
                    self.model_params,
                    source_tokens_cells,
                    source_tokens_lens,
                    source_centroids,
                    source_cell_lens,
                    source_idxs_embed,
                    targets_coords,
                    targets_coords_lens,
                    targets_coords_idxs,
                    forecast_dt,
                )

                loss = self.compute_loss(
                    self.loss_fcts,
                    source_tokens_cells,
                    target_tokens,
                    targets_coords,
                    target_token_lens,
                    preds,
                    losses_all,
                    stddev_all,
                    losses_chn,
                )

            # backward pass
            self.grad_scaler.scale(loss).backward()

            # gradient clipping
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.ddp_model.parameters(), max_norm=cf.grad_clip
            )

            # optimizer step
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.optimizer.zero_grad()

            # update learning rate
            self.lr_scheduler.step()

            self.losses_hist += [losses_all]
            self.stddev_hist += [stddev_all]
            self.losses_hist_chn += [losses_chn]

            perf_gpu, perf_mem = self.get_perf()
            self.perf_gpu = self.ddp_average(torch.tensor([perf_gpu])).item()
            self.perf_mem = self.ddp_average(torch.tensor([perf_mem])).item()

            self.log_terminal(bidx, epoch)
            self.log(bidx, epoch)
            # model checkpoint
            if bidx % self.checkpoint_freq == 0:
                self.save_model()

            self.cf.istep += cf.batch_size

        self.dataset.advance()

    ###########################################
    def validate(self, epoch):

        cf = self.cf
        self.ddp_model.eval()

        dataset_val_iter = iter(self.data_loader_validation)
        self.losses_hist, self.stddev_hist, self.losses_hist_chn = [], [], []

        with torch.no_grad():
            # print progress bar but only in interactive mode, i.e. when without ddp
            with tqdm.tqdm(
                total=len(self.data_loader_validation), disable=self.cf.with_ddp
            ) as pbar:
                for bidx, data in enumerate(dataset_val_iter):

                    data = self.input_to_device(data)
                    (
                        sources,
                        source_tokens_cells,
                        source_tokens_lens,
                        source_centroids,
                        source_cell_lens,
                        source_idxs_embed,
                        target_tokens,
                        target_token_lens,
                        targets_coords,
                        targets_coords_lens,
                        targets_coords_idxs,
                        forecast_dt,
                    ) = data

                    losses_all = (
                        torch.ones((len(self.loss_fcts_val), len(cf.streams)))
                        * torch.nan
                    )
                    losses_chn = (
                        torch.ones(
                            (
                                len(self.loss_fcts_val),
                                max(self.num_selected_chas),
                                len(cf.streams),
                            )
                        )
                        * torch.nan
                    )
                    stddev_all = torch.zeros(len(cf.streams)) * torch.nan

                    # evaluate model
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                        enabled=cf.with_mixed_precision,
                    ):
                        preds = self.ddp_model(
                            self.model_params,
                            source_tokens_cells,
                            source_tokens_lens,
                            source_centroids,
                            source_cell_lens,
                            source_idxs_embed,
                            targets_coords,
                            targets_coords_lens,
                            targets_coords_idxs,
                            forecast_dt,
                        )

                    # compute loss and log output
                    if bidx < cf.log_validation:

                        preds_all = [[] for _ in cf.streams]
                        targets_all, targets_coords_all = [[] for _ in cf.streams], [
                            [] for _ in cf.streams
                        ]
                        targets_lens = [[] for _ in cf.streams]

                        self.compute_loss(
                            self.loss_fcts_val,
                            source_tokens_cells,
                            target_tokens,
                            targets_coords,
                            target_token_lens,
                            preds,
                            losses_all,
                            stddev_all,
                            losses_chn,
                            preds_all,
                            targets_all,
                            targets_coords_all,
                            targets_lens,
                            mode="validation",
                        )

                        cols = [
                            ds[0][0].colnames
                            for ds in self.dataset_val.obs_datasets_norm
                        ]
                        write_validation(
                            self.cf,
                            self.path_run,
                            self.cf.rank,
                            epoch,
                            cols,
                            sources,
                            preds_all,
                            targets_all,
                            targets_coords_all,
                            targets_lens,
                        )

                    else:

                        self.compute_loss(
                            self.loss_fcts_val,
                            source_tokens_cells,
                            target_tokens,
                            targets_coords,
                            target_token_lens,
                            preds,
                            losses_all,
                            stddev_all,
                            losses_chn,
                            mode="validation",
                        )

                    self.losses_hist += [losses_all]
                    self.stddev_hist += [stddev_all]
                    self.losses_hist_chn += [losses_chn]

                    pbar.update(self.cf.batch_size_validation)

                losses_all = self.ddp_average(
                    torch.stack(self.losses_hist).to(torch.float64).nanmean(0)
                )
                losses_chn = self.ddp_average(
                    torch.stack(self.losses_hist_chn).to(torch.float64).nanmean(0)
                )
                stddev_all = self.ddp_average(
                    torch.stack(self.stddev_hist).to(torch.float64).nanmean(0)
                )

                if 0 == self.cf.rank and self.cf.istep >= 0:
                    loss_dict = {}
                    for j, (lname, _) in enumerate(cf.loss_fcts_val):
                        loss_dict[f"validation {lname}"] = torch.nanmean(
                            losses_all[j]
                        ).item()
                    loss_dict["validation std_dev"] = torch.nanmean(
                        stddev_all.mean()
                    ).item()
                    for i_obs, rt in enumerate(cf.streams):
                        loss_dict[
                            "validation {}".format(rt["name"].replace(",", ""))
                        ] = float(losses_all[0, i_obs])

                    for j, (lname, _) in enumerate(cf.loss_fcts_val):
                        for i_obs, rt in enumerate(cf.streams):
                            loss_dict[
                                "validation {} {}".format(
                                    rt["name"].replace(",", ""), lname
                                )
                            ] = losses_chn[j, :, i_obs].tolist()
                    # add data to plain logger
                    samples = cf.istep * cf.batch_size * cf.num_ranks
                    self.train_logger.add_val(
                        samples,
                        losses_all,
                        stddev_all,
                        losses_chn,
                        self.dataset_val.stream_channels,
                    )

                if 0 == self.cf.rank:
                    print(
                        "validation ({}) : {:03d} : loss = {:.4E}".format(
                            cf.run_id, epoch, torch.nanmean(losses_all[0])
                        ),
                        flush=True,
                    )
                    for i_obs, rt in enumerate(cf.streams):
                        print(
                            "{}".format(rt["name"]) + f" : {losses_all[0,i_obs]:0.4E}"
                        )

        # avoid that there is a systematic bias in the validation subset
        self.dataset_val.advance()

    ###########################################
    def input_to_device(self, data):

        (
            source,
            source_tokens_cells,
            source_tokens_lens,
            source_centroids,
            source_cell_lens,
            source_idxs_embed,
            target_tokens,
            target_token_lens,
            targets_coords,
            targets_coords_lens,
            targets_coords_idxs,
            forecast_dt,
        ) = data

        dev = self.devices[0]

        # source data
        source_tokens_cells = [
            [s.to(dev, non_blocking=True) for s in ss] for ss in source_tokens_cells
        ]
        source_centroids = [
            [c.to(dev, non_blocking=True) for c in cb] for cb in source_centroids
        ]
        source_cell_lens = source_cell_lens.to(dev, non_blocking=True)
        source_tokens_lens = source_tokens_lens.to(dev, non_blocking=True)
        source_idxs_embed[0] = [
            [s.to(dev, non_blocking=True) for s in ss] for ss in source_idxs_embed[0]
        ]

        # target data
        targets_coords = [
            [[t.to(dev, non_blocking=True) for t in tt] for tt in ttt]
            for ttt in targets_coords
        ]
        target_tokens = [
            [[t.to(dev, non_blocking=True) for t in tt] for tt in ttt]
            for ttt in target_tokens
        ]
        targets_coords_idxs[0] = [
            [s.to(dev, non_blocking=True) for s in ss] for ss in targets_coords_idxs[0]
        ]
        targets_coords_idxs[1] = [
            [s.to(dev, non_blocking=True) for s in ss] for ss in targets_coords_idxs[1]
        ]

        return (
            source,
            source_tokens_cells,
            source_tokens_lens,
            source_centroids,
            source_cell_lens,
            source_idxs_embed,
            target_tokens,
            target_token_lens,
            targets_coords,
            targets_coords_lens,
            targets_coords_idxs,
            forecast_dt,
        )

    ###########################################
    def save_model(self, epoch=-1, name=None):

        file_out = "./models/" + self.cf.run_id + "/{}_".format(self.cf.run_id)
        file_out += "latest" if epoch == -1 else "epoch{:05d}".format(epoch)
        file_out += ("_" + name) if name is not None else ""
        file_out += "{}.chkpt"

        if self.cf.with_ddp and self.cf.with_fsdp:
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                self.ddp_model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                state = self.ddp_model.state_dict()
        else:
            state = self.ddp_model.state_dict()

        if 0 == self.cf.rank:
            # save temp file (slow)
            torch.save(state, file_out.format("_temp"))
            # move file (which is changing the link in the file system and very fast)
            os.replace(file_out.format("_temp"), file_out.format(""))
            # save config
            self.cf.save(epoch)

    ###########################################
    def log(self, bidx, epoch):

        if bidx % self.log_freq == 0 and bidx > 0:

            l_avg = self.ddp_average(
                torch.nanmean(torch.stack(self.losses_hist), axis=0)
            )
            lchn_avg = self.ddp_average(
                torch.nanmean(torch.stack(self.losses_hist_chn), axis=0)
            )
            stddev_avg = self.ddp_average(
                torch.nanmean(torch.stack(self.stddev_hist), axis=0)
            )
            samples = self.cf.istep * self.cf.batch_size * self.cf.num_ranks

            if 0 == self.cf.rank:

                # logging
                loss_dict = {
                    "training mse": float(torch.nanmean(l_avg[0])),
                    "lr": self.lr_scheduler.get_lr(),
                }
                for i_obs, rt in enumerate(self.cf.streams):
                    loss_dict["training {}".format(rt["name"].replace(",", ""))] = (
                        float(l_avg[0, i_obs])
                    )

                for i_obs, rt in enumerate(self.cf.streams):
                    loss_dict["training {}".format(rt["name"].replace(",", ""))] = (
                        lchn_avg[0, :, i_obs].tolist()
                    )

                # plain logger
                self.train_logger.add_train(
                    samples,
                    self.lr_scheduler.get_lr(),
                    l_avg,
                    stddev_avg,
                    lchn_avg,
                    self.dataset_val.stream_channels,
                    self.perf_gpu,
                    self.perf_mem,
                )

            self.losses_hist, self.stddev_hist, self.losses_hist_chn = [], [], []

    ###########################################
    def log_terminal(self, bidx, epoch):

        if bidx % self.print_freq == 0 and bidx > 0:

            # compute from last iteration
            nanmean = torch.nanmean
            l_avg = self.ddp_average(
                nanmean(torch.stack(self.losses_hist[-self.print_freq :]), axis=0)
            )

            if 0 == self.cf.rank:

                # samples per sec
                dt = time.time() - self.t_start
                pstr = "{:03d} : {:05d}/{:05d} : {:06d} : loss = {:.4E} "
                pstr += "(lr={:.2E}, s/sec={:.3f})"
                len_dataset = len(self.data_loader) // self.cf.batch_size
                print(
                    pstr.format(
                        epoch,
                        bidx,
                        len_dataset,
                        self.cf.istep,
                        np.nanmean(l_avg[0]),
                        self.lr_scheduler.get_lr(),
                        (self.print_freq * self.cf.batch_size) / dt,
                    ),
                    flush=True,
                )
                print("\t", end="")
                for i_obs, rt in enumerate(self.cf.streams):
                    print(
                        "{}".format(rt["name"]) + f" : {l_avg[0,i_obs]:0.4E} \t", end=""
                    )
                print("\n", flush=True)

            self.t_start = time.time()
