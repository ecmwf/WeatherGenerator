# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data.distributed
import tqdm
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy  # default_auto_wrap_policy,

import weathergen.train.loss as losses
import weathergen.utils.config as config
from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler
from weathergen.model.model import Model, ModelParams
from weathergen.train.lr_scheduler import LearningRateScheduler
from weathergen.train.trainer_base import Trainer_Base
from weathergen.train.utils import get_run_id
from weathergen.utils.train_logger import TrainLogger
from weathergen.utils.validation_io import write_validation

_logger = logging.getLogger(__name__)


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
        elif run_id_contd is not None and not run_id_new:
            cf.run_id = run_id_contd
        assert cf.run_id is not None

        assert cf.samples_per_epoch % cf.batch_size == 0
        assert cf.samples_per_validation % cf.batch_size_validation == 0

        self.devices = self.init_torch()

        self.init_ddp(cf)

        # read configuration of data streams
        cf.streams = config.load_streams(Path(cf.streams_directory))

        # create output directory
        cf.run_path = cf.run_path if hasattr(cf, "run_path") else "./results"
        cf.model_path = cf.model_path if hasattr(cf, "model_path") else "./models"
        path_run = Path(cf.run_path) / cf.run_id
        path_model = Path(cf.model_path) / cf.run_id
        if self.cf.rank == 0:
            path_run.mkdir(exist_ok=True, parents=True)
            path_model.mkdir(exist_ok=True, parents=True)
        self.path_run = path_run

        self.init_perf_monitoring()

        self.train_logger = TrainLogger(cf, self.path_run)

    ###########################################
    def evaluate(self, cf, run_id_trained, epoch, run_id_new=False):
        # general initalization
        self.init(cf, run_id_trained, epoch, run_id_new, run_mode="evaluate")

        self.dataset_val = MultiStreamDataSampler(
            cf,
            cf.start_date_val,
            cf.end_date_val,
            cf.batch_size_validation,
            cf.samples_per_validation,
            shuffle=cf.shuffle,
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

        sources_size = self.dataset_val.get_sources_size()
        targets_num_channels = self.dataset_val.get_targets_num_channels()
        targets_coords_size = self.dataset_val.get_targets_coords_size()

        self.model = Model(cf, sources_size, targets_num_channels, targets_coords_size).create()
        self.model = self.model.to(self.devices[0])
        self.model.load(run_id_trained, epoch)
        print(f"Loaded model {run_id_trained} at epoch {epoch}.")
        self.ddp_model = self.model
        self.model_params = ModelParams().create(cf).to(self.devices[0])
        logging.getLogger("obslearn").info(f"Loaded model id={run_id_trained} at epoch={epoch}.")

        self.loss_fcts_val = []
        for name, w in cf.loss_fcts_val:
            self.loss_fcts_val += [[getattr(losses, name), w]]

        if self.cf.rank == 0:
            config.save(self.cf, epoch=None)

        # evaluate validation set
        self.validate(epoch=0)
        print(f"Finished evaluation run with id: {cf.run_id}")

    ###########################################
    def run(self, cf, run_id_contd=None, epoch_contd=None, run_id_new=False):
        # general initalization
        self.init(cf, run_id_contd, epoch_contd, run_id_new)

        self.dataset = MultiStreamDataSampler(
            cf, cf.start_date, cf.end_date, cf.batch_size, cf.samples_per_epoch, shuffle=True
        )
        self.dataset_val = MultiStreamDataSampler(
            cf,
            cf.start_date_val,
            cf.end_date_val,
            cf.batch_size_validation,
            cf.samples_per_validation,
            shuffle=True,
        )

        loader_params = {
            "batch_size": None,
            "batch_sampler": None,
            "shuffle": False,
            "num_workers": cf.loader_num_workers,
            "pin_memory": True,
        }
        self.data_loader = torch.utils.data.DataLoader(self.dataset, **loader_params, sampler=None)
        self.data_loader_validation = torch.utils.data.DataLoader(
            self.dataset_val, **loader_params, sampler=None
        )

        sources_size = self.dataset.get_sources_size()
        targets_num_channels = self.dataset.get_targets_num_channels()
        targets_coords_size = self.dataset.get_targets_coords_size()

        self.model = Model(cf, sources_size, targets_num_channels, targets_coords_size).create()
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
        if (self.cf.rank == 0 and not cf.with_fsdp) or not cf.with_ddp:
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

        assert len(self.dataset) > 0, f"No data found in {self.dataset}"

        # lr is updated after each batch so account for this
        # TODO: conf should be read-only, do not modify the conf in flight
        cf.lr_steps = int((len(self.dataset) * cf.num_epochs) / cf.batch_size)

        steps_decay = cf.lr_steps - cf.lr_steps_warmup - cf.lr_steps_cooldown
        _logger.debug(f"steps_decay={steps_decay} lr_steps={cf.lr_steps}")
        # ensure that steps_decay has a reasonable value
        if steps_decay < int(0.2 * cf.lr_steps):
            cf.lr_steps_warmup = int(0.1 * cf.lr_steps)
            cf.lr_steps_cooldown = int(0.05 * cf.lr_steps)
            steps_decay = cf.lr_steps - cf.lr_steps_warmup - cf.lr_steps_cooldown
            s = f"cf.lr_steps_warmup and cf.lr_steps_cooldown were larger than cf.lr_steps={cf.lr_steps}"
            s += f". The value have been adjusted to cf.lr_steps_warmup={cf.lr_steps_warmup} and "
            s += f" cf.lr_steps_cooldown={cf.lr_steps_cooldown} so that steps_decay={steps_decay}."
            logging.getLogger("obslearn").warning(s)
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

        if self.cf.istep > 0 and self.cf.rank == 0:
            str = f"Continuing run with learning rate: {self.lr_scheduler.get_lr()}"
            logging.getLogger("obslearn").info(str)

        # get function handles for loss function terms
        self.loss_fcts = [[getattr(losses, name), w] for name, w in cf.loss_fcts]
        self.loss_fcts_val = [[getattr(losses, name), w] for name, w in cf.loss_fcts_val]

        # recover epoch when continuing run
        epoch_base = int(self.cf.istep / len(self.data_loader))

        # torch.autograd.set_detect_anomaly(True)
        if cf.forecast_policy is not None:
            torch._dynamo.config.optimize_ddp = False

        if self.cf.rank == 0:
            config.save(self.cf, None)
            config.print_cf(self.cf)

        # training loop

        # validate once at the beginning as reference
        if cf.val_initial:
            self.validate(-1)

        for epoch in range(epoch_base, cf.num_epochs):
            _logger.info(f"Epoch {epoch} of {cf.num_epochs}: train.")
            self.train(epoch)
            _logger.info(f"Epoch {epoch} of {cf.num_epochs}: validate.")

            self.validate(epoch)
            _logger.info(f"Epoch {epoch} of {cf.num_epochs}: save_model.")

            self.save_model(epoch)

        # log final model
        self.save_model(cf.num_epochs)

    ###########################################
    def compute_loss(
        self,
        loss_fcts,
        forecast_steps,
        streams_data,
        preds,
        losses_all,
        stddev_all,
        mode="training",
        log_data=False,
    ):
        # merge across batch dimension (and keep streams)
        targets_rt = [
            [
                torch.cat([t[i].target_tokens[fstep] for t in streams_data])
                for i in range(len(self.cf.streams))
            ]
            for fstep in range(forecast_steps + 1)
        ]
        targets_coords_rt = [
            [
                torch.cat([t[i].target_coords[fstep] for t in streams_data])
                for i in range(len(self.cf.streams))
            ]
            for fstep in range(forecast_steps + 1)
        ]

        if log_data:
            fsteps = len(targets_rt)
            preds_all = [[[] for _ in self.cf.streams] for _ in range(fsteps)]
            targets_all = [[[] for _ in self.cf.streams] for _ in range(fsteps)]
            targets_lens = [[[] for _ in self.cf.streams] for _ in range(fsteps)]

            targets_coords_raw_rt = [
                [
                    torch.cat([t[i].target_coords_raw[fstep] for t in streams_data])
                    for i in range(len(self.cf.streams))
                ]
                for fstep in range(forecast_steps + 1)
            ]
            targets_times_raw_rt = [
                [
                    np.concatenate([t[i].target_times_raw[fstep] for t in streams_data])
                    for i in range(len(self.cf.streams))
                ]
                for fstep in range(forecast_steps + 1)
            ]

        ctr_ftarget = 0
        loss = torch.tensor(0.0, device=self.devices[0], requires_grad=True)

        # assert len(targets_rt) == len(preds) and len(preds) == len(self.cf.streams)
        for fstep in range(len(targets_rt)):
            for i_obs, (target, target_coords, si) in enumerate(
                zip(targets_rt[fstep], targets_coords_rt[fstep], self.cf.streams, strict=False)
            ):
                pred = preds[fstep][i_obs]

                num_channels = len(si["target_channels"])

                # set obs_loss_weight = 1. when not specified
                obs_loss_weight = si["loss_weight"] if "loss_weight" in si else 1.0
                channel_loss_weight = (
                    si["channel_weight"] if "channel_weight" in si else np.ones(num_channels)
                )
                # in validation mode, always unweighted loss is computed
                obs_loss_weight = 1.0 if mode == "validation" else obs_loss_weight
                channel_loss_weight = (
                    np.ones(num_channels) if mode == "validation" else channel_loss_weight
                )

                tok_spacetime = si["tokenize_spacetime"] if "tokenize_spacetime" in si else False

                if target.shape[0] > 0 and pred.shape[0] > 0:
                    # extract data/coords and remove token dimension if it exists
                    pred = pred.reshape([pred.shape[0], *target.shape])
                    assert pred.shape[1] > 0

                    mask_nan = ~torch.isnan(target)
                    if pred[:, mask_nan].shape[1] == 0:
                        continue
                    ens = pred.shape[0] > 1

                    # accumulate loss from different loss functions and channels
                    for j, (loss_fct, w) in enumerate(loss_fcts):
                        # compute per channel loss
                        # val_uw is unweighted loss for logging
                        val = torch.tensor(0.0, device=self.devices[0], requires_grad=True)
                        val_uw = 0.0
                        ctr_chs = 0.0

                        # loop over all channels
                        for i in range(target.shape[-1]):
                            # if stream is internal time step, compute loss separately per step
                            if tok_spacetime:
                                # iterate over time steps and compute loss separately for each
                                t_unique = torch.unique(target_coords[:, 1])
                                for _jj, t in enumerate(t_unique):
                                    mask_t = t == target_coords[:, 1]
                                    mask = torch.logical_and(mask_t, mask_nan[:, i])
                                    if mask.sum().item() > 0:
                                        temp = loss_fct(
                                            target[mask, i],
                                            pred[:, mask, i],
                                            pred[:, mask, i].mean(0),
                                            (pred[:, mask, i].std(0) if ens else torch.zeros(1)),
                                        )
                                        val_uw += temp.item()
                                        val = val + channel_loss_weight[i] * temp
                                        ctr_chs += 1

                            else:
                                # only compute loss is there are non-NaN values
                                if mask_nan[:, i].sum().item() > 0:
                                    temp = loss_fct(
                                        target[mask_nan[:, i], i],
                                        pred[:, mask_nan[:, i], i],
                                        pred[:, mask_nan[:, i], i].mean(0),
                                        (
                                            pred[:, mask_nan[:, i], i].std(0)
                                            if ens
                                            else torch.zeros(1)
                                        ),
                                    )
                                    val_uw += temp.item()
                                    val = val + channel_loss_weight[i] * temp
                                    ctr_chs += 1
                        val = val / ctr_chs if (ctr_chs > 0) else val
                        val_uw = val_uw / ctr_chs if (ctr_chs > 0) else val_uw

                        losses_all[j, i_obs] = val_uw
                        if self.cf.loss_fcts[j][0] == "stats" or self.cf.loss_fcts[j][0] == "kcrps":
                            stddev_all[i_obs] = pred[:, mask_nan].std(0).mean().item()
                        # ignore NaNs so that training can continue even if one pred-net diverges
                        loss = loss + (
                            (w * val * obs_loss_weight)
                            if not torch.isnan(val)
                            else torch.tensor(0.0, requires_grad=True)
                        )
                    ctr_ftarget += 1

                    # log data for analysis
                    if log_data:
                        targets_lens[fstep][i_obs] += [target.shape[0]]
                        dn_data = self.dataset_val.denormalize_target_channels

                        f32 = torch.float32
                        preds_all[fstep][i_obs] += [dn_data(i_obs, pred.to(f32)).detach().cpu()]
                        targets_all[fstep][i_obs] += [dn_data(i_obs, target.to(f32)).detach().cpu()]

        # normalize by all targets and forecast steps that were non-empty
        # (with each having an expected loss of 1 for an uninitalized neural net)
        loss /= ctr_ftarget

        return (
            loss,
            None
            if not log_data
            else [
                preds_all,
                targets_all,
                targets_coords_raw_rt,
                targets_times_raw_rt,
                targets_lens,
            ],
        )

    ###########################################
    def train(self, epoch):
        cf = self.cf
        self.ddp_model.train()

        dataset_iter = iter(self.data_loader)

        self.optimizer.zero_grad()
        self.losses_hist, self.stddev_hist = [], []

        # training loop
        self.t_start = time.time()
        for bidx, batch in enumerate(dataset_iter):
            forecast_steps = batch[-1]
            batch = self.batch_to_device(batch)

            losses_all = torch.ones((len(self.loss_fcts_val), len(cf.streams))) * torch.nan
            stddev_all = torch.zeros(len(cf.streams)) * torch.nan

            # evaluate model
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=cf.with_mixed_precision
            ):
                preds = self.ddp_model(self.model_params, batch, forecast_steps)

                loss, _ = self.compute_loss(
                    self.loss_fcts,
                    forecast_steps,
                    batch[0],
                    preds,
                    losses_all,
                    stddev_all,
                )

            # backward pass
            self.grad_scaler.scale(loss).backward()

            # gradient clipping
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), max_norm=cf.grad_clip)

            # optimizer step
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.optimizer.zero_grad()

            # update learning rate
            self.lr_scheduler.step()

            self.losses_hist += [losses_all]
            self.stddev_hist += [stddev_all]

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
        self.losses_hist, self.stddev_hist = [], []

        with torch.no_grad():
            # print progress bar but only in interactive mode, i.e. when without ddp
            with tqdm.tqdm(
                total=len(self.data_loader_validation), disable=self.cf.with_ddp
            ) as pbar:
                for bidx, batch in enumerate(dataset_val_iter):
                    forecast_steps = batch[-1]
                    batch = self.batch_to_device(batch)

                    losses_all = torch.ones((len(self.loss_fcts_val), len(cf.streams))) * torch.nan
                    stddev_all = torch.zeros(len(cf.streams)) * torch.nan

                    # evaluate model
                    with torch.autocast(
                        device_type="cuda", dtype=torch.float16, enabled=cf.with_mixed_precision
                    ):
                        preds = self.ddp_model(self.model_params, batch, forecast_steps)

                    # compute loss and log output
                    if bidx < cf.log_validation:
                        _, ret = self.compute_loss(
                            self.loss_fcts_val,
                            forecast_steps,
                            batch[0],
                            preds,
                            losses_all,
                            stddev_all,
                            mode="validation",
                            log_data=True,
                        )

                        (
                            preds_all,
                            targets_all,
                            targets_coords_all,
                            targets_times_all,
                            targets_lens,
                        ) = ret
                        sources = [[item.source_raw for item in b] for b in batch[0]]
                        write_validation(
                            self.cf,
                            self.path_run,
                            self.cf.rank,
                            epoch,
                            sources,
                            preds_all,
                            targets_all,
                            targets_coords_all,
                            targets_times_all,
                            targets_lens,
                        )

                    else:
                        self.compute_loss(
                            self.loss_fcts_val,
                            forecast_steps,
                            batch[0],
                            preds,
                            losses_all,
                            stddev_all,
                            mode="validation",
                        )

                    self.losses_hist += [losses_all]
                    self.stddev_hist += [stddev_all]

                    pbar.update(self.cf.batch_size_validation)

                losses_all = self.ddp_average(
                    torch.stack(self.losses_hist).to(torch.float64).nanmean(0)
                )
                stddev_all = self.ddp_average(
                    torch.stack(self.stddev_hist).to(torch.float64).nanmean(0)
                )

                if self.cf.rank == 0 and self.cf.istep >= 0:
                    loss_dict = {}
                    for j, (lname, _) in enumerate(cf.loss_fcts_val):
                        loss_dict[f"validation {lname}"] = torch.nanmean(losses_all[j]).item()
                    loss_dict["validation std_dev"] = torch.nanmean(stddev_all.mean()).item()
                    for i_obs, rt in enumerate(cf.streams):
                        loss_dict["validation {}".format(rt["name"].replace(",", ""))] = float(
                            losses_all[0, i_obs]
                        )

                    # add data to plain logger
                    samples = cf.istep * cf.batch_size * cf.num_ranks
                    self.train_logger.add_val(samples, losses_all, stddev_all)

                if self.cf.rank == 0:
                    print(
                        f"validation ({cf.run_id}) : {epoch:03d} : loss = {torch.nanmean(losses_all[0]):.4E}",
                        flush=True,
                    )
                    for i_obs, rt in enumerate(cf.streams):
                        print("{}".format(rt["name"]) + f" : {losses_all[0, i_obs]:0.4E}")

        # avoid that there is a systematic bias in the validation subset
        self.dataset_val.advance()

    ###########################################
    def batch_to_device(self, batch):
        # forecast_steps is dropped here from the batch
        return (
            [[d.to_device() for d in db] for db in batch[0]],
            batch[1].to("cuda"),
            [[b.to("cuda") for b in bf] for bf in batch[2]],
        )

    ###########################################
    def save_model(self, epoch=-1, name=None):
        if self.cf.with_ddp and self.cf.with_fsdp:
            _cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                self.ddp_model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                state = self.ddp_model.state_dict()
        else:
            state = self.ddp_model.state_dict()

        if self.cf.rank == 0:
            filename = "".join(
                [
                    self.cf.run_id,
                    "_",
                    "latest" if epoch == -1 else f"epoch{epoch:05d}",
                    ("_" + name) if name is not None else "",
                ]
            )
            base_path = Path(self.cf.model_path) / self.cf.run_id
            file_out: Path = base_path / (filename + ".chkpt")
            file_tmp: Path = base_path / (filename + "_tmp.chkpt")
            # save temp file (slow)
            torch.save(state, file_tmp)
            # move file (which is changing the link in the file system and very fast)
            file_tmp.replace(file_out)

            # save config
            config.save(self.cf, epoch)

    ###########################################
    def log(self, bidx, epoch):
        if bidx % self.log_freq == 0 and bidx > 0:
            l_avg = self.ddp_average(torch.nanmean(torch.stack(self.losses_hist), axis=0))
            stddev_avg = self.ddp_average(torch.nanmean(torch.stack(self.stddev_hist), axis=0))
            samples = self.cf.istep * self.cf.batch_size * self.cf.num_ranks

            if self.cf.rank == 0:
                # logging
                loss_dict = {
                    "training mse": float(torch.nanmean(l_avg[0])),
                    "lr": self.lr_scheduler.get_lr(),
                }
                for i_obs, rt in enumerate(self.cf.streams):
                    loss_dict["training {}".format(rt["name"].replace(",", ""))] = float(
                        l_avg[0, i_obs]
                    )

                # plain logger
                self.train_logger.add_train(
                    samples,
                    self.lr_scheduler.get_lr(),
                    l_avg,
                    stddev_avg,
                    self.perf_gpu,
                    self.perf_mem,
                )

            self.losses_hist, self.stddev_hist = [], []

    ###########################################
    def log_terminal(self, bidx, epoch):
        if bidx % self.print_freq == 0 and bidx > 0:
            # compute from last iteration
            nanmean = torch.nanmean
            l_avg = self.ddp_average(
                nanmean(torch.stack(self.losses_hist[-self.print_freq :]), axis=0)
            )

            if self.cf.rank == 0:
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
                        "{}".format(rt["name"]) + f" : {l_avg[0, i_obs]:0.4E} \t",
                        end="",
                    )
                print("\n", flush=True)

            self.t_start = time.time()
