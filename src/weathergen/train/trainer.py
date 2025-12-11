# ruff: noqa: T201

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

import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf
from torch import Tensor

# FSDP2
from torch.distributed.tensor import DTensor

import weathergen.common.config as config
from weathergen.common.config import Config
from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler
from weathergen.model.ema import EMAModel
from weathergen.model.model_interface import (
    get_target_aux_calculator,
    init_model_and_shard,
)
from weathergen.train.loss_calculator import LossCalculator
from weathergen.train.lr_scheduler import LearningRateScheduler
from weathergen.train.trainer_base import TrainerBase
from weathergen.utils.distributed import all_gather_vlen, ddp_average, is_root
from weathergen.utils.train_logger import TRAIN, VAL, Stage, TrainLogger
from weathergen.utils.utils import get_batch_size, get_dtype
from weathergen.utils.validation_io import write_output

logger = logging.getLogger(__name__)


class Trainer(TrainerBase):
    def __init__(self, train_log_freq: Config):
        TrainerBase.__init__(self)

        self.train_log_freq = train_log_freq

        self.data_loader: torch.utils.data.DataLoader | None = None
        self.data_loader_validation: torch.utils.data.DataLoader | None = None
        self.dataset: MultiStreamDataSampler | None = None
        self.dataset_val: MultiStreamDataSampler | None = None
        self.device: torch.device = None
        self.ema_model = None
        self.grad_scaler: torch.amp.GradScaler | None = None
        self.last_grad_norm = None
        self.loss_calculator: LossCalculator | None = None
        self.loss_calculator_val: LossCalculator | None = None
        self.loss_model_hist = []
        self.loss_unweighted_hist: dict = {}
        self.lr_scheduler: LearningRateScheduler | None = None
        self.model = None
        self.model_params = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.perf_gpu = None
        self.perf_mem = None
        self.stdev_unweighted_hist: dict = {}
        self.t_start: float = 0
        self.target_and_aux_calculator = None
        self.validate_with_ema: bool = False

    def init(self, cf: Config, devices):
        # pylint: disable=attribute-defined-outside-init
        self.cf = OmegaConf.merge(
            OmegaConf.create(
                {
                    "latent_noise_kl_weight": 0.0,
                    "latent_noise_gamma": 2.0,
                    "latent_noise_use_additive_noise": False,
                    "latent_noise_deterministic_latents": True,
                    "latent_noise_saturate_encodings": 5,
                }
            ),
            cf,
        )
        cf = self.cf

        self.freeze_modules = cf.get("freeze_modules", "")

        assert cf.samples_per_mini_epoch % cf.batch_size_per_gpu == 0
        assert cf.samples_per_validation % cf.batch_size_validation_per_gpu == 0
        config.validate_forecast_policy_and_steps(cf=cf)

        self.mixed_precision_dtype = get_dtype(cf.attention_dtype)

        self.devices = devices

        # Get world_size of previous, to be continued run before
        # world_size gets overwritten by current setting during init_ddp()
        self.world_size_original = cf.get("world_size_original", cf.get("world_size", None))
        cf.world_size_original = self.world_size_original

        self.log_grad_norms = cf.get("log_grad_norms", False)

        # create output directory
        if is_root():
            config.get_path_run(cf).mkdir(exist_ok=True, parents=True)
            config.get_path_model(cf).mkdir(exist_ok=True, parents=True)

        self.init_perf_monitoring()
        self.train_logger = TrainLogger(cf, config.get_path_run(self.cf))

    def inference(self, cf, devices, run_id_contd, mini_epoch_contd):
        # general initalization
        self.init(cf, devices)

        cf = self.cf
        device_type = torch.accelerator.current_accelerator()
        self.device = torch.device(f"{device_type}:{cf.local_rank}")
        self.ema_model = None

        # create data loader
        # only one needed since we only run the validation code path
        self.dataset = MultiStreamDataSampler(
            cf,
            cf.start_date_val,
            cf.end_date_val,
            cf.batch_size_validation_per_gpu,
            cf.samples_per_validation,
            stage=VAL,
            shuffle=cf.shuffle,
        )
        self.dataset_val = self.dataset

        # make sure number of loaders does not exceed requested samples
        loader_num_workers = min(cf.samples_per_validation, cf.loader_num_workers)
        loader_params = {
            "batch_size": None,
            "batch_sampler": None,
            "shuffle": False,
            "num_workers": loader_num_workers,
            "pin_memory": True,
        }
        self.data_loader_validation = torch.utils.data.DataLoader(
            self.dataset, **loader_params, sampler=None
        )

        self.model, self.model_params = init_model_and_shard(
            cf, self.dataset, run_id_contd, mini_epoch_contd, "student", devices[0]
        )

        self.target_and_aux_calculator = get_target_aux_calculator(
            cf, self.dataset, self.model, self.device
        )
        self.target_and_aux_calculator.to_device(self.device)

        self.loss_calculator_val = LossCalculator(cf=cf, stage=VAL, device=self.devices[0])

        if is_root():
            config.save(self.cf, mini_epoch=0)

        logger.info(f"Starting inference with id={self.cf.run_id}.")

        # inference validation set
        self.validate(mini_epoch=0)
        logger.info(f"Finished inference run with id: {cf.run_id}")

    def run(self, cf, devices, run_id_contd=None, mini_epoch_contd=None):
        # general initalization
        self.init(cf, devices)
        cf = self.cf

        device_type = torch.accelerator.current_accelerator()
        self.device = torch.device(f"{device_type}:{cf.local_rank}")

        # create data loaders
        self.dataset = MultiStreamDataSampler(
            cf,
            cf.start_date,
            cf.end_date,
            cf.batch_size_per_gpu,
            cf.samples_per_mini_epoch,
            stage=TRAIN,
            shuffle=cf.shuffle,
        )
        self.dataset_val = MultiStreamDataSampler(
            cf,
            cf.start_date_val,
            cf.end_date_val,
            cf.batch_size_validation_per_gpu,
            cf.samples_per_validation,
            stage=VAL,
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

        self.model, self.model_params = init_model_and_shard(
            cf, self.dataset, run_id_contd, mini_epoch_contd, "student", devices[0]
        )

        if cf.compile_model:
            self.model = torch.compile(self.model, dynamic=True)

        self.validate_with_ema = cf.get("validate_with_ema", False)
        self.ema_model = None
        if self.validate_with_ema:
            # validate_with_ema is incompatible with student-teacher
            meta_ema_model = init_model_and_shard(
                cf, self.dataset, run_id_contd, mini_epoch_contd, "student", devices[0]
            )[0]
            self.ema_model = EMAModel(
                self.model,
                meta_ema_model,
                halflife_steps=cf.get("ema_halflife_in_thousands", 1e-3),
                rampup_ratio=cf.get("ema_ramp_up_ratio", 0.09),
                is_model_sharded=(cf.with_ddp and cf.with_fsdp),
            )

        self.target_and_aux_calculator = get_target_aux_calculator(
            cf, self.dataset, self.model, self.device
        )

        self.target_and_aux_calculator.to_device(self.device)

        # if with_fsdp then parameter count is unreliable
        if is_root() and not cf.with_fsdp and not cf.with_ddp:
            self.model.print_num_parameters()

        # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
        # aiming for beta1=0.9 and beta2=0.95 following the MAE paper https://arxiv.org/pdf/2111.06377
        kappa = (
            cf.batch_size_per_gpu * cf.world_size
        )  # I doubt this holds for us from some anecdotal runs
        beta1 = max(
            0.5, 1.0 - kappa * (1.0 - 0.975)
        )  # aiming for beta1 = 0.9 at one node, ie kappa=B=4
        beta2 = 1.0 - kappa * (1.0 - 0.9875)  # aiming for beta2 = 0.95 at one node, ie B=4
        eps = 2e-08 / np.sqrt(kappa)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cf.lr_start,
            weight_decay=cf.weight_decay,
            betas=(beta1, beta2),
            eps=eps,
        )
        self.grad_scaler = torch.amp.GradScaler("cuda")

        assert len(self.dataset) > 0, f"No data found in {self.dataset}"

        # lr is updated after each batch so account for this
        # TODO: conf should be read-only, do not modify the conf in flight
        cf.lr_steps = int((len(self.dataset) * cf.num_mini_epochs) / cf.batch_size_per_gpu)

        steps_decay = cf.lr_steps - cf.lr_steps_warmup - cf.lr_steps_cooldown
        if is_root():
            logger.debug(f"steps_decay={steps_decay} lr_steps={cf.lr_steps}")
        # ensure that steps_decay has a reasonable value
        if steps_decay < int(0.2 * cf.lr_steps):
            cf.lr_steps_warmup = int(0.1 * cf.lr_steps)
            cf.lr_steps_cooldown = int(0.05 * cf.lr_steps)
            steps_decay = cf.lr_steps - cf.lr_steps_warmup - cf.lr_steps_cooldown
            s = (
                "cf.lr_steps_warmup and cf.lr_steps_cooldown",
                f" were larger than cf.lr_steps={cf.lr_steps}",
            )
            s += (
                f". The value have been adjusted to cf.lr_steps_warmup={cf.lr_steps_warmup} and ",
            )
            s += (
                f" cf.lr_steps_cooldown={cf.lr_steps_cooldown} so that steps_decay={steps_decay}.",
            )
            if is_root():
                logger.warning(s)
        self.lr_scheduler = LearningRateScheduler(
            self.optimizer,
            cf.batch_size_per_gpu,
            cf.world_size,
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

        if self.cf.istep > 0 and is_root():
            str = f"Continuing run with learning rate: {self.lr_scheduler.get_lr()}"
            if is_root():
                logger.info(str)

        # Instantiate loss calculator modules to compute losses
        self.loss_calculator = LossCalculator(cf=cf, stage=TRAIN, device=self.device)
        self.loss_calculator_val = LossCalculator(cf=cf, stage=VAL, device=self.device)

        # recover mini_epoch when continuing run
        if self.world_size_original is None:
            mini_epoch_base = int(self.cf.istep / len(self.data_loader))
        else:
            len_per_rank = (
                len(self.dataset) // (self.world_size_original * cf.batch_size_per_gpu)
            ) * cf.batch_size_per_gpu
            mini_epoch_base = int(
                self.cf.istep
                / (min(len_per_rank, cf.samples_per_mini_epoch) * self.world_size_original)
            )

        # torch.autograd.set_detect_anomaly(True)
        if cf.forecast_policy is not None:
            torch._dynamo.config.optimize_ddp = False

        if is_root():
            config.save(self.cf, None)
            logger.info(config.format_cf(self.cf))

        # training loop

        # validate once at the beginning as reference
        if cf.val_initial:
            self.validate(-1)

        for mini_epoch in range(mini_epoch_base, cf.num_mini_epochs):
            logger.info(f"Mini_epoch {mini_epoch} of {cf.num_mini_epochs}: train.")
            self.train(mini_epoch)

            logger.info(f"Mini_epoch {mini_epoch} of {cf.num_mini_epochs}: validate.")
            self.validate(mini_epoch)

            logger.info(f"Mini_epoch {mini_epoch} of {cf.num_mini_epochs}: save_model.")
            self.save_model(mini_epoch)

        # log final model
        self.save_model(cf.num_mini_epochs)

    def train(self, mini_epoch):
        cf = self.cf
        self.model.train()

        dataset_iter = iter(self.data_loader)

        self.optimizer.zero_grad()

        # training loop
        self.t_start = time.time()
        for bidx, batch in enumerate(dataset_iter):
            batch.to_device(self.device)

            with torch.autocast(
                device_type=f"cuda:{cf.local_rank}",
                dtype=self.mixed_precision_dtype,
                enabled=cf.with_mixed_precision,
            ):
                outputs = []
                for sample in batch.source_samples:
                    outputs.append(
                        self.model(
                            self.model_params,
                            sample,
                            cf.forecast_offset,
                            sample.get_forecast_steps(),
                        )
                    )

                targets_and_auxs = []
                for sample in batch.target_samples:
                    targets_and_auxs.append(
                        self.target_and_aux_calculator.compute(
                            sample,
                            self.model_params,
                            self.model,
                            cf.forecast_offset,
                            sample.get_forecast_steps(),
                        )
                    )
                # targets, aux = zip(*targets_and_auxs)
            loss, loss_values = self.loss_calculator.compute_loss(
                preds=outputs[0],
                targets=targets_and_auxs[0],
                # TOOD: view_metadata has to be part of targets and/or preds
                # view_metadata=(batch[-1].source2target_matching_idxs,
                #                 [sample.meta_info for sample in batch[-1].source_samples],
                #                 batch[-1].target2source_matching_idxs,
                #                 [sample.meta_info for sample in batch[-1].target_samples]
                #                ),
            )

            # TODO re-enable this, need to think on how to make it compatible with
            # TODO: CL, this should become a regular loss term
            # student-teacher training
            # if cf.latent_noise_kl_weight > 0.0:
            #     kl = torch.cat([posterior.kl() for posterior in output.latent["posteriors"]])
            #     loss_values.loss += cf.latent_noise_kl_weight * kl.mean()

            self.target_and_aux_calculator.update_state_pre_backward(
                self.cf.istep, batch, self.model
            )

            # backward pass
            self.optimizer.zero_grad()
            self.grad_scaler.scale(loss).backward()
            # loss_values.loss.backward()

            # gradient clipping
            self.grad_scaler.unscale_(self.optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=cf.grad_clip
            )

            # log gradient norms
            if self.log_grad_norms:
                if bidx % self.train_log_freq.terminal == 0:
                    self.last_grad_norm = self._get_tensor_item(total_norm)
                if bidx % self.train_log_freq.metrics == 0:
                    self._log_instant_grad_norms(TRAIN)

            # optimizer step
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            self.target_and_aux_calculator.update_state_post_opt_step(bidx, batch, self.model)

            # update learning rate
            self.lr_scheduler.step()

            # EMA update
            if self.validate_with_ema:
                self.ema_model.update(
                    self.cf.istep * get_batch_size(self.cf, self.world_size_original),
                    get_batch_size(self.cf, self.world_size_original),
                )

            # Collecting loss statistics for later inspection
            if bidx == 0:
                self.loss_unweighted_hist = {
                    loss_name: []
                    for _, calc_terms in loss_values.loss_terms.items()
                    for loss_name in calc_terms.losses_all.keys()
                }
                self.stdev_unweighted_hist = {
                    loss_name: []
                    for _, calc_terms in loss_values.loss_terms.items()
                    for loss_name in calc_terms.stddev_all.keys()
                }
                self.loss_model_hist = []
            for _, loss_terms in loss_values.loss_terms.items():
                for loss_name, losses_all in loss_terms.losses_all.items():
                    self.loss_unweighted_hist[loss_name].append(losses_all)
                for loss_name, stddev_all in loss_terms.stddev_all.items():
                    self.stdev_unweighted_hist[loss_name].append(stddev_all)
            self.loss_model_hist += [loss.item()]

            perf_gpu, perf_mem = self.get_perf()
            self.perf_gpu = ddp_average(torch.tensor([perf_gpu], device=self.device)).item()
            self.perf_mem = ddp_average(torch.tensor([perf_mem], device=self.device)).item()

            self._log_terminal(bidx, mini_epoch, TRAIN)
            if bidx % self.train_log_freq.metrics == 0:
                self._log(TRAIN)
                self.loss_unweighted_hist = {
                    loss_name: []
                    for _, calc_terms in loss_values.loss_terms.items()
                    for loss_name in calc_terms.losses_all.keys()
                }
                self.stdev_unweighted_hist = {
                    loss_name: []
                    for _, calc_terms in loss_values.loss_terms.items()
                    for loss_name in calc_terms.stddev_all.keys()
                }
                self.loss_model_hist = []

            # save model checkpoint (with designation _latest)
            if bidx % self.train_log_freq.checkpoint == 0 and bidx > 0:
                self.save_model(-1)
                self.loss_unweighted_hist = {
                    loss_name: []
                    for _, calc_terms in loss_values.loss_terms.items()
                    for loss_name in calc_terms.losses_all.keys()
                }
                self.stdev_unweighted_hist = {
                    loss_name: []
                    for _, calc_terms in loss_values.loss_terms.items()
                    for loss_name in calc_terms.stddev_all.keys()
                }
                self.loss_model_hist = []

            self.cf.istep += 1

        self.dataset.advance()

    def validate(self, mini_epoch):
        cf = self.cf
        self.model.eval()

        dataset_val_iter = iter(self.data_loader_validation)

        with torch.no_grad():
            # print progress bar but only in interactive mode, i.e. when without ddp
            with tqdm.tqdm(
                total=len(self.data_loader_validation), disable=self.cf.with_ddp
            ) as pbar:
                for bidx, batch in enumerate(dataset_val_iter):
                    batch.to_device(self.device)

                    # evaluate model
                    with torch.autocast(
                        device_type=f"cuda:{cf.local_rank}",
                        dtype=self.mixed_precision_dtype,
                        enabled=cf.with_mixed_precision,
                    ):
                        model_forward = (
                            self.model.forward
                            if self.ema_model is None
                            else self.ema_model.forward_eval
                        )
                        sample = batch.source_samples[0]
                        output = model_forward(
                            self.model_params,
                            sample,
                            cf.forecast_offset,
                            sample.get_forecast_steps(),
                        )
                        sample = batch.target_samples[0]
                        target_aux_output = self.target_and_aux_calculator.compute(
                            sample,
                            self.model_params,
                            self.model,
                            cf.forecast_offset,
                            sample.get_forecast_steps(),
                        )
                    loss, loss_values = self.loss_calculator_val.compute_loss(
                        preds=output,
                        targets=target_aux_output,
                    )

                    # log output
                    if bidx < cf.log_validation:
                        dn_data = self.dataset_val.denormalize_target_channels
                        write_output(
                            self.cf, mini_epoch, bidx, dn_data, batch, output, target_aux_output
                        )

                    # Collecting loss statistics for later inspection
                    if bidx == 0:
                        self.loss_unweighted_hist = {
                            loss_name: []
                            for _, calc_terms in loss_values.loss_terms.items()
                            for loss_name in calc_terms.losses_all.keys()
                        }
                        self.stdev_unweighted_hist = {
                            loss_name: []
                            for _, calc_terms in loss_values.loss_terms.items()
                            for loss_name in calc_terms.stddev_all.keys()
                        }
                        self.loss_model_hist = []
                    for _, loss_terms in loss_values.loss_terms.items():
                        for loss_name, losses_all in loss_terms.losses_all.items():
                            self.loss_unweighted_hist[loss_name].append(losses_all)
                        for loss_name, stddev_all in loss_terms.stddev_all.items():
                            self.stdev_unweighted_hist[loss_name].append(stddev_all)
                    self.loss_model_hist += [loss.item()]

                    pbar.update(self.cf.batch_size_validation_per_gpu)

                self._log_terminal(bidx, mini_epoch, VAL)
                self._log(VAL)

        # avoid that there is a systematic bias in the validation subset
        self.dataset_val.advance()

    def batch_to_device(self, batch):
        device_type = torch.accelerator.current_accelerator()
        self.device = torch.device(f"{device_type}:{self.cf.local_rank}")
        # forecast_steps is dropped here from the batch
        return (
            [[d.to_device(self.device) for d in db] for db in batch[0]],
            [b.to(self.device) for b in batch[1]],
            [[b.to(self.device) for b in bf] for bf in batch[2]],
        )

    def _get_full_model_state_dict(self):
        maybe_sharded_sd = (
            self.model.state_dict() if self.ema_model is None else self.ema_model.state_dict()
        )
        if self.cf.with_ddp and self.cf.with_fsdp:
            cpu_state_dict = {}
            for param_name, sharded_param in maybe_sharded_sd.items():
                full_param = sharded_param.full_tensor()
                if is_root():
                    cpu_state_dict[param_name] = full_param.cpu()
                else:
                    del full_param
            return cpu_state_dict
        else:
            return maybe_sharded_sd

    def _get_full_optimizer_state_dict(self):
        is_rank_zero = is_root()
        sharded_sd = self.optimizer.state_dict()
        sharded_state = sharded_sd["state"]
        full_state = {}
        for group_id, sharded_group in sharded_state.items():
            group_state = {}
            for attr, sharded_tensor in sharded_group.items():
                if isinstance(sharded_tensor, DTensor):
                    # "exp_avg" in AdamW is `DTensor`
                    full_tensor = sharded_tensor.full_tensor()
                else:
                    # "step" in AdamW is plain tensor
                    full_tensor = sharded_tensor
                if is_rank_zero:
                    group_state[attr] = full_tensor.cpu()
                else:
                    del full_tensor
            if is_rank_zero:
                full_state[group_id] = group_state
            else:
                del group_state
        if is_rank_zero:
            return {
                "param_groups": sharded_sd["param_groups"],
                "state": full_state,
            }
        else:
            return {}

    def save_model(self, mini_epoch: int, name=None):
        # Saving at mini_epoch == max_mini_epoch means that we are saving the latest checkpoint.
        max_mini_epoch = self.cf.num_mini_epochs
        assert mini_epoch <= max_mini_epoch, (mini_epoch, max_mini_epoch)
        model_state_dict = self._get_full_model_state_dict()

        if is_root():
            filename = "".join(
                [
                    self.cf.run_id,
                    "_",
                    "latest" if mini_epoch == -1 else f"chkpt{mini_epoch:05d}",
                    ("_" + name) if name is not None else "",
                ]
            )
            base_path = config.get_path_model(self.cf)
            file_out = base_path / (filename + ".chkpt")
            file_tmp = base_path / (filename + "_tmp.chkpt")
            # save temp file (slow)
            torch.save(model_state_dict, file_tmp)
            # move file (which is changing the link in the file system and very fast)
            file_tmp.replace(file_out)
            if is_root():
                logger.info(f"Saved model to {file_out}")

            # save config
            config.save(self.cf, mini_epoch)

    def _prepare_losses_for_logging(
        self,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Aggregates across ranks loss and standard deviation data for logging.

        Returns:
            real_loss (torch.Tensor): The scalar loss used for backpropagation.
            losses_all (dict[str, torch.Tensor]): Dictionary mapping each stream name to its
                per-channel loss tensor.
            stddev_all (dict[str, torch.Tensor]): Dictionary mapping each stream name to its
                per-channel standard deviation tensor.
        """

        losses_all: dict[str, Tensor] = {}
        stddev_all: dict[str, Tensor] = {}

        # Make list of losses into a tensor. This is individual tensor per rank
        real_loss = torch.tensor(self.loss_model_hist, device=self.device)
        # Gather all tensors from all ranks into a list and stack them into one tensor again
        real_loss = torch.cat(all_gather_vlen(real_loss))

        for loss_name, loss_values in self.loss_unweighted_hist.items():
            loss_values = torch.stack(loss_values).to(torch.float64)
            losses_all[loss_name] = torch.cat(all_gather_vlen(loss_values))
        for stddev_name, stddev_values in self.stdev_unweighted_hist.items():
            stddev_values = torch.stack(stddev_values).to(torch.float64)
            stddev_all[stddev_name] = torch.cat(all_gather_vlen(stddev_values))

        return real_loss, losses_all, stddev_all

    def _log(self, stage: Stage):
        """
        Logs training or validation metrics.

        Args:
            stage: Stage Is it's VAL, logs are treated as validation logs.
                        If TRAIN, logs are treated as training logs

        Notes:
            - This method only executes logging on the main process (rank 0).
            - After logging, historical loss and standard deviation records are cleared.
        """
        avg_loss, losses_all, stddev_all = self._prepare_losses_for_logging()
        samples = self.cf.istep * self.cf.batch_size_per_gpu * self.cf.world_size

        if is_root():
            # plain logger
            if stage == VAL:
                self.train_logger.add_val(samples, losses_all, stddev_all)

            elif self.cf.istep >= 0:
                self.train_logger.add_train(
                    samples,
                    self.lr_scheduler.get_lr(),
                    avg_loss,
                    losses_all,
                    stddev_all,
                    self.perf_gpu,
                    self.perf_mem,
                )

    def _get_tensor_item(self, tensor):
        """
        When using FSDP2, tensor is a DTensor and we need full_tensor().item() instead of .item(),
        see here: https://gist.github.com/Kai-46/a9835ef3f36e76d06afee6c11f388144
        """
        return tensor.full_tensor().item() if isinstance(tensor, DTensor) else tensor.item()

    def _log_instant_grad_norms(self, stage: Stage):
        """
        Log instantaneous grad norms, we do not average because of the cost and because we want to
        measure the actual values.
        """
        grad_norms = {"grad_norm.total": self.last_grad_norm}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norms["grad_norm." + name] = self._get_tensor_item(param.grad.norm())

        if is_root():
            self.train_logger.log_metrics(stage, grad_norms)

    def _log_terminal(self, bidx: int, mini_epoch: int, stage: Stage):
        print_freq = self.train_log_freq.terminal
        if bidx % print_freq == 0 and bidx > 0 or stage == VAL:
            # compute from last iteration
            avg_loss, losses_all, _ = self._prepare_losses_for_logging()

            if is_root():
                if stage == VAL:
                    logger.info(
                        f"""validation ({self.cf.run_id}) : {mini_epoch:03d} : 
                        {avg_loss.nanmean().item()}"""
                    )
                    for loss_name, loss_values in losses_all.items():
                        logger.info(
                            f"{loss_name}" + f" : {loss_values.nanmean():0.4E} \t",
                        )
                    logger.info("\n")

                elif stage == TRAIN:
                    # samples per sec
                    dt = time.time() - self.t_start
                    len_dataset = len(self.data_loader) // self.cf.batch_size_per_gpu
                    pstr = (
                        f"{mini_epoch:03d} : {bidx:05d}/{len_dataset:05d} : "
                        + f"{self.cf.istep:06d} : loss = {avg_loss.nanmean().item():.4E} "
                        + f"(lr={self.lr_scheduler.get_lr():.2E}, "
                    )
                    if self.log_grad_norms:
                        pstr += f"gradient norm={self.last_grad_norm:.3f}, "
                    pstr += f"s/sec={(print_freq * self.cf.batch_size_per_gpu) / dt:.3f})"
                    logger.info(pstr)
                    logger.info("\t")
                    for loss_name, loss_values in losses_all.items():
                        logger.info(
                            f"{loss_name}" + f" : {loss_values.nanmean():0.4E} \t",
                        )
                    logger.info("\n")

            self.t_start = time.time()
