# ruff: noqa: T201

# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import itertools
import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import tqdm
from omegaconf import OmegaConf
from torch import Tensor

# FSDP2
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.tensor import DTensor, distribute_tensor

import weathergen.common.config as config
from weathergen.common.config import Config
from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler
from weathergen.model.attention import (
    MultiCrossAttentionHeadVarlen,
    MultiCrossAttentionHeadVarlenSlicedQ,
    MultiSelfAttentionHead,
    MultiSelfAttentionHeadLocal,
    MultiSelfAttentionHeadVarlen,
)
from weathergen.model.ema import EMAModel
from weathergen.model.layers import MLP
from weathergen.model.model import Model, ModelParams
from weathergen.model.utils import freeze_weights
from weathergen.train.loss_calculator import LossCalculator
from weathergen.train.lr_scheduler import LearningRateScheduler
from weathergen.train.trainer_base import TrainerBase
from weathergen.utils.distributed import all_gather_vlen, ddp_average, is_root
from weathergen.utils.train_logger import TRAIN, VAL, Stage, TrainLogger
from weathergen.utils.utils import get_dtype
from weathergen.utils.validation_io import write_output

logger = logging.getLogger(__name__)


class Trainer(TrainerBase):
    def __init__(self, train_log_freq: Config):
        TrainerBase.__init__(self)

        self.train_log_freq = train_log_freq

    def init(self, cf: Config, devices):
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

        assert cf.samples_per_epoch % cf.batch_size_per_gpu == 0
        assert cf.samples_per_validation % cf.batch_size_validation_per_gpu == 0
        config.validate_forecast_policy_and_steps(cf=cf)

        self.mixed_precision_dtype = get_dtype(cf.attention_dtype)

        self.devices = devices

        # Get world_size of previous, to be continued run before
        # world_size gets overwritten by current setting during init_ddp()
        self.world_size_original = cf.get("world_size", None)

        # create output directory
        if is_root():
            config.get_path_run(cf).mkdir(exist_ok=True, parents=True)
            config.get_path_model(cf).mkdir(exist_ok=True, parents=True)

        self.init_perf_monitoring()
        self.train_logger = TrainLogger(cf, config.get_path_run(self.cf))

    def inference(self, cf, devices, run_id_trained, epoch):
        # general initalization
        self.init(cf, devices)

        cf = self.cf
        self.device_type = torch.accelerator.current_accelerator()
        self.device = torch.device(f"{self.device_type}:{cf.local_rank}")

        # !! modifies config: adds config.streams[i].<stage>_source_channels
        # and config.streams[i].<stage>_target_channels !!
        self.dataset_val = MultiStreamDataSampler(
            cf,
            cf.start_date_val,
            cf.end_date_val,
            cf.batch_size_validation_per_gpu,
            cf.samples_per_validation,
            stage=VAL,
            shuffle=cf.shuffle,
        )

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
            self.dataset_val, **loader_params, sampler=None
        )

        sources_size = self.dataset_val.get_sources_size()
        targets_num_channels = self.dataset_val.get_targets_num_channels()
        targets_coords_size = self.dataset_val.get_targets_coords_size()

        self.model = Model(cf, sources_size, targets_num_channels, targets_coords_size).create()
        self.model = self.model.to(self.devices[0])
        self.model.load(run_id_trained, epoch)
        logger.info(f"Loaded model {run_id_trained} at epoch {epoch}.")
        self.model_params = ModelParams(cf).create(cf)
        self.model_params = self.model_params.to(self.devices[0])
        logger.info(f"Loaded model id={run_id_trained} at epoch={epoch}.")

        self.loss_calculator_val = LossCalculator(cf=cf, stage=VAL, device=self.devices[0])

        if is_root():
            config.save(self.cf, epoch=0)

        logger.info(f"Starting inference with id={self.cf.run_id}.")

        # inference validation set
        self.validate(epoch=0)
        logger.info(f"Finished inference run with id: {cf.run_id}")

    def init_model_and_shard(self, cf, devices):
        sources_size = self.dataset.get_sources_size()
        targets_num_channels = self.dataset.get_targets_num_channels()
        targets_coords_size = self.dataset.get_targets_coords_size()

        with torch.device("meta"):
            model = Model(cf, sources_size, targets_num_channels, targets_coords_size).create()

        for name, module in model.named_modules():
            name = module.name if hasattr(module, "name") else name
            # avoid the whole model element which has name ''
            if name == "":
                continue
            if re.fullmatch(self.freeze_modules, name) is not None:
                freeze_weights(module)

        if cf.with_ddp and not cf.with_fsdp:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                broadcast_buffers=True,
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
                bucket_cap_mb=512,
            )

        if cf.with_ddp and cf.with_fsdp:
            fsdp_kwargs = {
                "mp_policy": (
                    MixedPrecisionPolicy(
                        param_dtype=self.mixed_precision_dtype,
                        reduce_dtype=torch.float32,
                    )
                    if cf.with_mixed_precision
                    else None
                ),
            }
            modules_to_shard = (
                MLP,
                MultiSelfAttentionHeadLocal,
                MultiSelfAttentionHead,
                MultiCrossAttentionHeadVarlen,
                MultiCrossAttentionHeadVarlenSlicedQ,
                MultiSelfAttentionHeadVarlen,
            )

            for module in model.ae_local_blocks.modules():
                if isinstance(module, modules_to_shard):
                    fully_shard(module, **fsdp_kwargs)

            for module in model.ae_adapter.modules():
                if isinstance(module, modules_to_shard):
                    fully_shard(module, **fsdp_kwargs)

            for module in model.ae_global_blocks.modules():
                if isinstance(module, modules_to_shard):
                    fully_shard(module, **fsdp_kwargs)

            for module in model.fe_blocks.modules():
                if isinstance(module, modules_to_shard):
                    fully_shard(module, **fsdp_kwargs)

            full_precision_fsdp_kwargs = {
                "mp_policy": (
                    MixedPrecisionPolicy(
                        param_dtype=torch.float32,
                        reduce_dtype=torch.float32,
                    )
                    if cf.with_mixed_precision
                    else None
                ),
            }
            for module in model.pred_adapter_kv.modules():
                if isinstance(module, modules_to_shard):
                    fully_shard(module, **full_precision_fsdp_kwargs)

            for module in model.target_token_engines.modules():
                if isinstance(module, modules_to_shard):
                    fully_shard(module, **full_precision_fsdp_kwargs)

        model_params = ModelParams(cf).create(cf)

        if cf.with_ddp and cf.with_fsdp:
            fully_shard(model)
            for tensor in itertools.chain(model.parameters(), model.buffers()):
                assert tensor.device == torch.device("meta")
        return model, model_params

    def run(self, cf, devices, run_id_contd=None, epoch_contd=None):
        # general initalization
        self.init(cf, devices)
        cf = self.cf

        # TODO: do not define new members outside of the init!!
        self.device_type = torch.accelerator.current_accelerator()
        self.device = torch.device(f"{self.device_type}:{cf.local_rank}")

        self.dataset = MultiStreamDataSampler(
            cf,
            cf.start_date,
            cf.end_date,
            cf.batch_size_per_gpu,
            cf.samples_per_epoch,
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

        self.model, self.model_params = self.init_model_and_shard(cf, devices)

        if run_id_contd is None:
            self.model.to_empty(device="cuda")
            self.model.reset_parameters()
        else:
            if is_root():
                logger.info(f"Continuing run with id={self.cf.from_run_id} at epoch {epoch_contd}.")
            self.load_model(self.cf.from_run_id, epoch_contd)
            if is_root():
                logger.info(f"Loaded model id={run_id_contd}.")
        self.model_params.reset_parameters(cf)
        self.model_params = self.model_params.to(self.device)

        if cf.compile_model:
            self.model = torch.compile(self.model, dynamic=True)

        self.validate_with_ema = cf.get("validate_with_ema", False)
        if self.validate_with_ema:
            meta_ema_model = self.init_model_and_shard(cf, devices)[0]
            self.ema_model = EMAModel(
                self.model,
                meta_ema_model,
                halflife_steps=cf.get("ema_halflife_in_thousands", 1e-3),
                rampup_ratio=cf.get("ema_ramp_up_ratio", 0.09),
                is_model_sharded=(cf.with_ddp and cf.with_fsdp),
            )

        # if with_fsdp then parameter count is unreliable
        if (is_root() and not cf.with_fsdp) or not cf.with_ddp:
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
        cf.lr_steps = int((len(self.dataset) * cf.num_epochs) / cf.batch_size_per_gpu)

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

        # recover epoch when continuing run
        if self.world_size_original is None:
            epoch_base = int(self.cf.istep / len(self.data_loader))
        else:
            len_per_rank = (
                len(self.dataset) // (self.world_size_original * cf.batch_size_per_gpu)
            ) * cf.batch_size_per_gpu
            epoch_base = int(
                self.cf.istep / (min(len_per_rank, cf.samples_per_epoch) * self.world_size_original)
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

        for epoch in range(epoch_base, cf.num_epochs):
            logger.info(f"Epoch {epoch} of {cf.num_epochs}: train.")
            self.train(epoch)

            logger.info(f"Epoch {epoch} of {cf.num_epochs}: validate.")
            self.validate(epoch)

            logger.info(f"Epoch {epoch} of {cf.num_epochs}: save_model.")
            self.save_model(epoch)

        # log final model
        self.save_model(cf.num_epochs)

    ###########################################
    def _prepare_logging(
        self,
        preds: list[list[Tensor]],
        forecast_offset: int,
        forecast_steps: int,
        streams_data: list[list[Any]],
    ):
        """Collects and denormalizes prediction and target data for logging.

        This function processes target and prediction tensors, extracts relevant
        coordinates and timestamps, denormalizes the data, and organizes it
        into a structured format suitable for logging or further analysis. It
        handles potential empty tensors and NaN values.

        Args:
            preds: A list of lists, where the outer list
                corresponds to forecast steps, and the inner list contains prediction
                tensors for each observation stream. Each prediction tensor is
                expected to be in the normalized latent or observation space,
                depending on the model's output.
            targets: A list of lists, where the outer list
                corresponds to forecast steps, and the inner list contains target
                tensors for each observation stream. Each target tensor is expected
                to be in the normalized observation space.
            forecast_offset: The starting offset for the forecast steps
                relative to the original data.
            forecast_steps: The number of forecast steps to consider.
            streams_data: A list of lists, where each inner list
                contains data objects (e.g., `BatchItem` instances) for each stream
                at a specific time step. These objects are expected to have
                `target_coords_raw` and `target_times_raw` attributes.

        Returns:
            tuple: A tuple containing:
                - preds_all: Denormalized
                predictions, organized by forecast step and observation stream.
                - targets_all: Denormalized
                targets, organized by forecast step and observation stream.
                - targets_coords_raw: Raw target coordinates,
                extracted and concatenated for each forecast step and stream.
                - targets_times_raw: Raw target timestamps,
                extracted and concatenated for each forecast step and stream.
                - targets_lens: A list of lists, where each
                inner list contains the original lengths (shape[0]) of the target
                tensors before any filtering.
        """

        #'''
        # TODO: Remove this function and port functionality to write_validation(), which then
        # extracts preds_all, targets_all,... itself directly from stream_data.
        # TODO: Undo list resorting
        # The following list operations realize a reshaping of the original tensors in streams_data
        # from shape [batch_sample][stream][fstep] into shape [fstep][stream][batch_sample]. When
        # removing the reshaping, make sure to index the tensors starting at forecast_offset, e.g.,
        # target_times_raw = streams_data[i_batch][i_strm].target_times_raw[forecast_offset+fstep],
        # when iterating over batch, stream, and fsteps.
        targets_rt = [
            [
                torch.cat([t[i].target_tokens[fstep] for t in streams_data])
                for i in range(len(self.cf.streams))
            ]
            for fstep in range(forecast_offset, forecast_offset + forecast_steps + 1)
        ]
        # TODO: Undo list resorting
        targets_coords_raw = [
            [
                torch.cat([t[i].target_coords_raw[fstep] for t in streams_data])
                for i in range(len(self.cf.streams))
            ]
            for fstep in range(forecast_offset, forecast_offset + forecast_steps + 1)
        ]
        # TODO: Undo list resorting
        targets_times_raw = [
            [
                np.concatenate([t[i].target_times_raw[fstep] for t in streams_data])
                for i in range(len(self.cf.streams))
            ]
            for fstep in range(forecast_offset, forecast_offset + forecast_steps + 1)
        ]

        # assert len(targets_rt) == len(preds) and len(preds) == len(self.cf.streams)
        fsteps = len(targets_rt)
        preds_all = [[[] for _ in self.cf.streams] for _ in range(fsteps)]
        targets_all = [[[] for _ in self.cf.streams] for _ in range(fsteps)]
        targets_lens = [[[] for _ in self.cf.streams] for _ in range(fsteps)]

        # TODO: iterate over batches here in future, and change loop order to batch, stream, fstep
        for fstep in range(len(targets_rt)):
            for i_strm, target in enumerate(targets_rt[fstep]):
                pred = preds[fstep][i_strm]

                if not (target.shape[0] > 0 and pred.shape[0] > 0):
                    continue

                # extract data/coords and remove token dimension if it exists
                pred = pred.reshape([pred.shape[0], *target.shape])
                assert pred.shape[1] > 0

                mask_nan = ~torch.isnan(target)
                if pred[:, mask_nan].shape[1] == 0:
                    continue

                targets_lens[fstep][i_strm] += [target.shape[0]]
                dn_data = self.dataset_val.denormalize_target_channels

                f32 = torch.float32
                preds_all[fstep][i_strm] += [dn_data(i_strm, pred.to(f32)).detach().cpu()]
                targets_all[fstep][i_strm] += [dn_data(i_strm, target.to(f32)).detach().cpu()]

        return (
            preds_all,
            targets_all,
            targets_coords_raw,
            targets_times_raw,
            targets_lens,
        )

    def train(self, epoch):
        cf = self.cf
        self.model.train()
        # torch.autograd.set_detect_anomaly(True)

        dataset_iter = iter(self.data_loader)

        self.optimizer.zero_grad()

        # Unweighted loss, real weighted loss, std for losses that need it
        self.loss_unweighted_hist, self.loss_model_hist, self.stdev_unweighted_hist = [], [], []

        # training loop
        self.t_start = time.time()
        for bidx, batch in enumerate(dataset_iter):
            forecast_steps = batch[-1]
            batch = self.batch_to_device(batch)

            # evaluate model
            with torch.autocast(
                device_type=f"cuda:{cf.local_rank}",
                dtype=self.mixed_precision_dtype,
                enabled=cf.with_mixed_precision,
            ):
                preds, posteriors = self.model(
                    self.model_params, batch, cf.forecast_offset, forecast_steps
                )
            loss_values = self.loss_calculator.compute_loss(
                preds=preds,
                streams_data=batch[0],
            )
            if cf.latent_noise_kl_weight > 0.0:
                kl = torch.cat([posterior.kl() for posterior in posteriors])
                loss_values.loss += cf.latent_noise_kl_weight * kl.mean()

            # backward pass
            self.optimizer.zero_grad()
            self.grad_scaler.scale(loss_values.loss).backward()
            # loss_values.loss.backward()

            # gradient clipping
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=cf.grad_clip)

            # optimizer step
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            # self.optimizer.step()

            # update learning rate
            self.lr_scheduler.step()

            # EMA update
            if self.validate_with_ema:
                self.ema_model.update(
                    self.cf.istep * self.world_size_original * self.cf.batch_size_per_gpu,
                    self.world_size_original * self.cf.batch_size_per_gpu,
                )

            self.loss_unweighted_hist += [loss_values.losses_all]
            self.loss_model_hist += [loss_values.loss.item()]
            self.stdev_unweighted_hist += [loss_values.stddev_all]

            perf_gpu, perf_mem = self.get_perf()
            self.perf_gpu = ddp_average(torch.tensor([perf_gpu], device=self.device)).item()
            self.perf_mem = ddp_average(torch.tensor([perf_mem], device=self.device)).item()

            self._log_terminal(bidx, epoch, TRAIN)
            if bidx % self.train_log_freq.metrics == 0:
                self._log(TRAIN)

            # save model checkpoint (with designation _latest)
            if bidx % self.train_log_freq.checkpoint == 0 and bidx > 0:
                self.save_model(-1)

            self.cf.istep += 1

        self.dataset.advance()

    def validate(self, epoch):
        cf = self.cf
        self.model.eval()

        dataset_val_iter = iter(self.data_loader_validation)
        self.loss_unweighted_hist, self.loss_model_hist, self.stdev_unweighted_hist = [], [], []

        with torch.no_grad():
            # print progress bar but only in interactive mode, i.e. when without ddp
            with tqdm.tqdm(
                total=len(self.data_loader_validation), disable=self.cf.with_ddp
            ) as pbar:
                for bidx, batch in enumerate(dataset_val_iter):
                    forecast_steps = batch[-1]
                    batch = self.batch_to_device(batch)

                    # evaluate model
                    with torch.autocast(
                        device_type=f"cuda:{cf.local_rank}",
                        dtype=self.mixed_precision_dtype,
                        enabled=cf.with_mixed_precision,
                    ):
                        preds, _ = self.ema_model.forward_eval(
                            self.model_params, batch, cf.forecast_offset, forecast_steps
                        )

                    # compute loss and log output
                    if bidx < cf.log_validation:
                        loss_values = self.loss_calculator_val.compute_loss(
                            preds=preds,
                            streams_data=batch[0],
                        )

                        # TODO: Move _prepare_logging into write_validation by passing streams_data
                        (
                            preds_all,
                            targets_all,
                            targets_coords_all,
                            targets_times_all,
                            targets_lens,
                        ) = self._prepare_logging(
                            preds=preds,
                            forecast_offset=cf.forecast_offset,
                            forecast_steps=cf.forecast_steps,
                            streams_data=batch[0],
                        )
                        sources = [[item.source_raw for item in b] for b in batch[0]]
                        write_output(
                            self.cf,
                            epoch,
                            bidx,
                            sources,
                            preds_all,
                            targets_all,
                            targets_coords_all,
                            targets_times_all,
                            targets_lens,
                        )

                    else:
                        loss_values = self.loss_calculator_val.compute_loss(
                            preds=preds,
                            streams_data=batch[0],
                        )

                    self.loss_unweighted_hist += [loss_values.losses_all]
                    self.loss_model_hist += [loss_values.loss.item()]
                    self.stdev_unweighted_hist += [loss_values.stddev_all]

                    pbar.update(self.cf.batch_size_validation_per_gpu)

                self._log_terminal(bidx, epoch, VAL)
                self._log(VAL)

        # avoid that there is a systematic bias in the validation subset
        self.dataset_val.advance()

    def batch_to_device(self, batch):
        # TODO: do not define new members outside of the init!!
        self.device_type = torch.accelerator.current_accelerator()
        self.device = torch.device(f"{self.device_type}:{self.cf.local_rank}")
        # forecast_steps is dropped here from the batch
        return (
            [[d.to_device(self.device) for d in db] for db in batch[0]],
            batch[1].to(self.device),
            [[b.to(self.device) for b in bf] for bf in batch[2]],
        )

    def load_model(self, run_id: str, epoch=-1):
        """Loads model state from checkpoint and checks for missing and unused keys.
        Args:
            run_id : model_id of the trained model
            epoch : The epoch to load. Default (-1) is the latest epoch
        """

        path_run = Path(self.cf.model_path) / run_id
        epoch_id = f"epoch{epoch:05d}" if epoch != -1 and epoch is not None else "latest"
        filename = f"{run_id}_{epoch_id}.chkpt"

        params = torch.load(
            path_run / filename, map_location=torch.device("cpu"), mmap=True, weights_only=True
        )

        model_state_dict = self.model.state_dict()
        params = {
            k: v
            for k, v in params.items()
            if k in model_state_dict and v.shape == model_state_dict[k].shape
        }

        is_model_sharded = self.cf.with_ddp and self.cf.with_fsdp
        if is_model_sharded:
            meta_sharded_sd = self.model.state_dict()
            maybe_sharded_sd = {}
            for param_name, full_tensor in params.items():
                sharded_meta_param = meta_sharded_sd.get(param_name)
                sharded_tensor = distribute_tensor(
                    full_tensor,
                    sharded_meta_param.device_mesh,
                    sharded_meta_param.placements,
                )
                maybe_sharded_sd[param_name.replace("module.", "")] = nn.Parameter(sharded_tensor)
        else:
            maybe_sharded_sd = {}
            for k in params.keys():
                maybe_sharded_sd[k.replace("module.", "")] = params[k]
        # choose `assign=True` for sharded model since we cannot call `copy_` on meta tensor
        mkeys, ukeys = self.model.load_state_dict(maybe_sharded_sd, strict=False, assign=True)

        if mkeys:
            # Get the unique parent modules for the missing parameters
            new_modules_to_init = {key.rsplit(".", 1)[0] for key in mkeys}

            # Find the highest-level "root" new modules to avoid redundant initializations
            root_new_modules = set()
            for path in sorted(list(new_modules_to_init)):
                if not any(path.startswith(root + ".") for root in root_new_modules):
                    root_new_modules.add(path)

            # Get all modules for quick lookup and initialize the new ones
            all_modules = dict(self.model.named_modules())
            for path in root_new_modules:
                if is_root():
                    logger.info(f"Initializing new module not found in checkpoint: {path}")
                module_to_init = all_modules[path]
                module_to_init.to_empty(device="cuda")
                module_to_init.reset_parameters()

        if not is_model_sharded:
            self.model = self.model.to(self.device)

        if len(mkeys) > 0:
            logger.warning(f"Missing keys when loading model: {mkeys}")

        if len(ukeys) > 0:
            logger.warning(f"Unused keys when loading model: {mkeys}")

    def load_optim(self):
        """
        TODO implement
        """
        pass
        # last_optim_checkpoint = (
        #     f"{self.folder}/{'dcp_api' if self.dcp_api else 'dtensor_api'}"
        #     f"/{self.last_training_time}/{OPTIM_CHECKPOINT}"
        # )
        # full_sd = torch.load(
        #     last_optim_checkpoint, mmap=True, weights_only=True, map_location="cpu"
        # )
        # _init_optim_state(opt)
        # param_groups = opt.state_dict()["param_groups"]
        # state = opt.state_dict()["state"]

        # full_param_groups = full_sd["param_groups"]
        # full_state = full_sd["state"]

        # for param_group, full_param_group in zip(param_groups, full_param_groups):
        #     for key, value in full_param_group.items():
        #         if key == PARAMS:
        #             continue
        #         param_group[key] = value
        #     for pid, full_pid in zip(param_group[PARAMS], full_param_group[PARAMS]):
        #         if pid not in state:
        #             continue
        #         param_state = state[pid]
        #         full_param_state = full_state[full_pid]
        #         for attr, full_tensor in full_param_state.items():
        #             sharded_tensor = param_state[attr]
        #             if isinstance(sharded_tensor, DTensor):
        #                 # exp_avg is DTensor
        #                 param_state[attr] = distribute_tensor(
        #                     full_tensor,
        #                     sharded_tensor.device_mesh,
        #                     sharded_tensor.placements,
        #                 )
        #             else:
        #                 # step is plain tensor
        #                 param_state[attr] = full_tensor
        # opt.load_state_dict(
        #     {
        #         "param_groups": param_groups,
        #         "state": state,
        #     }
        # )

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

    def save_model(self, epoch: int, name=None):
        # Saving at epoch == max_epoch means that we are saving the latest checkpoint.
        max_epoch = self.cf.num_epochs
        assert epoch <= max_epoch, (epoch, max_epoch)
        model_state_dict = self._get_full_model_state_dict()
        # optim_state_dict = self._get_full_optimizer_state_dict()

        if is_root():
            filename = "".join(
                [
                    self.cf.run_id,
                    "_",
                    "latest" if epoch == -1 else f"epoch{epoch:05d}",
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
            config.save(self.cf, epoch)

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

        for stream in self.cf.streams:  # Loop over all streams
            stream_hist = [losses_all[stream.name] for losses_all in self.loss_unweighted_hist]
            stream_all = torch.stack(stream_hist).to(torch.float64)
            losses_all[stream.name] = torch.cat(all_gather_vlen(stream_all))
            stream_hist = [stddev_all[stream.name] for stddev_all in self.stdev_unweighted_hist]
            stream_all = torch.stack(stream_hist).to(torch.float64)
            stddev_all[stream.name] = torch.cat(all_gather_vlen(stream_all))

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

        self.loss_unweighted_hist, self.loss_model_hist, self.stdev_unweighted_hist = [], [], []

    def _log_terminal(self, bidx: int, epoch: int, stage: Stage):
        print_freq = self.train_log_freq.terminal
        if bidx % print_freq == 0 and bidx > 0 or stage == VAL:
            # compute from last iteration
            avg_loss, losses_all, _ = self._prepare_losses_for_logging()

            if is_root():
                if stage == VAL:
                    logger.info(
                        f"validation ({self.cf.run_id}) : {epoch:03d} : {avg_loss.nanmean().item()}"
                    )
                    for _, st in enumerate(self.cf.streams):
                        logger.info(
                            "{}".format(st["name"])
                            + f" : {losses_all[st['name']].nanmean():0.4E} \t",
                        )
                    logger.info("\n")

                elif stage == TRAIN:
                    # samples per sec
                    dt = time.time() - self.t_start
                    pstr = "{:03d} : {:05d}/{:05d} : {:06d} : loss = {:.4E} "
                    pstr += "(lr={:.2E}, s/sec={:.3f})"
                    len_dataset = len(self.data_loader) // self.cf.batch_size_per_gpu
                    logger.info(
                        pstr.format(
                            epoch,
                            bidx,
                            len_dataset,
                            self.cf.istep,
                            avg_loss.nanmean().item(),
                            self.lr_scheduler.get_lr(),
                            (print_freq * self.cf.batch_size_per_gpu) / dt,
                        ),
                    )
                    logger.info("\t")
                    for _, st in enumerate(self.cf.streams):
                        logger.info(
                            "{}".format(st["name"])
                            + f" : {losses_all[st['name']].nanmean():0.4E} \t",
                        )
                    logger.info("\n")

            self.t_start = time.time()
