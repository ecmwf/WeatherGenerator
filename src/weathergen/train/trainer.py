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
from typing import Any

import numpy as np
import torch
import tqdm
from torch import Tensor
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,  # default_auto_wrap_policy,
)

import weathergen.utils.config as config
from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler
from weathergen.model.model import Model, ModelParams
from weathergen.train.loss_calculator import LossCalculator
from weathergen.train.lr_scheduler import LearningRateScheduler
from weathergen.train.trainer_base import TrainerBase
from weathergen.utils.config import Config, get_dtype
from weathergen.utils.distributed import all_gather_vlen, ddp_average, is_root
from weathergen.utils.train_logger import TRAIN, VAL, Stage, TrainLogger
from weathergen.utils.validation_io import write_output

_logger = logging.getLogger(__name__)


class Trainer(TrainerBase):
    def __init__(self, checkpoint_freq=250, print_freq=10):
        TrainerBase.__init__(self)

        self.checkpoint_freq = checkpoint_freq
        self.print_freq = print_freq

    def init(
        self,
        cf: Config,
    ):
        self.cf = cf

        assert cf.samples_per_epoch % cf.batch_size_per_gpu == 0
        assert cf.samples_per_validation % cf.batch_size_validation_per_gpu == 0
        assert cf.forecast_policy if cf.forecast_steps > 0 else True

        self.mixed_precision_dtype = get_dtype(cf.attention_dtype)

        self.devices = self.init_torch()

        # Get num_ranks of previous, to be continued run before
        # num_ranks gets overwritten by current setting during init_ddp()
        self.num_ranks_original = cf.get("num_ranks", None)

        # TODO remove num_ranks, rank, with_with ddp from config
        self.init_ddp(cf)

        # create output directory
        if is_root():
            config.get_path_run(cf).mkdir(exist_ok=True, parents=True)
            config.get_path_model(cf).mkdir(exist_ok=True, parents=True)

        self.init_perf_monitoring()
        self.train_logger = TrainLogger(cf, config.get_path_run(self.cf))

    def init_inference(self, cf: Config, run_id_trained: str, epoch: int, msds: MultiStreamDataSampler):
        """
        Initializes the trainer for inference mode. Does not write any config does not perform validation steps"""
        # general initalization
        self.init(cf)

        sources_size = msds.get_sources_size()
        targets_num_channels = msds.get_targets_num_channels()
        targets_coords_size = msds.get_targets_coords_size()

        self.model = Model(cf, sources_size, targets_num_channels, targets_coords_size).create()
        self.model = self.model.to(self.devices[0])
        self.model.load(run_id_trained, epoch)
        _logger.info(f"Loaded model {run_id_trained} at epoch {epoch}.")
        self.ddp_model = self.model
        self.model_params = ModelParams().create(cf).to(self.devices[0])
        _logger.info(f"Loaded model id={run_id_trained} at epoch={epoch}.")

    def inference(self, cf, run_id_trained, epoch):
        # general initalization
        self.init(cf)

        # !! modifies config: adds config.streams[i].<stage>_source_channels
        # and config.streams[i].<stage>_target_channels !!
        self.dataset_val = MultiStreamDataSampler(
            cf,
            cf.start_date_val,
            cf.end_date_val,
            cf.batch_size_validation_per_gpu,
            cf.samples_per_validation,
            train_logger=self.train_logger,
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
        _logger.info(f"Loaded model {run_id_trained} at epoch {epoch}.")
        self.ddp_model = self.model
        self.model_params = ModelParams().create(cf).to(self.devices[0])
        _logger.info(f"Loaded model id={run_id_trained} at epoch={epoch}.")

        self.loss_calculator_val = LossCalculator(cf=cf, stage=VAL, device=self.devices[0])

        if is_root():
            config.save(self.cf, epoch=0)

        _logger.info(f"Starting inference with id={self.cf.run_id}.")

        # inference validation set
        self.validate(epoch=0)
        _logger.info(f"Finished inference run with id: {cf.run_id}")

    def run(self, cf, run_id_contd=None, epoch_contd=None):
        # general initalization
        self.init(cf)

        self.dataset = MultiStreamDataSampler(
            cf,
            cf.start_date,
            cf.end_date,
            cf.batch_size_per_gpu,
            cf.samples_per_epoch,
            train_logger=self.train_logger,
            stage=TRAIN,
            shuffle=cf.shuffle,
        )
        self.dataset_val = MultiStreamDataSampler(
            cf,
            cf.start_date_val,
            cf.end_date_val,
            cf.batch_size_validation_per_gpu,
            cf.samples_per_validation,
            train_logger=self.train_logger,
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

        sources_size = self.dataset.get_sources_size()
        targets_num_channels = self.dataset.get_targets_num_channels()
        targets_coords_size = self.dataset.get_targets_coords_size()

        self.model = Model(cf, sources_size, targets_num_channels, targets_coords_size).create()
        # load model if specified
        if run_id_contd is not None:
            _logger.info(f"Continuing run with id={run_id_contd} at epoch {epoch_contd}.")
            self.model.load(run_id_contd, epoch_contd)
            _logger.info(f"Loaded model id={run_id_contd}.")

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
                else MixedPrecision(
                    param_dtype=self.mixed_precision_dtype, cast_forward_inputs=True
                )
            )
            mp = None
            self.ddp_model = FSDP(
                self.model,
                auto_wrap_policy=size_based_auto_wrap_policy,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                cpu_offload=None,
                sync_module_states=(run_id_contd is not None),
                mixed_precision=mp,
                use_orig_params=True,
            )

        self.model_params = ModelParams().create(cf).to("cuda")

        # if with_fsdp then parameter count is unreliable
        if (is_root() and not cf.with_fsdp) or not cf.with_ddp:
            self.model.print_num_parameters()

        # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
        # aiming for beta1=0.9 and beta2=0.95 following the MAE paper https://arxiv.org/pdf/2111.06377
        kappa = (
            cf.batch_size_per_gpu * cf.num_ranks
        )  # I doubt this holds for us from some anecdotal runs
        beta1 = max(
            0.5, 1.0 - kappa * (1.0 - 0.975)
        )  # aiming for beta1 = 0.9 at one node, ie kappa=B=4
        beta2 = 1.0 - kappa * (1.0 - 0.9875)  # aiming for beta2 = 0.95 at one node, ie B=4
        eps = 2e-08 / np.sqrt(kappa)

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
        cf.lr_steps = int((len(self.dataset) * cf.num_epochs) / cf.batch_size_per_gpu)

        steps_decay = cf.lr_steps - cf.lr_steps_warmup - cf.lr_steps_cooldown
        _logger.debug(f"steps_decay={steps_decay} lr_steps={cf.lr_steps}")
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
            _logger.warning(s)
        self.lr_scheduler = LearningRateScheduler(
            self.optimizer,
            cf.batch_size_per_gpu,
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

        if self.cf.istep > 0 and is_root():
            str = f"Continuing run with learning rate: {self.lr_scheduler.get_lr()}"
            _logger.info(str)

        # Instantiate loss calculator modules to compute losses
        self.loss_calculator = LossCalculator(cf=cf, stage=TRAIN, device=self.devices[0])
        self.loss_calculator_val = LossCalculator(cf=cf, stage=VAL, device=self.devices[0])

        # recover epoch when continuing run
        if self.num_ranks_original is None:
            epoch_base = int(self.cf.istep / len(self.data_loader))
        else:
            len_per_rank = (
                len(self.dataset) // (self.num_ranks_original * cf.batch_size_per_gpu)
            ) * cf.batch_size_per_gpu
            epoch_base = int(
                self.cf.istep / (min(len_per_rank, cf.samples_per_epoch) * self.num_ranks_original)
            )

        # torch.autograd.set_detect_anomaly(True)
        if cf.forecast_policy is not None:
            torch._dynamo.config.optimize_ddp = False

        if is_root():
            config.save(self.cf, None)
            _logger.info(config.format_cf(self.cf))

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
        self.ddp_model.train()
        log_interval = self.cf.train_log.log_interval

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
                device_type="cuda",
                dtype=self.mixed_precision_dtype,
                enabled=cf.with_mixed_precision,
            ):
                preds = self.ddp_model(self.model_params, batch, cf.forecast_offset, forecast_steps)
                loss_values = self.loss_calculator.compute_loss(
                    preds=preds,
                    streams_data=batch[0],
                )

            # backward pass
            self.grad_scaler.scale(loss_values.loss).backward()

            # gradient clipping
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), max_norm=cf.grad_clip)

            # optimizer step
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.optimizer.zero_grad()

            # update learning rate
            self.lr_scheduler.step()

            self.loss_unweighted_hist += [loss_values.losses_all]
            self.loss_model_hist += [loss_values.loss.item()]
            self.stdev_unweighted_hist += [loss_values.stddev_all]

            perf_gpu, perf_mem = self.get_perf()
            self.perf_gpu = ddp_average(torch.tensor([perf_gpu])).item()
            self.perf_mem = ddp_average(torch.tensor([perf_mem])).item()

            self._log_terminal(bidx, epoch, TRAIN)
            if bidx % log_interval == 0:
                self._log(TRAIN)

            # model checkpoint
            if bidx % self.checkpoint_freq == 0:
                self.save_model(-1)

            self.cf.istep += cf.batch_size_per_gpu

        self.dataset.advance()

    def validate(self, epoch):
        cf = self.cf
        self.ddp_model.eval()

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
                        device_type="cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=cf.with_mixed_precision,
                    ):
                        preds = self.ddp_model(
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
        # forecast_steps is dropped here from the batch
        return (
            [[d.to_device() for d in db] for db in batch[0]],
            batch[1].to("cuda"),
            [[b.to("cuda") for b in bf] for bf in batch[2]],
        )

    def save_model(self, epoch: int, name=None):
        # Saving at epoch == max_epoch means that we are saving the latest checkpoint.
        max_epoch = self.cf.num_epochs
        assert epoch <= max_epoch, (epoch, max_epoch)
        if self.cf.with_ddp and self.cf.with_fsdp:
            with FSDP.state_dict_type(
                self.ddp_model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                state = self.ddp_model.state_dict()
        else:
            state = self.ddp_model.state_dict()

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
            torch.save(state, file_tmp)
            # move file (which is changing the link in the file system and very fast)
            file_tmp.replace(file_out)
            _logger.info(f"Saved model to {file_out}")

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
        real_loss = torch.tensor(self.loss_model_hist, device=self.devices[0])
        # Gather all tensors from all ranks into a list and stack them into one tensor again
        real_loss = torch.cat(all_gather_vlen(real_loss))

        for stream in self.cf.streams:  # Loop over all steams
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
        samples = self.cf.istep * self.cf.batch_size_per_gpu * self.cf.num_ranks

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
        if bidx % self.print_freq == 0 and bidx > 0 or stage == VAL:
            # compute from last iteration
            avg_loss, losses_all, _ = self._prepare_losses_for_logging()

            if is_root():
                if stage == VAL:
                    print(
                        f"validation ({self.cf.run_id}) : {epoch:03d} : {avg_loss.nanmean().item()}"
                    )
                    for _, st in enumerate(self.cf.streams):
                        print(
                            "{}".format(st["name"])
                            + f" : {losses_all[st['name']].nanmean():0.4E} \t",
                            end="",
                        )
                    print("\n", flush=True)

                elif stage == TRAIN:
                    # samples per sec
                    dt = time.time() - self.t_start
                    pstr = "{:03d} : {:05d}/{:05d} : {:06d} : loss = {:.4E} "
                    pstr += "(lr={:.2E}, s/sec={:.3f})"
                    len_dataset = len(self.data_loader) // self.cf.batch_size_per_gpu
                    print(
                        pstr.format(
                            epoch,
                            bidx,
                            len_dataset,
                            self.cf.istep,
                            avg_loss.nanmean().item(),
                            self.lr_scheduler.get_lr(),
                            (self.print_freq * self.cf.batch_size_per_gpu) / dt,
                        ),
                        flush=True,
                    )
                    print("\t", end="")
                    for _, st in enumerate(self.cf.streams):
                        print(
                            "{}".format(st["name"])
                            + f" : {losses_all[st['name']].nanmean():0.4E} \t",
                            end="",
                        )
                    print("\n", flush=True)

            self.t_start = time.time()
