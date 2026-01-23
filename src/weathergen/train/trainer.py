# ruff: noqa: T201

# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import copy
import logging
import time

import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf

# FSDP2
from torch.distributed.tensor import DTensor

import weathergen.common.config as config
from weathergen.common.config import Config, merge_configs
from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler
from weathergen.model.ema import EMAModel
from weathergen.model.model_interface import (
    get_target_aux_calculator,
    init_model_and_shard,
)
from weathergen.train.loss_calculator import LossCalculator
from weathergen.train.lr_scheduler import LearningRateScheduler
from weathergen.train.trainer_base import TrainerBase
from weathergen.train.utils import (
    extract_batch_metadata,
    filter_config_by_enabled,
    get_batch_size_from_config,
    get_target_idxs_from_cfg,
)
from weathergen.utils.distributed import ddp_average, is_root
from weathergen.utils.train_logger import TRAIN, VAL, Stage, TrainLogger, prepare_losses_for_logging
from weathergen.utils.utils import get_dtype
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
        self.lr_scheduler: LearningRateScheduler | None = None
        self.model = None
        self.model_params = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.perf_gpu = None
        self.perf_mem = None
        self.t_start: float = 0
        self.target_and_aux_calculators = None
        self.validate_with_ema_cfg = None
        self.validate_with_ema: bool = False
        self.batch_size_per_gpu = -1
        self.batch_size_validation_per_gpu = -1
        self.batch_size_test_per_gpu = -1

    def get_batch_size_total(self, batch_size_per_gpu) -> int:
        """
        Get total, effective batch size across all DDP ranks
        """
        return self.world_size_original * batch_size_per_gpu

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

        # keys to filter for enabled/disabled
        keys_to_filter = ["losses", "model_input", "target_input"]

        # get training config and remove disabled options (e.g. because of overrides)
        self.training_cfg = cf.get("training_config")
        self.training_cfg = filter_config_by_enabled(self.training_cfg, keys_to_filter)

        # validation and test configs are training configs, updated by specified keys
        self.validation_cfg = merge_configs(self.training_cfg, cf.get("validation_config", {}))
        self.validation_cfg = filter_config_by_enabled(self.validation_cfg, keys_to_filter)
        # test cfg is derived from validation cfg with specified keys overwritten
        self.test_cfg = merge_configs(self.validation_cfg, cf.get("test_config", {}))
        self.test_cfg = filter_config_by_enabled(self.test_cfg, keys_to_filter)

        # batch sizes
        self.batch_size_per_gpu = get_batch_size_from_config(self.training_cfg)
        self.batch_size_validation_per_gpu = get_batch_size_from_config(self.validation_cfg)
        self.batch_size_test_per_gpu = get_batch_size_from_config(self.test_cfg)

        # config.validate_forecast_policy_and_steps(cf=cf)

        self.mixed_precision_dtype = get_dtype(cf.mixed_precision_dtype)

        self.devices = devices

        # Get world_size of previous, to be continued run before
        # world_size gets overwritten by current setting during init_ddp()
        self.world_size_original = cf.get("world_size_original", cf.get("world_size", None))
        cf.world_size_original = self.world_size_original

        self.log_grad_norms = self.training_cfg.optimizer.get("log_grad_norms", False)

        # create output directory
        if is_root():
            config.get_path_run(cf).mkdir(exist_ok=True, parents=True)
            config.get_path_model(cf).mkdir(exist_ok=True, parents=True)

        self.init_perf_monitoring()
        self.train_logger = TrainLogger(cf, config.get_path_run(self.cf))

    def get_target_aux_calculators(self, mode_cfg):
        """
        Get target_aux_calculators for given mode_cfg
        """

        batch_size = get_batch_size_from_config(mode_cfg)

        # get target_aux calculators for different loss terms
        target_and_aux_calculators = {}
        for loss_name, loss_cfg in mode_cfg.losses.items():
            target_and_aux_calculators[loss_name] = get_target_aux_calculator(
                self.cf, loss_cfg, self.dataset, self.model, self.device, batch_size
            ).to_device(self.device)

        return target_and_aux_calculators

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
            self.test_cfg,
            stage=VAL,
        )
        self.dataset_val = self.dataset

        # make sure number of loaders does not exceed requested samples
        loader_num_workers = min(self.test_cfg.samples_per_mini_epoch, cf.data_loading.num_workers)
        loader_params = {
            "batch_size": None,
            "batch_sampler": None,
            "shuffle": False,
            "num_workers": loader_num_workers,
        }
        self.data_loader_validation = torch.utils.data.DataLoader(
            self.dataset, **loader_params, sampler=None
        )

        self.model, self.model_params = init_model_and_shard(
            cf,
            self.dataset,
            run_id_contd,
            mini_epoch_contd,
            self.test_cfg.training_mode,
            devices[0],
        )

        # get target_aux calculators for different loss terms
        self.validate_with_ema_cfg = self.get_target_aux_calculators(self.test_cfg)

        self.loss_calculator_val = LossCalculator(cf, self.test_cfg, VAL, device=self.devices[0])

        if is_root():
            config.save(self.cf, mini_epoch=0)

        logger.info(f"Starting inference with id={self.cf.general.run_id}.")

        # inference validation set
        self.validate(0, self.test_cfg, self.batch_size_test_per_gpu)
        logger.info(f"Finished inference run with id: {cf.general.run_id}")

    def run(self, cf, devices, run_id_contd=None, mini_epoch_contd=None):
        # general initalization
        self.init(cf, devices)
        cf = self.cf

        device_type = torch.accelerator.current_accelerator()
        self.device = torch.device(f"{device_type}:{cf.local_rank}")

        # create data loaders
        self.dataset = MultiStreamDataSampler(cf, self.training_cfg, stage=TRAIN)
        self.dataset_val = MultiStreamDataSampler(cf, self.validation_cfg, stage=VAL)

        loader_params = {
            "batch_size": None,
            "batch_sampler": None,
            "shuffle": False,
            "num_workers": cf.data_loading.num_workers,
        }
        self.data_loader = torch.utils.data.DataLoader(self.dataset, **loader_params, sampler=None)
        self.data_loader_validation = torch.utils.data.DataLoader(
            self.dataset_val, **loader_params, sampler=None
        )

        self.model, self.model_params = init_model_and_shard(
            cf,
            self.dataset,
            run_id_contd,
            mini_epoch_contd,
            self.training_cfg.training_mode,
            devices[0],
        )

        validate_with_ema_cfg = self.validation_cfg.get("validate_with_ema")
        if validate_with_ema_cfg is not None:
            # if the config is specified and enabled not specified, then assume it is to be used
            self.validate_with_ema = validate_with_ema_cfg.get("enabled", True)
        else:
            self.validate_with_ema = False
        self.ema_model = None
        if self.validate_with_ema:
            meta_ema_model, _ = init_model_and_shard(
                cf,
                self.dataset,
                run_id_contd,
                mini_epoch_contd,
                cf.training_config.training_mode,
                devices[0],
                {},
            )
            self.ema_model = EMAModel(
                self.model,
                meta_ema_model,
                halflife_steps=validate_with_ema_cfg.get("ema_halflife_in_thousands", 1e-3),
                rampup_ratio=validate_with_ema_cfg.get("ema_ramp_up_ratio", 0.09),
                is_model_sharded=(cf.with_ddp and cf.with_fsdp),
            )

        # get target_aux calculators for different loss terms
        self.target_and_aux_calculators = self.get_target_aux_calculators(self.training_cfg)
        self.validate_with_ema_cfg = self.get_target_aux_calculators(self.validation_cfg)

        # if with_fsdp then parameter count is unreliable
        if is_root():
            if cf.with_fsdp:
                logger.warning("Trainable parameters are inaccurate with FSDP enabled.")
            self.model.print_num_parameters()

        # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
        # aiming for beta1=0.9 and beta2=0.95 following the MAE paper
        # https://arxiv.org/pdf/2111.06377
        kappa = self.get_batch_size_total(self.batch_size_per_gpu)
        # aiming for beta1 = 0.9 at one node, ie kappa=B=4
        beta1 = max(0.5, 1.0 - kappa * (1.0 - self.training_cfg.optimizer.adamw.beta1))
        # aiming for beta2 = 0.95 at one node, ie B=4
        beta2 = 1.0 - kappa * (1.0 - self.training_cfg.optimizer.adamw.beta2)
        eps = self.training_cfg.optimizer.adamw.get("eps", 2e-08) / np.sqrt(kappa)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_cfg.learning_rate_scheduling.lr_start,
            weight_decay=self.training_cfg.optimizer.weight_decay,
            betas=(beta1, beta2),
            eps=eps,
        )
        self.grad_scaler = torch.amp.GradScaler("cuda")

        assert len(self.dataset) > 0, f"No data found in {self.dataset}"

        # lr is updated after each batch so account for this
        # TODO: conf should be read-only, do not modify the conf in flight
        len_ds = len(self.dataset)
        lr_steps = int((len_ds * self.training_cfg.num_mini_epochs) / self.batch_size_per_gpu)
        self.lr_scheduler = LearningRateScheduler(
            self.optimizer,
            self.batch_size_per_gpu,
            cf.world_size,
            cf.general.istep,
            lr_steps,
            self.training_cfg.learning_rate_scheduling,
        )

        if self.cf.general.istep > 0 and is_root():
            str = f"Continuing run with learning rate: {self.lr_scheduler.get_lr()}"
            if is_root():
                logger.info(str)

        # Instantiate loss calculator modules to compute losses
        self.loss_calculator = LossCalculator(cf, self.training_cfg, TRAIN, device=self.device)
        val_cfg = self.validation_cfg
        self.loss_calculator_val = LossCalculator(cf, val_cfg, VAL, device=self.device)

        # recover mini_epoch when continuing run
        if self.world_size_original is None:
            mini_epoch_base = int(self.cf.general.istep / len(self.data_loader))
        else:
            len_per_rank = (
                len(self.dataset) // (self.world_size_original * self.batch_size_per_gpu)
            ) * self.batch_size_per_gpu
            mini_epoch_base = int(
                self.cf.general.istep
                / (
                    min(len_per_rank, self.training_cfg.samples_per_mini_epoch)
                    * self.world_size_original
                )
            )

        if is_root():
            config.save(self.cf, None)
            logger.info(config.format_cf(self.cf))

        # run validation before training if requested
        self.validate_before_training()

        # training loop

        for mini_epoch in range(mini_epoch_base, self.training_cfg.num_mini_epochs):
            logger.info(f"Mini_epoch {mini_epoch} of {self.training_cfg.num_mini_epochs}: train.")
            self.train(mini_epoch)

            logger.info(
                f"Mini_epoch {mini_epoch} of {self.training_cfg.num_mini_epochs}: validate."
            )
            self.validate(mini_epoch, self.validation_cfg, self.batch_size_validation_per_gpu)

            logger.info(
                f"Mini_epoch {mini_epoch} of {self.training_cfg.num_mini_epochs}: save_model."
            )
            self.save_model(mini_epoch)

        # log final model
        self.save_model(self.training_cfg.num_mini_epochs)

    def validate_before_training(self):
        """
        Perform validation before training (eg. to check validation pipeline or data normalization)
        if config parameters are set accordingly
        """

        # validate once at the beginning as reference
        if self.validation_cfg.get("validate_before_training", None) is not None:
            validate_before_training = self.validation_cfg.get("validate_before_training")
            batch_size = self.batch_size_validation_per_gpu
            if type(validate_before_training) is bool:
                if validate_before_training:
                    self.validate(-1, self.validation_cfg, batch_size)
            elif type(validate_before_training) is int:
                if validate_before_training > 0:
                    cfg = copy.deepcopy(self.validation_cfg)
                    cfg.samples_per_mini_epoch = validate_before_training
                    self.validate(-1, cfg, batch_size)
            else:
                assert False, "validate_before_training must be integer or boolean."

    def train(self, mini_epoch):
        """
        Perform training for one epoch
        """

        cf = self.cf
        self.model.train()

        dataset_iter = iter(self.data_loader)

        self.optimizer.zero_grad()

        # training loop
        self.t_start = time.time()
        for bidx, batch in enumerate(dataset_iter):
            if cf.data_loading.get("memory_pinning", False):
                # pin memory for faster CPU-GPU transfer
                batch = batch.pin_memory()

            batch.to_device(self.device)

            with torch.autocast(
                device_type=f"cuda:{cf.local_rank}",
                dtype=self.mixed_precision_dtype,
                enabled=cf.with_mixed_precision,
            ):
                preds = self.model(
                    self.model_params,
                    batch.get_source_samples(),
                    self.training_cfg.window_offset_prediction,
                )

                targets_and_auxs = {}
                for loss_name, target_aux in self.target_and_aux_calculators.items():
                    # find targets for this target-aux calculator
                    target_idxs = get_target_idxs_from_cfg(self.training_cfg, loss_name)
                    # apply target-aux calculator
                    targets_and_auxs[loss_name] = target_aux.compute(
                        self.cf.general.istep,
                        batch.get_target_samples(target_idxs),
                        self.model_params,
                        self.model,
                        self.training_cfg.window_offset_prediction,
                    )

            loss = self.loss_calculator.compute_loss(
                preds=preds,
                targets_and_aux=targets_and_auxs,
                metadata=extract_batch_metadata(batch),
            )
            # TODO re-enable this, need to think on how to make it compatible with
            # student-teacher training
            # if cf.latent_noise_kl_weight > 0.0:
            #     kl = torch.cat([posterior.kl() for posterior in output.latent["posteriors"]])
            #     loss_values.loss += cf.latent_noise_kl_weight * kl.mean()

            [
                target_aux.update_state_pre_backward(self.cf.general.istep, batch, self.model)
                for _, target_aux in self.target_and_aux_calculators.items()
            ]
            [
                target_aux.update_state_pre_backward(self.cf.general.istep, batch, self.model)
                for _, target_aux in self.validate_with_ema_cfg.items()
            ]

            # backward pass
            self.optimizer.zero_grad()
            self.grad_scaler.scale(loss).backward()

            # gradient clipping
            self.grad_scaler.unscale_(self.optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.training_cfg.optimizer.grad_clip
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

            # update learning rate
            self.lr_scheduler.step()

            batch_size_total = self.get_batch_size_total(self.batch_size_per_gpu)
            step = batch_size_total * self.cf.general.istep
            [
                target_aux.update_state_post_opt_step(step, batch, self.model)
                for _, target_aux in self.target_and_aux_calculators.items()
            ]
            [
                target_aux.update_state_post_opt_step(step, batch, self.model)
                for _, target_aux in self.validate_with_ema_cfg.items()
            ]
            # EMA update
            if self.validate_with_ema:
                self.ema_model.update(self.cf.general.istep * batch_size_total, batch_size_total)

            perf_gpu, perf_mem = self.get_perf()
            self.perf_gpu = ddp_average(torch.tensor([perf_gpu], device=self.device)).item()
            self.perf_mem = ddp_average(torch.tensor([perf_mem], device=self.device)).item()

            self._log_terminal(bidx, mini_epoch, TRAIN)
            if bidx % self.train_log_freq.metrics == 0:
                self._log(TRAIN)

            # save model checkpoint (with designation _latest)
            if bidx % self.train_log_freq.checkpoint == 0 and bidx > 0:
                self.save_model(-1)

            self.cf.general.istep += 1

        self.dataset.advance()

    def validate(self, mini_epoch, mode_cfg, batch_size):
        """
        Perform validation / test computation as specified by mode_cfg
        """

        cf = self.cf
        self.model.eval()

        dataset_val_iter = iter(self.data_loader_validation)

        num_samples_write = mode_cfg.get("output", {}).get("num_samples", 0) * batch_size

        with torch.no_grad():
            # print progress bar but only in interactive mode, i.e. when without ddp
            with tqdm.tqdm(total=mode_cfg.samples_per_mini_epoch, disable=self.cf.with_ddp) as pbar:
                for bidx, batch in enumerate(dataset_val_iter):
                    if cf.data_loading.get("memory_pinning", False):
                        # pin memory for faster CPU-GPU transfer
                        batch = batch.pin_memory()

                    batch.to_device(self.device)

                    # evaluate model
                    with torch.autocast(
                        device_type=f"cuda:{cf.local_rank}",
                        dtype=self.mixed_precision_dtype,
                        enabled=cf.with_mixed_precision,
                    ):
                        if self.ema_model is None:
                            preds = self.model(
                                self.model_params,
                                batch.get_source_samples(),
                                mode_cfg.window_offset_prediction,
                            )
                        else:
                            preds = self.ema_model.forward_eval(
                                self.model_params,
                                batch.get_source_samples(),
                                mode_cfg.window_offset_prediction,
                            )

                        targets_and_auxs = {}
                        for loss_name, target_aux in self.validate_with_ema_cfg.items():
                            target_idxs = get_target_idxs_from_cfg(self.training_cfg, loss_name)
                            targets_and_auxs[loss_name] = target_aux.compute(
                                self.cf.general.istep,
                                batch.get_target_samples(target_idxs),
                                self.model_params,
                                self.model,
                                mode_cfg.window_offset_prediction,
                            )

                    _ = self.loss_calculator_val.compute_loss(
                        preds=preds,
                        targets_and_aux=targets_and_auxs,
                        metadata=extract_batch_metadata(batch),
                    )

                    # log output
                    if bidx < num_samples_write:
                        # denormalization function for data
                        denormalize_data_fct = (
                            (lambda x0, x1: x1)
                            if mode_cfg.get("output", {}).get("normalized_samples", False)
                            else self.dataset_val.denormalize_target_channels
                        )
                        # write output
                        write_output(
                            self.cf,
                            mode_cfg,
                            batch_size,
                            mini_epoch,
                            bidx,
                            denormalize_data_fct,
                            batch,
                            preds,
                            targets_and_auxs,
                        )

                    pbar.update(batch_size)

                    if (bidx * batch_size) > mode_cfg.samples_per_mini_epoch:
                        break

                self._log_terminal(0, mini_epoch, VAL)
                self._log(VAL)

        # avoid that there is a systematic bias in the validation subset
        self.dataset_val.advance()

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
        max_mini_epoch = self.training_cfg.num_mini_epochs
        assert mini_epoch <= max_mini_epoch, (mini_epoch, max_mini_epoch)
        model_state_dict = self._get_full_model_state_dict()

        if is_root():
            filename = "".join(
                [
                    self.cf.general.run_id,
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
        loss_calculator = self.loss_calculator_val if stage == VAL else self.loss_calculator
        avg_loss, losses_all, stddev_all = prepare_losses_for_logging(
            loss_calculator.loss_hist,
            loss_calculator.losses_unweighted_hist,
            loss_calculator.stddev_unweighted_hist,
        )

        samples = self.cf.general.istep * self.get_batch_size_total(self.batch_size_per_gpu)

        if is_root():
            # plain logger
            if stage == VAL:
                self.train_logger.add_logs(stage, samples, losses_all, stddev_all)

            elif self.cf.general.istep >= 0:
                self.train_logger.add_logs(
                    stage,
                    samples,
                    losses_all,
                    stddev_all,
                    avg_loss=avg_loss,
                    lr=self.lr_scheduler.get_lr(),
                    perf_gpu=self.perf_gpu,
                    perf_mem=self.perf_mem,
                )

        loss_calculator.loss_hist = []
        loss_calculator.losses_unweighted_hist = []
        loss_calculator.stddev_unweighted_hist = []

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
            loss_calculator = self.loss_calculator_val if stage == VAL else self.loss_calculator
            avg_loss, losses_all, _ = prepare_losses_for_logging(
                loss_calculator.loss_hist,
                loss_calculator.losses_unweighted_hist,
                loss_calculator.stddev_unweighted_hist,
            )

            if is_root():
                if stage == VAL:
                    logger.info(
                        f"""validation ({self.cf.general.run_id}) : {mini_epoch:03d} : 
                        {np.nanmean(avg_loss)}"""
                    )

                elif stage == TRAIN:
                    # samples per sec
                    dt = time.time() - self.t_start
                    len_dataset = len(self.data_loader) // self.batch_size_per_gpu
                    pstr = (
                        f"{mini_epoch:03d} : {bidx:05d}/{len_dataset:05d} : "
                        + f"{self.cf.general.istep:06d} : loss = {np.nanmean(avg_loss):.4E} "
                        + f"(lr={self.lr_scheduler.get_lr():.2E}, "
                    )
                    if self.log_grad_norms:
                        pstr += f"gradient norm={self.last_grad_norm:.3f}, "
                    pstr += f"s/sec={(print_freq * self.batch_size_per_gpu) / dt:.3f})"
                    logger.info(pstr)
                    logger.info("\t")

                for key, value in losses_all.items():
                    if key.endswith("avg"):
                        logger.info(
                            f"{key} : {np.nanmean(value):0.4E} \t",
                        )
                logger.info("\n")

            self.t_start = time.time()
