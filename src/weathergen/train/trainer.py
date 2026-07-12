# ruff: noqa: T201

# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import contextlib
import copy
import logging
import time
from math import sqrt

import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf

# FSDP2
from torch.distributed.tensor import DTensor

import weathergen.common.config as config
from weathergen.common.config import Config
from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler
from weathergen.model.ema import EMAModel
from weathergen.model.model_interface import (
    init_model_and_shard,
)
from weathergen.model.utils import apply_fct_to_blocks, set_to_eval
from weathergen.train.collapse_monitor import CollapseMonitor
from weathergen.train.loss_calculator import LossCalculator
from weathergen.train.lr_scheduler import LearningRateScheduler
from weathergen.train.target_and_aux_utils import get_target_aux_calculator
from weathergen.train.trainer_base import TrainerBase
from weathergen.train.utils import (
    TRAIN,
    VAL,
    NoOpGradScaler,
    Stage,
    cfg_keys_to_filter,
    extract_batch_metadata,
    filter_config_by_enabled,
    get_active_stage_config,
    get_batch_size_from_config,
    get_target_idxs_from_cfg,
)
from weathergen.utils.distributed import is_root
from weathergen.utils.performance import NullThroughputTracker, ThroughputTracker, nvtx_range
from weathergen.utils.train_logger import TrainLogger, prepare_losses_for_logging
from weathergen.utils.utils import get_dtype
from weathergen.utils.validation_io import write_output

logger = logging.getLogger(__name__)

# cfg_keys_to_filter = ["losses", "model_input", "target_input"]


class Trainer(TrainerBase):
    def __init__(self, train_logging: Config):
        TrainerBase.__init__(self)

        self.train_logging = train_logging

        self.data_loader: torch.utils.data.DataLoader | None = None
        self.data_loader_validation: torch.utils.data.DataLoader | None = None
        self.dataset: MultiStreamDataSampler | None = None
        self.dataset_val: MultiStreamDataSampler | None = None
        self.device: torch.device = None
        self.ema_model = None
        self.grad_scaler: torch.amp.GradScaler | NoOpGradScaler = NoOpGradScaler()
        self.last_grad_norm = None
        self.loss_calculator: LossCalculator | None = None
        self.loss_calculator_val: LossCalculator | None = None
        self.lr_schedulers: list[LearningRateScheduler] = []
        self.model = None
        self.model_params = None
        self.optimizers: list[torch.optim.Optimizer] = []
        self.optimizer_names: list[str] = []
        self._muon_effective_lr_factor: float | None = None
        self.t_start: float = 0
        self.target_and_aux_calculators = None
        self.target_and_aux_calculators_val = None
        self.validate_with_ema_cfg = None
        self.validate_with_ema: bool = False
        self.batch_size_per_gpu = -1
        self.batch_size_validation_per_gpu = -1
        self.batch_size_test_per_gpu = -1
        self.collapse_monitor: CollapseMonitor | None = None
        self.perf_tracker: ThroughputTracker | NullThroughputTracker = NullThroughputTracker()
        self.t_training_start: float = 0
        self.training_loop_annotation_context = contextlib.nullcontext

    def get_batch_size_total(self, batch_size_per_gpu) -> int:
        """
        Get total, effective batch size across all DDP ranks
        """
        return self.world_size_original * batch_size_per_gpu

    def _current_lrs(self) -> dict[str, float]:
        """
        Current lr of each optimizer, keyed by name (e.g. {"adamw": ...} or
        {"muon": ..., "adamw": ...}).

        For muon, "muon" is the actual applied lr: the scheduled lr multiplied by a
        representative (median, across muon params) adjust_lr_fn factor. torch.optim.Muon
        applies this factor per-parameter, inside step(), based on each matrix's shape -- it
        never updates param_groups[...]["lr"], so the raw scheduled lr alone (identical to
        adamw's, since both share the same schedule) understates the actual per-parameter step
        size by ~10-25x for this model's matrix sizes and isn't what's really applied.
        """
        lrs = {
            name: scheduler.get_lr()
            for name, scheduler in zip(self.optimizer_names, self.lr_schedulers, strict=True)
        }
        if "muon" in lrs and self._muon_effective_lr_factor is not None:
            lrs["muon"] = lrs["muon"] * self._muon_effective_lr_factor
        return lrs

    @staticmethod
    def _muon_adjust_lr_factor(shape, adjust_lr_fn: str) -> float:
        """
        Mirrors torch.optim.Muon's internal per-parameter lr-adjustment factor
        (torch/optim/_muon.py::_adjust_lr), so a representative effective lr can be logged.
        """
        a, b = shape[0], shape[1]
        if adjust_lr_fn == "match_rms_adamw":
            return 0.2 * max(a, b) ** 0.5
        return max(1.0, a / b) ** 0.5

    @staticmethod
    def _scale_lr_cfg(lr_cfg, lr_max_override: float | None):
        """
        Rescale a learning_rate_scheduling config to a different peak lr, keeping the
        warmup/decay/cooldown timing and relative shape identical.
        """
        if lr_max_override is None:
            return lr_cfg
        scale = lr_max_override / lr_cfg.lr_max
        return OmegaConf.merge(
            lr_cfg,
            {
                "lr_start": lr_cfg.lr_start * scale,
                "lr_max": lr_max_override,
                "lr_final_decay": lr_cfg.lr_final_decay * scale,
                "lr_final": lr_cfg.lr_final * scale,
            },
        )

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

        # get training config and remove disabled options (e.g. because of overrides)
        self.training_cfg = cf.get("training_config")
        self.training_cfg = filter_config_by_enabled(self.training_cfg, cfg_keys_to_filter)
        assert len(self.training_cfg.model_input.keys()) != 0, (
            "You probably have no loss term enabled"
        )

        # validation and test configs are training configs, updated by specified keys
        self.validation_cfg = get_active_stage_config(
            self.training_cfg, cf.get("validation_config", {}), cfg_keys_to_filter
        )
        # test cfg is derived from validation cfg with specified keys overwritten
        self.test_cfg = get_active_stage_config(
            self.validation_cfg, cf.get("test_config", {}), cfg_keys_to_filter
        )

        # batch sizes
        self.batch_size_per_gpu = get_batch_size_from_config(self.training_cfg)
        self.batch_size_validation_per_gpu = get_batch_size_from_config(self.validation_cfg)
        self.batch_size_test_per_gpu = get_batch_size_from_config(self.test_cfg)

        for mode, mode_cfg in zip(
            ["training_config", "validation_config", "test_config"],
            [self.training_cfg, self.validation_cfg, self.test_cfg],
            strict=True,
        ):
            config.validate_forecast_policy_and_steps(mode_cfg.get("forecast", {}), mode)

        self.mixed_precision_dtype = get_dtype(cf.mixed_precision_dtype)

        self.devices = devices

        # Get world_size of previous, to be continued run before
        # world_size gets overwritten by current setting during init_ddp()
        self.world_size_original = cf.get("world_size_original", cf.get("world_size", None))
        cf.world_size_original = self.world_size_original

        self.log_grad_norms = cf.train_logging.get("log_grad_norms", False)

        # create output directory
        if is_root():
            config.get_path_run(cf).mkdir(exist_ok=True, parents=True)
            config.get_path_model(cf).mkdir(exist_ok=True, parents=True)

        self.train_logger = TrainLogger(cf, config.get_path_run(self.cf))

        # Initialize collapse monitor for SSL training
        collapse_config = cf.train_logging.get("collapse_monitoring", {})
        self.collapse_monitor = CollapseMonitor(collapse_config, None)  # device set later in run()

        if cf.train_logging.get("track_performance_metrics"):
            self.perf_tracker = ThroughputTracker(
                device=torch.device(self.devices[0]),
                warmup_steps=cf.train_logging.get("performance_tracking_warmup_steps", 2),
                batch_size_per_gpu=self.batch_size_per_gpu,
            )
        if cf.get("profiling", {}).get("nvtx_annotate", False):
            self.training_loop_annotation_context = nvtx_range

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
            "pin_memory": cf.data_loading.get("memory_pinning", False),
            "persistent_workers": cf.data_loading.get("persistent_workers", False),
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
            cf.with_ddp,
            cf.with_fsdp,
        )

        # get target_aux calculators for different loss terms
        self.target_and_aux_calculators_val = self.get_target_aux_calculators(self.test_cfg)

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

        # Update collapse monitor device
        self.collapse_monitor.device = self.device

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
            cf.with_ddp,
            cf.with_fsdp,
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
                cf.with_ddp,
                cf.with_fsdp,
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
        self.target_and_aux_calculators_val = self.get_target_aux_calculators(self.validation_cfg)

        # if with_fsdp then parameter count is unreliable
        if is_root():
            # ddp-wrapped model does not expose this function
            if not cf.with_ddp:
                self.model.print_num_parameters()

        # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
        # aiming for beta1=0.9 and beta2=0.95 following the MAE paper
        # https://arxiv.org/pdf/2111.06377
        kappa = self.get_batch_size_total(self.batch_size_per_gpu)
        # aiming for beta1 = 0.9 at one node, ie kappa=B=4
        beta1 = max(0.5, 1.0 - kappa * (1.0 - self.training_cfg.optimizer.adamw.beta1))
        # aiming for beta2 = 0.95 at one node, ie B=4
        beta2 = max(0.9, 1.0 - kappa * (1.0 - self.training_cfg.optimizer.adamw.beta2))
        eps = self.training_cfg.optimizer.adamw.get("eps", 2e-08) / np.sqrt(kappa)

        shared_lr_cfg = self.training_cfg.learning_rate_scheduling
        weight_decay = self.training_cfg.optimizer.weight_decay
        adamw_betas_eps = {"betas": (beta1, beta2), "eps": eps}

        # per-optimizer lr scheduler config, paired 1:1 with self.optimizers
        lr_cfgs = []

        optimizer_name = self.training_cfg.optimizer.get("name", "adamw").lower()
        if optimizer_name == "adamw":
            self.optimizers = [
                torch.optim.AdamW(
                    self.model.parameters(),
                    lr=shared_lr_cfg.lr_start,
                    weight_decay=weight_decay,
                    fused=True,
                    **adamw_betas_eps,
                )
            ]
            lr_cfgs = [shared_lr_cfg]
            self.optimizer_names = ["adamw"]
        elif optimizer_name == "muon":
            # muon orthogonalizes 2D weight matrices of hidden layers; all other parameters
            # (biases, norms, higher-dim tensors, ...) are left to adamw, as recommended by
            # https://kellerjordan.github.io/posts/muon/ (torch.optim.Muon also hard-requires
            # exactly 2D tensors and raises ValueError otherwise, e.g. for a [1, 1, 2048] param)
            muon_cfg = self.training_cfg.optimizer.muon
            muon_params = [p for p in self.model.parameters() if p.requires_grad and p.ndim == 2]
            adamw_params = [p for p in self.model.parameters() if p.requires_grad and p.ndim != 2]
            muon_lr_cfg = self._scale_lr_cfg(shared_lr_cfg, muon_cfg.get("lr_max", None))
            adjust_lr_fn = muon_cfg.get("adjust_lr_fn", None) or "original"
            self._muon_effective_lr_factor = float(
                np.median(
                    [self._muon_adjust_lr_factor(p.shape, adjust_lr_fn) for p in muon_params]
                )
            )
            if is_root():
                logger.info(
                    f"Using muon optimizer: {len(muon_params)} params (ndim == 2) via muon, "
                    f"{len(adamw_params)} params (ndim != 2) via adamw, "
                    f"muon lr_max={muon_lr_cfg.lr_max:.3g} "
                    f"(adamw lr_max={shared_lr_cfg.lr_max:.3g}), "
                    f"median {adjust_lr_fn} factor={self._muon_effective_lr_factor:.3g}"
                )
            self.optimizers = [
                torch.optim.Muon(
                    muon_params,
                    lr=muon_lr_cfg.lr_start,
                    weight_decay=weight_decay,
                    momentum=muon_cfg.get("momentum", 0.95),
                    nesterov=muon_cfg.get("nesterov", True),
                    ns_steps=muon_cfg.get("ns_steps", 5),
                    eps=muon_cfg.get("eps", 1e-7),
                    adjust_lr_fn=muon_cfg.get("adjust_lr_fn", None),
                ),
                torch.optim.AdamW(
                    adamw_params,
                    lr=shared_lr_cfg.lr_start,
                    weight_decay=weight_decay,
                    fused=True,
                    **adamw_betas_eps,
                ),
            ]
            lr_cfgs = [muon_lr_cfg, shared_lr_cfg]
            self.optimizer_names = ["muon", "adamw"]
        else:
            assert False, f"Unsupported optimizer '{optimizer_name}', expected 'adamw' or 'muon'"

        if cf.get("training_config").get("optimizer").get("grad_scaling", True):
            self.grad_scaler = torch.amp.GradScaler("cuda")
        assert len(self.dataset) > 0, f"No data found in {self.dataset}"

        # lr is updated after each batch so account for this
        # TODO: conf should be read-only, do not modify the conf in flight
        len_ds = len(self.dataset)
        lr_steps = int((len_ds * self.training_cfg.num_mini_epochs) / self.batch_size_per_gpu)
        self.lr_schedulers = [
            LearningRateScheduler(
                optimizer,
                self.batch_size_per_gpu,
                cf.world_size,
                cf.general.istep,
                lr_steps,
                lr_cfg,
            )
            for optimizer, lr_cfg in zip(self.optimizers, lr_cfgs, strict=True)
        ]

        if self.cf.general.istep > 0 and is_root():
            logger.info(f"Continuing run with learning rate: {self.lr_schedulers[0].get_lr()}")

        # Instantiate loss calculator modules to compute losses
        self.loss_calculator = LossCalculator(cf, self.training_cfg, TRAIN, device=self.device)
        val_cfg = self.validation_cfg
        self.loss_calculator_val = LossCalculator(cf, val_cfg, VAL, device=self.device)

        # recover mini_epoch when continuing run
        if self.world_size_original is None:
            mini_epoch_base = int(self.cf.general.istep / len(self.data_loader))
        else:
            len_per_rank = (
                max(1, len(self.dataset) // (self.world_size_original * self.batch_size_per_gpu))
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
        self.t_training_start = time.time()

        for mini_epoch in range(mini_epoch_base, self.training_cfg.num_mini_epochs):
            if is_root():
                logger.info(
                    f"Mini_epoch {mini_epoch} of {self.training_cfg.num_mini_epochs}: train."
                )
            self.train(mini_epoch)

            if is_root():
                logger.info(
                    f"Mini_epoch {mini_epoch} of {self.training_cfg.num_mini_epochs}: validate."
                )
            self.validate(mini_epoch, self.validation_cfg, self.batch_size_validation_per_gpu)

            if is_root():
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

        apply_fct_to_blocks(self.model, cf.freeze_modules, set_to_eval)

        dataset_iter = iter(self.data_loader)

        for optimizer in self.optimizers:
            optimizer.zero_grad()

        # training loop
        self.t_start = time.time()
        for bidx, batch in enumerate(dataset_iter):
            with self.training_loop_annotation_context(f"batch_{bidx}"):
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
                        model_params=self.model_params,
                        batch=batch.get_source_samples(),
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
                    for _, target_aux in self.target_and_aux_calculators_val.items()
                ]

                # backward pass
                for optimizer in self.optimizers:
                    optimizer.zero_grad()
                self.grad_scaler.scale(loss).backward()

                # gradient clipping
                for optimizer in self.optimizers:
                    self.grad_scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.training_cfg.optimizer.grad_clip
                )

                # log gradient norms
                if self.log_grad_norms:
                    if bidx % self.train_logging.terminal == 0:
                        self.last_grad_norm = self._get_tensor_item(total_norm)
                    if bidx % self.train_logging.metrics == 0:
                        self._log_instant_grad_norms(TRAIN)

                # optimizer step
                for optimizer in self.optimizers:
                    self.grad_scaler.step(optimizer)
                self.grad_scaler.update()

                # update learning rate
                for lr_scheduler in self.lr_schedulers:
                    lr_scheduler.step()

                batch_size_total = self.get_batch_size_total(self.batch_size_per_gpu)
                step = batch_size_total * self.cf.general.istep

                [
                    target_aux.update_state_post_opt_step(step, batch, self.model)
                    for _, target_aux in self.target_and_aux_calculators.items()
                ]
                [
                    target_aux.update_state_post_opt_step(step, batch, self.model)
                    for _, target_aux in self.target_and_aux_calculators_val.items()
                ]

            # EMA update
            if self.validate_with_ema:
                self.ema_model.update(self.cf.general.istep * batch_size_total, batch_size_total)

            self.perf_tracker.step(
                batch,
                self.cf.general.istep,
                log_fn=lambda m: self.train_logger.log_metrics(
                    TRAIN, m, step=self.cf.general.istep
                ),
            )
            # Compute collapse monitoring metrics
            if self.collapse_monitor.should_compute(self.cf.general.istep):
                self.collapse_monitor._compute_collapse_metrics(
                    self.cf,
                    batch_size_total,
                    self.target_and_aux_calculators,
                    preds,
                    targets_and_auxs,
                )

            self._log_terminal(bidx, mini_epoch, TRAIN)
            if bidx % self.train_logging.metrics == 0:
                self._log(TRAIN)
                # Log collapse metrics
                if self.collapse_monitor.should_log(self.cf.general.istep):
                    self._log_collapse_metrics(TRAIN)

            # save model checkpoint (with designation _latest)
            if bidx % self.train_logging.checkpoint == 0 and bidx > 0:
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
                        if self.ema_model is None:
                            preds = self.model(
                                self.model_params,
                                batch.get_source_samples(),
                            )
                        else:
                            preds = self.ema_model.forward_eval(
                                self.model_params,
                                batch.get_source_samples(),
                            )

                        targets_and_auxs = {}
                        for loss_name, target_aux in self.target_and_aux_calculators_val.items():
                            target_idxs = get_target_idxs_from_cfg(mode_cfg, loss_name)
                            targets_and_auxs[loss_name] = target_aux.compute(
                                self.cf.general.istep,
                                batch.get_target_samples(target_idxs),
                                self.model_params,
                                self.model,
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
        full_state_dicts = []
        for optimizer in self.optimizers:
            sharded_sd = optimizer.state_dict()
            sharded_state = sharded_sd["state"]
            full_state = {}
            for group_id, sharded_group in sharded_state.items():
                group_state = {}
                for attr, sharded_tensor in sharded_group.items():
                    if isinstance(sharded_tensor, DTensor):
                        # "exp_avg" in AdamW / momentum buffer in Muon is `DTensor`
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
                full_state_dicts.append(
                    {
                        "param_groups": sharded_sd["param_groups"],
                        "state": full_state,
                    }
                )
        return full_state_dicts if is_rank_zero else []

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
                elapsed_time = time.time() - self.t_training_start
                self.train_logger.add_logs(
                    stage,
                    samples,
                    losses_all,
                    stddev_all,
                    avg_loss=avg_loss,
                    lr=self._current_lrs(),
                    elapsed_training_time_seconds=elapsed_time,
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
                grad_norms["grad_norm." + name] = self._get_tensor_item(
                    param.grad.norm() / sqrt(param.numel())
                )

        if is_root():
            self.train_logger.log_metrics(stage, grad_norms)

    def _log_terminal(self, bidx: int, mini_epoch: int, stage: Stage):
        print_freq = self.train_logging.terminal
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
                    lr_str = ", ".join(
                        f"{name}={lr:.2E}" for name, lr in self._current_lrs().items()
                    )
                    pstr = (
                        f"{mini_epoch:03d} : {bidx:05d}/{len_dataset:05d} : "
                        + f"{self.cf.general.istep:06d} : loss = {np.nanmean(avg_loss):.4E} "
                        + f"(lr={lr_str}, "
                    )
                    if self.log_grad_norms:
                        pstr += f"gradient norm={self.last_grad_norm:.3f}, "
                    pstr += f"s/sec={(print_freq * self.batch_size_per_gpu) / dt:.3f})"
                    logger.info(pstr)
                    logger.info("\t")

                for key, value in losses_all.items():
                    if key.endswith("avg"):
                        val = np.nan if np.isnan(value).all() else f"{np.nanmean(value):0.4E}"
                        logger.info(
                            f"{key} : {val} \t",
                        )
                logger.info("\n")

            self.t_start = time.time()

    def _log_collapse_metrics(self, stage: Stage) -> None:
        """
        Log cached collapse monitoring metrics.
        """
        metrics = self.collapse_monitor.get_cached_metrics()
        if metrics and is_root():
            metrics["num_samples"] = self.cf.general.istep
            self.train_logger.log_metrics(stage, metrics)
