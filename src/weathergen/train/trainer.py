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
import json
import logging
import time
from collections import deque
from decimal import Decimal
from math import sqrt

import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf

# FSDP2
from torch.distributed.tensor import DTensor, distribute_tensor

import weathergen.common.config as config
from weathergen.common.config import Config
from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler
from weathergen.model.ema import EMAModel
from weathergen.model.model import ModelOutput
from weathergen.model.model_interface import (
    init_model_and_shard,
)
from weathergen.model.utils import apply_fct_to_blocks, set_to_eval
from weathergen.train.collapse_monitor import CollapseMonitor
from weathergen.train.loss_calculator import LossCalculator
from weathergen.train.lr_scheduler import LearningRateScheduler
from weathergen.train.optimizer import build_optimizer
from weathergen.train.target_and_aux_ssl_teacher import EMATeacher
from weathergen.train.target_and_aux_utils import get_target_aux_calculator
from weathergen.train.trainer_base import TrainerBase
from weathergen.train.utils import (
    TRAIN,
    VAL,
    Stage,
    cfg_keys_to_filter,
    extract_batch_metadata,
    filter_config_by_enabled,
    get_active_stage_config,
    get_batch_size_from_config,
    get_target_idxs_from_cfg,
)
from weathergen.utils.distributed import is_root
from weathergen.utils.performance import NullThroughputTracker, ThroughputTracker
from weathergen.utils.train_logger import TrainLogger, prepare_losses_for_logging
from weathergen.utils.utils import get_dtype
from weathergen.utils.validation_io import write_output

logger = logging.getLogger(__name__)

LOSS_SPIKE_DETECTION_DEFAULTS = {
    "enabled": False,
    "window_size": 50,
    "min_history": 20,
    "ratio_threshold": 5.0,
    "loss_threshold": 0.0,
    "skip_batch": True,
    "max_unique_times_per_step": 8,
    "file_name": "loss_spikes.jsonl",
}

# cfg_keys_to_filter = ["losses", "model_input", "target_input"]


def _expand_targets_to_match_preds(preds, targets_and_auxs: dict) -> None:
    """
    Replicate per-fstep entries in each TargetAuxOutput so its ``physical`` and ``latent``
    lists match the number of forecast steps in ``preds``.

    Diffusion inference produces one ``preds`` fstep per ODE denoising step, but the
    physical target is identical across the trajectory. Without this expansion the loss
    calculator (which zips preds and targets with ``strict=True``) raises a length
    mismatch.

    The expansion replicates references — no tensor copies are made — and is a no-op when
    the lengths already agree.
    """
    n_pred = len(preds.physical)

    def _expand_field(field: list[dict]) -> list[dict]:
        # Entries before the batch's forecast_offset are structural padding (empty
        # dicts), not real target data (see BatchSamples.output_idxs / TargetAuxOutput).
        # A naive index-proportional stretch (`i // (n_pred // n_tgt)`) replicates that
        # padding right alongside the real target, scattering empty dicts through the
        # expanded list instead of repeating the one target that is actually identical
        # across the whole trajectory. Diffusion inference supports a single physical
        # forecast step, so there is at most one real (non-empty) entry to replicate.
        real = [d for d in field if d]
        if len(real) > 1:
            logger.warning(
                "Cannot expand target/aux with %d real (non-empty) forecast steps to "
                "%d trajectory steps; leaving unchanged.",
                len(real),
                n_pred,
            )
            return field
        fill = real[0] if real else {}
        return [fill for _ in range(n_pred)]

    for t_aux in targets_and_auxs.values():
        n_tgt = len(t_aux.physical)
        if n_tgt == n_pred or n_tgt == 0:
            continue
        t_aux.physical = _expand_field(t_aux.physical)
        t_aux.latent = _expand_field(t_aux.latent)
        # output_idxs is consumed by validation IO via batch.get_output_idxs(), but we
        # keep the dataclass internally consistent in case other consumers read it.
        if t_aux.output_idxs is not None and len(t_aux.output_idxs) == 1:
            t_aux.output_idxs = [t_aux.output_idxs[0] for _ in range(n_pred)]


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
        self.grad_scaler: torch.amp.GradScaler | None = None
        self.last_grad_norm = None
        self.loss_calculator: LossCalculator | None = None
        self.loss_calculator_val: LossCalculator | None = None
        self.lr_schedulers: list[LearningRateScheduler] | None = None
        self.model = None
        self.model_params = None
        self.optimizers: list[torch.optim.Optimizer] | None = None
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
        self.loss_spike_cfg = None
        self.loss_spike_file = None
        self.loss_spike_history = deque()

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
        self._init_loss_spike_detection()

        # Initialize collapse monitor for SSL training
        collapse_config = cf.train_logging.get("collapse_monitoring", {})
        self.collapse_monitor = CollapseMonitor(collapse_config, None)  # device set later in run()

        if cf.train_logging.get("track_performance_metrics"):
            self.perf_tracker = ThroughputTracker(
                device=torch.device(self.devices[0]),
                warmup_steps=cf.train_logging.get("performance_tracking_warmup_steps", 2),
                batch_size_per_gpu=self.batch_size_per_gpu,
            )

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

    def _get_forecast_step_chunks(self, output_idxs: list[int], chunk_size: int) -> list[list[int]]:
        """Split the forecast steps into contiguous chunks of at most chunk_size steps."""
        assert chunk_size >= 1, f"forecast.chunk_size must be >= 1, got {chunk_size}."
        return [
            output_idxs[start : start + chunk_size]
            for start in range(0, len(output_idxs), chunk_size)
        ]

    def _offload_encoder_weights(self) -> None:
        """Move encoder parameter storage to CPU after the encoder has run.

        FSDP2 uses lazy all-gather: it only fetches parameters for modules that are
        actually called during a forward pass.  Continuation chunks (chunk_idx > 0)
        never call the encoder — they resume from the cached latent state — so FSDP2
        will never try to all-gather the encoder shards while they are on CPU.

        **FSDP2 DTensor parameters are intentionally skipped.**  Some encoder parameters
        (e.g. stream embedders in embed_engine) live in the *root* FSDP group rather than
        in individually-sharded sub-modules.  Moving their _local_tensor to CPU outside
        FSDP's knowledge causes FSDP's root pre-forward to trigger a spurious 1.95 GiB
        all-gather on the next chunk.  Safe offloading of DTensor params would require
        CPUOffload configured at shard time; we limit this method to the non-FSDP path.
        """
        try:
            from torch.distributed.tensor import DTensor

            base_model = getattr(self.model, "module", self.model)
            encoder = getattr(base_model, "encoder", None)
            if encoder is None:
                return
            for param in encoder.parameters():
                # DTensor = FSDP-managed shard — do NOT move; see docstring.
                if isinstance(param, DTensor):
                    continue
                if param.is_cuda:
                    param.data = param.data.cpu()
            torch.cuda.empty_cache()
        except Exception:
            pass  # best-effort; do not crash inference

    def _reload_encoder_weights(self) -> None:
        """Reload encoder parameter storage onto the inference device.

        Counterpart to ``_offload_encoder_weights()``.  Called just before chunk 0's
        model forward so that the encoder can run a fresh all-gather from its local
        (now GPU-resident) shards.  DTensor parameters are skipped for the same reason
        as in _offload_encoder_weights.
        """
        try:
            from torch.distributed.tensor import DTensor

            base_model = getattr(self.model, "module", self.model)
            encoder = getattr(base_model, "encoder", None)
            if encoder is None:
                return
            device = self.device
            for param in encoder.parameters():
                if isinstance(param, DTensor):
                    continue  # FSDP-managed; do not touch
                if not param.is_cuda:
                    param.data = param.data.to(device, non_blocking=True)
        except Exception:
            pass  # best-effort

    def _process_validation_chunks(
        self,
        batch,
        mode_cfg,
        batch_size,
        mini_epoch,
        bidx,
        targets_and_auxs,
        is_diffusion_trajectory: bool = False,
    ) -> ModelOutput:
        """Run the rollout in chunks and assemble the predictions for the whole batch."""
        forecast_cfg = mode_cfg.get("forecast", {})

        output_idxs = batch.get_output_idxs()
        # Trajectory-mode diffusion (diffusion_rollout=False) consumes the output's fstep
        # dimension for the ODE denoising trajectory, which neither the reassembly below nor a
        # per-chunk write can handle. Roll out in a single chunk and let validate() write the
        # output as it did before chunking. A diffusion *rollout* keeps the ordinary fstep
        # layout and carries its state across chunks, so it chunks like any other forecast.
        chunk_size = (
            len(output_idxs)
            if is_diffusion_trajectory
            else forecast_cfg.get("chunk_size", len(output_idxs))
        )
        chunks = self._get_forecast_step_chunks(output_idxs, chunk_size)

        num_samples_write = mode_cfg.get("output", {}).get("num_samples", 0) * batch_size
        # This assumes that you only have a physical loss, else this breaks
        compute_full_loss = mode_cfg.get("compute_full_validation_loss", True)
        should_write_output = not is_diffusion_trajectory and bidx < num_samples_write
        if should_write_output:
            denormalize_data_fct = (
                (lambda x0, x1: x1)
                if mode_cfg.get("output", {}).get("normalized_samples", False)
                else self.dataset_val.denormalize_target_channels
            )
            if not targets_and_auxs:
                raise ValueError(
                    "Writing validation output requires targets. "
                    "Configure validation losses or set output.num_samples=0."
                )

        physical_loss_names = [
            name for name, loss_cfg in mode_cfg.losses.items()
            if loss_cfg.type == "LossPhysical"
        ]
        assert len(physical_loss_names) == 1 or chunk_size == len(output_idxs), (
            "Chunked non-full validation requires one LossPhysical term."
        )

        physical, latent = [], []
        forecast_chunk = batch.get_source_samples()
       
        if not compute_full_loss:
            # shallow copy is sufficient: .physical and .output_idxs are reassigned before
            # any use, and the shared per-step target dicts are only read downstream
            target_aux = targets_and_auxs[physical_loss_names[0]]
            target_aux_chunk = copy.copy(target_aux)

        for chunk_idx, chunk in enumerate(chunks):
            # Return PyTorch's reserved-but-unallocated HBM cache to CUDA at the start
            # of every chunk.  This is critical for chunk 0: the previous batch's latent
            # conditioning tensor (~1 GiB) was freed by CPython ref-counting when 'batch'
            # was reassigned at the top of the outer loop, but it sits in PyTorch's
            # reserved cache until empty_cache() is called.  Without this, loading the new
            # batch's source tokens (~444 MiB) fails even though the total freed+free
            # exceeds the request.
            torch.cuda.empty_cache()

            if not compute_full_loss:
                batch.to_device_for_output_chunk(self.device, chunk)

            # Load source tokens to GPU just before the encoder runs (chunk 0 only).
            # They were intentionally left on CPU by to_device_for_chunked_inference
            # (include_source_tokens=False) to keep peak GPU HBM free during the
            # DataLoader prefetch and pre-forward target computation.
            if chunk_idx == 0 and not compute_full_loss:
                # Also reload encoder-only ModelParams embeddings (pe_global ~1-2 GiB,
                # pe_embed ~65 KiB) that were offloaded after the previous batch.
                if is_root():
                    print(
                        f"[memory] before encoder params + source tokens "
                        f"rank={self.cf.rank} device={self.device}\n"
                        + torch.cuda.memory_summary(device=self.device, abbreviated=True),
                        flush=True,
                    )
                self.model_params.load_encoder_params_to_device(self.device)
                self._reload_encoder_weights()  # bring encoder shards back to GPU
                batch.load_source_tokens_to_device(self.device)

            if self.ema_model is None:
                forecast_chunk = self.model(
                    self.model_params,
                    forecast_chunk,
                    chunk,
                )
            else:
                forecast_chunk = self.ema_model.forward_eval(
                    self.model_params,
                    forecast_chunk,
                    chunk,
                )

            # After the encoder has run on the first chunk, observation source tokens
            # are no longer needed on GPU (continuation chunks only use the latent
            # state and decoder coordinates).  Offloading them immediately keeps GPU
            # memory constant regardless of the number of forecast steps.
            if chunk_idx == 0 and not compute_full_loss:
                batch.offload_source_tokens()
                # Drop the now-CPU source tensors entirely (source_tokens_cells /
                # source_tokens_lens / source_idxs_embed / source_raw).  target_coords
                # and meta_info (LATENT_CONDITIONING_TOKENS) are preserved.
                batch.drop_source_observations()
                # Offload encoder-only ModelParams (pe_global ~1-2 GiB) — not needed
                # by the forecast engine or decoder during the rollout.
                self.model_params.offload_encoder_params()
                # Offload encoder weight shards (~244 MiB FSDP-sharded / ~975 MiB full).
                # FSDP2's lazy all-gather means it will not touch them on continuation
                # chunks; they are reloaded by _reload_encoder_weights before the next
                # sample's encoder forward.
                self._offload_encoder_weights()

            if should_write_output:
                if not compute_full_loss:
                    target_aux_chunk.physical = [None for _ in range(chunk[0])] + [
                        target_aux.physical[step] for step in chunk
                    ]
                    target_aux_chunk.output_idxs = chunk
                    target_output = { physical_loss_names[0]: target_aux_chunk }
                else:
                    target_output = targets_and_auxs
                write_output(
                    self.cf,
                    mode_cfg,
                    batch_size,
                    mini_epoch,
                    bidx,
                    denormalize_data_fct,
                    batch,
                    forecast_chunk,
                    target_output
                )

            if compute_full_loss:
                physical += forecast_chunk.physical
                latent += forecast_chunk.latent
            elif chunk_idx == len(chunks) - 1:
                physical = forecast_chunk.physical
                # Latent states are only needed for loss computation; skip accumulation
                # when losses are not computed (skip_target_values or chunked mode).
                latent = [] if mode_cfg.get("skip_target_values", False) else forecast_chunk.latent
            else:
                forecast_chunk.physical.clear()
                forecast_chunk.latent = [forecast_chunk.latent[-1]]

            if not compute_full_loss:
                batch.clear_output_chunk_coordinates(chunk)

        if is_diffusion_trajectory:
            # single chunk; its fstep dimension is the trajectory, not forecast steps
            return forecast_chunk, targets_and_auxs

        # Data for validation purposes => accumulates in memory!?
        # compute_full_loss keeps every chunk's predictions, so preds spans all output steps;
        # otherwise only the last chunk survives the loop and preds spans just that chunk.
        # (These coincide when there is a single chunk, which is why chunk alone used to do.)
        covered_steps = output_idxs if compute_full_loss else chunk
        preds = ModelOutput(covered_steps, output_idxs[0], batch.get_source_samples())
        assert len(physical) == len(preds.physical), (
            f"Chunks cover {len(physical)} forecast steps, expected {len(preds.physical)}."
        )
        preds.physical = physical
        preds.latent = latent


        if not compute_full_loss:
            # this modifies targets_and_auxs in place
            target_aux_chunk.physical = [target_aux.physical[step] for step in chunk]
            targets_and_auxs[physical_loss_names[0]] = target_aux_chunk

        return preds, targets_and_auxs

    def inference(self, cf, devices, run_id_contd, mini_epoch_contd):
        # general initalization
        self.init(cf, devices)

        cf = self.cf
        device_type = torch.accelerator.current_accelerator()
        self.device = torch.device(f"{device_type}:{cf.local_rank}")
        self.ema_model = None
        [stream.update({"max_num_targets": -1}) for _, stream in cf.streams.items()]

        # create data loader
        # only one needed since we only run the validation code path
        # Force full maps during inference by disabling target subsampling
        for stream_info in cf.streams.values():
            stream_info["max_num_targets"] = -1

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

        # Pre-emptively offload encoder-only ModelParams (pe_global ~1-2 GiB) to CPU.
        # They will be reloaded just-in-time before each sample's encoder forward and
        # offloaded again immediately after, keeping them off GPU during decoding and
        # between samples.
        self.model_params.offload_encoder_params()
        torch.cuda.empty_cache()

        # get target_aux calculators for different loss terms
        self.target_and_aux_calculators_val = self.get_target_aux_calculators(self.test_cfg)

        self.loss_calculator_val = LossCalculator(cf, self.test_cfg, VAL, device=self.devices[0])

        if is_root():
            config.save(self.cf, mini_epoch=0)

        if self.test_cfg.get("skip_target_values", False):
            logger.warning(
                "skip_target_values is active: target values are neither read nor written "
                "(the output zarr has no target datasets) and no validation losses are computed."
            )

        logger.info(f"Starting inference with id={self.cf.general.run_id}.")

        # inference validation set
        self.validate(0, self.test_cfg, self.batch_size_test_per_gpu)
        logger.info(f"Finished inference run with id: {cf.general.run_id}")

    def run(self, cf, devices, run_id_contd=None, mini_epoch_contd=None):
        # general initalization
        self.init(cf, devices)
        cf = self.cf

        # skip_target_values is inference-only: without target values there is nothing
        # to train against and validation losses would be meaningless
        if self.training_cfg.get("skip_target_values", False) or self.validation_cfg.get(
            "skip_target_values", False
        ):
            raise ValueError(
                "skip_target_values is only allowed in inference (test_config), "
                "not in training_config or validation_config."
            )

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
        loader_params["num_workers"] = cf.data_loading.get(
            "num_workers_validation", cf.data_loading.num_workers
        )
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

        # Restore EMA teacher weights when continuing from a checkpoint
        if run_id_contd is not None:
            self._load_ema_teacher_state(run_id_contd, mini_epoch_contd)

        # if with_fsdp then parameter count is unreliable
        if is_root():
            # ddp-wrapped model does not expose this function
            if not cf.with_ddp:
                self.model.print_num_parameters()

        kappa = self.get_batch_size_total(self.batch_size_per_gpu)
        shared_lr_cfg = self.training_cfg.learning_rate_scheduling

        built = build_optimizer(self.model, self.training_cfg.optimizer, shared_lr_cfg, kappa)
        self.optimizers = built.optimizers
        self.optimizer_names = built.optimizer_names
        self._muon_effective_lr_factor = built.muon_effective_lr_factor
        lr_cfgs = built.lr_cfgs

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

        # Restore optimizer momentum buffers when continuing from a checkpoint
        if run_id_contd is not None and self.cf.general.istep != 0:
            self._load_optimizer_state(run_id_contd, mini_epoch_contd)

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
                    samples_or_output=batch.get_source_samples(),
                    forecast_steps=batch.get_output_idxs(),
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
                istep=self.cf.general.istep,
            )
            loss_value = self._get_tensor_item(loss.detach())
            if self._maybe_log_loss_spike(loss_value, batch, mini_epoch, bidx):
                self._drop_latest_loss_record()
                for optimizer in self.optimizers:
                    optimizer.zero_grad()
                if is_root():
                    logger.warning(
                        "Skipping batch %s in mini_epoch %s due to loss spike: %.8E",
                        bidx,
                        mini_epoch,
                        loss_value,
                    )
                continue

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
        Perform validation / test computation as specified by mode_cfg.

        For diffusion models, runs separate validation passes for each noise level
        specified in ``validation_noise_levels`` (defaults to ``[0.0]``).
        Losses are logged with a per-noise-level suffix so they can be compared.
        """

        cf = self.cf
        self.model.eval()

        is_diffusion = cf.get("fe_diffusion_model", False)
        # Only trajectory-mode diffusion (diffusion_rollout=False) hands back a ModelOutput
        # whose fstep dimension is the ODE trajectory rather than the forecast steps. That
        # layout is what forces a single chunk and a deferred, whole-batch write below; a
        # diffusion rollout is an ordinary forecast as far as chunking and IO are concerned.
        is_diffusion_trajectory = is_diffusion and not cf.get("diffusion_rollout", False)
        noise_levels = list(mode_cfg.get("validation_noise_levels", [0.0]))
        if not is_diffusion:
            noise_levels = [0.0]
        else:
            # Always include a pass without fixed noise level (random sampling)
            noise_levels = [None] + noise_levels

        # Accumulate losses across noise levels with suffixed keys so they are
        # logged as a single "val" entry (e.g. LossLatentDiff.LossLatentDiff.mse.eta0.03)
        all_losses: dict[str, list] = {}
        all_stddev: dict[str, list] = {}

        for _noise_idx, noise_level in enumerate(noise_levels):
            if is_diffusion:
                self._set_validation_noise_level(noise_level)

            if noise_level is None:
                loss_suffix = ""
                stage_suffix = ""
            else:
                _d = Decimal(str(noise_level)).normalize()
                _sign, _digits, _exp = _d.as_tuple()
                eta_str = f"{'-' if _sign else ''}{''.join(map(str, _digits))}e{_exp}"
                loss_suffix = f".eta{eta_str}" if len(noise_levels) > 1 else ""
                stage_suffix = f"_eta{eta_str}" if len(noise_levels) > 1 else ""

            dataset_val_iter = iter(self.data_loader_validation)

            with torch.no_grad():
                # print progress bar but only in interactive mode, i.e. when without ddp
                with tqdm.tqdm(
                    total=len(self.data_loader_validation) * self.cf.world_size,
                    disable=self.cf.rank > 0,
                ) as pbar:
                    for bidx, batch in enumerate(dataset_val_iter):
                        compute_full_loss = mode_cfg.get("compute_full_validation_loss", True)
                        if compute_full_loss or is_diffusion_trajectory:
                            batch.to_device(self.device)
                        else:
                            output_idxs = batch.get_output_idxs()
                            forecast_cfg = mode_cfg.get("forecast", {})
                            chunk_size = forecast_cfg.get("chunk_size", len(output_idxs))
                            chunks = self._get_forecast_step_chunks(output_idxs, chunk_size)
                            # calculators that encode the target window (diffusion latent
                            # target, SSL teachers) read the target samples' source view
                            needs_target_source = any(
                                getattr(c, "encodes_target_samples", False)
                                for c in self.target_and_aux_calculators_val.values()
                            )
                            batch.to_device_for_chunked_inference(
                                self.device, chunks[-1], include_target_source=needs_target_source,
                                include_source_tokens=False,
                            )
                            print(">>>>>>> chunk_size: " + str(chunk_size))

                        # evaluate model
                        with torch.autocast(
                            device_type=f"cuda:{cf.local_rank}",
                            dtype=self.mixed_precision_dtype,
                            enabled=cf.with_mixed_precision,
                        ):
                            targets_and_auxs = {}
                            for (
                                loss_name,
                                target_aux,
                            ) in self.target_and_aux_calculators_val.items():
                                target_idxs = get_target_idxs_from_cfg(mode_cfg, loss_name)
                                # DiffusionLatentTargetEncoder calls the encoder on
                                # target samples, which needs pe_embed / pe_global on
                                # device.  Reload them if they were offloaded and the
                                # calculator encodes target samples.
                                if getattr(target_aux, "encodes_target_samples", False):
                                    self.model_params.load_encoder_params_to_device(self.device)
                                targets_and_auxs[loss_name] = target_aux.compute(
                                    self.cf.general.istep,
                                    batch.get_target_samples(target_idxs),
                                    self.model_params,
                                    self.model,
                                )

                            preds, targets_and_auxs = self._process_validation_chunks(
                                batch,
                                mode_cfg,
                                batch_size,
                                mini_epoch,
                                bidx,
                                targets_and_auxs,
                                is_diffusion_trajectory,
                            )
                            # Both diffusion modes produce more preds fsteps than target
                            # entries: trajectory mode inflates the fstep dimension to one
                            # entry per ODE step, and a rollout gets a single encoder latent
                            # target for all T forecast steps. The target is identical for
                            # every such step, so replicate target/aux entries to keep the
                            # downstream loss calculator and validation IO aligned.
                            if is_diffusion:
                                _expand_targets_to_match_preds(preds, targets_and_auxs)

                            # Write output for trajectory-mode diffusion —
                            # _process_validation_chunks skips writing there because chunked
                            # IO is incompatible with the ODE trajectory fstep layout. Do it
                            # here instead, mirroring the non-diffusion path. A diffusion
                            # rollout has already been written chunk by chunk.
                            num_samples_write = (
                                mode_cfg.get("output", {}).get("num_samples", 0) * batch_size
                            )
                            if (
                                is_diffusion_trajectory
                                and _noise_idx == 0
                                and bidx < num_samples_write
                            ):
                                denormalize_data_fct = (
                                    (lambda x0, x1: x1)
                                    if mode_cfg.get("output", {}).get("normalized_samples", False)
                                    else self.dataset_val.denormalize_target_channels
                                )
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

                        # skip_target_values: target values are not read (zero width),
                        # so no loss can be computed
                        if not mode_cfg.get("skip_target_values", False):
                            _ = self.loss_calculator_val.compute_loss(
                                preds=preds,
                                targets_and_aux=targets_and_auxs,
                                metadata=extract_batch_metadata(batch),
                            )
                        pbar.update(batch_size * self.cf.world_size)

                        if (bidx * batch_size) > mode_cfg.samples_per_mini_epoch:
                            break

                    # Terminal logging per noise level for progress visibility
                    self._log_terminal(0, mini_epoch, VAL, stage_suffix=stage_suffix)

            # Extract losses for this noise level, suffix keys, and accumulate
            loss_calc = self.loss_calculator_val
            _, losses_level, stddev_level = prepare_losses_for_logging(
                loss_calc.loss_hist,
                loss_calc.losses_unweighted_hist,
                loss_calc.stddev_unweighted_hist,
            )
            for key, value in losses_level.items():
                all_losses[f"{key}{loss_suffix}"] = value
            for key, value in stddev_level.items():
                all_stddev[f"{key}{loss_suffix}"] = value
            loss_calc.loss_hist = []
            loss_calc.losses_unweighted_hist = []
            loss_calc.stddev_unweighted_hist = []

        # Log all noise levels as a single "val" entry with suffixed loss keys
        samples = self.cf.general.istep * self.get_batch_size_total(self.batch_size_per_gpu)
        if is_root():
            self.train_logger.add_logs(VAL, samples, all_losses, all_stddev)

        # reset fixed noise level
        if is_diffusion:
            self._set_validation_noise_level(None)

        # avoid that there is a systematic bias in the validation subset
        self.dataset_val.advance()

    def _set_validation_noise_level(self, noise_level: float | None):
        """Set fixed noise level on diffusion components for validation.

        Args:
            noise_level: The eta value (standard normal space) to fix for validation.
                         sigma = exp(eta * p_std + p_mean). None resets to default (0.0).
        """
        # Unwrap DDP/FSDP to access the underlying model
        base_model = getattr(self.model, "module", self.model)
        # Set on the base model
        if hasattr(base_model, "forecast_engine") and hasattr(
            base_model.forecast_engine, "_fixed_noise_level"
        ):
            base_model.forecast_engine._fixed_noise_level = noise_level
        # Also set on the EMA model (separate model copy used during validation)
        if self.ema_model is not None:
            ema_net = getattr(self.ema_model.ema_model, "module", self.ema_model.ema_model)
            if hasattr(ema_net, "forecast_engine") and hasattr(
                ema_net.forecast_engine, "_fixed_noise_level"
            ):
                ema_net.forecast_engine._fixed_noise_level = noise_level
        for calc in self.target_and_aux_calculators_val.values():
            if hasattr(calc, "_fixed_noise_level"):
                calc._fixed_noise_level = noise_level

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
        # Gather full state dicts (collective ops for FSDP — all ranks must participate)
        model_state_dict = self._get_full_model_state_dict()
        optim_state_dict = self._get_full_optimizer_state_dict()

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
            logger.info(f"Saved model to {file_out}")

            # save optimizer state keyed by parameter name for robust resumption.
            # optim_state_dict has one entry per optimizer; each optimizer's "state" is
            # keyed by that optimizer's own param index, so map indices back to parameter
            # names via the parameter order of its param_groups.
            param_name_by_id = {id(p): n for n, p in self.model.named_parameters()}
            named_optim_state = {}
            for opt_sd, optimizer in zip(optim_state_dict, self.optimizers, strict=True):
                opt_param_names = [
                    param_name_by_id[id(p)]
                    for group in optimizer.param_groups
                    for p in group["params"]
                ]
                for idx, pname in enumerate(opt_param_names):
                    if idx in opt_sd["state"]:
                        named_optim_state[pname] = opt_sd["state"][idx]
            if named_optim_state:
                optim_out = base_path / (filename + ".optim")
                optim_tmp = base_path / (filename + "_tmp.optim")
                torch.save(named_optim_state, optim_tmp)
                optim_tmp.replace(optim_out)
                logger.info(f"Saved optimizer state to {optim_out}")

            # save EMA teacher state (weights + centering buffers) if present
            ema_teacher = self._get_ema_teacher()
            if ema_teacher is not None:
                ema_state = {
                    "ema_model": ema_teacher.ema_model.ema_model.state_dict(),
                    "postprocess_targets": {
                        name: module.state_dict()
                        for name, module in ema_teacher.postprocess_targets.items()
                    },
                }
                ema_out = base_path / (filename + ".ema_teacher")
                ema_tmp = base_path / (filename + "_tmp.ema_teacher")
                torch.save(ema_state, ema_tmp)
                ema_tmp.replace(ema_out)
                logger.info(f"Saved EMA teacher state to {ema_out}")

            # save config
            config.save(self.cf, mini_epoch)

    def _get_ema_teacher(self) -> EMATeacher | None:
        """Return the training EMATeacher calculator if one exists, else None."""
        if self.target_and_aux_calculators is None:
            return None
        for calc in self.target_and_aux_calculators.values():
            if isinstance(calc, EMATeacher):
                return calc
        return None

    @staticmethod
    def _get_ema_teachers_from(calculators) -> list[EMATeacher]:
        """Return all EMATeacher instances in *calculators*."""
        if calculators is None:
            return []
        return [c for c in calculators.values() if isinstance(c, EMATeacher)]

    def _load_ema_teacher_state(self, run_id: str, mini_epoch):
        """Load EMA teacher weights into both training and validation teachers."""
        all_teachers = self._get_ema_teachers_from(
            self.target_and_aux_calculators
        ) + self._get_ema_teachers_from(self.target_and_aux_calculators_val)
        if not all_teachers:
            return

        path_run = config.get_path_model(run_id=run_id)
        mini_epoch_id = f"chkpt{mini_epoch:05d}" if mini_epoch not in (-1, None) else "latest"
        ema_file = path_run / f"{run_id}_{mini_epoch_id}.ema_teacher"

        if not ema_file.exists():
            if is_root():
                logger.info(f"No EMA teacher state at {ema_file}, using reset from student.")
            return

        if is_root():
            logger.info(f"Loading EMA teacher state from {ema_file}")

        state = torch.load(ema_file, map_location=torch.device("cpu"), weights_only=True)

        for ema_teacher in all_teachers:
            # Restore EMA model weights
            mkeys, ukeys = ema_teacher.ema_model.ema_model.load_state_dict(
                state["ema_model"], strict=False
            )
            if is_root():
                if mkeys:
                    logger.warning(f"Missing keys in EMA teacher model: {mkeys}")
                if ukeys:
                    logger.warning(f"Unused keys in EMA teacher model: {ukeys}")

            # Restore postprocessing state (e.g. DINO/iBOT centering buffers)
            for name, module in ema_teacher.postprocess_targets.items():
                if name in state.get("postprocess_targets", {}):
                    module.load_state_dict(state["postprocess_targets"][name], strict=False)

        if is_root():
            logger.info(f"EMA teacher state restored into {len(all_teachers)} teacher(s).")

    def _load_optimizer_state(self, run_id: str, mini_epoch):
        """Load optimizer state from checkpoint if available.

        Restores AdamW momentum buffers (exp_avg, exp_avg_sq) so that training
        resumes smoothly when chaining jobs via train_continue.
        """
        path_run = config.get_path_model(run_id=run_id)
        mini_epoch_id = f"chkpt{mini_epoch:05d}" if mini_epoch not in (-1, None) else "latest"
        optim_file = path_run / f"{run_id}_{mini_epoch_id}.optim"

        if not optim_file.exists():
            if is_root():
                logger.info(f"No optimizer state found at {optim_file}, starting fresh.")
            return

        if is_root():
            logger.info(f"Loading optimizer state from {optim_file}")

        named_state = torch.load(
            optim_file, map_location=torch.device("cpu"), mmap=True, weights_only=True
        )
        is_model_sharded = self.cf.with_ddp and self.cf.with_fsdp

        # map each parameter to the optimizer that owns it, so state is restored on the
        # correct optimizer when the model is split across several (e.g. muon + adamw).
        optimizer_by_param_id = {
            id(p): optimizer
            for optimizer in self.optimizers
            for group in optimizer.param_groups
            for p in group["params"]
        }

        loaded = 0
        for name, param in self.model.named_parameters():
            if name not in named_state:
                continue
            optimizer = optimizer_by_param_id.get(id(param))
            if optimizer is None:
                continue
            entry = named_state[name]
            new_entry = {}
            for key, val in entry.items():
                if isinstance(val, torch.Tensor) and val.dim() > 0 and is_model_sharded:
                    new_entry[key] = distribute_tensor(val, param.device_mesh, param.placements)
                elif isinstance(val, torch.Tensor):
                    new_entry[key] = val.to(device=param.device)
                else:
                    new_entry[key] = val
            optimizer.state[param] = new_entry
            loaded += 1

        if is_root():
            total = sum(1 for _ in self.model.parameters())
            logger.info(f"Loaded optimizer state for {loaded}/{total} parameters.")

    def _log(self, stage: Stage, stage_suffix: str = ""):
        """
        Logs training or validation metrics.

        Args:
            stage: Stage Is it's VAL, logs are treated as validation logs.
                        If TRAIN, logs are treated as training logs
            stage_suffix: Optional suffix appended to the logged stage name
                          (e.g. "_eta0.00" for per-noise-level validation).

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
        log_stage = f"{stage}{stage_suffix}" if stage_suffix else stage

        if is_root():
            # plain logger
            if stage == VAL:
                self.train_logger.add_logs(log_stage, samples, losses_all, stddev_all)

            elif self.cf.general.istep >= 0:
                self.train_logger.add_logs(
                    log_stage,
                    samples,
                    losses_all,
                    stddev_all,
                    avg_loss=avg_loss,
                    lr=self.lr_schedulers[0].get_lr(),
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

    def _init_loss_spike_detection(self) -> None:
        configured_loss_spike_cfg = self.cf.train_logging.get("loss_spike_detection", {}) or {}
        self.loss_spike_cfg = OmegaConf.merge(
            OmegaConf.create(LOSS_SPIKE_DETECTION_DEFAULTS),
            configured_loss_spike_cfg,
        )
        window_size = int(self.loss_spike_cfg.window_size)
        self.loss_spike_history = deque(maxlen=window_size)
        self.loss_spike_file = None

        if not self.loss_spike_cfg.enabled:
            return

        self.loss_spike_file = config.get_path_run(self.cf) / self.loss_spike_cfg.file_name

    def _serialize_datetimes(self, datetimes) -> list[str]:
        if datetimes is None:
            return []

        datetimes_arr = np.asarray(datetimes).reshape(-1)
        if datetimes_arr.size == 0:
            return []

        max_unique = int(self.loss_spike_cfg.max_unique_times_per_step)
        return [str(dt) for dt in np.unique(datetimes_arr)[:max_unique]]

    @staticmethod
    def _to_python_indices(indices):
        if hasattr(indices, "astype") and hasattr(indices, "tolist"):
            return indices.astype(int).tolist()
        if isinstance(indices, list):
            return [int(idx) for idx in indices]
        if indices is None:
            return None
        return int(indices)

    @staticmethod
    def _to_bool_list(value) -> list[bool]:
        if isinstance(value, list):
            return [bool(item) for item in value]
        return [bool(value)]

    def _collect_sample_debug_info(self, sample, matching_indices) -> dict:
        streams = {}
        for stream_name, stream_data in sample.streams_data.items():
            if stream_data is None:
                continue

            source_raw = getattr(stream_data, "source_raw", [])
            target_times_raw = getattr(stream_data, "target_times_raw", [])
            source_start_idx = int(stream_data.sample_idx) - len(source_raw) + 1

            streams[stream_name] = {
                "sample_idx": int(stream_data.sample_idx),
                "source_is_spoof": self._to_bool_list(stream_data.source_is_spoof),
                "target_is_spoof": self._to_bool_list(stream_data.target_is_spoof),
                "source_step_indices": [source_start_idx + step for step in range(len(source_raw))],
                "target_step_indices": list(range(len(target_times_raw))),
                "source_step_datetimes": [
                    self._serialize_datetimes(getattr(raw_data, "datetimes", None))
                    for raw_data in source_raw
                ],
                "target_step_datetimes": [
                    self._serialize_datetimes(datetimes) for datetimes in target_times_raw
                ],
            }

        return {
            "matching_indices": self._to_python_indices(matching_indices),
            "streams": streams,
        }

    def _write_loss_spike_record(
        self, loss_value, baseline, ratio, batch, mini_epoch, bidx
    ) -> None:
        if self.loss_spike_file is None:
            return

        record = {
            "run_id": str(self.cf.general.run_id),
            "mini_epoch": int(mini_epoch),
            "batch_index": int(bidx),
            "global_step": int(self.cf.general.istep),
            "loss": float(loss_value),
            "loss_repr": f"{loss_value:.8E}",
            "baseline_median": float(baseline),
            "ratio_to_baseline": float(ratio),
            "skip_batch": bool(self.loss_spike_cfg.skip_batch),
            "source_samples": [
                self._collect_sample_debug_info(sample, batch.source2target_matching_idxs[sidx])
                for sidx, sample in enumerate(batch.source_samples.get_samples())
            ],
            "target_samples": [
                self._collect_sample_debug_info(sample, batch.target2source_matching_idxs[tidx])
                for tidx, sample in enumerate(batch.target_samples.get_samples())
            ],
        }

        with self.loss_spike_file.open("a", encoding="utf-8") as file_out:
            file_out.write(json.dumps(record) + "\n")

    def _sync_loss_spike_skip(self, should_skip: bool) -> bool:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            skip_flag = torch.tensor(
                [int(should_skip)], dtype=torch.int32, device=self.device or torch.device("cpu")
            )
            torch.distributed.all_reduce(skip_flag, op=torch.distributed.ReduceOp.MAX)
            should_skip = bool(skip_flag.item())

        return should_skip

    def _drop_latest_loss_record(self) -> None:
        for hist_name in ("loss_hist", "losses_unweighted_hist", "stddev_unweighted_hist"):
            hist = getattr(self.loss_calculator, hist_name)
            if hist:
                hist.pop()

    def _maybe_log_loss_spike(self, loss_value: float, batch, mini_epoch: int, bidx: int) -> bool:
        if not self.loss_spike_cfg.enabled:
            return False

        # each rank checks its local loss; the skip decision is then all-reduced (OR) so
        # that a spike / non-finite loss on any rank skips the batch on all ranks
        should_skip = False
        local_anomaly = False
        baseline, ratio = float("nan"), float("nan")
        is_finite = np.isfinite(loss_value)
        min_history = int(self.loss_spike_cfg.min_history)
        if len(self.loss_spike_history) >= min_history:
            baseline = float(np.median(self.loss_spike_history))
            ratio = loss_value / baseline if baseline > 0 else np.inf
            is_large_enough = loss_value >= float(self.loss_spike_cfg.loss_threshold)
            is_spike = ratio >= float(self.loss_spike_cfg.ratio_threshold)
            local_anomaly = (is_finite and is_large_enough and is_spike) or not is_finite
            should_skip = local_anomaly and bool(self.loss_spike_cfg.skip_batch)

        should_skip = self._sync_loss_spike_skip(should_skip)

        # logging stays rank-0-only; record fields are rank 0's local values
        if is_root() and (local_anomaly or should_skip):
            self._write_loss_spike_record(loss_value, baseline, ratio, batch, mini_epoch, bidx)

        if is_finite and not should_skip:
            self.loss_spike_history.append(float(loss_value))

        return should_skip

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

    def _log_terminal(self, bidx: int, mini_epoch: int, stage: Stage, stage_suffix: str = ""):
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
                        f"""validation{stage_suffix} ({self.cf.general.run_id}) : 
                        {mini_epoch:03d} : {np.nanmean(avg_loss)}"""
                    )

                elif stage == TRAIN:
                    # samples per sec
                    dt = time.time() - self.t_start
                    len_dataset = len(self.data_loader) // self.batch_size_per_gpu
                    pstr = (
                        f"{mini_epoch:03d} : {bidx:05d}/{len_dataset:05d} : "
                        + f"{self.cf.general.istep:06d} : loss = {np.nanmean(avg_loss):.4E} "
                        + f"(lr={self.lr_schedulers[0].get_lr():.2E}, "
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
