# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Trainer variant that measures a training run instead of (only) performing it.

Selected by the `profiling` and `performance_logging` config sections, see
`weathergen.run_train.get_trainer`.
"""

import contextlib
import logging
from collections.abc import Iterator
from itertools import islice

import torch

import weathergen.train.trainer as trainer_module
from weathergen.common.config import Config
from weathergen.datasets.batch import ModelBatch
from weathergen.train.trainer import Trainer
from weathergen.train.utils import TRAIN, VAL, Stage
from weathergen.utils.distributed import is_root
from weathergen.utils.nsys_windows import NsysWindows
from weathergen.utils.performance import (
    MemoryTracker,
    ThroughputTracker,
    nvtx_annotator,
    nvtx_batches,
    nvtx_next,
    nvtx_patched,
    nvtx_wrap,
)
from weathergen.utils.profiling import (
    ProfilingConfig,
    memory_snapshot_session,
    pytorch_profiler_session,
    wrap_module_forward_with_profiling,
)

logger = logging.getLogger(__name__)


class ProfilingTrainer(Trainer):
    """
    Trainer that measures the training loop (see `ProfilingConfig`, `NsysWindows`,
    `ThroughputTracker`, `MemoryTracker`).

    Only the iteration seams (`mini_epochs`, `train_batches`) are overridden, so the
    training step is the same as in a normal run. Measurement happens in `train_batches`,
    which regains control after each yielded batch has been trained on. The nvtx ranges of
    the phases of a step are added by wrapping the objects the step calls (see
    `_annotate_step_phases`), not by annotating `Trainer.train`; the same holds for
    validation (`validation_batches`, `_annotate_validation_phases`). Inference is a single
    `validate`, so only the memory snapshot, the peak-memory logging and the nvtx ranges of
    validation apply to it.

    Traces are collected on the root rank only; the other ranks run the same steps untraced
    so collectives stay matched.
    """

    def __init__(self, train_logging: Config, profiling_cfg: ProfilingConfig):
        super().__init__(train_logging)

        self.profiling_cfg = profiling_cfg
        self.throughput_tracker: ThroughputTracker | None = None
        self.memory_tracker: MemoryTracker | None = None
        self.nsys_windows: NsysWindows | None = None
        self.profiling_done: bool = False
        self.step_phases_annotated: bool = False
        # (id(obj), method) already in an nvtx range, as training and validation share some
        self.nvtx_wrapped: set[tuple[int, str]] = set()
        # `profiling.nvtx_annotate` without the master switch is a no-op
        self.annotate = nvtx_annotator(profiling_cfg.annotates_nvtx)

    def init(self, cf: Config, devices: list) -> None:
        super().init(cf, devices)

        self.throughput_tracker = ThroughputTracker.from_config(
            self.cf,
            device=torch.device(self.devices[0]),
            batch_size_per_gpu=self.batch_size_per_gpu,
        )
        self.memory_tracker = MemoryTracker.from_config(
            self.cf, device=torch.device(self.devices[0])
        )
        self.nsys_windows = NsysWindows.from_config(self.cf)
        logger.info(
            f"Profiling run: {self.profiling_cfg}, "
            f"nsys windows: {vars(self.nsys_windows) if self.nsys_windows else None}, "
            f"throughput logging: {self.throughput_tracker is not None}, "
            f"peak-memory logging: {self.memory_tracker is not None}"
        )

        profiling_cfg = self.profiling_cfg
        if profiling_cfg.enabled and not (
            profiling_cfg.collects_traces
            or profiling_cfg.annotates_nvtx
            or profiling_cfg.annotates_nvtx_modules
            or self.nsys_windows is not None
        ):
            logger.warning("Profiling is enabled, but no collector, nsys window or nvtx is.")
        if (
            self.nsys_windows is not None
            and profiling_cfg.collects_traces
            and profiling_cfg.stop_after_profiling
            and self.nsys_windows.last_step >= profiling_cfg.schedule.num_steps
        ):
            logger.warning(
                "stop_after_profiling ends the run before the last nsys window closes: "
                f"{profiling_cfg.schedule.num_steps} steps < {self.nsys_windows.last_step + 1}."
            )

    @property
    def profiling_only(self) -> bool:
        """Whether the run ends after the measured steps, without validation or checkpoints."""
        stops_after_profiling = (
            self.profiling_cfg.enabled and self.profiling_cfg.stop_after_profiling
        )
        stops_after_capture = self.nsys_windows is not None and self.nsys_windows.stop_after_capture
        return stops_after_profiling or stops_after_capture

    def mini_epochs(self, mini_epoch_base: int) -> Iterator[int]:
        """Run a single mini_epoch in a profiling-only run."""
        if self.profiling_only:
            yield mini_epoch_base
        else:
            yield from super().mini_epochs(mini_epoch_base)

    def train(self, mini_epoch: int) -> None:
        """Label the phases of the training step and open the nsys startup window."""
        if self.profiling_cfg.annotates_nvtx:
            self._annotate_step_phases()
        if self.nsys_windows is not None:
            self._open_nsys_startup_window()

        with contextlib.ExitStack() as stack:
            if self.profiling_cfg.annotates_nvtx:
                # a module function, so patched only while training
                stack.enter_context(
                    nvtx_patched(torch.nn.utils, "clip_grad_norm_", "clip_grad_norm")
                )
            super().train(mini_epoch)

    def train_batches(self, dataset_iter: Iterator) -> Iterator[tuple[int, ModelBatch]]:
        """Measure every training step, and trace the profiled stretch of them."""
        if self.profiling_cfg.annotates_nvtx:
            # the time the training loop waits for the data loader
            dataset_iter = nvtx_next(dataset_iter, "next_batch")
        batches = self._with_throughput(self._with_profiling(dataset_iter))
        # the nsys window is opened before and closed after the batch's nvtx range
        yield from self._with_nvtx(self._with_nsys(self._with_memory(batches)))

    def _with_nvtx(
        self, batches: Iterator[tuple[int, ModelBatch]]
    ) -> Iterator[tuple[int, ModelBatch]]:
        """Wrap each training step in an nvtx range, for nsys."""
        if not self.profiling_cfg.annotates_nvtx:
            yield from batches
            return

        for bidx, batch in nvtx_batches(batches):
            # the batch's methods, since there is a new batch each step
            nvtx_wrap(batch, "pin_memory", "pin_memory")
            nvtx_wrap(batch, "to_device", "to_device")
            yield bidx, batch

    def _with_nsys(
        self, batches: Iterator[tuple[int, ModelBatch]]
    ) -> Iterator[tuple[int, ModelBatch]]:
        """Open and close the nsys steady window, and end a capture-only run after it."""
        if self.nsys_windows is None:
            yield from batches
            return

        for bidx, batch in batches:
            self.nsys_windows.before_step(bidx)
            yield bidx, batch
            self.nsys_windows.after_step(bidx)
            if self.nsys_windows.ends_run:
                logger.info("All nsys capture windows done, ending the run.")
                return

    def _open_nsys_startup_window(self) -> None:
        """Capture from before the data loader starts until the first forward pass is done."""
        nsys_windows = self.nsys_windows
        assert nsys_windows is not None
        if not nsys_windows.startup or nsys_windows.done:
            return

        def close_after_first_forward(module, args, output) -> None:
            handle.remove()
            nsys_windows.after_forward(0)

        handle = self.model.register_forward_hook(close_after_first_forward)
        nsys_windows.before_data_iter()

    def _annotate_step_phases(self) -> None:
        """
        Put each phase of the training step into its own nvtx range.

        Wraps the methods of the objects that `Trainer.train` calls (instances only, once
        per run), so the step itself needs no annotations. Module forward/backward ranges
        come from `register_nvtx_hooks` (`nvtx_module_ranges`), the per-loss-term ranges from
        `LossCalculator`.
        """
        if self.step_phases_annotated:
            return
        self.step_phases_annotated = True

        wraps: list[tuple[object, str, str]] = [
            (self.model, "forward", "forward"),
            (self.loss_calculator, "compute_loss", "loss"),
            (self, "_get_tensor_item", "item (GPU sync)"),
            (self, "_maybe_log_loss_spike", "loss_spike_check"),
            (self.collapse_monitor, "_compute_collapse_metrics", "collapse_metrics"),
        ]
        for name, target_aux in self.target_and_aux_calculators.items():
            wraps += [
                (target_aux, "compute", f"target_aux:{name}"),
                (target_aux, "update_state_pre_backward", f"update_state_pre_backward:{name}"),
                (target_aux, "update_state_post_opt_step", f"update_state_post_opt_step:{name}"),
            ]
        for name, target_aux in self.target_and_aux_calculators_val.items():
            wraps += [
                (target_aux, "update_state_pre_backward", f"update_state_pre_backward:{name}"),
                (target_aux, "update_state_post_opt_step", f"update_state_post_opt_step:{name}"),
            ]
        wraps += [(optimizer, "zero_grad", "zero_grad") for optimizer in self.optimizers]
        wraps += [(scheduler, "step", "lr_scheduler_step") for scheduler in self.lr_schedulers]
        wraps += [
            (self.grad_scaler, "unscale_", "unscale_grads"),
            (self.grad_scaler, "step", "optimizer_step"),
            (self.grad_scaler, "update", "grad_scaler_update"),
        ]
        if self.ema_model is not None:
            wraps += [(self.ema_model, "update", "ema_update")]
        for method in ("_log_terminal", "_log", "_log_collapse_metrics", "_log_instant_grad_norms"):
            wraps += [(self, method, "log")]

        for obj, method, name in wraps:
            self._nvtx_wrap_once(obj, method, name)

        # backward runs on the scaled loss, a new tensor every step
        scale = self.grad_scaler.scale

        def scale_and_annotate_backward(outputs):
            scaled = scale(outputs)
            nvtx_wrap(scaled, "backward", "backward")
            return scaled

        self.grad_scaler.scale = scale_and_annotate_backward

    def validation_batches(self, dataset_iter: Iterator) -> Iterator[tuple[int, ModelBatch]]:
        """Put each validation step, and the wait for its batch, into an nvtx range."""
        if not self.profiling_cfg.annotates_nvtx:
            yield from super().validation_batches(dataset_iter)
            return

        batches = super().validation_batches(nvtx_next(dataset_iter, "next_batch"))
        for bidx, batch in nvtx_batches(batches):
            # the batch's methods, since there is a new batch each step
            for method in (
                "to_device",
                "to_device_for_chunked_inference",
                "to_device_for_output_chunk",
            ):
                nvtx_wrap(batch, method, method)
            yield bidx, batch

    def _annotate_validation_phases(self) -> None:
        """
        Put each phase of the validation step into its own nvtx range.

        Like `_annotate_step_phases`. The rollout runs one forward per chunk of forecast steps
        (`forecast.chunk_size`), so each chunk is a `forward` range. `write_output` is a module
        function, patched in `validate`.
        """
        wraps: list[tuple[object, str, str]] = [(self.model, "forward", "forward")]
        if self.ema_model is not None:
            wraps += [(self.ema_model, "forward_eval", "forward")]
        if self.loss_calculator_val is not None:
            wraps += [(self.loss_calculator_val, "compute_loss", "loss")]
        for name, target_aux in self.target_and_aux_calculators_val.items():
            wraps += [(target_aux, "compute", f"target_aux:{name}")]

        for obj, method, name in wraps:
            self._nvtx_wrap_once(obj, method, name)

    def _nvtx_wrap_once(self, obj: object, method: str, name: str) -> None:
        """`nvtx_wrap`, unless `obj.<method>` is already wrapped."""
        key = (id(obj), method)
        if key not in self.nvtx_wrapped:
            self.nvtx_wrapped.add(key)
            nvtx_wrap(obj, method, name)

    def _with_throughput(
        self, batches: Iterator[tuple[int, ModelBatch]]
    ) -> Iterator[tuple[int, ModelBatch]]:
        """Count each training step, and log the global counts at the metrics interval."""
        if self.throughput_tracker is None:
            yield from batches
            return

        for bidx, batch in batches:
            istep = self.cf.general.istep  # train() increments it as part of the step
            yield bidx, batch
            self.throughput_tracker.step(batch)
            if self._is_metrics_step(bidx):
                # collective, so on every rank
                self._log_metrics(TRAIN, self.throughput_tracker.compute_metrics(), istep)

    def _with_memory(
        self, batches: Iterator[tuple[int, ModelBatch]]
    ) -> Iterator[tuple[int, ModelBatch]]:
        """Log the peak memory of the training steps at the metrics interval."""
        if self.memory_tracker is None:
            yield from batches
            return

        for bidx, batch in batches:
            istep = self.cf.general.istep  # train() increments it as part of the step
            yield bidx, batch
            if self._is_metrics_step(bidx):
                # collective, so on every rank
                self._log_metrics(TRAIN, self.memory_tracker.collect("train"), istep)

    def _is_metrics_step(self, bidx: int) -> bool:
        """Whether train() logs its metrics at this batch."""
        return bidx % self.train_logging.metrics == 0

    def _log_metrics(self, stage: Stage, metrics: dict[str, float] | None, istep: int) -> None:
        """Log metrics as their own record, on the root rank."""
        if metrics and is_root():
            self.train_logger.log_metrics(stage, metrics, step=istep)

    def _close_memory_window(self, window: str) -> None:
        """Attribute the peak memory since the last read to `window` (local)."""
        if self.memory_tracker is not None:
            self.memory_tracker.close_window(window)

    def _log_peak_memory(self, stage: Stage, window: str) -> None:
        """Log the peak memory of `window` (collective)."""
        if self.memory_tracker is not None:
            self._log_metrics(stage, self.memory_tracker.collect(window), self.cf.general.istep)

    def _with_profiling(self, dataset_iter: Iterator) -> Iterator[tuple[int, ModelBatch]]:
        """Trace the profiled stretch, then continue (or stop) as configured."""
        if self.profiling_done or not self.profiling_cfg.enabled:
            # the stretch is profiled once per run, not once per mini_epoch
            yield from super().train_batches(dataset_iter)
            return

        self.profiling_done = True
        schedule = self.profiling_cfg.schedule

        if is_root() and self.profiling_cfg.pytorch_profiler:
            # the model only exists once run() has built it, hence not in init()
            wrap_module_forward_with_profiling(self.model, prefix="model")

        with contextlib.ExitStack() as stack:
            prof = None
            if self.profiling_cfg.pytorch_profiler:
                prof = stack.enter_context(pytorch_profiler_session(self.cf, schedule))

            for bidx, batch in enumerate(islice(dataset_iter, schedule.num_steps)):
                if bidx == schedule.steps_before_active and self.profiling_cfg.memory_snapshot:
                    # skip the wait and warmup steps, as the profiler does
                    stack.enter_context(memory_snapshot_session(self.cf))

                yield bidx, batch
                if prof is not None:
                    prof.step()

        # keep the other ranks in step with the root rank writing its traces
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if self.profiling_cfg.stop_after_profiling:
            logger.info(f"Profiled {schedule.num_steps} training steps, ending the run.")
            return

        yield from enumerate(dataset_iter, start=schedule.num_steps)

    def validate(self, mini_epoch, mode_cfg, batch_size) -> None:
        """
        Skipped in a profiling-only run; otherwise its peak memory is logged on its own.

        In inference, validation is the whole run: it is never skipped, and it is recorded in
        a memory snapshot if one is configured (the schedule does not apply).
        """
        inference = self.cf.stage == "inference"
        if self.profiling_only and not inference:
            return

        # the training steps since the last metrics interval belong to the training window
        self._close_memory_window("train")
        with contextlib.ExitStack() as stack:
            if inference and self.profiling_cfg.enabled and self.profiling_cfg.memory_snapshot:
                stack.enter_context(memory_snapshot_session(self.cf))
            if self.profiling_cfg.annotates_nvtx:
                self._annotate_validation_phases()
                # a module function, so patched only while validating
                stack.enter_context(nvtx_patched(trainer_module, "write_output", "write_output"))
                stack.enter_context(self.annotate("validation"))
            super().validate(mini_epoch, mode_cfg, batch_size)
        self._log_peak_memory(VAL, "validation")

    def save_model(self, mini_epoch: int, name=None) -> None:
        """
        Skipped in a profiling-only run; otherwise its peak memory is logged on its own.

        Gathering the full state dict for the checkpoint is often the peak of the whole run.
        """
        if self.profiling_only:
            return

        self._close_memory_window("train")
        with self.annotate("checkpoint"):
            super().save_model(mini_epoch, name)
        self._log_peak_memory(TRAIN, "save_model")
