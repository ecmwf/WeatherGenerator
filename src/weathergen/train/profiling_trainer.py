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

from weathergen.common.config import Config
from weathergen.datasets.batch import ModelBatch
from weathergen.train.trainer import Trainer
from weathergen.train.utils import TRAIN
from weathergen.utils.distributed import is_root
from weathergen.utils.performance import ThroughputTracker, nvtx_batches
from weathergen.utils.profiling import (
    ProfilingConfig,
    memory_snapshot_session,
    pytorch_profiler_session,
    wrap_module_forward_with_profiling,
)

logger = logging.getLogger(__name__)


class ProfilingTrainer(Trainer):
    """
    Trainer that measures the training loop (see `ProfilingConfig`, `ThroughputTracker`).

    Only the iteration seams (`mini_epochs`, `train_batches`) are overridden, so the
    training step is the same as in a normal run. Measurement happens in `train_batches`,
    which regains control after each yielded batch has been trained on.

    Traces are collected on the root rank only; the other ranks run the same steps untraced
    so collectives stay matched.
    """

    def __init__(self, train_logging: Config, profiling_cfg: ProfilingConfig):
        super().__init__(train_logging)

        self.profiling_cfg = profiling_cfg
        self.throughput_tracker: ThroughputTracker | None = None
        self.profiling_done: bool = False

    def init(self, cf: Config, devices: list) -> None:
        super().init(cf, devices)

        self.throughput_tracker = ThroughputTracker.from_config(
            self.cf,
            device=torch.device(self.devices[0]),
            batch_size_per_gpu=self.batch_size_per_gpu,
        )
        logger.info(
            f"Profiling run: {self.profiling_cfg}, "
            f"throughput logging: {self.throughput_tracker is not None}"
        )

        profiling_cfg = self.profiling_cfg
        if profiling_cfg.enabled and not (
            profiling_cfg.collects_traces or profiling_cfg.annotates_nvtx
        ):
            logger.warning("Profiling is enabled, but no collector or nvtx_annotate is.")

    @property
    def profiling_only(self) -> bool:
        """Whether the run ends after the profiled stretch, without validation or checkpoints."""
        return self.profiling_cfg.enabled and self.profiling_cfg.stop_after_profiling

    def mini_epochs(self, mini_epoch_base: int) -> Iterator[int]:
        """Run a single mini_epoch in a profiling-only run."""
        if self.profiling_only:
            yield mini_epoch_base
        else:
            yield from super().mini_epochs(mini_epoch_base)

    def train_batches(self, dataset_iter: Iterator) -> Iterator[tuple[int, ModelBatch]]:
        """Measure every training step, and trace the profiled stretch of them."""
        yield from self._with_nvtx(self._with_throughput(self._with_profiling(dataset_iter)))

    def _with_nvtx(
        self, batches: Iterator[tuple[int, ModelBatch]]
    ) -> Iterator[tuple[int, ModelBatch]]:
        """Wrap each training step in an nvtx range, for nsys."""
        if not self.profiling_cfg.annotates_nvtx:
            yield from batches
            return

        yield from nvtx_batches(batches)

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
                self._log_train_metrics(self.throughput_tracker.compute_metrics(), istep)

    def _is_metrics_step(self, bidx: int) -> bool:
        """Whether train() logs its metrics at this batch."""
        return bidx % self.train_logging.metrics == 0

    def _log_train_metrics(self, metrics: dict[str, float] | None, istep: int) -> None:
        """Log metrics as their own training record, on the root rank."""
        if metrics and is_root():
            self.train_logger.log_metrics(TRAIN, metrics, step=istep)

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

        if self.profiling_only:
            logger.info(f"Profiled {schedule.num_steps} training steps, ending the run.")
            return

        yield from enumerate(dataset_iter, start=schedule.num_steps)

    def validate(self, mini_epoch, mode_cfg, batch_size) -> None:
        """Skipped in a profiling-only run."""
        if self.profiling_only:
            return

        super().validate(mini_epoch, mode_cfg, batch_size)

    def save_model(self, mini_epoch: int, name=None) -> None:
        """Skipped in a profiling-only run."""
        if self.profiling_only:
            return

        super().save_model(mini_epoch, name)
