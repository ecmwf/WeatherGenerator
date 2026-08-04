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
from weathergen.utils.distributed import is_root
from weathergen.utils.performance import ThroughputTracker, nvtx_range
from weathergen.utils.profiling import (
    PerformanceLoggingConfig,
    ProfilingConfig,
    memory_snapshot_session,
    pytorch_profiler_session,
    wrap_module_forward_with_profiling,
)

logger = logging.getLogger(__name__)


class ProfilingTrainer(Trainer):
    """
    Trainer that measures the training loop, configured by `profiling` and
    `performance_logging`.

    The training step itself is inherited unchanged from `Trainer`: only the iteration
    seams (`mini_epochs`, `train_batches`) are overridden, so the measured code path and
    the code path of a normal run cannot drift apart.

    `profiling` traces the profiled stretch (`schedule.num_steps` training steps) on the
    root rank, while the other ranks run the same steps untraced so that collectives stay
    matched. The PyTorch profiler steps through the schedule; the memory snapshot records
    from the first active step onwards. With `stop_after_profiling` (the default) the run
    ends once the stretch is done, without validating or checkpointing, so that the traces
    cover the training step and nothing else.

    `performance_logging` covers the whole run on all ranks, and is cheap: a run with only
    this enabled trains exactly as a plain `Trainer` run would, and just logs more.
    """

    def __init__(self, train_logging: Config):
        super().__init__(train_logging)

        self.profiling_cfg = ProfilingConfig()
        self.performance_cfg = PerformanceLoggingConfig()
        self.profiling_done: bool = False

    def init(self, cf: Config, devices: list) -> None:
        super().init(cf, devices)

        self.profiling_cfg = ProfilingConfig.from_config(self.cf)
        self.performance_cfg = PerformanceLoggingConfig.from_config(self.cf)
        logger.info(f"Profiling run: {self.profiling_cfg}, {self.performance_cfg}")

        if self.performance_cfg.throughput:
            self.perf_tracker = ThroughputTracker(
                device=torch.device(self.devices[0]),
                warmup_steps=self.performance_cfg.throughput_warmup_steps,
                batch_size_per_gpu=self.batch_size_per_gpu,
            )
        if self.profiling_cfg.nvtx_annotate:
            self.training_loop_annotation_context = nvtx_range

    @property
    def stops_after_profiling(self) -> bool:
        """Whether the run exists only to be profiled, and ends once it is."""
        return self.profiling_cfg.enabled and self.profiling_cfg.stop_after_profiling

    def mini_epochs(self, mini_epoch_base: int) -> Iterator[int]:
        """Run a single mini_epoch when the run only exists to be profiled."""
        if not self.stops_after_profiling:
            yield from super().mini_epochs(mini_epoch_base)
            return

        yield mini_epoch_base

    def train_batches(self, dataset_iter: Iterator) -> Iterator[tuple[int, ModelBatch]]:
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

        if self.stops_after_profiling:
            logger.info(f"Profiled {schedule.num_steps} training steps, ending the run.")
            return

        yield from enumerate(dataset_iter, start=schedule.num_steps)

    def validate(self, mini_epoch, mode_cfg, batch_size) -> None:
        """Skipped while the run only exists to be profiled."""
        if self.stops_after_profiling:
            return

        super().validate(mini_epoch, mode_cfg, batch_size)

    def save_model(self, mini_epoch: int, name=None) -> None:
        """Skipped while the run trains too few steps for its checkpoints to be useful."""
        if self.stops_after_profiling:
            return

        super().save_model(mini_epoch, name)
