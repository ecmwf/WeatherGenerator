# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Performance logging: cheap metrics of the whole run (throughput, peak memory), and nvtx
ranges.

Tracing a bounded stretch of training lives in `weathergen.utils.profiling`. Both are
driven by `weathergen.train.profiling_trainer.ProfilingTrainer`.
"""

import logging
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import torch

from weathergen.common.config import Config

logger = logging.getLogger(__name__)

_GIB = 1024**3


def _throughput_section(cf: Config) -> Config | dict:
    return (cf.get("performance_logging") or {}).get("throughput") or {}


def _memory_section(cf: Config) -> Config | dict:
    return (cf.get("performance_logging") or {}).get("memory") or {}


class ThroughputTracker:
    """Tracks training throughput as cumulative global counts.

    `step` only accumulates this rank's batch, sample and source-megabyte counts, without
    device synchronization or collectives. `compute_metrics` reduces them across ranks once
    per logging interval. Throughput is the difference between two logged records divided
    by the difference of their timestamps, which also discards the warmup steps.

    Configured by `performance_logging.throughput.enabled`, which is absent from
    `config/default_config.yml`. Cheap and does not change training, so safe for full runs.
    """

    @staticmethod
    def is_enabled(cf: Config) -> bool:
        """Whether the run config asks for throughput logging."""
        return _throughput_section(cf).get("enabled", False)

    @classmethod
    def from_config(
        cls, cf: Config, device: torch.device, batch_size_per_gpu: int
    ) -> "ThroughputTracker | None":
        """The tracker the run config asks for, or None if throughput logging is off."""
        if not cls.is_enabled(cf):
            return None

        return cls(device=device, batch_size_per_gpu=batch_size_per_gpu)

    def __init__(self, device: torch.device, batch_size_per_gpu: int) -> None:
        self._device = device
        self.batch_size_per_gpu = batch_size_per_gpu
        self._total_batches: int = 0
        self._total_samples: int = 0
        self._total_mb: float = 0.0

    def step(self, batch) -> None:
        """Accumulate one training step's counts on this rank. No sync, no collective.

        Args:
            batch: The current training batch (must expose ``get_source_samples()``).
        """
        # computed per step since the number of source tokens varies across batches
        self.update(compute_source_bytes(batch.get_source_samples()) / 1e6)

    def update(self, source_mb: float) -> None:
        """Accumulate one batch with `source_mb` megabytes of source tokens on this rank."""
        self._total_batches += 1
        self._total_samples += self.batch_size_per_gpu
        self._total_mb += source_mb

    def compute_metrics(self) -> dict[str, float] | None:
        """Return the cumulative global counts, or None before the first step.

        Collective: a single all-reduce, so it must be called on every rank at the same
        point in the training loop. The returned dict is identical on all ranks.

        Returns:
            Dict of ``"performance.throughput.global.<count>": value`` pairs.
        """
        if self._total_batches == 0:
            return None

        totals = [self._total_batches, self._total_samples, self._total_mb]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            reduced = torch.tensor(totals, dtype=torch.float64, device=self._device)
            torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
            totals = reduced.tolist()

        batches, samples, mb = totals
        return {
            "performance.throughput.global.batches": batches,
            "performance.throughput.global.samples": samples,
            "performance.throughput.global.mb": mb,
        }


class MemoryTracker:
    """Tracks the global (max across all ranks) peak GPU memory per window.

    Reads the CUDA caching allocator's high-water marks and resets them, so each window gets
    the peak since the previous read. `close_window` folds that peak into a window locally,
    e.g. to keep a checkpoint's peak out of the training window; `collect` reports a window,
    with the MAX across all ranks.

    `max_memory_allocated` is the peak of memory occupied by live tensors (parameters,
    gradients, optimizer states, activations): the model's actual demand, comparable to
    analytic memory estimates. `max_memory_reserved` is the peak of memory the caching
    allocator has claimed from the device; it also counts cached blocks that are free but not
    returned to CUDA, so reserved >= allocated and the gap measures fragmentation / caching
    slack. An OOM is raised when a new cudaMalloc fails, so reserved (plus non-PyTorch usage
    such as NCCL buffers and cuBLAS workspaces, which neither counter sees) is what determines
    headroom against the device's capacity.

    Configured by `performance_logging.memory.enabled`, which is absent from
    `config/default_config.yml`. Cheap and does not change training, so safe for full runs.
    """

    @staticmethod
    def is_enabled(cf: Config) -> bool:
        """Whether the run config asks for peak-memory logging."""
        return _memory_section(cf).get("enabled", False)

    @classmethod
    def from_config(cls, cf: Config, device: torch.device) -> "MemoryTracker | None":
        """The tracker the run config asks for, or None if it is off or CUDA is unavailable."""
        if not cls.is_enabled(cf):
            return None
        if not torch.cuda.is_available():
            logger.warning("Peak-memory logging requested, but CUDA is unavailable.")
            return None

        return cls(device=device)

    def __init__(self, device: torch.device) -> None:
        self._device = device
        # local (allocated, reserved) peaks of windows closed but not yet collected
        self._pending: dict[str, tuple[int, int]] = {}
        # start with a clean window so the first collect reports its own peak
        torch.cuda.reset_peak_memory_stats(device)

    def close_window(self, window: str) -> None:
        """Attribute the peak since the last read to `window`, without reporting it yet.

        Local: no synchronization across ranks.
        """
        max_allocated = torch.cuda.max_memory_allocated(self._device)
        max_reserved = torch.cuda.max_memory_reserved(self._device)
        torch.cuda.reset_peak_memory_stats(self._device)

        pending_allocated, pending_reserved = self._pending.get(window, (0, 0))
        self._pending[window] = (
            max(pending_allocated, max_allocated),
            max(pending_reserved, max_reserved),
        )

    def collect(self, window: str) -> dict[str, float]:
        """Close `window` and return its peak-memory metrics since it was last collected.

        Collective: must be called on every rank at the same point in the training loop
        (the cross-rank MAX reduction and the peak-stat reset run on every rank). The
        returned dict is identical on all ranks.

        Args:
            window: what the window covers (e.g. ``"train"``, ``"validation"``,
                ``"save_model"``); becomes part of the metric key.

        Returns:
            Dict of ``"performance.memory.<window>.max_{allocated,reserved}_gib"`` pairs.
        """
        self.close_window(window)
        max_allocated, max_reserved = self._pending.pop(window)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            peaks = torch.tensor(
                [max_allocated, max_reserved], dtype=torch.int64, device=self._device
            )
            torch.distributed.all_reduce(peaks, op=torch.distributed.ReduceOp.MAX)
            max_allocated, max_reserved = peaks.tolist()

        return {
            f"performance.memory.{window}.max_allocated_gib": max_allocated / _GIB,
            f"performance.memory.{window}.max_reserved_gib": max_reserved / _GIB,
        }


def compute_source_bytes(source_samples) -> int:
    """Count total bytes of all source token tensors in a batch.

    Args:
        source_samples: Result of sample_batch.get_source_samples(), containing
                        a list of samples each with per-stream source token cells.

    Returns:
        Total byte count across all streams and cells in the batch.
    """
    total = 0
    for sample in source_samples.samples:
        for stream_data in sample.streams_data.values():
            for t in stream_data.source_tokens_cells:
                total += t.nbytes
    return total


@contextmanager
def nvtx_range(name):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def nvtx_batches[T](batches: Iterable[tuple[int, T]]) -> Iterator[tuple[int, T]]:
    """Wrap the consumer's work on each yielded (index, batch) in an nvtx range."""
    for bidx, batch in batches:
        # the range stays open while the consumer runs the step on the yielded batch
        with nvtx_range(f"batch_{bidx}"):
            yield bidx, batch


def _nvtx_push(name: str):
    torch.cuda.nvtx.range_push(name)


def _nvtx_pop():
    torch.cuda.nvtx.range_pop()


def register_nvtx_hooks(model, scope: str = "global"):
    torch.nn.modules.module.register_module_forward_pre_hook(
        lambda m, args: _nvtx_push(f"{m.__class__.__name__}.forward")
    )
    torch.nn.modules.module.register_module_forward_hook(
        lambda m, input, output: _nvtx_pop(), always_call=True
    )
    torch.nn.modules.module.register_module_full_backward_pre_hook(
        lambda m, args: _nvtx_push(f"{m.__class__.__name__}.backward")
    )
    torch.nn.modules.module.register_module_full_backward_hook(lambda m, input, output: _nvtx_pop())
