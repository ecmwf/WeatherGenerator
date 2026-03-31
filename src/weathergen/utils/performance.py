# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Utilities for measuring and computing model performance metrics (MFU, HFU)."""

import logging
import time
from collections.abc import Callable
from contextlib import contextmanager, nullcontext

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode
from lightning.fabric.utilities.throughput import get_available_flops

from weathergen.utils.distributed import is_root

logger = logging.getLogger(__name__)


class ThroughputTracker:
    """Tracks training throughput and hardware utilisation metrics.

    Accumulates per-batch FLOPs and source-byte counts across ranks, with the warmup
    / accumulation logic required to produce stable MFU / HFU estimates and global
    throughput metrics.
    """

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        world_size: int,
        warmup_steps: int,
        recompute_factor: float = 4 / 3,
    ) -> None:
        self._available_flops = get_available_flops(device, dtype=dtype)
        self._device = device
        if is_root():
            if self._available_flops:
                logger.info(f"GPU peak FLOPS: {self._available_flops:.2e}")
            else:
                logger.warning(
                    "GPU peak FLOPS not recognized — MFU will not be available."
                )
        self.flops_per_batch_fwd: int | None = None
        self.flops_per_batch: int | None = None
        self._warmup_steps = warmup_steps
        self._t0: float | None = None
        self._warmup_done: bool = False
        self._total_batches: int = 0
        self._total_samples: int = 0
        self._total_mb: float = 0.0
        self._world_size = world_size
        self._recompute_factor = recompute_factor
        # Cached results from the last _sync() call (set on all ranks, read on root).
        self._synced_elapsed: float | None = None
        self._synced_global_batches: int = 0
        self._synced_global_samples: int = 0
        self._synced_global_mb: float = 0.0

    @contextmanager
    def step_context(self):
        """Context manager spanning the full training step (fwd + bwd + optimizer).

        Wrap the entire training step body with this to count all compute ops.
        Nest ``flop_ctx()`` inside it around the model forward to additionally
        capture forward-only FLOPs for MFU:

            with self.perf_tracker.step_ctx():
                with self.perf_tracker.flop_ctx():
                    preds = model(...)
                loss = loss_fn(preds)
                loss.backward()
                optimizer.step()
        """
        with FlopCounterMode(display=False) as counter:
            yield
        self.flops_per_batch = counter.get_total_flops()

    @contextmanager
    def forward_context(self):
        """Context manager that counts forward-only FLOPs. Nest inside ``step_ctx()``.

        Wrap the model forward pass with this to separately capture forward FLOPs
        for MFU, while the enclosing ``step_ctx()`` accumulates the full total:

            with self.perf_tracker.flop_ctx():
                preds = model(params, batch.get_source_samples())
        """
        with FlopCounterMode(display=False) as counter:
            yield
        self.flops_per_batch_fwd = counter.get_total_flops()

    def step(
        self,
        batch,
        batch_size_per_gpu: int,
        istep: int,
        log_fn: Callable[[dict[str, float]], None] | None = None,
    ) -> None:
        """Record one training step and optionally log metrics.

        Wrapper around ``update`` and ``compute_metrics`` that also computes
        source bytes from the batch on the fly. When metrics are available and
        the current rank is root, ``log_fn`` is called with the metrics dict.

        Args:
            batch: The current training batch (must expose ``get_source_samples()``).
            batch_size_per_gpu: Number of samples processed on this rank.
            istep: Global training step index.
            log_fn: Called with the metrics dict on the root rank once warmup is
                    complete. Typically ``lambda m: logger.log_metrics(stage, m, step=istep)``.
        """
        source_mb = compute_source_bytes(batch.get_source_samples()) / 1e6
        self.update(batch_size_per_gpu, istep, source_mb)
        self._sync()  # collective: all ranks must participate
        if log_fn is not None and is_root():
            metrics = self.compute_metrics()
            if metrics is not None:
                log_fn(metrics)

    def update(self, batch_size_per_gpu: int, istep: int, source_mb: float) -> None:
        """Record one training step, handling warmup internally.

        Args:
            batch_size_per_gpu: Number of samples processed on this rank.
            istep: Global training step index (used for warmup countdown).
            source_mb: Source tensor megabytes for this batch. Should be computed
                       fresh each step via ``compute_source_bytes`` as batch sizes
                       can vary across samples.
        """
        torch.cuda.synchronize()
        if not self._warmup_done:
            if istep >= self._warmup_steps - 1:
                self._t0 = time.time()
                self._warmup_done = True
        else:
            self._total_batches += 1
            self._total_samples += batch_size_per_gpu
            self._total_mb += source_mb

    def _sync(self) -> None:
        """Collective: reduce per-rank counters across all ranks and cache the result.

        Must be called on every rank at the same point in the training loop.
        The cached values are later read by ``compute_metrics()`` on the root rank.
        """
        if self._total_batches == 0 or self._t0 is None:
            return

        elapsed = time.time() - self._t0

        global_batches = torch.tensor(self._total_batches, dtype=torch.int64, device=self._device)
        global_samples = torch.tensor(self._total_samples, dtype=torch.int64, device=self._device)
        global_total_mb = torch.tensor(self._total_mb, dtype=torch.float32, device=self._device)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            elapsed_tensor = torch.tensor(elapsed, dtype=torch.float32, device=self._device)
            torch.distributed.all_reduce(elapsed_tensor, op=torch.distributed.ReduceOp.AVG)
            elapsed = elapsed_tensor.item()

            torch.distributed.all_reduce(global_batches)
            torch.distributed.all_reduce(global_samples)
            torch.distributed.all_reduce(global_total_mb)

        self._synced_elapsed = elapsed
        self._synced_global_batches = int(global_batches.item())
        self._synced_global_samples = int(global_samples.item())
        self._synced_global_mb = global_total_mb.item()

    def compute_metrics(self) -> dict[str, float] | None:
        """Return performance metrics dict, or None if warmup is not yet complete.

        Returns:
            Dict of ``"performance.<key>": value`` pairs, or None if no data yet.
        """
        if self._total_batches == 0 or self._t0 is None:
            return None
        elapsed = time.time() - self._t0
        steps_per_sec = self._total_batches / elapsed if elapsed > 0 else 0.0

        metrics = {}
        metrics.update(self._compute_throughput_metrics(elapsed))
        metrics.update(self._compute_utilization_metrics(steps_per_sec))
        return metrics

    def _compute_throughput_metrics(self, elapsed: float) -> dict[str, float]:
        """Compute throughput metrics (samples/sec, batches/sec, MB/s).

        Measures both device-level (this rank) and global (all ranks) throughput.
        Global values come from ``_sync()``, which must have been called on all ranks
        before this method is invoked on the root rank.

        Args:
            elapsed: Device-local seconds elapsed since tracking started (after warmup).

        Returns:
            Dict of ``"performance.throughput.*"`` metrics.
        """
        metrics: dict[str, float] = {}

        if elapsed <= 0 or self._synced_elapsed is None or self._synced_elapsed <= 0:
            return metrics

        # Device-level throughput (this rank only).
        metrics["performance.throughput.device.batches_per_sec"] = self._total_batches / elapsed
        metrics["performance.throughput.device.samples_per_sec"] = self._total_samples / elapsed
        metrics["performance.throughput.device.mb_per_sec"] = self._total_mb / elapsed

        # Global throughput: use values already reduced across all ranks by _sync().
        synced_elapsed = self._synced_elapsed
        metrics["performance.throughput.global.batches_per_sec"] = self._synced_global_batches / synced_elapsed
        metrics["performance.throughput.global.samples_per_sec"] = self._synced_global_samples / synced_elapsed
        metrics["performance.throughput.global.mb_per_sec"] = self._synced_global_mb / synced_elapsed

        return metrics

    def _compute_utilization_metrics(self, steps_per_sec: float) -> dict[str, float]:
        """Compute model and hardware FLOPs utilization (MFU and HFU).

        Args:
            steps_per_sec: Training steps per second.

        Returns:
            Dict of ``"performance.utilization.*"`` metrics (empty if data unavailable).
        """
        metrics: dict[str, float] = {}

        if not self._available_flops or steps_per_sec <= 0:
            return metrics

        if self.flops_per_batch_fwd:
            # MFU = 3 × fwd_flops × steps/sec / available_flops
            mfu = 3 * self.flops_per_batch_fwd * steps_per_sec / self._available_flops
            metrics["performance.utilization.device.mfu"] = mfu

        if self.flops_per_batch:
            # HFU = total_flops × recompute_factor × steps/sec / available_flops
            hfu = (
                self.flops_per_batch
                * self._recompute_factor
                * steps_per_sec
                / self._available_flops
            )
            metrics["performance.utilization.device.hfu"] = hfu

        return metrics



class NullThroughputTracker:
    """No-op throughput tracker used when performance tracking is disabled.

    Implements the same interface as ``ThroughputTracker`` so call sites in the
    training loop need no ``if`` guards.
    """

    def step_context(self):
        return nullcontext()

    def forward_context(self):
        return nullcontext()

    def step(self, batch, batch_size_per_gpu: int, istep: int, log_fn=None) -> None:
        pass


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



