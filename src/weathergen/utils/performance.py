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
from lightning.fabric.utilities.throughput import Throughput, get_available_flops
from torch.utils.flop_counter import FlopCounterMode

from weathergen.utils.distributed import is_root

logger = logging.getLogger(__name__)


class ThroughputTracker:
    """Tracks training throughput and hardware utilisation metrics.

    Encapsulates Lightning's Throughput, per-batch FLOPs and source-byte
    bookkeeping, and the warmup / accumulation logic required to produce
    stable MFU / HFU estimates.
    """

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        world_size: int,
        window_size: int = 2,
        warmup_steps: int = 2,
        recompute_factor: float = 4 / 3,
    ) -> None:
        self._available_flops = get_available_flops(device, dtype=dtype)
        if is_root():
            if self._available_flops:
                logger.info(f"GPU peak FLOPS: {self._available_flops:.2e}")
            else:
                logger.warning(
                    "GPU peak FLOPS not recognized by Lightning — MFU will not be available."
                )
        self._throughput = Throughput(
            available_flops=self._available_flops,
            world_size=world_size,
            window_size=window_size,
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
            if istep == self._warmup_steps - 1:
                self._t0 = time.time()
                self._warmup_done = True
        else:
            self._total_batches += 1
            self._total_samples += batch_size_per_gpu
            self._total_mb += source_mb
            self._throughput.update(
                time=time.time() - self._t0,
                batches=self._total_batches,
                samples=self._total_samples,
            )

    def compute_metrics(self) -> dict[str, float] | None:
        """Return performance metrics dict, or None if warmup is not yet complete.

        Returns:
            Dict of ``"performance.<key>": value`` pairs, or None if no data yet.
        """
        if self._total_batches == 0 or self._t0 is None:
            return None
        elapsed = time.time() - self._t0
        return build_performance_metrics(
            self._throughput.compute(),
            elapsed,
            self._total_batches,
            self._total_mb,
            world_size=self._world_size,
            flops_fwd=self.flops_per_batch_fwd,
            flops_total=self.flops_per_batch,
            available_flops=self._available_flops,
            recompute_factor=self._recompute_factor,
        )



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


def build_performance_metrics(
    lightning_metrics: dict[str, float],
    elapsed: float,
    total_batches: int,
    total_mb: float,
    world_size: int = 1,
    flops_fwd: int | None = None,
    flops_total: int | None = None,
    available_flops: float | None = None,
    recompute_factor: float = 4 / 3,
) -> dict[str, float]:
    """Build the ``performance.*`` metrics dict ready for logging.

    Combines Lightning's Throughput.compute() output with MB/s, MFU, and HFU.
    Lightning's own ``mfu`` label is dropped: FlopCounterMode does not capture
    activation-checkpoint recomputation (see compute_hfu), so MFU and HFU are
    recomputed explicitly here using the theoretical recompute_factor.

    Args:
        lightning_metrics: Output of ``Throughput.compute()``.
        elapsed: Seconds since the start of throughput tracking (after warmup).
        total_batches: Number of tracked batches (equals steps when batch_size=1/rank).
        total_mb: Cumulative megabytes of source data processed on this device.
        world_size: Number of data-parallel ranks. Used to compute global MB/s.
        flops_fwd: Forward-only FLOPs per step from measure_model_flops, or None.
        flops_total: Forward + backward FLOPs per step from measure_model_flops, or None.
        available_flops: Peak device FLOP/s from get_available_flops, or None.
        recompute_factor: Recompute correction for HFU. Default 4/3 (full per-layer ckpt).

    Returns:
        Dict of ``"performance.<key>": value`` pairs.
    """
    steps_per_sec = total_batches / elapsed if elapsed > 0 else 0.0

    # Lift Lightning metrics, dropping its "mfu" label (recomputed below).
    metrics = {
        f"performance.throughput.{k.replace('/', '.')}": v
        for k, v in lightning_metrics.items()
        if isinstance(v, int | float) and k not in {"device/mfu", "mfu"}
    }

    device_mb_per_sec = total_mb / elapsed if elapsed > 0 else 0.0
    metrics["performance.throughput.device.mb_per_sec"] = device_mb_per_sec
    metrics["performance.throughput.mb_per_sec"] = device_mb_per_sec * world_size

    util = compute_utilisation_metrics(
        flops_fwd, flops_total, steps_per_sec, available_flops, recompute_factor
    )
    metrics.update({f"performance.utilization.{k}": v for k, v in util.items()})

    return metrics


def compute_utilisation_metrics(
    flops_fwd: int | None,
    flops_total: int | None,
    steps_per_sec: float,
    available_flops: float | None,
    recompute_factor: float = 4 / 3,
) -> dict[str, float]:
    """Compute MFU and HFU from measured FLOPs and device peak throughput.

    Returns a dict with keys ``"device.mfu"`` and ``"device.hfu"`` (both absent
    when the required inputs are None or zero).

    Args:
        flops_fwd: Forward-only FLOPs per step from measure_model_flops.
        flops_total: Forward + backward FLOPs per step from measure_model_flops.
        steps_per_sec: Training steps per second.
        available_flops: Peak device FLOP/s from get_available_flops.
        recompute_factor: See compute_hfu. Default 4/3 (full per-layer checkpointing).

    Returns:
        Dict with zero or more of ``{"device.mfu": float, "device.hfu": float}``.
    """
    if not available_flops or steps_per_sec <= 0:
        return {}

    metrics: dict[str, float] = {}
    if flops_fwd:
        metrics["device.mfu"] = compute_mfu(flops_fwd, steps_per_sec, available_flops)
    if flops_total:
        metrics["device.hfu"] = compute_hfu(
            flops_total, steps_per_sec, available_flops, recompute_factor
        )
    return metrics


def compute_mfu(flops_fwd: int, steps_per_sec: float, available_flops: float) -> float:
    """Compute Model FLOPs Utilization (MFU).

    MFU = (3 × fwd_flops × steps/sec) / available_flops

    The factor 3 accounts for forward (1×) plus backward (2×), with no
    activation-checkpoint recomputation overhead. This is the conventional
    MFU definition (PaLM/Chinchilla: 6N FLOPs per token = 3 × 2N model forward)
    and allows cross-model comparison independent of checkpointing strategy.

    Args:
        flops_fwd: Forward-only FLOPs per training step.
        steps_per_sec: Training steps per second.
        available_flops: Peak device FLOP/s.

    Returns:
        MFU as a fraction (typically in [0, 1]).
    """
    return 3 * flops_fwd * steps_per_sec / available_flops


def compute_hfu(
    flops_total: int,
    steps_per_sec: float,
    available_flops: float,
    recompute_factor: float = 1.0,
) -> float:
    """Compute Hardware FLOPs Utilization (HFU).

    HFU = (total_flops × recompute_factor × steps/sec) / available_flops

    **Recomputation overhead**

    PyTorch's FlopCounterMode (used by measure_flops) does NOT count activation-checkpoint
    recomputation when torch.utils.checkpoint is called with use_reentrant=False.
    This is because the non-reentrant implementation recomputes activations via a saved-tensor
    hook that bypasses the normal dispatch path that FlopCounterMode instruments.

    As a result, the flops_total argument here reflects only measured forward + backward FLOPs
    — it is a lower bound on the true hardware FLOPs executed per step.

    To account for recomputation, pass a recompute_factor > 1:
    - No checkpointing:       recompute_factor = 1.0  (flops_total ≈ 3 × fwd)
    - Full per-layer ckpt:    recompute_factor ≈ 4/3  (true total ≈ 4 × fwd instead of 3 ×)
    - Partial checkpointing:  recompute_factor between 1.0 and 4/3, proportional to the
                              fraction of forward FLOPs in checkpointed modules.

    The factor 4/3 comes from: without recompute the training step costs 3 × fwd FLOPs
    (fwd + 2 × bwd); with full per-layer recompute an extra fwd pass is re-executed during
    backward, raising the total to 4 × fwd FLOPs, giving a ratio of 4/3.

    HFU > MFU precisely when recompute_factor × flops_total > 3 × flops_fwd.

    Args:
        flops_total: Measured FLOPs per training step (fwd + bwd, from measure_model_flops).
        steps_per_sec: Training steps per second.
        available_flops: Peak device FLOP/s.
        recompute_factor: Multiplier to account for activation-checkpoint recomputation
                          (default 1.0 = no correction; use 4/3 for full per-layer checkpointing).

    Returns:
        HFU as a fraction (typically in [0, 1]).
    """
    return flops_total * recompute_factor * steps_per_sec / available_flops
