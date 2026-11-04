# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""MonitorManager: scheduling, hook lifecycle, fused reduction and logging.

Trainer call order per step:

    manager.begin_step(istep)
    with manager.forward():
        preds = model(...)
    # loss-spike skip: manager.abort_step(); continue
    loss.backward(); unscale; total_norm = clip_grad_norm_(...)
    manager.on_after_clip(total_norm)
    optimizer.step(); lr / EMA updates
    manager.end_step()

The active monitors are decided from `istep` alone, so every rank runs the same collectives in
`end_step` without communicating.
"""

from __future__ import annotations

import contextlib
import logging

import torch

from weathergen.train.monitoring.base import Monitor, StatBuffer
from weathergen.train.monitoring.stats import local_tensor
from weathergen.utils.distributed import is_root

logger = logging.getLogger(__name__)

TRAIN_STAGE = "train"


class NullMonitorManager:
    """Monitoring disabled: every call is a no-op."""

    def begin_step(self, istep: int) -> bool:
        return False

    def forward(self):
        return contextlib.nullcontext()

    def on_after_clip(self, total_norm) -> None:
        pass

    def end_step(self) -> None:
        pass

    def abort_step(self) -> None:
        pass


class MonitorManager:
    def __init__(self, model: torch.nn.Module, monitors: list[Monitor], train_logger):
        self.model = model
        self.monitors = monitors
        self.train_logger = train_logger
        self.device = self._infer_device(model)
        self._active: list[Monitor] = []
        self._buf: StatBuffer | None = None
        self._istep: int | None = None

    @staticmethod
    def _infer_device(model) -> torch.device:
        for p in model.parameters():
            return local_tensor(p).device
        return torch.device("cpu")

    def begin_step(self, istep: int) -> bool:
        """Decide the active monitors for this step. Returns True if any is active."""
        if self._active:  # previous step was neither ended nor aborted
            self.abort_step()
        self._istep = istep
        self._active = [m for m in self.monitors if m.is_active(istep)]
        if self._active:
            self._buf = StatBuffer(self.device)
        return bool(self._active)

    def forward(self):
        """Context manager around the model forward: hooks/probes exist only inside it."""
        if not self._active:
            return contextlib.nullcontext()
        return self._monitored_forward()

    @contextlib.contextmanager
    def _monitored_forward(self):
        started = []
        try:
            for monitor in self._active:
                started.append(monitor)
                monitor.on_forward_start(self._buf)
            yield
        finally:
            for monitor in started:
                monitor.on_forward_end()

    def on_after_clip(self, total_norm) -> None:
        for monitor in self._active:
            monitor.on_after_clip(self._buf, total_norm)

    def end_step(self) -> None:
        """Weights / Δw stats, one all-reduce per op, one D2H copy, log on root."""
        if not self._active:
            return
        try:
            for monitor in self._active:
                monitor.on_step_end(self._buf)
            metrics = self._buf.reduce()
            if metrics and is_root():
                self.train_logger.log_metrics(
                    TRAIN_STAGE, {f"mon/{k}": v for k, v in metrics.items()}, step=self._istep
                )
        finally:
            self._clear()

    def abort_step(self) -> None:
        """Skipped step: drop hooks, probes, snapshots and stats. No collectives, no logging."""
        self._clear()

    def _clear(self) -> None:
        for monitor in self._active:
            monitor.reset()
        self._active = []
        self._buf = None
