# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Module output stats via forward hooks that live only inside `manager.forward()`."""

from __future__ import annotations

import math

import torch

from weathergen.train.monitoring.base import Monitor, StatBuffer, register_monitor
from weathergen.train.monitoring.stats import absmax, local_tensor, stats_context, sumsq


def _first_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, list | tuple):
        for o in output:
            if isinstance(o, torch.Tensor):
                return o
    return None


@register_monitor("activations")
class ActivationStats(Monitor):
    """rms, absmax and outlier fraction over the elements of each selected module's output,
    accumulated over all calls of the module within one forward.

    `outlier_frac` is the fraction of elements with |x| > `outlier_k` * rms, where rms is that
    of the same call on this rank (two passes over the output; disabled if `outlier_k` is null).
    `max_tokens` takes a strided subsample of at most that many rows (last dim = features) per
    call, to bound the cost on large outputs; all stats are then estimates from the subsample.

    Hooks are registered on entering `manager.forward()` and removed on leaving it, i.e. before
    backward, so activation-checkpoint recomputation does not count twice. In `group_by: block`
    mode only block-level modules (e.g. `fe_blocks.3`) are hooked.
    """

    def __init__(self, cfg, model, selection):
        super().__init__(cfg, model, selection)
        self.modules = selection.modules(model, block_level_only=True)
        outlier_k = cfg.get("outlier_k", None)
        self.outlier_k = float(outlier_k) if outlier_k else None
        max_tokens = cfg.get("max_tokens", None)
        self.max_tokens = int(max_tokens) if max_tokens else None
        self._handles = []
        self._buf: StatBuffer | None = None

    def on_forward_start(self, buf):
        self._buf = buf
        for _, module, group in self.modules:
            self._register_group(buf, group)
            self._handles.append(module.register_forward_hook(self._make_hook(group)))

    def on_forward_end(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._buf = None

    def _keys(self, group):
        return {s: self.key(group, f"#{s}") for s in ("sumsq", "count", "absmax", "outliers")}

    def _register_group(self, buf, group):
        k = self._keys(group)
        buf.register(k["sumsq"], "sum")
        buf.register(k["count"], "sum")
        buf.register(k["absmax"], "max")

        def out(fn):
            return lambda r: fn(r) if r[k["count"]] > 0 else None

        buf.output(self.key(group, "rms"), out(lambda r: math.sqrt(r[k["sumsq"]] / r[k["count"]])))
        buf.output(self.key(group, "absmax"), out(lambda r: r[k["absmax"]]))
        if self.outlier_k is not None:
            buf.register(k["outliers"], "sum")
            buf.output(
                self.key(group, "outlier_frac"), out(lambda r: r[k["outliers"]] / r[k["count"]])
            )

    def _subsample(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_tokens is None or x.dim() < 2:
            return x
        rows = x.reshape(-1, x.shape[-1])
        stride = math.ceil(rows.shape[0] / self.max_tokens)
        return rows[::stride] if stride > 1 else rows

    def _make_hook(self, group):
        k = self._keys(group)

        def hook(module, args, output):
            x = _first_tensor(output)
            buf = self._buf
            if buf is None or x is None or not x.is_floating_point() or x.numel() == 0:
                return
            with stats_context(x.device):
                x = self._subsample(local_tensor(x))
                sq = sumsq(x)
                buf.accumulate(k["sumsq"], sq, "sum")
                buf.accumulate(k["count"], x.numel(), "sum")
                buf.accumulate(k["absmax"], absmax(x), "max")
                if self.outlier_k is not None:
                    threshold = self.outlier_k * (sq / x.numel()).sqrt()
                    buf.accumulate(k["outliers"], (x.abs() > threshold).sum(), "sum")

        return hook
