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
    """rms and absmax over all elements of each selected module's output, accumulated over all
    calls of the module within one forward.

    Hooks are registered on entering `manager.forward()` and removed on leaving it, i.e. before
    backward, so activation-checkpoint recomputation does not count twice. In `group_by: block`
    mode only block-level modules (e.g. `fe_blocks.3`) are hooked.
    """

    def __init__(self, cfg, model, selection):
        super().__init__(cfg, model, selection)
        self.modules = selection.modules(model, block_level_only=True)
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

    def _register_group(self, buf, group):
        k_sq, k_n, k_amax = (self.key(group, s) for s in ("#sumsq", "#count", "#absmax"))
        buf.register(k_sq, "sum")
        buf.register(k_n, "sum")
        buf.register(k_amax, "max")
        buf.output(
            self.key(group, "rms"),
            lambda r: math.sqrt(r[k_sq] / r[k_n]) if r[k_n] > 0 else None,
        )
        buf.output(self.key(group, "absmax"), lambda r: r[k_amax] if r[k_n] > 0 else None)

    def _make_hook(self, group):
        k_sq, k_n, k_amax = (self.key(group, s) for s in ("#sumsq", "#count", "#absmax"))

        def hook(module, args, output):
            x = _first_tensor(output)
            buf = self._buf
            if buf is None or x is None or not x.is_floating_point() or x.numel() == 0:
                return
            with stats_context(x.device):
                x = local_tensor(x)
                buf.accumulate(k_sq, sumsq(x), "sum")
                buf.accumulate(k_n, x.numel(), "sum")
                buf.accumulate(k_amax, absmax(x), "max")

        return hook
