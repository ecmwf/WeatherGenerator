# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Gradient stats after unscaling and clipping."""

from __future__ import annotations

import torch

from weathergen.train.monitoring.base import Monitor, register_monitor
from weathergen.train.monitoring.stats import DTensor, local_tensor
from weathergen.train.monitoring.weights import accumulate_norm_stats

TOTAL_GROUP = "all"


@register_monitor("grads")
class GradStats(Monitor):
    """Post-clip l2 / rms / absmax / min / max / nonfinite of the selected grads per group, plus
    the pre-clip total norm returned by `clip_grad_norm_` as `mon/grads/all/total_norm`."""

    def __init__(self, cfg, model, selection):
        super().__init__(cfg, model, selection)
        self.params = selection.params(model)

    @torch.no_grad()
    def on_after_clip(self, buf, total_norm):
        entries = [
            (g, local_tensor(p.grad) if p.grad is not None else None) for _, p, g in self.params
        ]
        accumulate_norm_stats(buf, self, entries)

        if total_norm is not None:
            if DTensor is not None and isinstance(total_norm, DTensor):
                total_norm = total_norm.full_tensor()  # collective; every rank calls this
            if not isinstance(total_norm, torch.Tensor):
                total_norm = torch.tensor(float(total_norm))
            key = self.key(TOTAL_GROUP, "total_norm")
            # identical on all ranks, so a max-reduction leaves it unchanged
            buf.accumulate(key, total_norm, "max")
            buf.output(key, lambda r, k=key: r[k])
