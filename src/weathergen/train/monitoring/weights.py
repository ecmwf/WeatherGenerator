# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Weight norms (post optimizer step) and the update-to-weight ratio ‖Δw‖/(‖w‖+ε)."""

from __future__ import annotations

import math

import torch

from weathergen.train.monitoring.base import Monitor, StatBuffer, register_monitor
from weathergen.train.monitoring.stats import foreach_sumsq_absmax, local_tensor


def accumulate_norm_stats(
    buf: StatBuffer, monitor: Monitor, entries: list[tuple[str, torch.Tensor | None]]
) -> None:
    """Accumulate l2 / rms / absmax / min / max / nonfinite per group from local shards
    `(group, tensor)`.

    `nonfinite` counts the tensors of the group whose norm is not finite (contains nan/inf);
    it is derived from the sum of squares, so it costs no extra pass over the data.

    `None` or empty shards (FSDP2 uneven sharding, unused params) contribute identity values,
    so every rank adds the same set of keys. Outputs are only logged for groups with data.
    """
    nonempty = [(g, t) for g, t in entries if t is not None and t.numel() > 0]
    if not monitor.adds_param_stats():
        nonempty = []
    sq, amax = foreach_sumsq_absmax([t for _, t in nonempty])
    for (group, t), s, a in zip(nonempty, sq, amax, strict=True):
        t_min, t_max = torch.aminmax(t)
        buf.accumulate(monitor.key(group, "#sumsq"), s, "sum")
        buf.accumulate(monitor.key(group, "#absmax"), a, "max")
        buf.accumulate(monitor.key(group, "#min"), t_min, "min")
        buf.accumulate(monitor.key(group, "#max"), t_max, "max")
        buf.accumulate(monitor.key(group, "#nonfinite"), (~torch.isfinite(s)).float(), "sum")
        buf.accumulate(monitor.key(group, "#count"), t.numel(), "sum")

    for group in dict.fromkeys(g for g, _ in entries):
        k_sq, k_amax, k_n = (monitor.key(group, s) for s in ("#sumsq", "#absmax", "#count"))
        k_min, k_max, k_nf = (monitor.key(group, s) for s in ("#min", "#max", "#nonfinite"))
        buf.register(k_sq, "sum")
        buf.register(k_amax, "max")
        buf.register(k_min, "min")
        buf.register(k_max, "max")
        buf.register(k_nf, "sum")
        buf.register(k_n, "sum")

        def has_data(r, k_n=k_n):
            return r.get(k_n, 0) > 0

        for stat, k in (("min", k_min), ("max", k_max), ("nonfinite", k_nf)):
            buf.output(monitor.key(group, stat), lambda r, k=k, h=has_data: r[k] if h(r) else None)

        buf.output(
            monitor.key(group, "l2"),
            lambda r, k=k_sq, h=has_data: math.sqrt(r[k]) if h(r) else None,
        )
        buf.output(
            monitor.key(group, "rms"),
            lambda r, k=k_sq, n=k_n, h=has_data: math.sqrt(r[k] / r[n]) if h(r) else None,
        )
        buf.output(
            monitor.key(group, "absmax"),
            lambda r, k=k_amax, h=has_data: r[k] if h(r) else None,
        )


@register_monitor("weights")
class WeightStats(Monitor):
    """l2 / rms / absmax / min / max / nonfinite of the selected weights after the optimizer
    step."""

    def __init__(self, cfg, model, selection):
        super().__init__(cfg, model, selection)
        self.params = selection.params(model)

    @torch.no_grad()
    def on_step_end(self, buf):
        accumulate_norm_stats(buf, self, [(g, local_tensor(p)) for _, p, g in self.params])


@register_monitor("updates")
class UpdateRatio(Monitor):
    """‖Δw‖/(‖w‖+ε) per group, Δw = change made by this step's optimizer update.

    The pre-step local shards are snapshotted in `on_after_clip` (nothing modifies the student
    weights between clipping and `optimizer.step()`), only on monitor steps. ε avoids inf/NaN
    for zero-initialised weights (e.g. AdaLNZero).
    """

    def __init__(self, cfg, model, selection):
        super().__init__(cfg, model, selection)
        self.params = selection.params(model)
        self.eps = float(cfg.get("eps", 1e-12))
        self._snapshot: list[torch.Tensor] | None = None

    @torch.no_grad()
    def on_after_clip(self, buf, total_norm):
        if self.adds_param_stats():
            self._snapshot = [local_tensor(p).clone() for _, p, _ in self.params]
        else:
            self._snapshot = []  # replica: only registers the (identity) keys

    @torch.no_grad()
    def on_step_end(self, buf):
        if self._snapshot is None:
            return
        new = [local_tensor(p) for _, p, _ in self.params] if self._snapshot else []
        idx = [i for i, t in enumerate(new) if t.numel() > 0]
        dw_sq, w_sq = [], []
        if idx:
            new_ne = [new[i] for i in idx]
            dw = torch._foreach_sub(new_ne, [self._snapshot[i] for i in idx])
            dw_sq = [d.square() for d in torch._foreach_norm(dw, 2, dtype=torch.float32)]
            w_sq = [w.square() for w in torch._foreach_norm(new_ne, 2, dtype=torch.float32)]
            del dw
        self._snapshot = None
        for j, i in enumerate(idx):
            group = self.params[i][2]
            buf.accumulate(self.key(group, "#dw_sumsq"), dw_sq[j], "sum")
            buf.accumulate(self.key(group, "#w_sumsq"), w_sq[j], "sum")
            buf.accumulate(self.key(group, "#count"), new[i].numel(), "sum")

        for group in dict.fromkeys(g for _, _, g in self.params):
            k_dw, k_w, k_n = (self.key(group, s) for s in ("#dw_sumsq", "#w_sumsq", "#count"))
            buf.register(k_dw, "sum")
            buf.register(k_w, "sum")
            buf.register(k_n, "sum")
            buf.output(
                self.key(group, "ratio"),
                lambda r, k_dw=k_dw, k_w=k_w, k_n=k_n: (
                    math.sqrt(r[k_dw]) / (math.sqrt(r[k_w]) + self.eps) if r[k_n] > 0 else None
                ),
            )

    def reset(self):
        self._snapshot = None
