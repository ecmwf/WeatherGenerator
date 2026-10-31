# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Adam(W) optimizer-state stats: second-moment magnitude vs ε and first-moment rms."""

from __future__ import annotations

import math

import torch

from weathergen.train.monitoring.base import Monitor, register_monitor
from weathergen.train.monitoring.stats import local_tensor, sumsq


@register_monitor("adam_state")
class AdamStateStats(Monitor):
    """Per group, from the (bias-uncorrected) Adam state after the optimizer step:

    - `sqrt_v_mean`: mean of √v (≈ gradient rms seen by Adam);
    - `frac_sqrt_v_below_eps`: fraction of elements with √v < `eps_mult` * ε, where updates are
      dominated by ε (Wortsman et al. 2023);
    - `m_rms`: rms of the first moment.

    Params without `exp_avg_sq` in their optimizer state (e.g. Muon) are skipped.
    """

    def __init__(self, cfg, model, selection):
        super().__init__(cfg, model, selection)
        self.params = selection.params(model)
        self.eps_mult = float(cfg.get("eps_mult", 10.0))
        self._opt_of: dict[int, tuple[torch.optim.Optimizer, float]] = {}

    def bind_optimizers(self, optimizers):
        for opt in optimizers:
            for group in opt.param_groups:
                eps = float(group.get("eps", 0.0))
                for p in group["params"]:
                    self._opt_of[id(p)] = (opt, eps)

    def _keys(self, group):
        return {s: self.key(group, f"#{s}") for s in ("sqrt_v_sum", "below", "m_sumsq", "count")}

    @torch.no_grad()
    def on_step_end(self, buf):
        for _, p, group in self.params if self.adds_param_stats() else []:
            entry = self._opt_of.get(id(p))
            if entry is None:
                continue
            opt, eps = entry
            state = opt.state.get(p, {})
            if "exp_avg_sq" not in state:
                continue
            v = local_tensor(state["exp_avg_sq"])
            if v.numel() == 0:
                continue
            k = self._keys(group)
            sqrt_v = v.float().sqrt()
            buf.accumulate(k["sqrt_v_sum"], sqrt_v.sum(), "sum")
            buf.accumulate(k["below"], (sqrt_v < self.eps_mult * eps).sum(), "sum")
            buf.accumulate(k["count"], v.numel(), "sum")
            if "exp_avg" in state:
                buf.accumulate(k["m_sumsq"], sumsq(local_tensor(state["exp_avg"])), "sum")

        for group in dict.fromkeys(g for _, _, g in self.params):
            k = self._keys(group)
            for name in ("sqrt_v_sum", "below", "m_sumsq", "count"):
                buf.register(k[name], "sum")

            def out(fn, k=k):
                return lambda r: fn(r) if r[k["count"]] > 0 else None

            buf.output(
                self.key(group, "sqrt_v_mean"),
                out(lambda r, k=k: r[k["sqrt_v_sum"]] / r[k["count"]]),
            )
            buf.output(
                self.key(group, "frac_sqrt_v_below_eps"),
                out(lambda r, k=k: r[k["below"]] / r[k["count"]]),
            )
            buf.output(
                self.key(group, "m_rms"),
                out(lambda r, k=k: math.sqrt(r[k["m_sumsq"]] / r[k["count"]])),
            )
