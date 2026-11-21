# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Attention probe: per-head max logit and softmax entropy, recomputed on a small sample.

Attention modules expose a `probe` attribute (None by default) and, after q/k are computed
(post QK-norm / RoPE, before the fused kernel), call
`if self.probe is not None: self.probe(q, k, scale, softcap, cu_seqlens_q, cu_seqlens_k)`
with q: [T_q, H, D], k: [T_k, H, D] in varlen layout. The manager sets the probes only on
monitor steps, inside `manager.forward()`.
"""

from __future__ import annotations

import math

import torch

from weathergen.train.monitoring.base import Monitor, StatBuffer, register_monitor
from weathergen.train.monitoring.stats import stats_context


def _first_nonempty_sequence(cu_seqlens_q, cu_seqlens_k):
    """First sequence with queries and >= 2 keys (a single key has entropy 0 by definition,
    e.g. one-token healpix cells in the local engine), else the first with any keys."""
    cu_q = cu_seqlens_q.tolist() if isinstance(cu_seqlens_q, torch.Tensor) else list(cu_seqlens_q)
    cu_k = cu_seqlens_k.tolist() if isinstance(cu_seqlens_k, torch.Tensor) else list(cu_seqlens_k)
    fallback = None
    for i in range(min(len(cu_q), len(cu_k)) - 1):
        len_q, len_k = cu_q[i + 1] - cu_q[i], cu_k[i + 1] - cu_k[i]
        if len_q > 0 and len_k > 0:
            seq = (int(cu_q[i]), int(cu_q[i + 1])), (int(cu_k[i]), int(cu_k[i + 1]))
            if len_k > 1:
                return seq
            fallback = fallback or seq
    return fallback


def attention_probe_stats(
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float | None,
    softcap: float | None,
    cu_seqlens_q,
    cu_seqlens_k,
    num_queries: int | None = None,
) -> dict[str, torch.Tensor] | None:
    """Per-head attention stats of the first varlen sequence with >= 2 keys, in fp32.

    Returns `{"max_logit": [H], "entropy": [H], "max_logit_bound": [H]}` with
    - max_logit: max of q·k·scale over (sampled) queries and all keys, before the softcap;
    - entropy: softmax entropy of the (softcapped, `softcap*tanh(l/softcap)` if softcap > 0)
      logits, averaged over queries and normalised by log(L_k) (1 = uniform attention);
    - max_logit_bound: max‖q‖·max‖k‖·scale over the whole sequence (cheap upper bound).
    Falls back to a single-key sequence (entropy 0) if there is none, and returns None if
    all sequences are empty. At most `num_queries` (evenly strided) query
    rows are used. Reads `cu_seqlens` on the host (one sync if they live on the GPU).
    """
    with stats_context(q.device):
        return _attention_probe_stats(q, k, scale, softcap, cu_seqlens_q, cu_seqlens_k, num_queries)


def _attention_probe_stats(q, k, scale, softcap, cu_seqlens_q, cu_seqlens_k, num_queries):
    seq = _first_nonempty_sequence(cu_seqlens_q, cu_seqlens_k)
    if seq is None:
        return None
    (q0, q1), (k0, k1) = seq
    if scale is None:
        scale = q.shape[-1] ** -0.5
    qs = q[q0:q1]
    ks = k[k0:k1].float()

    q_norm = torch.linalg.vector_norm(qs, dim=-1, dtype=torch.float32).amax(dim=0)
    k_norm = torch.linalg.vector_norm(ks, dim=-1).amax(dim=0)
    bound = q_norm * k_norm * scale

    if num_queries is not None and qs.shape[0] > num_queries:
        qs = qs[:: math.ceil(qs.shape[0] / num_queries)]
    logits = torch.einsum("qhd,khd->hqk", qs.float(), ks) * scale  # [H, Lq, Lk]
    max_logit = logits.amax(dim=(1, 2))
    if softcap:
        logits = softcap * torch.tanh(logits / softcap)
    log_probs = logits.log_softmax(dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean(dim=-1)  # [H]
    len_k = k1 - k0
    entropy = entropy / math.log(len_k) if len_k > 1 else torch.zeros_like(entropy)
    return {"max_logit": max_logit, "entropy": entropy, "max_logit_bound": bound}


class _Probe:
    """Callable set as `module.probe` on monitor steps; records once per step (`fired`).

    One persistent instance per module, so that a compiled caller sees the same object.
    """

    def __init__(self, monitor: AttentionProbe, group: str):
        self.monitor = monitor
        self.group = group
        self.fired = False

    def __call__(self, q, k, scale, softcap, cu_seqlens_q, cu_seqlens_k):
        buf = self.monitor._buf
        if self.fired or buf is None:
            return
        # no_grad: saving tensors for backward would break activation-checkpoint recompute
        with stats_context(q.device):
            stats = attention_probe_stats(
                q.detach(),
                k.detach(),
                scale,
                softcap,
                cu_seqlens_q,
                cu_seqlens_k,
                num_queries=self.monitor.num_queries,
            )
            if stats is None:
                return
            self.fired = True
            key = self.monitor.key
            g = self.group
            buf.accumulate(key(g, "#max_logit"), stats["max_logit"].max(), "max")
            buf.accumulate(key(g, "#max_logit_bound"), stats["max_logit_bound"].max(), "max")
            buf.accumulate(key(g, "#entropy_sum"), stats["entropy"].mean(), "sum")
            buf.accumulate(key(g, "#entropy_min"), stats["entropy"].min(), "min")
            buf.accumulate(key(g, "#n"), 1, "sum")


@register_monitor("attention")
class AttentionProbe(Monitor):
    """Per group: `max_logit` and `max_logit_bound` (max over heads), `entropy` (mean over heads
    and probed modules/ranks) and `entropy_min` (min over heads), from one sampled sequence per
    module per step. Modules are those with a `probe` attribute."""

    def __init__(self, cfg, model, selection):
        super().__init__(cfg, model, selection)
        self.num_queries = cfg.get("num_queries", None)
        modules = selection.modules(model, predicate=lambda m: hasattr(m, "probe"))
        self.probes = [(module, _Probe(self, group)) for _, module, group in modules]
        self._buf: StatBuffer | None = None

    def on_forward_start(self, buf):
        self._buf = buf
        for module, probe in self.probes:
            self._register_group(buf, probe.group)
            probe.fired = False
            module.probe = probe

    def on_forward_end(self):
        for module, probe in self.probes:
            module.probe = None
            probe.fired = False
        self._buf = None

    def _register_group(self, buf, g):
        k = {s: self.key(g, f"#{s}") for s in ("max_logit", "max_logit_bound", "entropy_sum")}
        k["entropy_min"], k["n"] = self.key(g, "#entropy_min"), self.key(g, "#n")
        buf.register(k["max_logit"], "max")
        buf.register(k["max_logit_bound"], "max")
        buf.register(k["entropy_sum"], "sum")
        buf.register(k["entropy_min"], "min")
        buf.register(k["n"], "sum")

        def out(fn):
            return lambda r: fn(r) if r[k["n"]] > 0 else None

        buf.output(self.key(g, "max_logit"), out(lambda r: r[k["max_logit"]]))
        buf.output(self.key(g, "max_logit_bound"), out(lambda r: r[k["max_logit_bound"]]))
        buf.output(self.key(g, "entropy"), out(lambda r: r[k["entropy_sum"]] / r[k["n"]]))
        buf.output(self.key(g, "entropy_min"), out(lambda r: r[k["entropy_min"]]))
