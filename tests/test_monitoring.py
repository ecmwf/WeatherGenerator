# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""TDD tests for training-time monitoring (`weathergen.train.monitoring`).

Written before the implementation, following the plan in
`weathergen-research/docs/notes/sessions/2026-10-22-training-monitoring-plan.md` ("Testing ->
Long-term tests"). They go through the trainer-facing API only and run on CPU in a single
process (no `torch.distributed` process group).
"""

import copy
import math

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.checkpoint import checkpoint

from weathergen.train.monitoring import NullMonitorManager, build_monitor_manager
from weathergen.train.monitoring.attention import attention_probe_stats

DIM = 16
NUM_HEADS = 2
NUM_TOKENS = 12
NUM_STEPS = 3
GRAD_CLIP = 0.1  # small enough that clipping is active

# Reference quantities checked against the logged metrics (group_by: param).
REF_PARAM = "blocks.0.mlp.0.weight"
REF_BLOCK = "blocks.0"

MONITORING_CFG = {
    "enabled": True,
    "every": 1,
    "group_by": "param",
    "include": ["*"],
    "exclude": [],
    "monitors": {
        "grads": {"enabled": True, "every": 1},
        "weights": {"enabled": True, "every": 1},
        "updates": {"enabled": True, "every": 1},
        "activations": {"enabled": True, "every": 1, "include": ["blocks.*"]},
        "attention": {"enabled": True, "every": 1, "include": ["*"]},
    },
}

FP32_TOL = {"rtol": 1e-5, "atol": 1e-6}


class FakeTrainLogger:
    """Records `log_metrics` calls (same signature as `TrainLogger.log_metrics`)."""

    def __init__(self):
        self.calls = []

    def log_metrics(self, stage, metrics, step=None):
        self.calls.append((stage, dict(metrics)))


class ToyAttention(torch.nn.Module):
    """Plain softmax self-attention over a single sequence, with the monitoring probe hook."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = torch.nn.Linear(dim, 3 * dim, bias=False)
        self.proj = torch.nn.Linear(dim, dim)
        self.probe = None

    def forward(self, x):
        n = x.shape[0]
        q, k, v = self.qkv(x).reshape(n, 3, self.num_heads, self.head_dim).unbind(1)
        scale = self.head_dim**-0.5
        cu_seqlens = torch.tensor([0, n], dtype=torch.int32)
        if self.probe is not None:
            self.probe(q, k, scale, 0.0, cu_seqlens, cu_seqlens)
        probs = (torch.einsum("qhd,khd->hqk", q, k) * scale).softmax(dim=-1)
        out = torch.einsum("hqk,khd->qhd", probs, v).reshape(n, -1)
        return self.proj(out)


class ToyBlock(torch.nn.Module):
    def __init__(self, dim, with_attention=False):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, 2 * dim), torch.nn.GELU(), torch.nn.Linear(2 * dim, dim)
        )
        self.attn = ToyAttention(dim, NUM_HEADS) if with_attention else None

    def forward(self, x):
        x = x + self.mlp(self.norm(x))
        if self.attn is not None:
            x = x + self.attn(x)
        return x


class ToyModel(torch.nn.Module):
    def __init__(self, dim=DIM):
        super().__init__()
        self.embed = torch.nn.Linear(4, dim)
        self.blocks = torch.nn.ModuleList(
            [ToyBlock(dim), ToyBlock(dim), ToyBlock(dim, with_attention=True)]
        )
        self.head = torch.nn.Linear(dim, 1)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False)
        return self.head(x)


def _make_batches(num_steps):
    gen = torch.Generator().manual_seed(1)
    return [
        (torch.randn(NUM_TOKENS, 4, generator=gen), torch.randn(NUM_TOKENS, 1, generator=gen))
        for _ in range(num_steps)
    ]


def _assert_no_forward_hooks_or_probes(model):
    for name, module in model.named_modules():
        assert not module._forward_hooks, f"forward hook left on '{name}'"
        assert not module._forward_pre_hooks, f"forward pre-hook left on '{name}'"
        assert getattr(module, "probe", None) is None, f"probe left on '{name}'"
    assert not torch.nn.modules.module._global_forward_hooks
    assert not torch.nn.modules.module._global_forward_pre_hooks


def _rms(x):
    return x.detach().float().pow(2).mean().sqrt()


def _run_unmonitored(model, batches):
    """Plain training steps; records losses and reference stats with a test-owned hook."""
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    params = dict(model.named_parameters())
    ref_block = model.get_submodule(REF_BLOCK)
    out = {"loss": [], "grad_l2": [], "weight_l2": [], "act_rms": []}
    for x, y in batches:
        handle = ref_block.register_forward_hook(lambda m, i, o: out["act_rms"].append(_rms(o)))
        pred = model(x)
        handle.remove()  # before backward, so checkpoint recompute does not fire it again
        loss = torch.nn.functional.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        out["grad_l2"].append(params[REF_PARAM].grad.detach().norm())
        opt.step()
        out["weight_l2"].append(params[REF_PARAM].detach().norm())
        out["loss"].append(loss.detach())
    assert len(out["act_rms"]) == len(batches)
    out["grads"] = {n: p.grad.clone() for n, p in model.named_parameters()}
    return out


def _run_monitored(model, batches, train_logger):
    """Same steps, driving the monitor manager in trainer order, plus one aborted step."""
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    manager = build_monitor_manager(OmegaConf.create(MONITORING_CFG), model, [opt], train_logger)
    out = {"loss": [], "logged": []}
    for istep, (x, y) in enumerate(batches):
        num_calls = len(train_logger.calls)
        assert manager.begin_step(istep)
        manager.on_forward_start()
        pred = model(x)
        manager.on_forward_end()
        _assert_no_forward_hooks_or_probes(model)
        loss = torch.nn.functional.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        manager.on_after_clip(total_norm)
        manager.on_before_opt_step()
        opt.step()
        manager.on_after_opt_step()
        manager.end_step()
        _assert_no_forward_hooks_or_probes(model)

        logged = {}
        for stage, metrics in train_logger.calls[num_calls:]:
            assert stage == "train"
            logged.update(metrics)
        out["logged"].append(logged)
        out["loss"].append(loss.detach())
    out["grads"] = {n: p.grad.clone() for n, p in model.named_parameters()}

    # Loss-spike skip path: abort right after the forward, no backward / optimizer step.
    num_calls = len(train_logger.calls)
    x, _ = batches[0]
    manager.begin_step(len(batches))
    manager.on_forward_start()
    model(x)
    manager.on_forward_end()
    manager.abort_step()
    _assert_no_forward_hooks_or_probes(model)
    assert len(train_logger.calls) == num_calls, "abort_step must not log"
    return out


def test_monitoring_is_numerically_invisible_and_correct():
    torch.manual_seed(0)
    model_ref = ToyModel()
    model_mon = copy.deepcopy(model_ref)
    _assert_no_forward_hooks_or_probes(model_ref)
    batches = _make_batches(NUM_STEPS)

    ref = _run_unmonitored(model_ref, batches)
    train_logger = FakeTrainLogger()
    mon = _run_monitored(model_mon, batches, train_logger)

    # Bitwise-identical training with and without monitoring.
    for loss_ref, loss_mon in zip(ref["loss"], mon["loss"], strict=True):
        assert torch.equal(loss_ref, loss_mon)
    for (name, p_ref), p_mon in zip(
        model_ref.named_parameters(), model_mon.parameters(), strict=True
    ):
        assert torch.equal(p_ref, p_mon), f"parameter '{name}' differs"
        assert torch.equal(ref["grads"][name], mon["grads"][name]), f"grad of '{name}' differs"

    # Logged values match references, step by step.
    for istep, logged in enumerate(mon["logged"]):
        expected = {
            f"mon/grads/{REF_PARAM}/l2": ref["grad_l2"][istep],
            f"mon/weights/{REF_PARAM}/l2": ref["weight_l2"][istep],
            f"mon/activations/{REF_BLOCK}/rms": ref["act_rms"][istep],
        }
        for key, value in expected.items():
            assert key in logged, f"step {istep}: '{key}' not logged"
            torch.testing.assert_close(torch.tensor(float(logged[key])), value, **FP32_TOL)

        attention_keys = [k for k in logged if k.startswith("mon/attention/")]
        assert attention_keys, f"step {istep}: no 'mon/attention/' metric logged"
        assert any("max_logit" in k for k in attention_keys)
        assert any("entropy" in k for k in attention_keys)
        assert all(math.isfinite(float(v)) for k, v in logged.items() if k.startswith("mon/"))


@pytest.mark.parametrize("as_dictconfig", [False, True])
def test_disabled_monitoring_returns_null_manager(as_dictconfig):
    cfg = {"enabled": False}
    cfg = OmegaConf.create(cfg) if as_dictconfig else cfg
    model = ToyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    train_logger = FakeTrainLogger()
    manager = build_monitor_manager(cfg, model, [opt], train_logger)
    assert isinstance(manager, NullMonitorManager)

    assert not manager.begin_step(0)
    manager.on_forward_start()
    model(torch.randn(NUM_TOKENS, 4)).sum().backward()
    manager.on_forward_end()
    manager.on_after_clip(torch.tensor(1.0))
    manager.on_before_opt_step()
    opt.step()
    manager.on_after_opt_step()
    manager.end_step()
    manager.abort_step()
    assert train_logger.calls == []


def _explicit_probe_reference(q, k, scale, softcap):
    """fp32 per-head max logit (pre-softcap) and log(L_k)-normalised mean softmax entropy."""
    logits = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale  # [H, Lq, Lk]
    max_logit = logits.amax(dim=(1, 2))
    if softcap:
        logits = softcap * torch.tanh(logits / softcap)
    log_probs = logits.log_softmax(dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)  # [H, Lq]
    return max_logit, entropy.mean(dim=-1) / math.log(k.shape[0])


@pytest.mark.parametrize("softcap", [0.0, 5.0])
def test_attention_probe_matches_explicit_softmax(softcap):
    gen = torch.Generator().manual_seed(0)
    head_dim = 8
    # Varlen batch of 3 sequences with different query and key lengths.
    cu_seqlens_q = torch.tensor([0, 4, 10, 13], dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, 5, 12, 14], dtype=torch.int32)
    # Scale q up so that logits exceed the softcap and the cap matters.
    q = 3.0 * torch.randn(int(cu_seqlens_q[-1]), NUM_HEADS, head_dim, generator=gen)
    k = torch.randn(int(cu_seqlens_k[-1]), NUM_HEADS, head_dim, generator=gen)
    scale = head_dim**-0.5

    stats = attention_probe_stats(q, k, scale, softcap, cu_seqlens_q, cu_seqlens_k)

    q0, k0 = q[: cu_seqlens_q[1]], k[: cu_seqlens_k[1]]
    ref_max_logit, ref_entropy = _explicit_probe_reference(q0, k0, scale, softcap)
    if softcap:
        # Guard that the test is sensitive to the softcap.
        assert ref_max_logit.max() > softcap
        _, ref_entropy_uncapped = _explicit_probe_reference(q0, k0, scale, 0.0)
        assert not torch.allclose(ref_entropy, ref_entropy_uncapped, **FP32_TOL)

    assert stats["max_logit"].shape == (NUM_HEADS,)
    assert stats["entropy"].shape == (NUM_HEADS,)
    torch.testing.assert_close(stats["max_logit"].float(), ref_max_logit, **FP32_TOL)
    torch.testing.assert_close(stats["entropy"].float(), ref_entropy, **FP32_TOL)

    # Only the first sequence is used: changing the other sequences leaves the result unchanged.
    q_other, k_other = q.clone(), k.clone()
    q_other[cu_seqlens_q[1] :] = torch.randn_like(q_other[cu_seqlens_q[1] :])
    k_other[cu_seqlens_k[1] :] = torch.randn_like(k_other[cu_seqlens_k[1] :])
    stats_other = attention_probe_stats(
        q_other, k_other, scale, softcap, cu_seqlens_q, cu_seqlens_k
    )
    torch.testing.assert_close(stats_other["max_logit"], stats["max_logit"])
    torch.testing.assert_close(stats_other["entropy"], stats["entropy"])
