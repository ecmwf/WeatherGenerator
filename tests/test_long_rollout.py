# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for long forecast rollouts: pushforward, step checkpointing and ensembles."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from weathergen.datasets.multi_stream_data_sampler import (
    MultiStreamDataSampler,
    draw_random_forecast_steps,
)
from weathergen.model.model import Model, ModelOutput
from weathergen.utils.rollout import get_ensemble_size, get_rollout_plan

DIM = 8
NUM_TOKENS = 5
STREAM = "ERA5"


@pytest.mark.parametrize(
    "forecast_cfg, training, no_grad, loss",
    [
        # standard multi-step training and all evaluation: every step has gradient and loss
        ({}, True, [], [1, 2, 3, 4]),
        ({"pushforward": True}, False, [], [1, 2, 3, 4]),
        # classic pushforward: gradient and loss on the last step only
        ({"pushforward": True}, True, [1, 2, 3], [4]),
        ({"pushforward": True, "rollout_grad_steps": 2}, True, [1, 2], [3, 4]),
        (
            {"pushforward": True, "rollout_grad_steps": -1, "rollout_loss_steps": 1},
            True,
            [],
            [4],
        ),
        # more gradient steps than rollout steps
        ({"pushforward": True, "rollout_grad_steps": 8}, True, [], [1, 2, 3, 4]),
        (
            {"pushforward": True, "rollout_grad_steps": 3, "rollout_loss_steps": 5},
            True,
            [1],
            [2, 3, 4],
        ),
        # anchored: the first steps also get gradient and loss
        (
            {"pushforward": True, "rollout_grad_steps": 1, "rollout_anchor_steps": 1},
            True,
            [2, 3],
            [1, 4],
        ),
        # anchor and final steps overlap: nothing without gradient, every step scored once
        (
            {"pushforward": True, "rollout_grad_steps": 2, "rollout_anchor_steps": 3},
            True,
            [],
            [1, 2, 3, 4],
        ),
        ({"pushforward": True, "rollout_anchor_steps": 2}, False, [], [1, 2, 3, 4]),
    ],
)
def test_rollout_plan(forecast_cfg, training, no_grad, loss):
    plan = get_rollout_plan(forecast_cfg, [1, 2, 3, 4], training)
    assert plan.no_grad_steps == frozenset(no_grad)
    assert plan.loss_steps == frozenset(loss)


def test_rollout_plan_masking():
    # pure masking / JEPA: a single output step 0, unaffected by rollout options
    plan = get_rollout_plan({"pushforward": True}, [0], True)
    assert plan.no_grad_steps == frozenset() and plan.loss_steps == frozenset([0])


@pytest.mark.parametrize(
    "cfg",
    [
        {"pushforward": True, "rollout_grad_steps": 0},
        {"pushforward": True, "rollout_anchor_steps": -1},
    ],
)
def test_rollout_plan_invalid(cfg):
    with pytest.raises(AssertionError):
        get_rollout_plan(cfg, [1, 2], True)


def test_ensemble_size():
    assert get_ensemble_size({}) == 1
    assert get_ensemble_size({"ensemble_size": 4}) == 4
    with pytest.raises(AssertionError):
        get_ensemble_size({"ensemble_size": 0})


def test_forecast_steps_seed_independent_of_rank(monkeypatch):
    """The number of forecast steps must be drawn identically on all ranks."""
    seeds = {}
    for rank in range(3):
        sampler = MultiStreamDataSampler.__new__(MultiStreamDataSampler)
        sampler.rank, sampler.len, sampler.mini_epoch = rank, 16, 2
        sampler.data_loader_rng_seed = sampler.forecast_steps_rng_seed = 1234
        monkeypatch.setattr(
            torch.utils.data,
            "get_worker_info",
            lambda: SimpleNamespace(id=1, num_workers=2),
        )
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_rank", lambda r=rank: r)
        sampler.worker_workset()
        seeds[rank] = (sampler.data_loader_rng_seed, sampler.forecast_steps_rng_seed)

    data_seeds = {s[0] for s in seeds.values()}
    steps_seeds = {s[1] for s in seeds.values()}
    assert len(data_seeds) == 3, "data RNG should still differ between ranks"
    assert len(steps_seeds) == 1, "forecast steps RNG must be identical on all ranks"
    # but differs between workers and mini epochs
    seed = MultiStreamDataSampler.get_forecast_steps_rng_seed
    assert seed(1234, 0, 2) != seed(1234, 1, 2) != seed(1234, 1, 3)


def test_random_forecast_steps():
    steps = np.array([8, 16, 24, 32, 40], dtype=np.int32)
    # random_choice: only the listed values, all about equally often, reproducible
    fs = draw_random_forecast_steps("random_choice", steps, 40, 50000, 1234)
    assert fs.dtype == np.int64 and set(np.unique(fs)) == set(steps.tolist())
    assert np.allclose(np.bincount(fs)[steps] / len(fs), 0.2, atol=0.01)
    assert np.array_equal(fs, draw_random_forecast_steps("random_choice", steps, 40, 50000, 1234))
    # random: unchanged, uniform in [min, fsm]
    fs = draw_random_forecast_steps("random", np.array([8, 40]), 40, 1000, 1234)
    ref = np.random.default_rng(1234).integers(low=8, high=41, size=1000, dtype=np.int64)
    assert np.array_equal(fs, ref)


def test_random_choice_lead_time_coverage():
    """With K from [8, 16, 24, 32, 40] and the loss on the last 8 steps, every lead time is
    scored equally often."""
    steps = [8, 16, 24, 32, 40]
    cfg = {"pushforward": True, "rollout_grad_steps": 8}
    counts = np.zeros(41)
    for k in steps:
        plan = get_rollout_plan(cfg, list(range(1, k + 1)), True)
        counts[sorted(plan.loss_steps)] += 1
        assert plan.no_grad_steps == frozenset(range(1, k - 7))
    assert np.all(counts[1:] == 1)


def test_anchored_lead_time_coverage():
    """With an anchor of 8 steps, leads 1-8 are scored in every batch, later leads as before."""
    steps = [8, 16, 24, 32, 40]
    cfg = {"pushforward": True, "rollout_grad_steps": 8, "rollout_anchor_steps": 8}
    counts = np.zeros(41)
    for k in steps:
        plan = get_rollout_plan(cfg, list(range(1, k + 1)), True)
        counts[sorted(plan.loss_steps)] += 1
        assert plan.no_grad_steps == frozenset(range(9, k - 7))
        assert len(plan.loss_steps) == (8 if k == 8 else 16)
    assert np.all(counts[1:9] == 5) and np.all(counts[9:] == 1)


class StubForecastEngine(torch.nn.Module):
    """Small stand-in for ForecastingEngine with dropout and latent noise, recording calls."""

    def __init__(self, dropout=0.0, noise=0.0):
        super().__init__()
        self.lin = torch.nn.Linear(DIM, DIM)
        self.dropout = torch.nn.Dropout(dropout)
        self.noise = noise
        self.calls = []

    def forward(
        self, tokens, fstep, meta_info=None, noise_emb=None, conditioning=None, coords=None
    ):
        self.calls.append((fstep, torch.is_grad_enabled(), self.training))
        if self.training and self.noise > 0:
            tokens = tokens + torch.randn_like(tokens) * self.noise
        return tokens + self.dropout(torch.tanh(self.lin(tokens)))


class StubEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(DIM, DIM)

    def forward(self, model_params, batch):
        tokens = self.lin(batch.inputs)
        return tokens, None, []


class StubBatch:
    """Stand-in for BatchSamples with what Model.forward uses."""

    def __init__(self, num_samples, num_steps, plan=None, ensemble_size=1, offset=1):
        self.inputs = torch.randn(num_samples, NUM_TOKENS, DIM)
        self.output_idxs = list(range(offset, offset + num_steps))
        self.rollout_plan = plan
        self.ensemble_size = ensemble_size
        self.samples = [SimpleNamespace(meta_info={}) for _ in range(num_samples)]

    def __len__(self):
        return len(self.samples)

    def get_output_len(self):
        return self.output_idxs[-1] + 1

    def get_output_idxs(self):
        return self.output_idxs

    def get_num_steps(self):
        return 1


def stub_predict_decoders(model_params, step, tokens, batch, output, out_step=None):
    # like predict_decoders: one prediction per row of tokens, with a leading ensemble dim of 1,
    # and samples indexed modulo the batch size; 2 or 3 target points per sample
    lens = [2 + (i_b % len(batch)) % 2 for i_b in range(tokens.shape[0])]
    pred = torch.cat([tokens[i, :n] for i, n in enumerate(lens)]).unsqueeze(0)
    output.add_physical_prediction(step, STREAM, torch.split(pred, lens, dim=1))
    return output


def make_model(forecast_cfg, dropout=0.0, noise=0.0, training_mode=("masking",)):
    model = Model.__new__(Model)
    torch.nn.Module.__init__(model)
    model.cf = OmegaConf.create(
        {
            "training_config": {
                "training_mode": list(training_mode),
                "forecast": forecast_cfg,
            }
        }
    )
    model.encoder = StubEncoder()
    model.forecast_engine = StubForecastEngine(dropout, noise)
    model.predict_decoders = stub_predict_decoders
    return model


def run(model, batch, training=True):
    model.train(training)
    params = SimpleNamespace(rope_coords=None)
    with torch.set_grad_enabled(training):
        return model(params, batch)


def loss_of(output: ModelOutput):
    return sum(p.sum() for step in output.physical for preds in step.values() for p in preds)


def test_default_rollout_unchanged():
    """Without rollout options, every step is computed with gradient and decoded."""
    torch.manual_seed(0)
    model = make_model({})
    batch = StubBatch(2, 3)
    out = run(model, batch)
    assert [c[:2] for c in model.forecast_engine.calls] == [(1, True), (2, True), (3, True)]
    assert all(STREAM in out.physical[s] for s in [1, 2, 3])

    # same as a plain autoregressive loop
    tokens = model.encoder(None, batch)[0]
    for step in [1, 2, 3]:
        tokens = model.forecast_engine(tokens, step)
        ref = stub_predict_decoders(None, step, tokens, batch, ModelOutput(4))
        for p, r in zip(out.physical[step][STREAM], ref.physical[step][STREAM], strict=True):
            assert torch.equal(p, r)


@pytest.mark.parametrize("num_steps", [1, 4])
def test_pushforward(num_steps):
    torch.manual_seed(0)
    model = make_model({"pushforward": True})
    batch = StubBatch(
        2, num_steps, get_rollout_plan({"pushforward": True}, list(range(1, num_steps + 1)), True)
    )
    out = run(model, batch)

    grads = [c[1] for c in model.forecast_engine.calls]
    assert grads == [False] * (num_steps - 1) + [True]
    decoded = [s for s in range(len(out.physical)) if STREAM in out.physical[s]]
    assert decoded == [num_steps]

    loss_of(out).backward()
    assert model.forecast_engine.lin.weight.grad is not None
    # the encoder only gets gradient when there is no step without gradient before
    encoder_grad = model.encoder.lin.weight.grad
    if num_steps == 1:
        assert encoder_grad is not None and encoder_grad.abs().sum() > 0
    else:
        assert encoder_grad is None


def test_pushforward_grad_window():
    torch.manual_seed(0)
    cfg = {"pushforward": True, "rollout_grad_steps": 3, "rollout_loss_steps": 2}
    model = make_model(cfg)
    batch = StubBatch(2, 5, get_rollout_plan(cfg, [1, 2, 3, 4, 5], True))
    out = run(model, batch)
    assert [c[1] for c in model.forecast_engine.calls] == [False, False, True, True, True]
    assert [s for s in range(len(out.physical)) if STREAM in out.physical[s]] == [4, 5]


def test_pushforward_matches_detached_rollout():
    """Pushforward gives the same values and gradients as an explicitly detached rollout."""
    cfg = {"pushforward": True, "rollout_grad_steps": 2}
    torch.manual_seed(0)
    model = make_model(cfg)
    batch = StubBatch(2, 4, get_rollout_plan(cfg, [1, 2, 3, 4], True))
    out = run(model, batch)
    loss_of(out).backward()
    grad = model.forecast_engine.lin.weight.grad.clone()

    model.zero_grad()
    tokens = model.encoder(None, batch)[0].detach()
    ref = ModelOutput(5)
    for step in [1, 2, 3, 4]:
        tokens = model.forecast_engine(tokens, step)
        tokens = tokens.detach() if step < 3 else tokens
        if step >= 3:
            ref = stub_predict_decoders(None, step, tokens, batch, ref)
    loss_ref = loss_of(ref)
    loss_ref.backward()
    assert torch.allclose(loss_of(out), loss_ref)
    assert torch.allclose(grad, model.forecast_engine.lin.weight.grad)


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("checkpoint_steps", [False, True])
def test_pushforward_under_autocast(device, checkpoint_steps):
    """
    Under autocast (mixed precision training with fp32 weights, DDP), the steps without gradient
    must not leave cached weight casts without autograd history for the steps with gradient.
    """
    cfg = {"pushforward": True, "rollout_grad_steps": 2, "checkpoint_steps": checkpoint_steps}
    torch.manual_seed(0)
    model = make_model(cfg).to(device)
    batch = StubBatch(2, 4, get_rollout_plan(cfg, [1, 2, 3, 4], True))
    batch.inputs = batch.inputs.to(device)
    with torch.autocast(device, dtype=torch.bfloat16):
        out = run(model, batch)
        loss = loss_of(out)
    loss.backward()
    grad = model.forecast_engine.lin.weight.grad
    assert grad is not None and grad.abs().sum() > 0

    # reference: rollout with gradient everywhere, detached after the steps without gradient
    grad = grad.clone()
    model.zero_grad()
    with torch.autocast(device, dtype=torch.bfloat16):
        # sum over the input steps as in Model.forward (in fp32 under autocast on cuda)
        tokens = model.encoder(None, batch)[0].detach().unsqueeze(1).sum(axis=1)
        ref = ModelOutput(5)
        for step in [1, 2, 3, 4]:
            tokens = model.forecast_engine(tokens, step)
            tokens = tokens.detach() if step < 3 else tokens
            if step >= 3:
                ref = stub_predict_decoders(None, step, tokens, batch, ref)
        loss_ref = loss_of(ref)
    loss_ref.backward()
    assert torch.equal(loss.detach(), loss_ref.detach())
    assert torch.equal(grad, model.forecast_engine.lin.weight.grad)
    assert torch.is_autocast_cache_enabled()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("checkpoint_steps", [False, True])
def test_anchored_pushforward(device, checkpoint_steps):
    """
    Anchored pushforward (gradient and loss on the first and the last steps, the steps between
    without gradient) gives the values and gradients of an explicitly detached rollout, also
    under autocast, where steps with gradient come both before and after the ones without.
    """
    cfg = {
        "pushforward": True,
        "rollout_grad_steps": 1,
        "rollout_anchor_steps": 1,
        "checkpoint_steps": checkpoint_steps,
    }
    torch.manual_seed(0)
    model = make_model(cfg).to(device)
    batch = StubBatch(2, 4, get_rollout_plan(cfg, [1, 2, 3, 4], True))
    batch.inputs = batch.inputs.to(device)
    with torch.autocast(device, dtype=torch.bfloat16):
        out = run(model, batch)
        loss = loss_of(out)
    # (before backward, which recomputes the checkpointed steps)
    assert [c[1] for c in model.forecast_engine.calls] == [True, False, False, True]
    assert [s for s in range(len(out.physical)) if STREAM in out.physical[s]] == [1, 4]
    loss.backward()
    grad = model.forecast_engine.lin.weight.grad.clone()

    # reference: rollout with gradient everywhere, detached after the steps without gradient
    model.zero_grad()
    with torch.autocast(device, dtype=torch.bfloat16):
        tokens = model.encoder(None, batch)[0].detach().unsqueeze(1).sum(axis=1)
        ref = ModelOutput(5)
        for step in [1, 2, 3, 4]:
            tokens = model.forecast_engine(tokens, step)
            if step in (1, 4):
                ref = stub_predict_decoders(None, step, tokens, batch, ref)
            else:
                tokens = tokens.detach()
        loss_ref = loss_of(ref)
    loss_ref.backward()
    assert torch.equal(loss.detach(), loss_ref.detach())
    assert torch.equal(grad, model.forecast_engine.lin.weight.grad)
    assert torch.is_autocast_cache_enabled()


def test_checkpoint_steps_identical():
    """Step checkpointing gives identical outputs and gradients, including dropout and noise."""
    grads, losses = [], []
    for checkpoint_steps in [False, True]:
        torch.manual_seed(0)
        model = make_model({"checkpoint_steps": checkpoint_steps}, dropout=0.3, noise=0.1)
        torch.manual_seed(1)
        batch = StubBatch(2, 4)
        out = run(model, batch)
        loss = loss_of(out)
        loss.backward()
        losses.append(loss.detach())
        grads.append([p.grad.clone() for p in model.parameters()])
    assert torch.equal(losses[0], losses[1])
    for g0, g1 in zip(*grads, strict=True):
        assert torch.allclose(g0, g1)


def test_ensemble_training():
    torch.manual_seed(0)
    model = make_model({}, dropout=0.3)
    batch = StubBatch(2, 2, ensemble_size=3)
    out = run(model, batch)
    for step in [1, 2]:
        preds = out.physical[step][STREAM]
        assert len(preds) == 2
        # members along the ensemble dimension, points of each sample
        assert preds[0].shape == (3, 2, DIM) and preds[1].shape == (3, 3, DIM)
        # dropout makes members differ
        assert not torch.allclose(preds[0][0], preds[0][1])
    loss_of(out).backward()


def test_ensemble_members_are_per_sample():
    """Without stochasticity all members equal the deterministic forecast of their sample."""
    torch.manual_seed(0)
    model = make_model({})
    batch = StubBatch(2, 2, ensemble_size=3)
    out = run(model, batch, training=False)
    batch.ensemble_size = 1
    out_det = run(model, batch, training=False)
    for step in [1, 2]:
        for b, pred in enumerate(out.physical[step][STREAM]):
            for m in range(3):
                assert torch.equal(pred[m], pred[0])
            assert torch.equal(pred[0], out_det.physical[step][STREAM][b][0])


def test_ensemble_eval_mc_dropout():
    torch.manual_seed(0)
    model = make_model({}, dropout=0.3)
    model.eval()

    # single member: deterministic evaluation, dropout off
    batch = StubBatch(1, 2)
    out_a = run(model, batch, training=False)
    out_b = run(model, batch, training=False)
    assert torch.equal(out_a.physical[2][STREAM][0], out_b.physical[2][STREAM][0])
    assert all(not c[2] for c in model.forecast_engine.calls)

    # ensemble: dropout active during the rollout, eval mode restored afterwards
    batch.ensemble_size = 4
    out = run(model, batch, training=False)
    pred = out.physical[2][STREAM][0]
    assert pred.shape[0] == 4 and not torch.allclose(pred[0], pred[1])
    assert model.forecast_engine.calls[-1][2]
    assert not model.forecast_engine.training and not model.forecast_engine.dropout.training


def test_ensemble_rejects_student_teacher():
    model = make_model({}, training_mode=("student_teacher",))
    with pytest.raises(AssertionError):
        run(model, StubBatch(1, 1, ensemble_size=2))


# small forecast engine with the nhv6tkln settings (swiglu, XSA, qk norm, 2D RoPE, flash
# attention, a LayerNorm between blocks)
FE_CFG = {
    "fe_diffusion_model": False,
    "ae_global_dim_embed": 64,
    "ae_local_num_queries": 1,
    "ae_global_block_factor": 64,
    "fe_num_blocks": 2,
    "fe_num_heads": 4,
    "fe_dropout_rate": 0.1,
    "fe_with_qk_lnorm": True,
    "fe_layer_norm_after_blocks": [0],
    "fe_impute_latent_noise_std": 1e-5,
    "forecast_att_dense_rate": 1.0,
    "rope_2D": True,
    "mlp_type": "swiglu",
    "use_xsa": True,
    "with_flash_attention": True,
    "attention_dtype": "bf16",
    "mlp_norm_eps": 1e-5,
    "norm_eps": 1e-4,
    "norm_type": "LayerNorm",
    "qk_norm_type": "RMSNorm",
}


@pytest.fixture(scope="module")
def single_rank_process_group():
    import os
    import socket

    import torch.distributed as dist

    if dist.is_initialized():
        yield
        return
    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def _fsdp_rollout(
    no_grad_steps,
    num_grad_steps,
    checkpoint_steps,
    step_checkpointing,
    fe_cfg_overrides=None,
    reshard_after_forward=False,
):
    """
    Loss and gradients of a rollout of a small forecast engine sharded with FSDP2 as in
    model_interface (with mixed precision), after an fp32 input layer (stand-in for the encoder)
    and with the RoPE coordinates as in Model.forward.
    """
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    from weathergen.model.attention import MultiSelfAttentionHead, MultiSelfAttentionHeadLocal
    from weathergen.model.engines import ForecastingEngine
    from weathergen.model.layers import MLP
    from weathergen.model.model_interface import shard_forecast_engine

    cf = OmegaConf.create({**FE_CFG, **(fe_cfg_overrides or {})})
    dim, num_cells = cf.ae_global_dim_embed, 256

    class Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = torch.nn.Linear(dim, dim)
            self.norm = torch.nn.LayerNorm(dim)
            mode_cfg = OmegaConf.create({"forecast": {"policy": "fixed"}})
            self.forecast_engine = ForecastingEngine(cf, mode_cfg, num_cells)

        def forward(self, x, coords):
            # fp32 latent state, as the encoder output under autocast
            tokens = self.norm(self.inp(x))
            for step in range(no_grad_steps):
                with torch.no_grad():
                    tokens = self.forecast_engine(tokens, step, coords=coords)
            loss = 0.0
            for step in range(no_grad_steps, no_grad_steps + num_grad_steps):
                if step_checkpointing:
                    tokens = torch.utils.checkpoint.checkpoint(
                        self.forecast_engine, tokens, step, coords=coords, use_reentrant=False
                    )
                else:
                    tokens = self.forecast_engine(tokens, step, coords=coords)
                loss = loss + tokens.float().pow(2).mean()
            return loss

    torch.manual_seed(0)
    root = Root().cuda().train()
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    shard_forecast_engine(
        root.forecast_engine,
        root.forecast_engine.fe_blocks,
        (MLP, MultiSelfAttentionHead, MultiSelfAttentionHeadLocal),
        checkpoint_steps=checkpoint_steps,
        reshard_after_forward=reshard_after_forward,
        mp_policy=mp_policy,
    )
    fully_shard(root, mp_policy=mp_policy)

    x = torch.randn(1, num_cells, dim, device="cuda")
    coords = (torch.rand(1, num_cells, 2, device="cuda") * 3.0).to(torch.bfloat16)
    torch.manual_seed(1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = root(x, coords)
    loss.backward()
    # parameters before pushforward (no-grad) steps get no gradient
    grads = {n: p.grad.full_tensor() for n, p in root.named_parameters() if p.grad is not None}
    return loss.item(), grads


def _rel_diff(grads, grads_ref):
    diff = torch.cat([(grads[n] - grads_ref[n]).flatten() for n in grads_ref])
    return (diff.norm() / torch.cat([g.flatten() for g in grads_ref.values()]).norm()).item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FSDP2 test needs a GPU")
@pytest.mark.parametrize("reshard_after_forward", [False, True])
def test_checkpoint_steps_fsdp_bitwise(single_rank_process_group, reshard_after_forward):
    """
    The forecast engine sharded for checkpoint_steps (one FSDP unit, inputs cast per block),
    with and without step checkpointing, gives bitwise the same loss and gradients as the
    default per-block FSDP units without step checkpointing (the standard fine-tuning setup),
    here after a pushforward step and without blocks that get fp32 inputs (see below).
    Per-block units with step checkpointing fail with a DTensor error or a CheckpointError.
    """
    no_ln = {"fe_layer_norm_after_blocks": []}
    loss_ref, grads_ref = _fsdp_rollout(1, 3, False, False, no_ln, reshard_after_forward)
    for step_checkpointing in [False, True]:
        loss, grads = _fsdp_rollout(1, 3, True, step_checkpointing, no_ln, reshard_after_forward)
        assert loss == loss_ref
        assert grads.keys() == grads_ref.keys()
        assert all(torch.equal(g, grads_ref[n]) for n, g in grads.items())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FSDP2 test needs a GPU")
@pytest.mark.parametrize("no_grad_steps", [0, 2])
def test_checkpoint_steps_fsdp(single_rank_process_group, no_grad_steps):
    """
    As test_checkpoint_steps_fsdp_bitwise, with blocks that get fp32 inputs: the first block
    for the fp32 encoder output, and the block after a LayerNorm between blocks. With per-block
    units, FSDP casts the inputs of such a block to bf16 in the forward pass but not in the
    recomputation of its (per-block) activation checkpoint, so its gradients are computed from
    slightly different activations; the single unit casts in both. The forward pass is identical
    and step checkpointing itself is exact.
    """
    loss_ref, grads_ref = _fsdp_rollout(no_grad_steps, 3, False, False)
    _, grads_unit = _fsdp_rollout(no_grad_steps, 3, True, False)
    loss, grads = _fsdp_rollout(no_grad_steps, 3, True, True)
    assert loss == loss_ref
    assert grads.keys() == grads_ref.keys()
    assert all(torch.equal(g, grads_unit[n]) for n, g in grads.items())
    assert _rel_diff(grads, grads_ref) < 1e-4


def _prepare_losses_worker(rank: int, world_size: int, init_file: str):
    import datetime

    from weathergen.utils.train_logger import prepare_losses_for_logging

    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        # anchored pushforward: rank 1 scores step 9 in addition to step 1, rank 0 does not
        losses = []
        for _ in range(2):
            steps = {"1": {"mse": float(rank)}} | ({"9": {"mse": 5.0}} if rank == 1 else {})
            losses.append({"ERA5": steps | {"loss_avg": torch.tensor(10.0 + rank)}})
        stddev = [{"ERA5": {"stddev_avg": 0.0 if rank == 0 else 3.0}}] * 2
        _, losses_all, stddev_all = prepare_losses_for_logging([], losses, stddev)
        assert dict(losses_all) == {
            "ERA5.1.mse": [0.0, 0.0, 1.0, 1.0],
            "ERA5.9.mse": [5.0, 5.0],
            "ERA5.loss_avg": [10.0, 10.0, 11.0, 11.0],
        }
        assert dict(stddev_all) == {"ERA5.stddev_avg": [3.0, 3.0]}
    finally:
        torch.distributed.destroy_process_group()


def test_prepare_losses_for_logging_different_keys_per_rank(tmp_path):
    """
    Ranks that log different loss keys (different numbers of scored steps with anchored
    pushforward) must not deadlock, and the values are pooled by key over ranks and batches.
    """
    world_size = 2
    torch.multiprocessing.spawn(
        _prepare_losses_worker, args=(world_size, str(tmp_path / "pg")), nprocs=world_size
    )
