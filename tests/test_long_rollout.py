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

import pytest
import torch
from omegaconf import OmegaConf

from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler
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


@pytest.mark.parametrize("cfg", [{"pushforward": True, "rollout_grad_steps": 0}])
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
