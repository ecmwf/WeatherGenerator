# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for the flow-matching Gaussian probability path, its conversions, and the
EDM equivalence of the ve path.

Given the *conditional* quantity for a single ``(z, eps, t)`` (which the network target
equals), every converter must recover a mutually consistent ``(z, eps)`` and hence each
other quantity (course Prop. 1 / Remark 16). This validates the algebra without a network.
Engine methods are called unbound on a minimal stand-in, so the heavy module is not built.
"""

import types

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from weathergen.datasets.masking import sample_noise_level
from weathergen.model.diffusion import rho_sigma_steps
from weathergen.model.flow_matching import FlowMatchingForecastEngine, GaussianPath
from weathergen.train.loss_modules.loss_module_flow_matching import LossFlowMatching

_TIMES = [1e-3, 0.1, 0.3, 0.5, 0.7, 0.9, 1 - 1e-3]
_SIGMAS = [1e-3, 0.1, 1.0, 7.0, 80.0]
_PREDICTION_TYPES = ["velocity", "noise", "score", "denoiser"]


def _fake_engine(**over):
    """A minimal stand-in exposing just the attributes the seam methods read."""
    ns = types.SimpleNamespace(
        path=GaussianPath("ve"),
        sigma_data=1.0,
        sigma_min=0.4,
        sigma_max=50.0,
        rho=7,
        fm_c_in=1.0,
        edm_noise_time_scale=1.0,
        edm_preconditioning=False,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


@pytest.fixture
def z_eps():
    torch.manual_seed(0)
    return torch.randn(2, 5, 8), torch.randn(2, 5, 8)


# --------------------------------------------------------------------------------------------------
# 1. condot path conversions
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("t_val", _TIMES)
def test_condot_velocity_is_z_minus_eps(z_eps, t_val):
    z, eps = z_eps
    v = GaussianPath("condot").conditional_target(z, eps, torch.tensor(t_val), "velocity")
    assert torch.allclose(v, z - eps, atol=1e-5)


@pytest.mark.parametrize("t_val", _TIMES)
@pytest.mark.parametrize("prediction_type", ["velocity", "noise", "score"])
def test_conversions_recover_consistent_quantities(z_eps, t_val, prediction_type):
    path = GaussianPath("condot")
    z, eps = z_eps
    t = torch.tensor(t_val)
    alpha, beta, _, _ = path.coeffs(t)
    x = alpha * z + beta * eps

    # Ground-truth conditional quantities and the matching network prediction.
    v = z - eps  # CondOT velocity
    s = -eps / beta  # score
    pred = {"velocity": v, "noise": eps, "score": s}[prediction_type]

    # Reconstructing conversions divide by alpha (near t=0) or beta (near t=1); float32
    # rounding is amplified ~1/min(t, 1-t). A genuine algebra bug is O(1) and still fails.
    tol = 1e-4 * max(1.0, 1.0 / min(t_val, 1.0 - t_val))
    assert torch.allclose(path.to_velocity(pred, x, t, prediction_type), v, atol=tol)
    assert torch.allclose(path.to_score(pred, x, t, prediction_type), s, atol=tol)
    assert torch.allclose(path.to_denoiser(pred, x, t, prediction_type), z, atol=tol)


# --------------------------------------------------------------------------------------------------
# 2. x0-space <-> parameterization-space loss equivalence (engine returns D, loss rescales)
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind,s_val", [("condot", t) for t in _TIMES[1:-1]] + [("ve", 0.5)])
@pytest.mark.parametrize("prediction_type", _PREDICTION_TYPES)
def test_denoiser_to_target_scale(kind, s_val, prediction_type):
    """pred(D_hat) - conditional_target == a_t * (D_hat - z) for an imperfect D_hat."""
    path = GaussianPath(kind)
    torch.manual_seed(2)
    z, eps = torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    s = torch.tensor(s_val)
    alpha, beta, _, _ = path.coeffs(s)
    x = alpha * z + beta * eps
    d_hat = z + 0.3 * torch.randn_like(z)

    pred = {
        "velocity": path.to_velocity(d_hat, x, s, "denoiser"),
        "noise": path._to_eps(d_hat, x, s, "denoiser"),
        "score": path.to_score(d_hat, x, s, "denoiser"),
        "denoiser": d_hat,
    }[prediction_type]
    target = path.conditional_target(z, eps, s, prediction_type)
    scale = path.denoiser_to_target_scale(s, prediction_type)
    assert torch.allclose(pred - target, scale * (d_hat - z), atol=1e-4, rtol=1e-4)


def test_path_variable():
    noise_level = np.array([0.0, 0.5, 1.0])
    t = GaussianPath("condot", t_eps=1e-3).path_variable(noise_level, "cpu")
    assert t.shape == (3, 1, 1)
    assert torch.allclose(t.reshape(-1), torch.tensor([1e-3, 0.5, 1 - 1e-3]))
    sigma = GaussianPath("ve").path_variable(np.array([1e-3, 80.0]), "cpu")
    assert torch.equal(sigma.reshape(-1), torch.tensor([1e-3, 80.0]))


def test_uniform_noise_level_is_path_time():
    rng = np.random.default_rng(0)
    cfg = {"distribution": "uniform", "t_min": 0.2, "t_max": 0.4}
    draws = [sample_noise_level(rng, cfg) for _ in range(100)]
    assert min(draws) >= 0.2 and max(draws) <= 0.4


# --------------------------------------------------------------------------------------------------
# 3. ve / denoiser conversions (EDM)
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("sig", _SIGMAS)
def test_ve_coeffs(sig):
    a, b, ad, bd = GaussianPath("ve").coeffs(torch.tensor(sig))
    assert a.item() == 1.0 and b.item() == pytest.approx(sig)
    assert ad.item() == 0.0 and bd.item() == 1.0


@pytest.mark.parametrize("sig", _SIGMAS)
def test_ve_denoiser_conversions(sig):
    p = GaussianPath("ve")
    torch.manual_seed(0)
    z, eps = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
    s = torch.tensor(sig)
    x = z + sig * eps  # ve: x = alpha z + beta eps = z + sigma eps

    # A perfect denoiser predicts d_pred = z.
    d_pred = z
    # to_velocity(d_pred) == (x - d_pred)/sigma == EDM's ODE slope == eps here.
    assert torch.allclose(p.to_velocity(d_pred, x, s, "denoiser"), (x - d_pred) / sig, atol=1e-5)
    assert torch.allclose(p.to_velocity(d_pred, x, s, "denoiser"), eps, atol=1e-4)
    # conditional target for denoiser is z; to_denoiser short-circuits to the prediction.
    assert torch.allclose(p.conditional_target(z, eps, s, "denoiser"), z)
    assert torch.allclose(p.to_denoiser(d_pred, x, s, "denoiser"), d_pred)
    # score = -eps/sigma. At the data end (sigma -> 0) it genuinely diverges and is reconstructed
    # via (x - z)/sigma; the machine-eps cancellation is amplified by ~1/sigma^2, so elements where
    # eps_i ~ 0 are dominated by float noise (intrinsic, not a bug). The EDM ODE never uses the
    # score (only to_velocity = (x-d)/sigma, checked above) and ve+sde is untested, so this does not
    # affect EDM sampling — only assert it away from the sigma->0 ill-conditioning.
    if sig >= 0.1:
        assert torch.allclose(
            p.to_score(d_pred, x, s, "denoiser"), -eps / sig, rtol=1e-3, atol=1e-4
        )


@pytest.mark.parametrize("sig", _SIGMAS)
def test_ve_eps_recovery_roundtrip(sig):
    """Feeding an imperfect denoiser: recovered eps and re-derived quantities stay consistent."""
    p = GaussianPath("ve")
    torch.manual_seed(1)
    z, eps = torch.randn(1, 4, 8), torch.randn(1, 4, 8)
    s = torch.tensor(sig)
    x = z + sig * eps
    d_pred = z + 0.3 * torch.randn_like(z)  # imperfect prediction
    eps_hat = (x - d_pred) / sig
    # velocity from denoiser == eps_hat; score == -eps_hat/sigma; denoiser back == d_pred.
    assert torch.allclose(p.to_velocity(d_pred, x, s, "denoiser"), eps_hat, atol=1e-5)
    assert torch.allclose(p.to_score(d_pred, x, s, "denoiser"), -eps_hat / sig, atol=1e-5)
    assert torch.allclose(p.to_denoiser(d_pred, x, s, "denoiser"), d_pred, atol=0)


# --------------------------------------------------------------------------------------------------
# 4. schedule and preconditioner match EDM
# --------------------------------------------------------------------------------------------------
def test_sampling_nodes():
    fe = _fake_engine()
    assert torch.equal(
        FlowMatchingForecastEngine._sampling_nodes(fe, 18, "cpu"),
        rho_sigma_steps(fe.sigma_min, fe.sigma_max, fe.rho, 18, "cpu"),
    )
    fe = _fake_engine(path=GaussianPath("condot", t_eps=1e-3))
    got = FlowMatchingForecastEngine._sampling_nodes(fe, 10, "cpu")
    assert torch.equal(got, torch.linspace(1e-3, 1.0 - 1e-3, 11, dtype=torch.float32))


@pytest.mark.parametrize("sig", _SIGMAS)
def test_ve_preconditioner_matches_edm(sig):
    fe = _fake_engine()
    s = torch.tensor(sig).reshape(1, 1, 1)
    # c_in = 1/sqrt(sigma^2 + sigma_data^2), as in EDMForecastEngine.denoise
    assert torch.allclose(
        FlowMatchingForecastEngine._c_in(fe, s), 1.0 / (s**2 + fe.sigma_data**2).sqrt()
    )
    # embedder input = log(sigma)/4 * scale; default scale 1.0
    assert torch.allclose(FlowMatchingForecastEngine._emb_input(fe, s), s.reshape(-1).log() / 4.0)
    fe = _fake_engine(edm_noise_time_scale=100.0)
    assert torch.allclose(
        FlowMatchingForecastEngine._emb_input(fe, s), (s.reshape(-1).log() / 4.0) * 100.0
    )


@pytest.mark.parametrize("sig", _SIGMAS)
def test_ve_output_preconditioning(sig):
    """ve with edm_preconditioning is the original Karras c_skip/c_out; without it (diffusion.py's
    default) c_skip=0, c_out=1; condot is always identity."""
    s = torch.tensor(sig)
    raw = torch.randn(1, 4, 8)
    x = torch.randn(1, 4, 8)

    fe = _fake_engine(edm_preconditioning=True)
    c_skip = fe.sigma_data**2 / (s**2 + fe.sigma_data**2)
    c_out = s * fe.sigma_data / (s**2 + fe.sigma_data**2).sqrt()
    assert torch.allclose(
        FlowMatchingForecastEngine._precondition_output(fe, raw, x, s), c_skip * x + c_out * raw
    )
    # EDM unit-variance property: lambda(sigma) * c_out^2 == 1.
    lam = (s**2 + fe.sigma_data**2) / (s * fe.sigma_data) ** 2
    assert torch.allclose(lam * c_out**2, torch.ones_like(c_out), atol=1e-5)

    fe = _fake_engine(edm_preconditioning=False)
    assert torch.equal(FlowMatchingForecastEngine._precondition_output(fe, raw, x, s), raw)
    fe = _fake_engine(path=GaussianPath("condot"), edm_preconditioning=True)
    assert torch.equal(FlowMatchingForecastEngine._precondition_output(fe, raw, x, s), raw)


# --------------------------------------------------------------------------------------------------
# 5. sampler trajectory equivalence (load-bearing)
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("num_steps", [10, 20, 50])
def test_sampler_trajectory_equivalence(num_steps):
    """The generic Euler+Heun loop over GaussianPath.to_velocity(denoiser) must reproduce EDM's
    probability-flow ODE loop (EDMForecastEngine._run_sampler) to float precision, for a fixed
    mock net."""
    p = GaussianPath("ve")
    nodes = FlowMatchingForecastEngine._sampling_nodes(_fake_engine(), num_steps, "cpu")

    def net(x, s):  # deterministic mock denoiser
        return x / (1.0 + s**2)

    torch.manual_seed(0)
    x0 = torch.randn(1, 4, 8) * nodes[0].float()

    # --- flow framework: generic Euler + Heun using to_velocity(denoiser) ---
    xf = x0.clone()
    for i in range(num_steps):
        sc, sn = nodes[i].float(), nodes[i + 1].float()
        h = sn - sc
        u = p.to_velocity(net(xf, sc), xf, sc, "denoiser")
        xn = xf + h * u
        if i < num_steps - 1:
            u2 = p.to_velocity(net(xn, sn), xn, sn, "denoiser")
            xn = xf + h * 0.5 * (u + u2)
        xf = xn

    # --- EDM: Euler + Heun with d_cur = (x - D)/sigma ---
    xe = x0.clone()
    for i in range(num_steps):
        sc, sn = nodes[i].float(), nodes[i + 1].float()
        d_cur = (xe - net(xe, sc)) / sc
        xn = xe + (sn - sc) * d_cur
        if i < num_steps - 1:
            d_prime = (xn - net(xn, sn)) / sn
            xn = xe + (sn - sc) * 0.5 * (d_cur + d_prime)
        xe = xn

    assert torch.allclose(xf, xe, atol=1e-6)


# --------------------------------------------------------------------------------------------------
# 6. loss
# --------------------------------------------------------------------------------------------------
def _loss(path: str, prediction_type: str, stage: str) -> LossFlowMatching:
    cf = OmegaConf.create(
        {
            "num_register_tokens": 0,
            "num_class_tokens": 0,
            "diffusion": {
                "type": "flow_matching",
                "flow_matching": {"path": path, "prediction_type": prediction_type},
            },
        }
    )
    return LossFlowMatching(cf, None, stage, "cpu", mse={})


@pytest.mark.parametrize("sig", [0.1, 1.0, 7.0])
def test_loss_noise_weight(sig):
    """ve reproduces EDM lambda(sigma) (LossLatentDiffusion); condot is unweighted."""
    s = torch.tensor([sig])
    lam_ref = (s**2 + 1.0**2) / (s * 1.0) ** 2
    assert torch.allclose(_loss("ve", "denoiser", "train")._get_noise_weight(s), lam_ref)
    assert _loss("condot", "velocity", "train")._get_noise_weight(s) is None


def test_loss_space():
    """Training rescales x0 to the parameterization space, validation stays in x0 space."""
    tokens = torch.randn(2, 5, 8)
    t = np.array([0.25, 0.5])
    got = _loss("condot", "velocity", "train")._to_loss_space(tokens, t)
    assert torch.allclose(got, tokens / (1.0 - torch.tensor(t, dtype=torch.float32)[:, None, None]))
    assert torch.equal(_loss("condot", "velocity", "val")._to_loss_space(tokens, t), tokens)
