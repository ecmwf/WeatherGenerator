# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""EDM-implant equivalence tests for the flow-matching framework.

These verify that selecting the ``ve`` path + ``denoiser`` parameterization reproduces
``DiffusionForecastEngine`` (diffusion.py): the conversions, the rho-spaced sigma schedule, the
preconditioner, the loss weight, and — the load-bearing one — that the generic Euler+Heun sampler
produces the *same trajectory* as EDM's probability-flow ODE loop. Reference EDM quantities are
transcribed inline from diffusion.py so the tests do not depend on constructing the heavy engine.
"""

import math
import types

import pytest
import torch

from weathergen.model.flow_matching import FlowMatchingForecastEngine, GaussianPath
from weathergen.train.loss_modules.loss_module_flow_matching import LossFlowMatching

_SIGMAS = [1e-3, 0.1, 1.0, 7.0, 80.0]


def _fake_engine(**over):
    """A minimal stand-in exposing just the attributes the seam methods read, so we can call the
    unbound FlowMatchingForecastEngine methods without building the full nn.Module."""
    ns = types.SimpleNamespace(
        path=GaussianPath("ve"),
        sigma_data=1.0,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7,
        p_mean=1.5,
        p_std=1.2,
        sigma_min_quantile=0.05,
        t_eps=1e-3,
        time_scale=1000.0,
        edm_noise_time_scale=1.0,
        no_skip_connection=False,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def _edm_ref_nodes(num_steps, p_mean, p_std, sigma_min, sigma_max, sigma_data, rho, q):
    """Transcription of the sigma schedule in diffusion.py:442-476."""
    sigma_max_eff = min(sigma_max, math.exp(p_mean + 3.0 * p_std))
    z = {0.01: -2.326, 0.025: -1.960, 0.05: -1.645, 0.10: -1.282}.get(q, -1.645)
    sigma_min_eff = max(sigma_min, math.exp(p_mean + z * p_std), sigma_data * 0.01)
    si = torch.arange(num_steps, dtype=torch.float64)
    t = (
        sigma_max_eff ** (1 / rho)
        + si / (num_steps - 1) * (sigma_min_eff ** (1 / rho) - sigma_max_eff ** (1 / rho))
    ) ** rho
    return torch.cat([t, torch.zeros_like(t[:1])])


# --------------------------------------------------------------------------------------------------
# 1. ve / denoiser conversions
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
# 2. sigma schedule bit-match vs diffusion.py
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("num_steps", [10, 18, 50])
@pytest.mark.parametrize("pm,ps", [(1.5, 1.2), (0.5, 1.0), (2.0, 0.8)])
@pytest.mark.parametrize("q", [0.05, 0.01, 0.10])
def test_ve_schedule_bit_matches_diffusion(num_steps, pm, ps, q):
    fe = _fake_engine(p_mean=pm, p_std=ps, sigma_min_quantile=q)
    got = FlowMatchingForecastEngine._sampling_nodes(fe, num_steps, "cpu")
    ref = _edm_ref_nodes(
        num_steps, pm, ps, fe.sigma_min, fe.sigma_max, fe.sigma_data, fe.rho, q
    )
    assert got.dtype == ref.dtype == torch.float64
    assert torch.equal(got, ref)  # bit-identical


def test_condot_schedule_unchanged():
    fe = _fake_engine(path=GaussianPath("condot"))
    got = FlowMatchingForecastEngine._sampling_nodes(fe, 10, "cpu")
    ref = torch.linspace(fe.t_eps, 1.0 - fe.t_eps, 11, dtype=torch.float32)
    assert torch.equal(got, ref)


# --------------------------------------------------------------------------------------------------
# 3. preconditioner matches EDM
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("sig", _SIGMAS)
def test_ve_preconditioner_matches_edm(sig):
    fe = _fake_engine()
    s = torch.tensor(sig)
    # c_in = 1/sqrt(sigma^2 + sigma_data^2)  (diffusion.py:270)
    assert torch.allclose(
        FlowMatchingForecastEngine._c_in(fe, s), 1.0 / (s**2 + fe.sigma_data**2).sqrt()
    )
    # embedder input = log(sigma)/4 * scale  (diffusion.py:271); default scale 1.0
    assert torch.allclose(
        FlowMatchingForecastEngine._emb_input(fe, s), s.reshape(1).log() / 4.0
    )


@pytest.mark.parametrize("sig", _SIGMAS)
def test_ve_output_preconditioning_karras_default(sig):
    """Default ve output preconditioning is the original Karras c_skip/c_out; the no-skip flag
    restores diffusion.py's c_skip=0, c_out=1; condot is always identity."""
    s = torch.tensor(sig)
    raw = torch.randn(1, 4, 8)
    x = torch.randn(1, 4, 8)

    # Default (Karras): D = c_skip*x + c_out*raw
    fe = _fake_engine()
    c_skip = fe.sigma_data**2 / (s**2 + fe.sigma_data**2)
    c_out = s * fe.sigma_data / (s**2 + fe.sigma_data**2).sqrt()
    assert torch.allclose(
        FlowMatchingForecastEngine._precondition_output(fe, raw, x, s), c_skip * x + c_out * raw
    )
    # EDM unit-variance property: lambda(sigma) * c_out^2 == 1.
    lam = (s**2 + fe.sigma_data**2) / (s * fe.sigma_data) ** 2
    assert torch.allclose(lam * c_out**2, torch.ones_like(c_out), atol=1e-5)

    # no_skip_connection: identity.
    fe_ns = _fake_engine(no_skip_connection=True)
    assert torch.equal(FlowMatchingForecastEngine._precondition_output(fe_ns, raw, x, s), raw)

    # condot: always identity (no EDM preconditioning), regardless of the flag.
    fe_c = _fake_engine(path=GaussianPath("condot"), no_skip_connection=False)
    assert torch.equal(FlowMatchingForecastEngine._precondition_output(fe_c, raw, x, s), raw)


def test_edm_noise_time_scale_applies():
    fe = _fake_engine(edm_noise_time_scale=100.0)
    s = torch.tensor(7.0)
    assert torch.allclose(
        FlowMatchingForecastEngine._emb_input(fe, s), (s.reshape(1).log() / 4.0) * 100.0
    )


# --------------------------------------------------------------------------------------------------
# 4. sampler trajectory equivalence (load-bearing)
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("num_steps", [10, 20, 50])
def test_sampler_trajectory_equivalence(num_steps):
    """The generic Euler+Heun loop over GaussianPath.to_velocity(denoiser) must reproduce EDM's
    probability-flow ODE loop (diffusion.py:500-525) to float precision, for a fixed mock net."""
    p = GaussianPath("ve")
    fe = _fake_engine()
    nodes = FlowMatchingForecastEngine._sampling_nodes(fe, num_steps, "cpu")

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

    # --- EDM: diffusion.py Euler + Heun with d_cur = (x - D)/sigma ---
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
# 5. training-quantity + loss-weight equivalence
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("eta_val", [-2.0, 0.0, 0.7, 2.0])
def test_ve_training_quantities_match_edm(eta_val):
    p = GaussianPath("ve")
    fe = _fake_engine()
    torch.manual_seed(0)
    z, eps = torch.randn(1, 4, 8), torch.randn(1, 4, 8)
    eta = torch.tensor(eta_val)
    sigma = (eta * fe.p_std + fe.p_mean).exp()  # diffusion training sigma

    # x_t = alpha z + beta eps == z + sigma eps  (== diffusion's y + n with n = randn*sigma)
    a, b, _, _ = p.coeffs(sigma)
    assert torch.allclose(a * z + b * eps, z + sigma * eps)
    # regression target for denoiser is the clean latent z (== LossLatentDiffusion's target)
    assert torch.allclose(p.conditional_target(z, eps, sigma, "denoiser"), z)
    # preconditioner quantities match EDM
    assert torch.allclose(
        FlowMatchingForecastEngine._c_in(fe, sigma), 1.0 / (sigma**2 + fe.sigma_data**2).sqrt()
    )


@pytest.mark.parametrize("sig", [0.1, 1.0, 7.0])
def test_loss_lambda_weight(sig):
    """LossFlowMatching._noise_weight reproduces EDM lambda(sigma) for ve/train, 1.0 for val."""
    s = torch.tensor(sig)
    lam_ref = (s**2 + 1.0**2) / (s * 1.0) ** 2

    train = types.SimpleNamespace(path=GaussianPath("ve"), sigma_data=1.0, stage="train")
    got = LossFlowMatching._noise_weight(train, s)
    assert torch.allclose(got, lam_ref)

    # Validation is unweighted.
    val = types.SimpleNamespace(path=GaussianPath("ve"), sigma_data=1.0, stage="val")
    assert LossFlowMatching._noise_weight(val, s) == 1.0

    # condot is unweighted (flow-matching MSE).
    condot = types.SimpleNamespace(path=GaussianPath("condot"), sigma_data=1.0, stage="train")
    assert LossFlowMatching._noise_weight(condot, s) == 1.0
