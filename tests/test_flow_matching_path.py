# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for the flow-matching Gaussian probability path and its conversions.

Given the *conditional* quantity for a single ``(z, eps, t)`` (which the network target
equals), every converter must recover a mutually consistent ``(z, eps)`` and hence each
other quantity (course Prop. 1 / Remark 16). This validates the algebra without a network.
"""

import pytest
import torch

from weathergen.model.flow_matching import GaussianPath

_TIMES = [1e-3, 0.1, 0.3, 0.5, 0.7, 0.9, 1 - 1e-3]


@pytest.fixture
def path():
    return GaussianPath("condot")


@pytest.fixture
def z_eps():
    torch.manual_seed(0)
    return torch.randn(2, 5, 8), torch.randn(2, 5, 8)


@pytest.mark.parametrize("t_val", _TIMES)
def test_condot_velocity_is_z_minus_eps(path, z_eps, t_val):
    z, eps = z_eps
    t = torch.tensor(t_val)
    v = path.conditional_target(z, eps, t, "velocity")
    assert torch.allclose(v, z - eps, atol=1e-5)


@pytest.mark.parametrize("t_val", _TIMES)
@pytest.mark.parametrize("prediction_type", ["velocity", "noise", "score"])
def test_conversions_recover_consistent_quantities(path, z_eps, t_val, prediction_type):
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
