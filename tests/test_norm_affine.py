# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""`norm_affine=False` and `qk_norm_scale` (strip_norm_affine, applied at the end of Model.create).

norm_affine=False removes the weight and bias of every LayerNorm/RMSNorm. qk_norm_scale removes the
affine of the query/key norms and multiplies their output by a fixed scale s, so that every
attention logit q.k / sqrt(d) is bounded by s**2 * sqrt(d).
"""

import pytest
import torch

from weathergen.model.layers import MLP
from weathergen.model.norms import FixedScaleNorm, RMSNorm, strip_norm_affine

DIM, HEADS = 64, 4


def _model():
    pytest.importorskip("flash_attn")  # attention.py imports it at module level
    from weathergen.model import attention as A

    att = A.MultiSelfAttentionHead(
        DIM, HEADS, dropout_rate=0.0, norm_type="LayerNorm", qk_norm_type="RMSNorm"
    )
    mlp = MLP(DIM, DIM, with_residual=True, norm_type="LayerNorm")
    return torch.nn.ModuleDict({"att": att, "mlp": mlp, "ln": torch.nn.LayerNorm(DIM)})


def _norm_params(model):
    norms = (torch.nn.LayerNorm, RMSNorm)
    return [
        f"{n}.{p}"
        for n, m in model.named_modules()
        if isinstance(m, norms)
        for p, _ in m.named_parameters(recurse=False)
    ]


def test_defaults_change_nothing():
    model = _model()
    keys = set(model.state_dict())
    strip_norm_affine(model, norm_affine=True, qk_norm_scale=None)
    assert set(model.state_dict()) == keys
    assert not any(isinstance(m, FixedScaleNorm) for m in model.modules())


def test_no_affine_removes_every_norm_parameter():
    model = _model()
    assert _norm_params(model)  # the MLP norm, the plain LayerNorm and the QK RMSNorms are affine
    strip_norm_affine(model, norm_affine=False, qk_norm_scale=None)
    assert _norm_params(model) == []
    assert not any(isinstance(m, FixedScaleNorm) for m in model.modules())
    # all remaining parameters are Linear weights/biases
    assert all(
        ".proj" in k or "layers" in k or k.endswith(("weight", "bias")) for k in model.state_dict()
    )


def test_qk_scale_only_touches_qk_norms():
    model = _model()
    before = set(_norm_params(model))
    strip_norm_affine(model, norm_affine=True, qk_norm_scale=1.3)
    after = set(_norm_params(model))
    assert {k for k in before - after} == {"att.lnorm_q.weight", "att.lnorm_k.weight"}
    assert isinstance(model["att"].lnorm_q, FixedScaleNorm)
    assert isinstance(model["att"].lnorm_k, FixedScaleNorm)


def test_qk_scale_output_and_logit_bound():
    model = _model()
    strip_norm_affine(model, norm_affine=False, qk_norm_scale=1.3)
    d = DIM // HEADS
    x = 50 * torch.randn(1000, d)
    q = model["att"].lnorm_q(x)
    k = model["att"].lnorm_k(x.flip(0))
    torch.testing.assert_close(
        q.pow(2).mean(-1).sqrt(), torch.full((1000,), 1.3), rtol=1e-3, atol=1e-3
    )
    logits = q @ k.T / d**0.5
    assert logits.abs().max() <= 1.3**2 * d**0.5 * (1 + 1e-4)
    # identical q and k reach the bound
    torch.testing.assert_close(
        (q * q).sum(-1) / d**0.5, torch.full((1000,), 1.3**2 * d**0.5), rtol=1e-3, atol=1e-3
    )
    assert list(model["att"].lnorm_q.parameters()) == []


def test_rmsnorm_without_weight_resets_and_runs():
    norm = RMSNorm(8)
    strip_norm_affine(torch.nn.ModuleDict({"n": norm}), norm_affine=False, qk_norm_scale=None)
    norm.reset_parameters()
    x = torch.randn(3, 8)
    torch.testing.assert_close(norm(x), x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps))


def test_layernorm_without_affine_matches_functional():
    ln = torch.nn.LayerNorm(DIM)
    strip_norm_affine(torch.nn.ModuleDict({"n": ln}), norm_affine=False, qk_norm_scale=None)
    ln.reset_parameters()
    x = torch.randn(5, DIM)
    torch.testing.assert_close(ln(x), torch.nn.functional.layer_norm(x, (DIM,)))


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="flash attention needs CUDA")


@cuda
def test_stripped_attention_forward_backward():
    model = _model().cuda().to(torch.bfloat16)
    strip_norm_affine(model, norm_affine=False, qk_norm_scale=1.3)
    x = torch.randn(2, 32, DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = model["mlp"](model["att"](x))
    y.float().pow(2).mean().backward()
    assert torch.isfinite(x.grad).all()
