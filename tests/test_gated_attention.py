# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Headwise gated attention (Qiu et al. 2025, "Gated Attention for Large Language Models").

With `use_gated_attention`, each attention head's output is scaled per token by
sigmoid(x_norm @ W_g)_h before the output projection, where x_norm is the (query-side)
input of the q projection and W_g is `proj_gate.weight` of shape [num_heads, dim_in].
"""

import ast
import pathlib

import pytest
import torch

MODEL_DIR = pathlib.Path(__file__).parents[1] / "src" / "weathergen" / "model"

# the attention classes the model builds (MultiSelfAttentionHeadVarlenFlex is unused)
GATED_CLASSES = {
    "MultiSelfAttentionHeadVarlen",
    "MultiSelfAttentionHeadLocal",
    "MultiSelfAttentionHead",
    "MultiCrossAttentionHeadVarlen",
    "MultiCrossAttentionHeadVarlenSlicedQ",
    "MultiCrossAttentionHead",
}


def test_every_attention_call_site_passes_the_flag():
    """Each construction of an attention class gets `use_gated_attention`, either as a
    keyword or through an `attention_kwargs` splat, and every `attention_kwargs` dict has it."""
    missing = []
    for path in MODEL_DIR.glob("*.py"):
        if path.name == "attention.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", getattr(node.func, "attr", None))
                if name in GATED_CLASSES:
                    kws = {k.arg for k in node.keywords}
                    if "use_gated_attention" not in kws and None not in kws:
                        missing.append(f"{path.name}:{node.lineno} {name}")
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
                targets = [getattr(t, "id", None) for t in node.targets]
                if "attention_kwargs" in targets:
                    keys = {getattr(k, "value", None) for k in node.value.keys}
                    if "use_gated_attention" not in keys:
                        missing.append(f"{path.name}:{node.lineno} attention_kwargs")
    assert not missing, missing


# ---- module tests: flash attention and flex attention need a GPU ----

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="flash/flex attention need CUDA")

DIM, DIM_KV, HEADS = 64, 48, 4
N, NK, B = 32, 24, 2
SLICES = 2


def _make(cls_name, gated, use_xsa=False):
    from weathergen.model import attention as A

    kw = dict(dropout_rate=0.0, with_residual=True, use_gated_attention=gated)
    if cls_name == "MultiSelfAttentionHeadVarlen":
        return A.MultiSelfAttentionHeadVarlen(DIM, HEADS, use_xsa=use_xsa, **kw)
    if cls_name == "MultiSelfAttentionHeadLocal":
        return A.MultiSelfAttentionHeadLocal(
            DIM, HEADS, qkv_len=N, block_factor=8, use_xsa=use_xsa, **kw
        )
    if cls_name == "MultiSelfAttentionHead":
        return A.MultiSelfAttentionHead(DIM, HEADS, use_xsa=use_xsa, **kw)
    if cls_name == "MultiCrossAttentionHeadVarlen":
        return A.MultiCrossAttentionHeadVarlen(DIM, DIM_KV, HEADS, **kw)
    if cls_name == "MultiCrossAttentionHeadVarlenSlicedQ":
        return A.MultiCrossAttentionHeadVarlenSlicedQ(DIM, DIM_KV, SLICES, HEADS, **kw)
    if cls_name == "MultiCrossAttentionHead":
        return A.MultiCrossAttentionHead(DIM, DIM_KV, HEADS, **kw)
    raise ValueError(cls_name)


def _inputs(cls_name, device="cuda"):
    g = torch.Generator().manual_seed(0)

    def r(*s):
        return torch.randn(*s, generator=g).to(device)

    lens = torch.tensor([N // 2, N // 2], dtype=torch.int32, device=device)
    lens_kv = torch.tensor([NK // 2, NK // 2], dtype=torch.int32, device=device)
    if cls_name == "MultiSelfAttentionHeadVarlen":
        return (r(N, DIM), lens), {}
    if cls_name == "MultiSelfAttentionHeadLocal":
        return (r(B, N, DIM),), {}
    if cls_name == "MultiSelfAttentionHead":
        return (r(B, N, DIM),), {}
    if cls_name == "MultiCrossAttentionHeadVarlen":
        return (r(N, DIM), r(NK, DIM_KV), lens, lens_kv), {}
    if cls_name == "MultiCrossAttentionHeadVarlenSlicedQ":
        return (r(N, SLICES, DIM), r(NK, DIM_KV), lens, lens_kv), {}
    if cls_name == "MultiCrossAttentionHead":
        return (r(B, N, DIM), r(B, NK, DIM_KV)), {}
    raise ValueError(cls_name)


def _q_proj_modules(m):
    return (
        list(m.proj_heads_q)
        if isinstance(m.proj_heads_q, torch.nn.ModuleList)
        else [m.proj_heads_q]
    )


def _capture(m, args):
    """Run m and capture the input of proj_out, of the q projection(s) and of proj_gate."""
    cap = {"q_in": []}
    hooks = [
        m.proj_out.register_forward_pre_hook(lambda _m, a: cap.__setitem__("out_in", a[0].float()))
    ]
    for p in _q_proj_modules(m):
        hooks.append(p.register_forward_pre_hook(lambda _m, a: cap["q_in"].append(a[0].float())))
    if getattr(m, "proj_gate", None) is not None:
        hooks.append(
            m.proj_gate.register_forward_pre_hook(
                lambda _m, a: cap.__setitem__("gate_in", a[0].float())
            )
        )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(*args)
    for h in hooks:
        h.remove()
    return out, cap


@pytest.mark.parametrize("cls_name", sorted(GATED_CLASSES))
def test_flag_off_adds_no_parameters(cls_name):
    """Existing checkpoints load unchanged: without the flag there is no gate parameter."""
    m = _make(cls_name, gated=False)
    assert getattr(m, "proj_gate", None) is None
    assert not any("gate" in k for k in m.state_dict())


@pytest.mark.parametrize("cls_name", sorted(GATED_CLASSES))
def test_gate_is_headwise_and_only_new_parameter(cls_name):
    torch.manual_seed(0)
    ungated = _make(cls_name, gated=False)
    gated = _make(cls_name, gated=True)
    res = gated.load_state_dict(ungated.state_dict(), strict=False)
    assert res.missing_keys == ["proj_gate.weight"] and not res.unexpected_keys
    assert gated.proj_gate.weight.shape == (HEADS, DIM)
    assert gated.proj_gate.bias is None


@cuda
@pytest.mark.parametrize("use_xsa", [False, True])
@pytest.mark.parametrize("cls_name", sorted(GATED_CLASSES))
def test_gate_scales_each_head_by_sigmoid_of_query_input(cls_name, use_xsa):
    """proj_out input (gated) == proj_out input (ungated) * sigmoid(x_norm W_g^T) per head,
    and the gate reads the same normalised tensor as the q projection."""
    xsa = use_xsa and cls_name.startswith("MultiSelfAttention")
    if use_xsa and not xsa:
        pytest.skip("XSA only exists for self-attention")
    torch.manual_seed(0)
    ungated = _make(cls_name, gated=False, use_xsa=xsa).cuda().eval()
    gated = _make(cls_name, gated=True, use_xsa=xsa).cuda().eval()
    gated.load_state_dict(ungated.state_dict(), strict=False)
    with torch.no_grad():
        gated.proj_gate.weight.normal_(0.0, 1.0)  # gates spread well away from 0.5
    args, _ = _inputs(cls_name)

    with torch.no_grad():
        _, cap_u = _capture(ungated, args)
        _, cap_g = _capture(gated, args)

    # gate input is the q-projection input (for sliced Q: one gate shared by all slices)
    q_in = torch.stack(cap_g["q_in"], dim=-2) if len(cap_g["q_in"]) > 1 else cap_g["q_in"][0]
    torch.testing.assert_close(cap_g["gate_in"], q_in)

    o_u, o_g = cap_u["out_in"], cap_g["out_in"]
    gate = torch.sigmoid(cap_g["gate_in"] @ gated.proj_gate.weight.float().T)  # [..., H]
    o_u = o_u.reshape(*gate.shape, -1)
    o_g = o_g.reshape(*gate.shape, -1)
    assert gate.min() < 0.2 and gate.max() > 0.8
    torch.testing.assert_close(o_g, o_u * gate[..., None], rtol=2e-2, atol=2e-3)


@cuda
@pytest.mark.parametrize("cls_name", sorted(GATED_CLASSES))
def test_zero_gate_weights_halve_attention_update(cls_name):
    """W_g = 0 gives gate 0.5 everywhere: out - x_in is exactly half the ungated update."""
    torch.manual_seed(0)
    ungated = _make(cls_name, gated=False).cuda().eval()
    gated = _make(cls_name, gated=True).cuda().eval()
    gated.load_state_dict(ungated.state_dict(), strict=False)
    with torch.no_grad():
        gated.proj_gate.weight.zero_()
    args, _ = _inputs(cls_name)
    x_in = args[0]
    with torch.no_grad():
        out_u, _ = _capture(ungated, args)
        out_g, _ = _capture(gated, args)
    d_u = (out_u.float() - x_in.reshape(out_u.shape)).float()
    d_g = (out_g.float() - x_in.reshape(out_g.shape)).float()
    torch.testing.assert_close(d_g, 0.5 * d_u, rtol=2e-2, atol=2e-3)


@cuda
@pytest.mark.parametrize("cls_name", sorted(GATED_CLASSES))
def test_gate_gets_finite_nonzero_gradient(cls_name):
    torch.manual_seed(0)
    m = _make(cls_name, gated=True).cuda().train()
    args, _ = _inputs(cls_name)
    out, _ = _capture(m, args)
    out.float().pow(2).mean().backward()
    g = m.proj_gate.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
