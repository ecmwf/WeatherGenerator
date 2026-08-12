# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch

from weathergen.model import attention


def test_varlen_attention_matches_independent_sequence_attention():
    torch.manual_seed(0)
    qs = torch.randn(5, 2, 4)
    ks = torch.randn(7, 2, 4)
    vs = torch.randn(7, 2, 4)
    q_lens = torch.tensor([0, 2, 3])
    kv_lens = torch.tensor([0, 4, 3])

    actual = attention._varlen_attention(qs, ks, vs, q_lens, kv_lens)

    expected = []
    q_start = kv_start = 0
    for q_len, kv_len in zip(q_lens[1:].tolist(), kv_lens[1:].tolist(), strict=True):
        q = qs[q_start : q_start + q_len].transpose(0, 1).unsqueeze(0)
        k = ks[kv_start : kv_start + kv_len].transpose(0, 1).unsqueeze(0)
        v = vs[kv_start : kv_start + kv_len].transpose(0, 1).unsqueeze(0)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        expected.append(out.squeeze(0).transpose(0, 1))
        q_start += q_len
        kv_start += kv_len

    torch.testing.assert_close(actual, torch.cat(expected))


def test_varlen_softcap_attention_applies_dropout():
    torch.manual_seed(1)
    qs = torch.randn(3, 2, 4)
    ks = torch.randn(5, 2, 4)
    vs = torch.randn(5, 2, 4)
    softcap = 2.0
    dropout_rate = 0.25

    torch.manual_seed(2)
    actual = attention._varlen_attention(
        qs,
        ks,
        vs,
        torch.tensor([0, 3]),
        torch.tensor([0, 5]),
        dropout_rate=dropout_rate,
        softcap=softcap,
    )

    q = qs.transpose(0, 1).unsqueeze(0)
    k = ks.transpose(0, 1).unsqueeze(0)
    v = vs.transpose(0, 1).unsqueeze(0)
    scores = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    scores = softcap * torch.tanh(scores / softcap)
    weights = torch.softmax(scores, dim=-1)
    torch.manual_seed(2)
    weights = torch.nn.functional.dropout(weights, p=dropout_rate, training=True)
    expected = torch.matmul(weights, v).squeeze(0).transpose(0, 1)

    torch.testing.assert_close(actual, expected)


def test_with_flash_requires_flash_attention(monkeypatch):
    monkeypatch.setattr(attention, "FLASH_ATTN_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="with_flash=True requires flash-attn"):
        attention.MultiSelfAttentionHeadVarlen(4, 2, with_flash=True)


def test_with_flash_false_forces_pytorch_fallback(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("flash-attn should not be called")

    monkeypatch.setattr(attention, "FLASH_ATTN_AVAILABLE", True)
    monkeypatch.setattr(attention, "flash_attn_varlen_func", fail_if_called)

    layer = attention.MultiSelfAttentionHeadVarlen(
        4,
        2,
        with_flash=False,
        with_residual=False,
    )
    layer.eval()

    output = layer(torch.randn(3, 4), torch.tensor([0, 2, 1]))

    assert layer.with_flash is False
    assert output.dtype == torch.float32
    assert output.shape == (3, 4)
