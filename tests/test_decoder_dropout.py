# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for the configurable dropout of the decoder (pred_dropout_rate)."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from weathergen.model.engines import TargetPredictionEngineClassic

REPO = Path(__file__).parent.parent
DIM = 32
DIM_KV = 64
DIM_COORD = 8


def _engine(**options):
    cf = OmegaConf.create(
        {
            "ae_global_dim_embed": DIM_KV,
            "with_flash_attention": True,  # construction only, no forward pass
            "norm_type": "LayerNorm",
            "norm_eps": 1e-5,
            "mlp_norm_eps": 1e-5,
            "attention_dtype": "fp32",
            "pred_self_attention": True,
            "pred_mlp_adaln": True,
            **options,
        }
    )
    stream_config = {"name": "ERA5", "target_readout": {"num_heads": 4}}
    torch.manual_seed(0)
    return TargetPredictionEngineClassic(
        cf, [DIM] * 3, DIM_COORD, None, 2, "mlp", 0.0, stream_config=stream_config
    )


def _dropout_rates(engine):
    rates = []
    for m in engine.modules():
        if isinstance(m, torch.nn.Dropout):
            rates.append(m.p)
        elif isinstance(getattr(m, "dropout_rate", None), float):
            rates.append(m.dropout_rate)
    return rates


def test_default_unchanged():
    """Without the option (configs from before it existed) the decoder keeps dropout 0.1."""
    rates = _dropout_rates(_engine())
    assert len(rates) > 0 and set(rates) == {0.1}
    assert _dropout_rates(_engine(pred_dropout_rate=0.1)) == rates


def test_default_config_unchanged():
    cfg = OmegaConf.load(REPO / "config" / "default_config.yml")
    assert cfg.pred_dropout_rate == 0.1


@pytest.mark.parametrize("rate", [0.0, 0.05])
def test_rate_applies_to_all_blocks(rate):
    rates = _dropout_rates(_engine(pred_dropout_rate=rate))
    assert len(rates) > 0 and set(rates) == {rate}


def test_state_dict_unchanged():
    """Checkpoints trained with dropout 0.1 load into a decoder without dropout and vice versa."""
    sd_ref = _engine().state_dict()
    engine = _engine(pred_dropout_rate=0.0)
    assert list(engine.state_dict()) == list(sd_ref)
    engine.load_state_dict(sd_ref, strict=True)
