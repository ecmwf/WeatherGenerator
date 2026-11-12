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
from omegaconf import OmegaConf

from weathergen.model.norms import RMSNorm
from weathergen.train.optimizer.optimizer import (
    AdamW,
    Muon,
    _is_muon_param,
    _weight_decay_param_groups,
)

LR_CFG = OmegaConf.create(
    {"lr_start": 1e-6, "lr_max": 1e-4, "lr_final_decay": 1e-6, "lr_final": 0.0}
)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)
        self.ln = torch.nn.LayerNorm(4)
        self.q_norm = RMSNorm(4)
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4, bias=False), torch.nn.RMSNorm(4)])
        self.ln_plain = torch.nn.LayerNorm(4, elementwise_affine=False)


def _cfg(no_weight_decay):
    return OmegaConf.create(
        {
            "weight_decay": 0.1,
            "no_weight_decay": no_weight_decay,
            "adamw": {"beta1": 0.9, "beta2": 0.95, "eps": 1e-10},
            "muon": {},
        }
    )


def _names_by_decay(model, groups):
    name_of = {id(p): n for n, p in model.named_parameters()}
    return {g["weight_decay"]: sorted(name_of[id(p)] for p in g["params"]) for g in groups}


def test_default_is_single_decayed_group():
    model = _Model()
    for no_weight_decay in (None, []):
        groups = _weight_decay_param_groups(model, model.parameters(), _cfg(no_weight_decay))
        assert len(groups) == 1 and groups[0]["weight_decay"] == 0.1
        assert len(groups[0]["params"]) == len(list(model.parameters()))


def test_norms():
    model = _Model()
    groups = _weight_decay_param_groups(model, model.parameters(), _cfg(["norms"]))
    assert _names_by_decay(model, groups) == {
        0.1: ["blocks.0.weight", "lin.bias", "lin.weight"],
        0.0: ["blocks.1.weight", "ln.bias", "ln.weight", "q_norm.weight"],
    }


def test_norms_and_biases():
    model = _Model()
    groups = _weight_decay_param_groups(model, model.parameters(), _cfg(["norms", "biases"]))
    assert _names_by_decay(model, groups) == {
        0.1: ["blocks.0.weight", "lin.weight"],
        0.0: ["blocks.1.weight", "lin.bias", "ln.bias", "ln.weight", "q_norm.weight"],
    }


def test_unknown_kind_raises():
    model = _Model()
    with pytest.raises(AssertionError):
        _weight_decay_param_groups(model, model.parameters(), _cfg(["embeddings"]))


@pytest.mark.parametrize("optimizer_cls", [AdamW, Muon])
def test_optimizers_use_groups_and_eps(optimizer_cls):
    if optimizer_cls is Muon and not hasattr(torch.optim, "Muon"):
        pytest.skip("torch.optim.Muon not available")
    model = _Model()
    built = optimizer_cls(model, _cfg(["norms", "biases"]), LR_CFG, kappa=1.0)
    adamw = built.optimizers[-1]
    assert [g["weight_decay"] for g in adamw.param_groups] == [0.1, 0.0]
    assert all(g["eps"] == 1e-10 for g in adamw.param_groups)
    no_decay = {id(p) for p in adamw.param_groups[1]["params"]}
    assert id(model.q_norm.weight) in no_decay and id(model.lin.bias) in no_decay
    # every trainable parameter is optimized exactly once
    all_ids = [id(p) for opt in built.optimizers for g in opt.param_groups for p in g["params"]]
    assert sorted(all_ids) == sorted(id(p) for p in model.parameters())


# 2D parameter names of the d0sqd131-base model (layer indices are examples), and whether muon
# takes them with optimizer.muon.hidden_only
_HIDDEN_ONLY_CASES = {
    "module.embed_target_coords.ERA5.linear.weight": False,
    "module.encoder.ae_global_engine.ae_global_blocks.3.layers.0.weight": True,
    "module.encoder.ae_global_engine.ae_global_blocks.3.proj_heads_q.weight": True,
    "module.encoder.ae_local_global_engine.ae_adapter.0.proj_heads_q.0.weight": True,
    "module.encoder.embed_engine.embeds.AVHRR.embed.weight": False,
    "module.encoder.embed_engine.embeds.AVHRR.unembed.0.weight": False,
    "module.encoder.embed_engine.embeds.AVHRR.layers.1.proj_out.weight": True,
    "module.encoder.embed_engine.embeds.AVHRR.layers.1.layers.0.weight": True,
    "module.encoder.embed_engine.embeds.ERA5_in.layer.weight": False,
    "module.encoder.q_cells": False,
    "module.encoder.q_aux": False,
    "module.forecast_engine.fe_blocks.7.proj_heads_v.weight": True,
    "module.pred_heads.ERA5.pred_heads.0.0.weight": False,
    "module.target_token_engines.ERA5.tte.0.layers.0.embed_aux.0.weight": False,
    "module.target_token_engines.ERA5.tte.0.lnorm_in_q.embed_aux.0.weight": False,
    "module.target_token_engines.ERA5.tte.0.layers.0.weight": True,
    "module.target_token_engines.ERA5.tte.0.proj_heads_k.weight": True,
}


@pytest.mark.parametrize("name,is_muon", _HIDDEN_ONLY_CASES.items())
def test_muon_hidden_only(name, is_muon):
    p = torch.zeros(4, 4)
    assert _is_muon_param(name, p, hidden_only=True) == is_muon
    assert _is_muon_param(name, p, hidden_only=False)  # default: all 2D params


def test_muon_hidden_only_optimizer_split():
    if not hasattr(torch.optim, "Muon"):
        pytest.skip("torch.optim.Muon not available")
    model = _Model()
    model.pred_heads = torch.nn.Linear(4, 4)
    cfg = _cfg(None)
    cfg.muon.hidden_only = True
    opt = Muon(model, cfg, LR_CFG, kappa=1.0)
    muon_ids = {id(p) for g in opt.optimizers[0].param_groups for p in g["params"]}
    assert id(model.pred_heads.weight) not in muon_ids
    assert id(model.lin.weight) in muon_ids
