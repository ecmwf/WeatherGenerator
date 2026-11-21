# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch

from weathergen.train.optimizer.optimizer import AdamW, Muon, OptimizerBase

_OPTIMIZER_CLASSES = {"adamw": AdamW, "muon": Muon}


def build_optimizer(model: torch.nn.Module, optimizer_cfg, lr_cfg, kappa: float) -> OptimizerBase:
    """
    Builds the optimizer for a model, according to optimizer_cfg.name ("adamw" or "muon").
    Returns a torch.optim.Optimizer (an OptimizerBase subclass), which drives one optimizer per
    class of parameters internally but behaves as a single optimizer with a single learning rate.
    """
    optimizer_name = optimizer_cfg.get("name", "adamw").lower()
    assert optimizer_name in _OPTIMIZER_CLASSES, (
        f"Unsupported optimizer '{optimizer_name}', expected one of {list(_OPTIMIZER_CLASSES)}"
    )
    return _OPTIMIZER_CLASSES[optimizer_name](model, optimizer_cfg, lr_cfg, kappa)
