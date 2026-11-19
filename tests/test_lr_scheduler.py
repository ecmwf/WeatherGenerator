# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Optimizer momentum after the cosine (OneCycleLR) warmup. Runs on CPU."""

import pytest
import torch
from omegaconf import OmegaConf

from weathergen.train.lr_scheduler import LearningRateScheduler

NUM_STEPS_WARMUP = 10


def _lr_cfg(**kwargs):
    cfg = {
        "lr_start": 1e-6,
        "lr_max": 5e-5,
        "lr_final_decay": 1e-6,
        "lr_final": 0.0,
        "num_steps_warmup": NUM_STEPS_WARMUP,
        "num_steps_cooldown": 10,
        "policy_warmup": "cosine",
        "policy_decay": "constant",
        "policy_cooldown": "linear",
        # sqrt with batch_size 1 and world_size 1 thus no lr scaling
        "parallel_scaling_policy": "sqrt",
    }
    cfg.update(kwargs)
    return OmegaConf.create(cfg)


def _run_past_warmup(optimizer, lr_cfg):
    scheduler = LearningRateScheduler(
        optimizer, batch_size=1, world_size=1, istep=0, lr_steps=40, lr_cfg=lr_cfg
    )
    for _ in range(NUM_STEPS_WARMUP + 5):
        optimizer.step()
        scheduler.step()


def _adamw():
    return torch.optim.AdamW([torch.nn.Parameter(torch.zeros(4))], lr=1e-6, betas=(0.9, 0.95))


def test_adamw_beta1_stays_as_configured_after_warmup():
    optimizer = _adamw()
    _run_past_warmup(optimizer, _lr_cfg())
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.95)


def test_cycle_momentum_true_keeps_old_beta1():
    # old behaviour: OneCycleLR moves beta1 to 0.85 in warmup and constant decay keeps it
    optimizer = _adamw()
    _run_past_warmup(optimizer, _lr_cfg(cycle_momentum=True))
    assert optimizer.param_groups[0]["betas"][0] == pytest.approx(0.85, abs=0.01)
