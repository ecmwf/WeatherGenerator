# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Rollout options for forecast training (all under `forecast` in the mode config):

    pushforward : bool, default False
        In training, run the first steps of the rollout without gradient and without decoding.
        Validation and inference always decode every step.
    rollout_grad_steps : int, default 1
        With pushforward, number of final rollout steps that get gradient (-1 for all steps).
    rollout_loss_steps : int, default rollout_grad_steps
        With pushforward, number of final rollout steps that are decoded and scored; at most
        rollout_grad_steps. Targets are only loaded for these steps.
    checkpoint_steps : bool, default False
        Activation checkpointing of whole forecast steps (on top of the per-block checkpointing
        in the forecast engine), so memory grows with the latent state per step only.
    ensemble_size : int, default 1
        Number of ensemble members rolled out from each latent state, e.g. for CRPS losses. For
        validation and inference, the forecast engine's dropout is kept active (MC dropout) when
        ensemble_size > 1.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import torch

ROLLOUT_DEFAULTS = {
    "pushforward": False,
    "rollout_grad_steps": 1,
    "rollout_loss_steps": None,
    "checkpoint_steps": False,
    "ensemble_size": 1,
}


@dataclass(frozen=True)
class RolloutPlan:
    """Which output steps run without gradient and which are decoded and scored."""

    no_grad_steps: frozenset[int]
    loss_steps: frozenset[int]


def get_rollout_plan(forecast_cfg, output_idxs: list[int], training: bool) -> RolloutPlan:
    """
    Split the output steps of one batch into steps without gradient and steps with a loss.

    Without pushforward (or outside training) every step has gradient and is scored, which is
    the standard multi-step forecast training.
    """
    steps = sorted(output_idxs)
    pushforward = bool(forecast_cfg.get("pushforward", False)) if forecast_cfg else False
    if not (training and pushforward) or len(steps) == 0:
        return RolloutPlan(no_grad_steps=frozenset(), loss_steps=frozenset(steps))

    num_steps = len(steps)
    num_grad = forecast_cfg.get("rollout_grad_steps", ROLLOUT_DEFAULTS["rollout_grad_steps"])
    num_grad = num_steps if num_grad == -1 else num_grad
    assert isinstance(num_grad, int) and num_grad > 0, (
        f"rollout_grad_steps must be -1 or a positive integer, got {num_grad}."
    )
    num_grad = min(num_grad, num_steps)

    num_loss = forecast_cfg.get("rollout_loss_steps", None)
    num_loss = num_grad if num_loss is None else num_loss
    num_loss = num_steps if num_loss == -1 else num_loss
    assert isinstance(num_loss, int) and num_loss > 0, (
        f"rollout_loss_steps must be -1 or a positive integer, got {num_loss}."
    )
    num_loss = min(num_loss, num_grad)

    return RolloutPlan(
        no_grad_steps=frozenset(steps[: num_steps - num_grad]),
        loss_steps=frozenset(steps[num_steps - num_loss :]),
    )


def get_ensemble_size(forecast_cfg) -> int:
    ensemble_size = forecast_cfg.get("ensemble_size", 1) if forecast_cfg else 1
    assert isinstance(ensemble_size, int) and ensemble_size >= 1, (
        f"ensemble_size must be a positive integer, got {ensemble_size}."
    )
    return ensemble_size


@contextmanager
def autocast_cache_disabled():
    """
    Disable the autocast cache of casted weights, for forward passes without gradient inside an
    autocast region with gradient (pushforward).

    Autocast caches the low precision cast of each fp32 weight for the whole autocast region,
    also when the cast happens under torch.no_grad. Later forward passes with gradient then reuse
    the cast without autograd history, so the weights get no gradient (or, with activation
    checkpointing, the recomputation differs from the forward pass: CheckpointError).
    """
    enabled = torch.is_autocast_cache_enabled()
    torch.set_autocast_cache_enabled(False)
    try:
        yield
    finally:
        torch.set_autocast_cache_enabled(enabled)
