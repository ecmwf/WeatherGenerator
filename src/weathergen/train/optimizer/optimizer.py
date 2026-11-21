# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import numpy as np
import torch

from weathergen.utils.distributed import is_root

logger = logging.getLogger(__name__)


def _adamw_betas_eps(optimizer_cfg, kappa: float) -> dict:
    """
    DDP-scaled adamw betas/eps, shared by AdamW and Muon (which falls back to adamw for
    non-2D parameters).

    https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    aiming for beta1=0.9 and beta2=0.95 following the MAE paper https://arxiv.org/pdf/2111.06377
    """
    # aiming for beta1 = 0.9 at one node, ie kappa=B=4
    beta1 = max(0.5, 1.0 - kappa * (1.0 - optimizer_cfg.adamw.beta1))
    # aiming for beta2 = 0.95 at one node, ie B=4
    beta2 = max(0.9, 1.0 - kappa * (1.0 - optimizer_cfg.adamw.beta2))
    eps = optimizer_cfg.adamw.get("eps", 2e-08) / np.sqrt(kappa)
    return {"betas": (beta1, beta2), "eps": eps}


def _muon_adjust_lr_factor(shape, adjust_lr_fn: str) -> float:
    """
    Mirrors torch.optim.Muon's internal per-parameter lr-adjustment factor
    (torch/optim/_muon.py::_adjust_lr), so a representative effective lr can be logged.
    """
    a, b = shape[0], shape[1]
    if adjust_lr_fn == "match_rms_adamw":
        return 0.2 * max(a, b) ** 0.5
    return max(1.0, a / b) ** 0.5


class OptimizerBase(torch.optim.Optimizer):
    """
    Base class of the optimizers used for training. Some optimizers (see Muon) need one
    torch optimizer per class of parameters; this presents them to the trainer as a single
    torch.optim.Optimizer, so that the trainer, the learning rate scheduler and the grad scaler
    all keep working with one optimizer object and one learning rate.

    The param groups of the wrapped optimizers are shared (not copied) into self.param_groups,
    so that a learning rate scheduler stepping this object writes the lr straight through to
    them. Subclasses build the wrapped optimizers and pass them to __init__.
    """

    def __init__(self, optimizers: list[torch.optim.Optimizer], names: list[str], lr: float):
        self.optimizers = optimizers
        self.names = names

        params = [p for opt in optimizers for group in opt.param_groups for p in group["params"]]
        super().__init__(params, {"lr": lr})

        # replace the param group created by Optimizer.__init__ by the wrapped optimizers' own
        # group dicts; they are shared by reference, so an lr written here is seen by them
        self.param_groups = [group for opt in optimizers for group in opt.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None) -> None:
        assert closure is None, "closures are not supported"
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self) -> dict:
        """
        State dict of all wrapped optimizers, merged into a single, standard optimizer state
        dict: parameter indices of each wrapped optimizer are offset by the number of parameters
        of the preceding ones, so that they stay unique.
        """
        merged_state, merged_groups, offset = {}, [], 0
        for optimizer in self.optimizers:
            state_dict = optimizer.state_dict()
            merged_state.update({idx + offset: s for idx, s in state_dict["state"].items()})
            merged_groups += [
                {**group, "params": [idx + offset for idx in group["params"]]}
                for group in state_dict["param_groups"]
            ]
            offset += sum(len(group["params"]) for group in state_dict["param_groups"])
        return {"state": merged_state, "param_groups": merged_groups}

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Split a state dict merged by state_dict() back over the wrapped optimizers.
        """
        param_offset, group_offset = 0, 0
        for optimizer in self.optimizers:
            groups = state_dict["param_groups"][
                group_offset : group_offset + len(optimizer.param_groups)
            ]
            num_params = sum(len(group["params"]) for group in groups)
            optimizer.load_state_dict(
                {
                    "state": {
                        idx - param_offset: s
                        for idx, s in state_dict["state"].items()
                        if param_offset <= idx < param_offset + num_params
                    },
                    "param_groups": [
                        {**group, "params": [idx - param_offset for idx in group["params"]]}
                        for group in groups
                    ],
                }
            )
            param_offset += num_params
            group_offset += len(optimizer.param_groups)


class AdamW(OptimizerBase):
    """
    Single torch.optim.AdamW optimizer over all of the model's parameters.
    """

    def __init__(self, model: torch.nn.Module, optimizer_cfg, lr_cfg, kappa: float):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr_cfg.lr_start,
            weight_decay=optimizer_cfg.weight_decay,
            fused=True,
            **_adamw_betas_eps(optimizer_cfg, kappa),
        )

        super().__init__([optimizer], ["adamw"], lr_cfg.lr_start)


class Muon(OptimizerBase):
    """
    Muon optimizer for the model's 2D weight matrices (hidden layers), paired with a separate
    AdamW optimizer for all other (non-2D) parameters -- biases, norms, ... -- as recommended by
    https://kellerjordan.github.io/posts/muon/ (torch.optim.Muon also hard-requires exactly 2D
    tensors and raises ValueError otherwise, e.g. for a [1, 1, 2048] param).

    step() therefore runs twice: once over the 2D parameters with muon, once over the remaining
    parameters with adamw.

    Both share a single learning rate and hence a single schedule: torch.optim.Muon rescales the
    lr per parameter inside step(), and with adjust_lr_fn="match_rms_adamw" that rescaling is
    exactly what makes the RMS of muon's update match adamw's at the same nominal lr
    (`Muon is Scalable for LLM Training`, https://arxiv.org/abs/2502.16982).
    """

    def __init__(self, model: torch.nn.Module, optimizer_cfg, lr_cfg, kappa: float):
        muon_cfg = optimizer_cfg.muon
        muon_params = [p for p in model.parameters() if p.requires_grad and p.ndim == 2]
        adamw_params = [p for p in model.parameters() if p.requires_grad and p.ndim != 2]

        adjust_lr_fn = muon_cfg.get("adjust_lr_fn", None) or "original"
        # muon never writes the adjusted lr back to param_groups[...]["lr"], so this is the only
        # place where the lr that is really applied to the 2D parameters is visible
        self.muon_effective_lr_factor: float = float(
            np.median([_muon_adjust_lr_factor(p.shape, adjust_lr_fn) for p in muon_params])
        )

        if is_root():
            logger.info(
                f"Using muon optimizer: {len(muon_params)} params (ndim == 2) via muon, "
                f"{len(adamw_params)} params (ndim != 2) via adamw, "
                f"shared lr_max={lr_cfg.lr_max:.3g}, median {adjust_lr_fn} "
                f"factor={self.muon_effective_lr_factor:.3g} "
                f"(muon effective lr_max={lr_cfg.lr_max * self.muon_effective_lr_factor:.3g})"
            )

        muon_optimizer = torch.optim.Muon(
            muon_params,
            lr=lr_cfg.lr_start,
            weight_decay=optimizer_cfg.weight_decay,
            momentum=muon_cfg.get("momentum", 0.95),
            nesterov=muon_cfg.get("nesterov", True),
            ns_steps=muon_cfg.get("ns_steps", 5),
            eps=muon_cfg.get("eps", 1e-7),
            adjust_lr_fn=muon_cfg.get("adjust_lr_fn", None),
        )
        adamw_optimizer = torch.optim.AdamW(
            adamw_params,
            lr=lr_cfg.lr_start,
            weight_decay=optimizer_cfg.weight_decay,
            fused=True,
            **_adamw_betas_eps(optimizer_cfg, kappa),
        )

        super().__init__([muon_optimizer, adamw_optimizer], ["muon", "adamw"], lr_cfg.lr_start)
