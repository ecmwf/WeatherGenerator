# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import re

import numpy as np
import torch
from omegaconf import OmegaConf

from weathergen.model.norms import RMSNorm
from weathergen.utils.distributed import is_root

logger = logging.getLogger(__name__)

# modules whose (affine) parameters count as "norms" for optimizer.no_weight_decay
_NORM_MODULES = (
    torch.nn.LayerNorm,
    torch.nn.RMSNorm,
    torch.nn.GroupNorm,
    RMSNorm,
)
_NO_WEIGHT_DECAY_KINDS = ("norms", "biases")

# with optimizer.muon.hidden_only, parameters whose name matches one of these go to adamw even if
# 2D: input/output embeddings and coordinate/aux embeddings, prediction heads, learned queries
_MUON_NON_HIDDEN_PATTERNS = (
    r"(^|\.)embed_target_coords\.",  # target coordinate embedding
    r"(^|\.)embeds\.[^.]+\.(embed|unembed|layer)\.",  # stream input embed / unembed
    r"(^|\.)embed_aux\.",  # coordinate conditioning in target token engines
    r"(^|\.)(pred_heads|latent_heads)\.",
    r"(^|\.)q_(cells|aux)$",
)


def _adamw_betas_eps(optimizer_cfg, kappa: float) -> dict:
    """
    DDP-scaled adamw betas/eps, shared by AdamW and Muon (muon falls back to adamw for
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


def _no_weight_decay_param_ids(model: torch.nn.Module, optimizer_cfg) -> set[int]:
    """
    Ids of the parameters excluded from weight decay by optimizer.no_weight_decay, a list of
    - "norms": all parameters of normalization layers (gains and biases, incl. QK-norms);
    - "biases": all parameters named `bias`.
    """
    kinds = list(optimizer_cfg.get("no_weight_decay", None) or [])
    unknown = set(kinds) - set(_NO_WEIGHT_DECAY_KINDS)
    assert not unknown, (
        f"Unsupported optimizer.no_weight_decay {sorted(unknown)}, "
        f"expected a subset of {list(_NO_WEIGHT_DECAY_KINDS)}"
    )
    ids = set()
    if "norms" in kinds:
        for module in model.modules():
            if isinstance(module, _NORM_MODULES):
                ids.update(id(p) for p in module.parameters(recurse=False))
    if "biases" in kinds:
        ids.update(id(p) for n, p in model.named_parameters() if n.split(".")[-1] == "bias")
    return ids


def _weight_decay_param_groups(model: torch.nn.Module, params, optimizer_cfg) -> list[dict]:
    """
    Splits params into a decayed group (optimizer.weight_decay) and, if
    optimizer.no_weight_decay selects any of them, a group without weight decay.
    """
    no_decay_ids = _no_weight_decay_param_ids(model, optimizer_cfg)
    params = list(params)
    decay = [p for p in params if id(p) not in no_decay_ids]
    no_decay = [p for p in params if id(p) in no_decay_ids]
    groups = [{"params": decay, "weight_decay": optimizer_cfg.weight_decay}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
        if is_root():
            logger.info(
                f"No weight decay ({list(optimizer_cfg.no_weight_decay)}) for {len(no_decay)} "
                f"params ({sum(p.numel() for p in no_decay)} elements), weight decay "
                f"{optimizer_cfg.weight_decay} for {len(decay)} params."
            )
    return groups


def _scale_lr_cfg(lr_cfg, lr_max_override: float | None):
    """
    Rescale a learning_rate_scheduling config to a different peak lr, keeping the
    warmup/decay/cooldown timing and relative shape identical.
    """
    if lr_max_override is None:
        return lr_cfg
    scale = lr_max_override / lr_cfg.lr_max
    return OmegaConf.merge(
        lr_cfg,
        {
            "lr_start": lr_cfg.lr_start * scale,
            "lr_max": lr_max_override,
            "lr_final_decay": lr_cfg.lr_final_decay * scale,
            "lr_final": lr_cfg.lr_final * scale,
        },
    )


def _muon_adjust_lr_factor(shape, adjust_lr_fn: str) -> float:
    """
    Mirrors torch.optim.Muon's internal per-parameter lr-adjustment factor
    (torch/optim/_muon.py::_adjust_lr), so a representative effective lr can be logged.
    """
    a, b = shape[0], shape[1]
    if adjust_lr_fn == "match_rms_adamw":
        return 0.2 * max(a, b) ** 0.5
    return max(1.0, a / b) ** 0.5


class AdamW:
    """
    Single torch.optim.AdamW optimizer over all of the model's parameters.
    """

    def __init__(self, model: torch.nn.Module, optimizer_cfg, shared_lr_cfg, kappa: float):
        optimizer = torch.optim.AdamW(
            _weight_decay_param_groups(model, model.parameters(), optimizer_cfg),
            lr=shared_lr_cfg.lr_start,
            fused=True,
            **_adamw_betas_eps(optimizer_cfg, kappa),
        )

        self.optimizers: list[torch.optim.Optimizer] = [optimizer]
        self.optimizer_names: list[str] = ["adamw"]
        self.lr_cfgs: list = [shared_lr_cfg]
        self.muon_effective_lr_factor: float | None = None


def _is_muon_param(name: str, param: torch.Tensor, hidden_only: bool) -> bool:
    """
    Muon takes 2D parameters; with hidden_only, only the transformer hidden matrices, i.e. not
    those matching _MUON_NON_HIDDEN_PATTERNS (embeddings, heads, learned queries).
    """
    if param.ndim != 2:
        return False
    return not (hidden_only and any(re.search(pat, name) for pat in _MUON_NON_HIDDEN_PATTERNS))


class Muon:
    """
    Muon optimizer for the model's 2D weight matrices (hidden layers), paired with a separate
    AdamW optimizer for all other (non-2D) parameters -- biases, norms, ... -- as recommended by
    https://kellerjordan.github.io/posts/muon/ (torch.optim.Muon also hard-requires exactly 2D
    tensors and raises ValueError otherwise, e.g. for a [1, 1, 2048] param).
    With optimizer.muon.hidden_only, 2D embeddings and heads go to AdamW as well.
    """

    def __init__(self, model: torch.nn.Module, optimizer_cfg, shared_lr_cfg, kappa: float):
        muon_cfg = optimizer_cfg.muon
        hidden_only = bool(muon_cfg.get("hidden_only", False))
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        muon_params = [p for n, p in named if _is_muon_param(n, p, hidden_only)]
        adamw_params = [p for n, p in named if not _is_muon_param(n, p, hidden_only)]

        muon_lr_cfg = _scale_lr_cfg(shared_lr_cfg, muon_cfg.get("lr_max", None))
        adjust_lr_fn = muon_cfg.get("adjust_lr_fn", None) or "original"
        self.muon_effective_lr_factor: float = float(
            np.median([_muon_adjust_lr_factor(p.shape, adjust_lr_fn) for p in muon_params])
        )

        if is_root():
            logger.info(
                f"Using muon optimizer (hidden_only={hidden_only}): "
                f"{len(muon_params)} params ({sum(p.numel() for p in muon_params)} elements) "
                f"via muon, {len(adamw_params)} params "
                f"({sum(p.numel() for p in adamw_params)} elements) via adamw, "
                f"muon lr_max={muon_lr_cfg.lr_max:.3g} "
                f"(adamw lr_max={shared_lr_cfg.lr_max:.3g}), "
                f"median {adjust_lr_fn} factor={self.muon_effective_lr_factor:.3g}"
            )

        muon_optimizer = torch.optim.Muon(
            muon_params,
            lr=muon_lr_cfg.lr_start,
            weight_decay=optimizer_cfg.weight_decay,
            momentum=muon_cfg.get("momentum", 0.95),
            nesterov=muon_cfg.get("nesterov", True),
            ns_steps=muon_cfg.get("ns_steps", 5),
            eps=muon_cfg.get("eps", 1e-7),
            adjust_lr_fn=muon_cfg.get("adjust_lr_fn", None),
        )
        adamw_optimizer = torch.optim.AdamW(
            _weight_decay_param_groups(model, adamw_params, optimizer_cfg),
            lr=shared_lr_cfg.lr_start,
            fused=True,
            **_adamw_betas_eps(optimizer_cfg, kappa),
        )

        self.optimizers: list[torch.optim.Optimizer] = [muon_optimizer, adamw_optimizer]
        self.optimizer_names: list[str] = ["muon", "adamw"]
        self.lr_cfgs: list = [muon_lr_cfg, shared_lr_cfg]
