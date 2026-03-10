# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import torch

from weathergen.common.config import Config
from weathergen.utils.distributed import is_root

logger = logging.getLogger(__name__)


def build_param_groups(
    model: torch.nn.Module,
    stream_optimizer_cfgs: dict[str, Config],
    optimizer_cfg: Config,
) -> list[dict]:
    """Build per-stream + shared optimizer parameter groups.

    Each stream's embedding parameters get their own group with optional lr_scale and weight_decay.
    All remaining parameters go into a "shared" group.

    :param model: The model (may be DDP-wrapped).
    :param stream_optimizer_cfgs: {stream_name: optimizer_cfg} from each stream's config.
    :param optimizer_cfg: Global optimizer config (provides default weight_decay).
    :return: List of param group dicts for torch.optim.AdamW.
    """
    # unwrap DDP if necessary
    raw_model = model.module if hasattr(model, "module") else model
    embeds = raw_model.encoder.embed_engine.embeds

    default_wd = optimizer_cfg.weight_decay
    stream_param_ids: set[int] = set()
    groups: list[dict] = []

    for stream_name, stream_opt_cfg in stream_optimizer_cfgs.items():
        if stream_name not in embeds:
            continue
        params = [p for p in embeds[stream_name].parameters() if p.requires_grad]
        if not params:
            continue
        for p in params:
            stream_param_ids.add(id(p))

        lr_scale = stream_opt_cfg.get("lr_scale", 1.0)
        wd = stream_opt_cfg.get("weight_decay", default_wd)
        groups.append(
            {
                "params": params,
                "lr_scale": lr_scale,
                "weight_decay": wd,
                "name": f"embed_{stream_name}",
            }
        )

    # shared group: everything not assigned to a stream
    shared_params = [
        p for p in model.parameters() if id(p) not in stream_param_ids and p.requires_grad
    ]
    if shared_params:
        groups.append(
            {
                "params": shared_params,
                "lr_scale": 1.0,
                "weight_decay": default_wd,
                "name": "shared",
            }
        )

    # sanity check: no parameter missed or double-counted
    n_grouped = sum(len(g["params"]) for g in groups)
    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    assert n_grouped == n_total, (
        f"Parameter group mismatch: {n_grouped} grouped vs {n_total} total trainable params"
    )

    if is_root():
        for g in groups:
            logger.info(
                f"Param group '{g['name']}': {len(g['params'])} params, "
                f"lr_scale={g['lr_scale']}, weight_decay={g['weight_decay']}"
            )

    return groups
