# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from weathergen.common.config import get_path_model
from weathergen.model.engines import (
    LatentPredictionHeadIdentity,
    LatentPredictionHeadMLP,
    LatentPredictionHeadTransformer,
)

logger = logging.getLogger(__name__)


def _create_teacher_heads(
    name: str, head_type: str, dim_embed: int, loss_conf, cf=None
) -> nn.Module:
    """Create a latent prediction head for a given SSL loss type.

    Mirrors Model._create_latent_pred_head() logic with per-loss-type token settings:
        iBOT: use_class_token=True, use_patch_token=True
        DINO: use_class_token=True, use_patch_token=False
    """
    if name == "iBOT":
        use_class_token, use_patch_token = True, True
    elif name == "DINO":
        use_class_token, use_patch_token = True, False
    else:
        raise ValueError(f"_create_teacher_heads does not support loss type {name!r}")

    if head_type == "mlp":
        return LatentPredictionHeadMLP(
            f"{name}-head", dim_embed, loss_conf, use_class_token, use_patch_token
        )
    elif head_type == "transformer":
        if cf is None:
            raise ValueError("LatentPredictionHeadTransformer requires a global config (cf)")
        return LatentPredictionHeadTransformer(
            cf, f"{name}-head", dim_embed, loss_conf, use_class_token, use_patch_token
        )
    elif head_type == "identity":
        return LatentPredictionHeadIdentity()
    else:
        raise ValueError(f"Unknown latent prediction head type {head_type!r}")


def prepare_encoder_teacher(model: nn.Module, training_cfg, override_cfg) -> None:
    """Strip a model to encoder-only and create fresh SSL latent heads.

    Modifies model in-place:
    1. Removes forecast_engine, decoders, pred_heads, embed_target_coords
    2. Ensures latent_pre_norm exists
    3. Creates fresh latent_heads based on the student's SSL loss config
    """
    # Strip non-encoder components
    teacher_dim_embed = override_cfg.ae_global_dim_embed
    model.forecast_engine = None
    model.embed_target_coords = nn.ModuleDict()
    model.target_token_engines = nn.ModuleDict()
    model.pred_heads = nn.ModuleDict()

    # Ensure latent_pre_norm exists (teacher may not have had SSL training)
    if model.latent_pre_norm is None:
        model.latent_pre_norm = nn.LayerNorm(teacher_dim_embed)

    # Create fresh latent heads from student's SSL config
    model.latent_heads = nn.ModuleDict()
    ssl_losses = [
        v for v in training_cfg.losses.values() if v.type == "LossLatentSSLStudentTeacher"
    ]
    for ssl_loss in ssl_losses:
        for name, conf in ssl_loss.loss_fcts.items():
            if name == "JEPA":
                model.latent_heads[name] = LatentPredictionHeadIdentity()
            elif name in ("iBOT", "DINO"):
                head_type = conf.get("head", "mlp").lower()
                model.latent_heads[name] = _create_teacher_heads(
                    name, head_type, teacher_dim_embed, conf
                )
            else:
                logger.warning(f"Unknown SSL loss type {name!r} in teacher setup, skipping.")


def load_encoder_from_checkpoint(
    model: nn.Module,
    cf,
    teacher_run_id: str,
    teacher_mini_epoch: int | None,
    device: torch.device | str,
) -> None:
    """Load only encoder weights from a checkpoint into a model.

    Filters checkpoint to encoder.* and latent_pre_norm* keys only, then loads with
    strict=False. Moves the model to the given device afterwards.
    """
    path_run = Path(cf.get("model_path", get_path_model(run_id=teacher_run_id))) / teacher_run_id
    mini_epoch_id = (
        f"chkpt{teacher_mini_epoch:05d}"
        if teacher_mini_epoch is not None and teacher_mini_epoch != -1
        else "latest"
    )
    filename = f"{teacher_run_id}_{mini_epoch_id}.chkpt"

    params = torch.load(path_run / filename, map_location="cpu", mmap=True, weights_only=True)

    # Filter to encoder + latent_pre_norm only
    encoder_params = {
        k: v for k, v in params.items() if k.startswith(("encoder.", "latent_pre_norm"))
    }

    mkeys, ukeys = model.load_state_dict(encoder_params, strict=False)
    model.to(device)

    logging.info(f"Teacher: Loaded encoder weights from checkpoint {filename}")
    if mkeys is not None:
        logger.info(f"Number of missing keys: {len(mkeys)}")
        logger.debug(f"Missing keys: {mkeys}")
    if ukeys is not None:
        logger.info(f"Number of unused keys: {len(ukeys)}")
        logger.debug(f"Unused keys: {ukeys}")
    if mkeys is None and ukeys is None:
        logger.info("All keys in checkpoint matched successfully.")
