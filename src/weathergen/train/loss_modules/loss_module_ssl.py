# ruff: noqa: T201

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
import torch.nn.functional as F
from omegaconf import DictConfig

import weathergen.train.loss_modules.loss_functions as loss_fns
from weathergen.train.loss_modules.loss_module_base import LossModuleBase, LossValues
from weathergen.utils.train_logger import Stage

logger = logging.getLogger(__name__)


class LossLatentSSLStudentTeacher(LossModuleBase):
    """
    Manages and computes the overall loss for a WeatherGenerator model pretraining using
    DINO/iBOT/JEPA/BYOL style losses.

    This class handles the initialization and application of various loss functions,
    It provides both the main loss for backpropagation and detailed loss metrics for logging.
    """

    valid_loss_names = set(["DINO", "iBOT", "JEPA"])

    def __init__(self, cf: DictConfig, mode_cfg: DictConfig, stage: Stage, device: str, **losses):
        LossModuleBase.__init__(self)
        self.cf = cf
        self.stage = stage
        self.device = device
        self.num_class_tokens = cf.num_class_tokens
        self.name = "LossLatentSSLStudentTeacher"

        # Dynamically load loss functions based on configuration and stage
        self.losses = {
            name: (local_conf["weight"], get_loss_function_ssl(name), local_conf["loss_extra_args"])
            for name, local_conf in losses.items()
        }

        # Deep SSL level weights
        deep_ssl_cfg = mode_cfg.get("deep_ssl", None)
        if deep_ssl_cfg and deep_ssl_cfg.get("tap_after"):
            self.level_weights = list(deep_ssl_cfg.get("level_weights", []))
            num_levels = len(deep_ssl_cfg.tap_after) + 1
            if not self.level_weights:
                self.level_weights = [1.0 / num_levels] * num_levels
            assert len(self.level_weights) == num_levels, (
                f"level_weights length ({len(self.level_weights)}) must match "
                f"number of levels ({num_levels})"
            )
        else:
            self.level_weights = None

        # Context loss (V-JEPA 2.1): L1 on context (visible) tokens with linear warmup
        context_cfg = mode_cfg.get("context_loss", None)
        if context_cfg and context_cfg.get("enabled", True):
            self.context_loss_weight = context_cfg.get("weight", 1.0)
            self.context_loss_warmup_steps = context_cfg.get("warmup_steps", 0)
        else:
            self.context_loss_weight = 0.0
            self.context_loss_warmup_steps = 0

    def compute_loss(self, preds, targets, metadata, istep=0, **kwargs) -> LossValues:
        # gradient loss
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # initialize dictionaries for detailed loss tracking and standard deviation statistics
        # create tensor for each stream
        losses_all: dict[str, float] = {loss_name: 0.0 for loss_name in self.losses}

        source2target_matching_idxs, output_info, target2source_matching_idxs, _ = metadata

        deep_preds = preds.latent_deep
        deep_targets = targets.latent_deep
        preds_latent = preds.latent[0]  # [0] because we always want the first fstep
        target_info = targets.aux_outputs
        targets_latent = targets.latent

        has_deep_ssl = (
            self.level_weights is not None and deep_preds is not None and deep_targets is not None
        )

        # Normalize both paths: name -> [(level_weight, student_data, teacher_data)]
        if has_deep_ssl:
            deep_preds_fstep0 = deep_preds[0]  # fstep 0
            levels_by_name = {
                name: list(
                    zip(
                        self.level_weights, deep_preds_fstep0[name], deep_targets[name], strict=True
                    )
                )
                for name in self.losses
                if name in deep_preds_fstep0 and name in deep_targets
            }
        else:
            levels_by_name = {
                name: [(1.0, preds_latent[name], targets_latent[name])] for name in self.losses
            }

        # Context loss warmup (V-JEPA 2.1)
        has_context_loss = self.context_loss_weight > 0.0 and "JEPA" in levels_by_name
        if has_context_loss:
            warmup_factor = (
                min(1.0, istep / self.context_loss_warmup_steps)
                if self.context_loss_warmup_steps > 0
                else 1.0
            )
            ctx_weight = self.context_loss_weight * warmup_factor
            losses_all["context_warmup"] = warmup_factor

        # Unified loss loop over all levels
        for name, (weight, loss_fn, extra_args) in self.losses.items():
            if name not in levels_by_name:
                continue
            for level_idx, (level_w, s_data, t_data) in enumerate(levels_by_name[name]):
                preds_for_loss = self.gather_preds_for_loss(
                    name, s_data, output_info, target2source_matching_idxs
                )
                targets_for_loss = self.gather_targets_for_loss(
                    name, t_data, target_info, target2source_matching_idxs
                )

                loss_value = loss_fn(**preds_for_loss, **targets_for_loss, **extra_args).mean()
                suffix = f"_L{level_idx}" if has_deep_ssl else ""
                loss = loss + level_w * weight * loss_value
                losses_all[f"{name}{suffix}"] = loss_value.item()

                # Context loss reuses already-gathered tensors
                if name == "JEPA" and has_context_loss:
                    ctx_loss_value = context_loss(
                        student_patches=preds_for_loss["student_patches_masked"],
                        student_masks=preds_for_loss["student_masks"],
                        teacher_patches=targets_for_loss["teacher_patches_masked"],
                        teacher_masks=targets_for_loss["teacher_masks"],
                    )
                    loss = loss + level_w * ctx_weight * ctx_loss_value
                    losses_all[f"context{suffix}"] = ctx_loss_value.item()

        return LossValues(loss=loss, losses_all=losses_all, stddev_all={})

    def gather_preds_for_loss(self, name, preds, metadata, target2source_matching_idxs):
        if name == "JEPA":
            """
            Important this assumes that there is 1 masked version for each global view
            ie. student_patches_masked.shape[0] == teacher_patches_masked.shape[0]
            """
            return {
                "student_patches_masked": torch.stack(
                    [
                        p
                        for p, info in zip(preds, metadata, strict=False)
                        if "JEPA" in info.global_params["loss"]
                    ],
                    dim=0,
                ),
                "student_masks": torch.stack(
                    [info.mask for info in metadata if "JEPA" in info.global_params["loss"]],
                    dim=0,
                ).unsqueeze(1),
            }
        elif name == "iBOT":
            """
            Important this assumes that there is 1 masked version for each global view
            ie. student_patches_masked.shape[0] == teacher_patches_masked.shape[0]

            Note the class token of iBOT is still missing
            """
            return {
                "student_patches_masked": torch.stack(
                    [
                        p[self.num_class_tokens :]
                        for p, info in zip(preds, metadata, strict=False)
                        if "iBOT" in info.global_params["loss"]
                    ],
                    dim=0,
                ),
                "student_masks": torch.stack(
                    [info.mask for info in metadata if "iBOT" in info.global_params["loss"]],
                    dim=0,
                ).unsqueeze(1),
                "student_class_masked": torch.stack(
                    [
                        p[: self.num_class_tokens]
                        for p, info in zip(preds, metadata, strict=False)
                        if "iBOT" in info.global_params["loss"]
                    ],
                    dim=0,
                ),
            }
        elif name == "DINO":
            local2global_dino_student = []
            for student_indices in target2source_matching_idxs:
                local_preds = [
                    preds[sidx]
                    for sidx in student_indices
                    if "DINO" in metadata[sidx].global_params["loss"]
                    and metadata[sidx].global_params["relationship"] != "identity"
                ]
                local2global_dino_student.append(local_preds)
            local2global_dino_student = [
                torch.stack(latents, dim=0)
                for latents in zip(*local2global_dino_student, strict=False)
            ]
            return {
                "local2global_dino_student": local2global_dino_student,
                "global2global_dino_student": torch.stack(
                    [
                        p
                        for p, info in zip(preds, metadata, strict=False)
                        if "DINO" in info.global_params["loss"]
                        and info.global_params["relationship"] == "identity"
                    ],
                    dim=0,
                ),
            }
        else:
            raise NotImplementedError(
                f"{name} is not an implemented loss for the LossLatentSSLStudentTeacher"
            )

    def gather_targets_for_loss(self, name, targets, metadata, target2source_matching_idxs):
        if name == "JEPA":
            """
            Important this assumes that there is 1 masked version for each global view
            ie. student_patches_masked.shape[0] == teacher_patches_masked.shape[0]
            """
            return {
                "teacher_patches_masked": torch.stack(
                    [
                        p
                        for p, info in zip(targets, metadata, strict=True)
                        if "JEPA" in info.global_params["loss"]
                    ],
                    dim=0,
                ),
                "teacher_masks": torch.stack(
                    [info.mask for info in metadata if "JEPA" in info.global_params["loss"]],
                    dim=0,
                ).unsqueeze(1),
            }
        elif name == "iBOT":
            """
            Important this assumes that there is 1 masked version for each global view
            ie. student_patches_masked.shape[0] == teacher_patches_masked.shape[0]

            Note the class token of iBOT is still missing
            """
            return {
                "teacher_patches_masked": torch.stack(
                    [
                        p[self.num_class_tokens :]
                        for p, info in zip(targets, metadata, strict=False)
                    ],
                    dim=0,
                ),
                "teacher_masks": torch.stack(
                    [info.mask for info in metadata if "iBOT" in info.global_params["loss"]],
                    dim=0,
                ).unsqueeze(1),
                "teacher_class_masked": torch.stack(
                    [
                        p[: self.num_class_tokens]
                        for p, info in zip(targets, metadata, strict=False)
                    ],
                    dim=0,
                ),
            }
        elif name == "DINO":
            return {
                "local2global_dino_teacher": torch.stack(
                    [p for p, info in zip(targets, metadata, strict=False)],
                    dim=0,
                ),
                "global2global_dino_teacher": torch.stack(
                    list(reversed([p for p, info in zip(targets, metadata, strict=False)])),
                    dim=0,
                ),
            }
        else:
            raise NotImplementedError(
                f"{name} is not an implemented loss for the LossLatentSSLStudentTeacher"
            )


def _masked_l1_loss_per_sample(
    student_patches, teacher_patches, mask, empty_warning: str | None = None
):
    """Average L1 on a masked token set, normalized on the set's own support.

    Reduction order:
    1. Mean over feature dimension for each selected token.
    2. Mean over selected tokens per sample.
    3. Mean over samples with at least one selected token.
    """

    assert mask.shape[0] == student_patches.shape[0], (
        "mask.shape[0], batch dimension, has to match batch dimension for student_patches."
    )
    assert teacher_patches.shape[0] in (1, mask.shape[0]), (
        "teacher_patches batch dimension must be 1 or match mask batch dimension."
    )

    if mask.sum() == 0:
        if empty_warning is not None:
            logger.warning(empty_warning)
        return student_patches.sum() * 0.0

    teacher_patches = teacher_patches.expand((mask.shape[0], -1, -1))
    per_token_loss = F.l1_loss(student_patches, teacher_patches, reduction="none").mean(dim=-1)

    mask_f = mask.to(per_token_loss.dtype)
    token_counts = mask_f.sum(dim=-1)
    valid_samples = token_counts > 0
    # torch.where (not multiply): NaN at mask-excluded positions must not poison the sum.
    per_sample_loss = torch.where(mask.bool(), per_token_loss, 0.0).sum(dim=-1) / (
        token_counts.clamp(min=1.0)
    )

    return per_sample_loss[valid_samples].mean()


def jepa_loss(
    student_patches_masked, student_masks, teacher_patches_masked, teacher_masks, temporal=False
):
    # TODO remove as we deal with batch dimension
    assert teacher_masks.shape[0] == 1 or teacher_masks.shape[0] == student_masks.shape[0]
    student_masks = student_masks.squeeze(dim=1)
    teacher_masks = teacher_masks.squeeze(dim=1)

    if temporal:
        # Temporal JEPA: predict teacher's representation at ALL teacher-visible cells
        # (no spatial masking exclusion since the prediction task is across time, not space)
        mask = teacher_masks
    else:
        # Standard JEPA: predict only at cells the teacher sees but the student doesn't
        mask = torch.logical_and(teacher_masks, torch.logical_not(student_masks))

    return _masked_l1_loss_per_sample(
        student_patches_masked,
        teacher_patches_masked,
        mask,
        empty_warning="jepa_loss mask is all zeros, likely incorrect masking config.",
    )


def context_loss(student_patches, student_masks, teacher_patches, teacher_masks):
    """V-JEPA 2.1 context loss: L1 on context tokens, normalized on context support."""
    student_masks = student_masks.squeeze(dim=1)
    teacher_masks = teacher_masks.squeeze(dim=1)

    # Context = positions visible to the student (AND visible to teacher)
    mask = torch.logical_and(student_masks, teacher_masks)
    return _masked_l1_loss_per_sample(student_patches, teacher_patches, mask)


def ibot_loss(
    student_patches_masked,
    student_masks,
    teacher_patches_masked,
    teacher_masks,
    student_class_masked,
    teacher_class_masked,
    student_temp,
):
    student_masks = student_masks.squeeze(dim=1)
    teacher_masks = teacher_masks.squeeze(dim=1)
    loss = loss_fns.masked_student_teacher_patch_softmax(
        student_patches_masked, teacher_patches_masked, student_masks, teacher_masks, student_temp
    )
    loss = loss + loss_fns.student_teacher_softmax(
        student_class_masked, teacher_class_masked, student_temp
    )
    return loss / 2


def dino_loss(
    local2global_dino_student,
    local2global_dino_teacher,
    global2global_dino_student,
    global2global_dino_teacher,
    student_temp,
):
    loss = loss_fns.student_teacher_global_softmax(
        local2global_dino_student, local2global_dino_teacher, student_temp
    ) + loss_fns.student_teacher_softmax(
        global2global_dino_student, global2global_dino_teacher, student_temp
    )
    return loss / 2


def get_loss_function_ssl(name):
    if name == "iBOT":
        return ibot_loss
    elif name == "DINO":
        return dino_loss
    elif name == "JEPA":
        return jepa_loss
    else:
        raise NotImplementedError(
            f"{name} is not an implemented loss for the LossLatentSSLStudentTeacher"
        )
