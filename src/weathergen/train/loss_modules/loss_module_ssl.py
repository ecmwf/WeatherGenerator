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

        if not has_deep_ssl:
            # Standard single-level SSL loss
            for name, (weight, loss_fn, extra_args) in self.losses.items():
                preds_for_loss = self.gather_preds_for_loss(
                    name, preds_latent[name], output_info, target2source_matching_idxs
                )
                targets_for_loss = self.gather_targets_for_loss(
                    name, targets_latent[name], target_info, target2source_matching_idxs
                )

                loss_value = loss_fn(**preds_for_loss, **targets_for_loss, **extra_args).mean()
                loss = loss + (weight * loss_value)
                losses_all[name] = loss_value.item()
        else:
            # Deep SSL: per-level loss replaces the single-level loss
            deep_preds_fstep0 = deep_preds[0]  # fstep 0
            for name, (weight, loss_fn, extra_args) in self.losses.items():
                if name not in deep_preds_fstep0 or name not in deep_targets:
                    continue
                student_levels = deep_preds_fstep0[name]
                teacher_levels = deep_targets[name]
                for level_idx, (s_level, t_level) in enumerate(
                    zip(student_levels, teacher_levels, strict=True)
                ):
                    preds_for_loss = self.gather_preds_for_loss(
                        name, s_level, output_info, target2source_matching_idxs
                    )
                    targets_for_loss = self.gather_targets_for_loss(
                        name, t_level, target_info, target2source_matching_idxs
                    )
                    level_loss = loss_fn(**preds_for_loss, **targets_for_loss, **extra_args).mean()
                    level_w = self.level_weights[level_idx]
                    loss = loss + level_w * weight * level_loss
                    losses_all[f"{name}_L{level_idx}"] = level_loss.item()

        # Context loss (V-JEPA 2.1): L1 on context (visible) tokens
        if self.context_loss_weight > 0.0 and "JEPA" in self.losses:
            warmup_factor = (
                min(1.0, istep / self.context_loss_warmup_steps)
                if self.context_loss_warmup_steps > 0
                else 1.0
            )
            ctx_weight = self.context_loss_weight * warmup_factor

            preds_for_ctx = self.gather_preds_for_loss(
                "JEPA", preds_latent["JEPA"], output_info, target2source_matching_idxs
            )
            targets_for_ctx = self.gather_targets_for_loss(
                "JEPA", targets_latent["JEPA"], target_info, target2source_matching_idxs
            )
            ctx_loss_value = context_loss(
                student_patches=preds_for_ctx["student_patches_masked"],
                student_masks=preds_for_ctx["student_masks"],
                teacher_patches=targets_for_ctx["teacher_patches_masked"],
                teacher_masks=targets_for_ctx["teacher_masks"],
            ).mean()
            loss = loss + ctx_weight * ctx_loss_value
            losses_all["context"] = ctx_loss_value.item()
            losses_all["context_warmup"] = warmup_factor

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

    if mask.sum() == 0:
        logger.warning("jepa_loss mask is all zeros, likely incorrect masking config.")

    masks_weight = (
        (1 / mask.sum(-1).clamp(min=1.0))
        .unsqueeze(-1)
        .expand_as(mask)
    )

    assert mask.shape[0] == student_patches_masked.shape[0], (
        "mask.shape[0], batch dimension, has to match batch dimension for student_patches_masked."
    )
    # expand/repeat teacher_masks to match number of student samples
    teacher_patches = teacher_patches_masked.expand((mask.shape[0], -1, -1))
    # compute loss
    loss = F.l1_loss(student_patches_masked[mask], teacher_patches[mask])
    loss = loss * masks_weight[mask]

    return loss.sum()  # / student_masks.shape[0]


def context_loss(student_patches, student_masks, teacher_patches, teacher_masks):
    """V-JEPA 2.1 context loss: L1 on context (student-visible) tokens."""
    student_masks = student_masks.squeeze(dim=1)
    teacher_masks = teacher_masks.squeeze(dim=1)

    # Context = positions visible to the student (AND visible to teacher)
    mask = torch.logical_and(student_masks, teacher_masks)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=student_patches.device, requires_grad=True)

    teacher_patches = teacher_patches.expand((mask.shape[0], -1, -1))
    loss = F.l1_loss(student_patches[mask], teacher_patches[mask], reduction="mean")
    return loss


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


class LossLeJEPA(LossModuleBase):
    """LeJEPA loss: invariance (MSE) + SIGReg regularization.

    The invariance term is the MSE between the student's MLP prediction
    and the target view's raw encoder latent. SIGReg prevents representation
    collapse by penalizing deviation from a standard Gaussian distribution
    via the empirical characteristic function.
    """

    def __init__(self, cf: DictConfig, mode_cfg: DictConfig, stage: Stage, device: str, **losses):
        LossModuleBase.__init__(self)
        self.cf = cf
        self.device = device
        self.name = "LossLeJEPA"

        jepa_conf = losses["JEPA"]
        self.temporal = jepa_conf.get("loss_extra_args", {}).get("temporal", False)
        self.sigreg_weight = jepa_conf.get("sigreg_weight", 0.02)
        self.sigreg_num_projections = jepa_conf.get("sigreg_num_projections", 256)
        self._step = 0

    def compute_loss(self, preds, targets, metadata, **kwargs) -> LossValues:
        _source2target, output_info, _target2source, _ = metadata

        # Student MLP prediction and raw encoder latent
        student_pred = preds.latent[0]["JEPA"]
        student_raw = preds.latent[0]["latent_state"].patch_tokens

        # Target raw encoder latent (from SelfTeacher, carries gradients)
        target_raw = targets.latent["JEPA"]

        # Masks
        student_masks = torch.stack(
            [info.mask for info in output_info if "JEPA" in info.global_params["loss"]],
            dim=0,
        )
        teacher_masks = torch.stack(
            [info.mask for info in targets.aux_outputs if "JEPA" in info.global_params["loss"]],
            dim=0,
        )

        if self.temporal:
            mask = teacher_masks
        else:
            mask = teacher_masks & ~student_masks

        # Invariance loss: MSE between student prediction and target latent
        target_expanded = target_raw.expand(student_pred.shape[0], -1, -1)
        if mask.sum() > 0:
            invariance = F.mse_loss(
                student_pred[mask].to(target_expanded.dtype), target_expanded[mask]
            )
        else:
            logger.warning("LeJEPA mask is all zeros — no cells for invariance loss.")
            invariance = student_pred.sum() * 0.0

        # SIGReg on both views' raw latents (over cells as sample dimension)
        sigreg_s = sigreg_loss(
            student_raw.reshape(-1, student_raw.shape[-1]), self.sigreg_num_projections
        )
        sigreg_t = sigreg_loss(
            target_raw.reshape(-1, target_raw.shape[-1]), self.sigreg_num_projections
        )
        sigreg = (sigreg_s + sigreg_t) / 2

        loss = invariance + self.sigreg_weight * sigreg

        # TODO: remove — quick gaussianity sanity check
        self._step += 1
        if self._step % 10 == 0:
            with torch.no_grad():
                z = student_raw.reshape(-1, student_raw.shape[-1]).float()
                std = z.std(0).mean().item()
                mean_abs = z.mean(0).abs().mean().item()
                kurt = ((z - z.mean(0)).pow(4).mean(0) / z.std(0).pow(4).clamp(min=1e-8) - 3).mean().item()
                print(f"[LeJEPA] step={self._step}  |mean|={mean_abs:.3f}  std={std:.3f}  excess_kurt={kurt:.3f}")

        return LossValues(
            loss=loss,
            losses_all={"invariance": invariance.item(), "sigreg": sigreg.item()},
            stddev_all={},
        )


def sigreg_loss(
    z: torch.Tensor, num_projections: int = 256, knots: int = 17
) -> torch.Tensor:
    """SIGReg regularization (Balestriero & LeCun, 2025).

    Penalizes deviation of the representation distribution from a standard
    Gaussian by comparing the empirical characteristic function (ECF) against
    the Gaussian CF along random 1D projections at multiple frequencies.

    Args:
        z: [N, D] tensor of representations (N samples, D dimensions).
        num_projections: Number of random unit-vector projection directions.
        knots: Number of frequency evaluation points in [0, 3].

    Returns:
        Scalar loss.
    """
    z = z.float()  # float32 for cos/sin precision
    n, d = z.shape

    # Trapezoidal quadrature weights with Gaussian window on [0, 3]
    t = torch.linspace(0, 3, knots, device=z.device)
    dt = 3.0 / (knots - 1)
    weights = torch.full((knots,), 2 * dt, device=z.device)
    weights[0] = dt
    weights[-1] = dt
    phi = torch.exp(-t.square() / 2.0)  # Gaussian CF at frequencies t
    weights = weights * phi

    # Random unit projection directions: [D, num_projections]
    a = torch.randn(d, num_projections, device=z.device)
    a = a / a.norm(p=2, dim=0, keepdim=True)

    # 1D projections at multiple frequencies
    proj_1d = z @ a  # [N, num_projections]
    x_t = proj_1d.unsqueeze(-1) * t  # [N, num_projections, knots]

    # ECF vs Gaussian CF
    ecf_cos = x_t.cos().mean(0)  # [num_projections, knots]
    ecf_sin = x_t.sin().mean(0)  # [num_projections, knots]
    err = (ecf_cos - phi).square() + ecf_sin.square()

    # Weighted quadrature, scaled by sample count
    statistic = (err @ weights) * n
    return statistic.mean()
