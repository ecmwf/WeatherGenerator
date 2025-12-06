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

_logger = logging.getLogger(__name__)


class LossLatentSSLStudentTeacher(LossModuleBase):
    """
    Manages and computes the overall loss for a WeatherGenerator model pretraining using
    DINO/iBOT/JEPA/BYOL style losses.

    This class handles the initialization and application of various loss functions,
    It provides both the main loss for backpropagation and detailed loss metrics for logging.
    """

    valid_loss_names = set(["DINO", "iBOT", "JEPA"])

    def __init__(self, cf: DictConfig, stage: Stage, device: str, **losses):
        LossModuleBase.__init__(self)
        self.cf = cf
        self.stage = stage
        self.device = device
        self.name = "LossLatentSSLStudentTeacher"

        # Dynamically load loss functions based on configuration and stage
        self.losses = {
            name: (local_conf["weight"], get_loss_function_ssl(name), local_conf["loss_extra_args"])
            for name, local_conf in losses.items()
            # if name in self.valid_loss_names
        }

    def compute_loss(self, preds: dict, targets: dict, metadata) -> LossValues:
        # gradient loss
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # initialize dictionaries for detailed loss tracking and standard deviation statistics
        # create tensor for each stream
        # losses_all: dict[str, Tensor] = {loss: 0.0 for loss in self.losses}

        source2target_matching_idxs, output_info, target2source_matching_idxs, target_info = (
            metadata
        )
        for name, (weight, loss_fn, extra_args) in self.losses.items():
            preds_for_loss = gather_preds_for_loss(
                name, preds, output_info, source2target_matching_idxs
            )
            targets_for_loss = gather_targets_for_loss(
                name, targets, target_info, target2source_matching_idxs
            )
            loss_value = loss_fn(**preds_for_loss, **targets_for_loss, **extra_args).mean()
            loss = loss + (weight * loss_value)
            # losses_all[name] = loss_value.item()

        return LossValues(loss=loss, losses_all={}, stddev_all={})


def jepa_loss(student_patches_masked, student_masks, teacher_patches_masked):
    masks_weight = (
        (1 / student_masks.sum(-1).clamp(min=1.0))
        .unsqueeze(-1)
        .expand_as(student_masks)  # [student_masks_flat]
    )
    loss = F.l1_loss(student_patches_masked, teacher_patches_masked)
    loss = loss * student_masks * masks_weight
    return loss.sum() / student_masks.shape[0]


def ibot_loss(
    student_patches_masked,
    student_masks,
    teacher_patches_masked,
    student_class_masked,
    teacher_class_masked,
    student_temp,
):
    import pdb

    pdb.set_trace()
    loss = loss_fns.masked_student_teacher_patch_softmax(
        student_patches_masked, teacher_patches_masked, student_masks, student_temp
    ) + loss_fns.student_teacher_softmax(student_class_masked, teacher_class_masked, student_temp)
    return loss / 2


def dino_loss(
    local2global_dino_student,
    local2global_dino_teacher,
    global2global_dino_student,
    global2global_dino_teacher,
    student_temp,
):
    import pdb

    pdb.set_trace()
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


def gather_preds_for_loss(name, preds, metadata, source2target_matching_idxs):
    if name == "JEPA":
        """
        Important this assumes that there is 1 masked version for each global view
        ie. student_patches_masked.shape[0] == teacher_patches_masked.shape[0]
        """
        return {
            "student_patches_masked": torch.stack(
                [
                    p.latent[name]
                    for p, info in zip(preds, metadata, strict=False)
                    if info.params["loss"] == "jepa"
                ],
                dim=0,
            ),
            # TODO remove the [:, :2049]
            "student_masks": torch.stack(
                [info.mask.to("cuda")[:2049] for info in metadata if info.params["loss"] == "jepa"],
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
                    p.latent[name]
                    for p, info in zip(preds, metadata, strict=False)
                    if info.params["loss"] == "ibot"
                ],
                dim=0,
            ),
            # TODO remove the [:, :2049]
            "student_masks": torch.stack(
                [info.mask.to("cuda")[:2049] for info in metadata if info.params["loss"] == "ibot"],
                dim=0,
            ).unsqueeze(1),
            "student_class_masked": torch.stack(
                [
                    p.latent[name]
                    for p, info in zip(preds, metadata, strict=False)
                    if info.params["loss"] == "ibot"
                ],
                dim=0,
            ),
        }
    elif name == "DINO":
        return {
            "local2global_dino_student": torch.stack(
                [
                    p.latent[name]
                    for p, info in zip(preds, metadata, strict=False)
                    if info.params["loss"] == "dino"
                ],
                dim=0,
            ),
            "global2global_dino_student": torch.stack(
                [
                    p.latent[name]
                    for p, info in zip(preds, metadata, strict=False)
                    if info.params["loss"] == "dino"
                ],
                dim=0,
            ),
        }
    else:
        raise NotImplementedError(
            f"{name} is not an implemented loss for the LossLatentSSLStudentTeacher"
        )


def gather_targets_for_loss(name, targets, metadata, target2source_matching_idxs):
    if name == "JEPA":
        """
        Important this assumes that there is 1 masked version for each global view
        ie. student_patches_masked.shape[0] == teacher_patches_masked.shape[0]
        """
        return {
            "teacher_patches_masked": torch.stack(
                [p.latent[name] for p, info in zip(targets, metadata, strict=False)],
                dim=0,
            ),
        }
    elif name == "iBOT":
        """
        Important this assumes that there is 1 masked version for each global view
        ie. student_patches_masked.shape[0] == teacher_patches_masked.shape[0]

        Note the class token of iBOT is still missing
        """
        import pdb

        pdb.set_trace()
        return {
            "teacher_patches_masked": torch.stack(
                [p.latent[name] for p, info in zip(targets, metadata, strict=False)],
                dim=0,
            ),
            # TODO remove the [:, :2049]
            "teacher_class_masked": torch.stack(
                [p.latent[name] for p, info in zip(targets, metadata, strict=False)],
                dim=0,
            ),
        }
    elif name == "DINO":
        return {
            "local2global_dino_teacher": torch.stack(
                [p.latent[name] for p, info in zip(targets, metadata, strict=False)],
                dim=0,
            ),
            "global2global_dino_teacher": torch.stack(
                list(
                    reversed([p.latent[name] for p, info in zip(targets, metadata, strict=False)])
                ),
                dim=0,
            ),
        }
    else:
        raise NotImplementedError(
            f"{name} is not an implemented loss for the LossLatentSSLStudentTeacher"
        )
