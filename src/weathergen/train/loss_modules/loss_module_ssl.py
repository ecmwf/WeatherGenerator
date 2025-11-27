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
from torch import Tensor

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
            name: (local_conf["weight"], get_loss_function_ssl(name))
            for name, local_conf in losses.items()
            # if name in self.valid_loss_names
        }

    def compute_loss(self, preds: dict, targets: dict, metadata) -> LossValues:
        # gradient loss
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # initialize dictionaries for detailed loss tracking and standard deviation statistics
        # create tensor for each stream
        losses_all: dict[str, Tensor] = {loss: 0.0 for loss in self.losses}

        source_target_matching_idxs, output_info, target_source_matching_idxs, target_info = (
            metadata
        )

        import pdb; pdb.set_trace()

        for name, (weight, loss_fn) in self.losses.items():
            preds_for_loss = gather_preds_for_loss(name, preds, output_info)
            targets_for_loss = gather_targets_for_loss(name, targets, target_info)
            loss_value = loss_fn(preds_for_loss, targets_for_loss).mean()
            loss += weight * loss_value
            losses_all[name] = loss_value.item()

        return loss


def get_loss_function_ssl(name):
    if name == "iBOT":
        return loss_fns.masked_student_teacher_patch_softmax
    elif name == "DINO":
        return loss_fns.student_teacher_global_softmax
    elif name == "JEPA":
        return F.l1_loss
    else:
        raise NotImplementedError(
            f"{name} is not an implemented loss for the LossLatentSSLStudentTeacher"
        )


def gather_preds_for_loss(name, preds, metadata):
    if name == "iBOT" or name == "JEPA":
        return {
            "stident_patches_masked": torch.stack(
                [
                    p.latent[name]
                    for p, info in zip(preds, metadata, strict=False)
                    # TODO filter for loss if info.strategy == "masking"
                ],
                dim=0,
            ),
            "student_masks_flat": torch.stack(
                [info['ERA5'].mask.to("cuda") for info in metadata], dim=0
            ),
        }
    elif name == "DINO":
        # TODO deal with DINO having a local and global component
        local2global_dino = torch.stack(
            [
                p.latent[name]
                for p, info in zip(preds, metadata, strict=False)
                # TODO if info.strategy == "cropping"
            ],
            dim=0,
        )
        global2global_dino = torch.stack(
            [
                p.latent[name]
                for p, info in zip(preds, view_metadata, strict=False)
                if info.strategy == "pure"
            ],
            dim=0,
        )
        return local2global_dino, global2global_dino
    else:
        raise NotImplementedError(
            f"{name} is not an implemented loss for the LossLatentSSLStudentTeacher"
        )


def gather_targets_for_loss(name, targets, metadata):
    if name == "iBOT" or name == "JEPA":
        return torch.stack(
            [p[name] for p, info in zip(targets, metadata, strict=False)], dim=0
        )
    if name == "DINO":
        local2global_dino = torch.stack(
            [p[name] for p, info in zip(targets, metadata, strict=False)], dim=0
        )
        global2global_dino = torch.stack(
            reversed([p[name] for p, info in zip(targets, metadata, strict=False)]), dim=0
        )
        return local2global_dino, global2global_dino
    else:
        raise NotImplementedError(
            f"{name} is not an implemented loss for the LossLatentSSLStudentTeacher"
        )
