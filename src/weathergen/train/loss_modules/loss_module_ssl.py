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
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from collections import deque

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
        
        self.latent_buffer = deque(maxlen=cf.training_config.loss_module.ssl_memory_buffer_size)

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
                name, preds, output_info, target2source_matching_idxs
            )
            targets_for_loss = gather_targets_for_loss(
                name, targets, target_info, target2source_matching_idxs
            )
            loss_value = loss_fn(**preds_for_loss, **targets_for_loss, **extra_args).mean()
            loss = loss + (weight * loss_value)
            # losses_all[name] = loss_value.item()

        # TODO update to preds and targets
        #self.latent_buffer.append({"preds": preds.detach(), "targets": targets.detach()})
        #past = list(self.latent_buffer)

        return LossValues(loss=loss, losses_all={}, stddev_all={})


def jepa_loss(student_patches_masked, student_masks, teacher_patches_masked, teacher_masks):
    masks_weight = (
        (1 / student_masks.sum(-1).clamp(min=1.0))
        .unsqueeze(-1)
        .expand_as(student_masks)  # [student_masks_flat]
    )
    mask = torch.logical_and(teacher_masks, torch.logical_not(student_masks))
    loss = F.l1_loss(student_patches_masked[mask], teacher_patches_masked[mask])
    loss = loss * masks_weight[mask]
    return loss.sum() # / student_masks.shape[0]


def ibot_loss(
    student_patches_masked,
    student_masks,
    teacher_patches_masked,
    teacher_masks,
    student_class_masked,
    teacher_class_masked,
    student_temp,
):
    loss = loss_fns.masked_student_teacher_patch_softmax(
        student_patches_masked, teacher_patches_masked, student_masks, teacher_masks, student_temp
    ) + loss_fns.student_teacher_softmax(student_class_masked, teacher_class_masked, student_temp)
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


def gather_preds_for_loss(name, preds, metadata, target2source_matching_idxs):     
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
                    if info[0].global_params["loss"] == "jepa"
                ],
                dim=0,
            ),
            "student_masks": torch.stack(
                [info[0].mask.to("cuda") for info in metadata if info[0].global_params["loss"] == "jepa"],
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
                    p.latent[name][:, 1:]
                    for p, info in zip(preds, metadata, strict=False)
                    if info.params["loss"] == "ibot"
                ],
                dim=0,
            ),
            "student_masks": torch.stack(
                [info.mask.to("cuda") for info in metadata if info.params["loss"] == "ibot"],
                dim=0,
            ).unsqueeze(1),
            "student_class_masked": torch.stack(
                [
                    p.latent[name][:, :1]
                    for p, info in zip(preds, metadata, strict=False)
                    if info.params["loss"] == "ibot"
                ],
                dim=0,
            ),
        }
    elif name == "DINO":
        local2global_dino_student = []
        for student_indices in target2source_matching_idxs:
            local_preds = [
                preds[sidx].latent[name]
                for sidx in student_indices
                if metadata[sidx].params["loss"] == "dino"
            ]
            local2global_dino_student.append(local_preds)
        local2global_dino_student = [
            torch.stack(latents, dim=0) for latents in zip(*local2global_dino_student, strict=False)
        ]
        return {
            "local2global_dino_student": local2global_dino_student,
            "global2global_dino_student": torch.stack(
                [
                    p.latent[name]
                    for p, info in zip(preds, metadata, strict=False)
                    if info.params["loss"] == "dino" and info.params["relationship"] == "identity"
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
            "teacher_masks": torch.stack(
                [info[0].mask.to("cuda") for info in metadata],
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
                [p.latent[name][:, 1:] for p, info in zip(targets, metadata, strict=False)],
                dim=0,
            ),
            "teacher_masks": torch.stack(
                [info.mask.to("cuda") for info in metadata],
                dim=0,
            ).unsqueeze(1),
            "teacher_class_masked": torch.stack(
                [p.latent[name][:, :1] for p, info in zip(targets, metadata, strict=False)],
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


# LeJEPA loss

class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device="cuda")
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()
    

class ProjectionHead(torch.nn.Module):
    def __init__(self, input_dim, proj_dim=256):
        super().__init__()
        # make sure input dim is correct
        self.proj = MLP(512, [12288, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, emb):
        # get embeddings from the encoder
        # TODO: reshape here
        return self.proj(emb).reshape(N, V, -1).transpose(0, 1)   


def lejepa_loss(emb, proj):

    inv_loss = (proj.mean(0) - proj).square().mean()
    sigreg = SIGReg().to("cuda")
    sigreg_loss = sigreg(proj)
    lejepa_loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)
    y_rep, yhat = y.repeat_interleave(cfg.V), probe(emb.detach())
    probe_loss = F.cross_entropy(yhat, y_rep)
    loss = lejepa_loss + probe_loss
    return loss


class MLP(nn.Sequential):
    def __init__(self, in_channels, hidden_channels, norm_layer=None, activation_layer=nn.ReLU):
        layers = []
        in_dim = in_channels
        
        for h_dim in hidden_channels[:-1]:
            layers.append(nn.Linear(in_dim, h_dim))
            if norm_layer is not None:
                layers.append(norm_layer(h_dim))
            layers.append(activation_layer(inplace=True))
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, hidden_channels[-1]))
        
        super().__init__(*layers)