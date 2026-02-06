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

import torch
from omegaconf import DictConfig

from weathergen.train.loss_modules.loss_module_base import LossModuleBase, LossValues
from weathergen.train.sigreg import SIGReg
from weathergen.utils.train_logger import Stage

logger = logging.getLogger(__name__)


class LossLatentLeJEPA(LossModuleBase):
    """LeJEPA loss: SIGReg regularization + invariance loss.

    Loss = lamb * SIGReg(student_embeddings) + (1 - lamb) * MSE(student, teacher)

    SIGReg constrains embeddings toward isotropic Gaussian N(0, I) to prevent
    collapse, replacing EMA/centering/sharpening heuristics.

    Config params (passed via loss_fcts.LeJEPA):
        lamb: Weight for SIGReg vs invariance (default 0.5).
        num_slices: Number of random projections for SIGReg (default 1024).
        num_points: Quadrature points for EppsPulley (default 17).
        t_max: Upper integration bound for EppsPulley (default 3.0).
    """

    def __init__(self, cf: DictConfig, mode_cfg: DictConfig, stage: Stage, device: str, **losses):
        LossModuleBase.__init__(self)
        self.cf = cf
        self.stage = stage
        self.device = device
        self.name = "LossLatentLeJEPA"

        # There should be exactly one loss entry: "LeJEPA"
        assert "LeJEPA" in losses, f"LossLatentLeJEPA requires a 'LeJEPA' loss entry, got {losses}"
        lejepa_conf = losses["LeJEPA"]

        self.lamb = lejepa_conf.get("lamb", 0.5)
        self.sigreg = SIGReg(
            num_slices=lejepa_conf.get("num_slices", 1024),
            n_points=lejepa_conf.get("num_points", 17),
            t_max=lejepa_conf.get("t_max", 3.0),
        ).to(device)

    def compute_loss(self, preds, targets, metadata) -> LossValues:
        source2target_matching_idxs, output_info, target2source_matching_idxs, _ = metadata

        student_latents = preds.latent[0]  # first fstep
        target_info = targets.aux_outputs
        teacher_latents = targets.latent

        # Gather student and teacher embeddings for the LeJEPA loss
        student_preds = self._gather_preds(student_latents["LeJEPA"], output_info)
        teacher_tgts = self._gather_targets(teacher_latents["LeJEPA"], target_info)

        # Invariance loss: MSE on masked region (student predicts teacher)
        student_patches = student_preds["student_patches"]
        student_masks = student_preds["student_masks"]
        teacher_patches = teacher_tgts["teacher_patches"]
        teacher_masks = teacher_tgts["teacher_masks"]

        # Mask: teacher sees but student doesn't (same as JEPA)
        student_masks_sq = student_masks.squeeze(dim=1)
        teacher_masks_sq = teacher_masks.squeeze(dim=1)
        mask = torch.logical_and(teacher_masks_sq, torch.logical_not(student_masks_sq))

        teacher_patches_expanded = teacher_patches.expand((mask.shape[0], -1, -1))
        invariance_loss = ((student_patches[mask] - teacher_patches_expanded[mask]) ** 2).mean()

        # SIGReg loss: compute on all student embeddings (flattened across batch)
        # Reshape to (N_total, D) for SIGReg
        all_student = torch.cat([p for p in student_latents["LeJEPA"]], dim=0)  # (total_patches, D)
        sigreg_loss = self.sigreg(all_student.unsqueeze(0).float())

        # Combined loss
        loss = self.lamb * sigreg_loss + (1 - self.lamb) * invariance_loss

        losses_all = {
            "LeJEPA": loss.item(),
            "invariance": invariance_loss.item(),
            "sigreg": sigreg_loss.item(),
        }

        return LossValues(loss=loss, losses_all=losses_all, stddev_all={})

    def _gather_preds(self, preds, metadata):
        """Gather student predictions for LeJEPA (same pattern as JEPA)."""
        return {
            "student_patches": torch.stack(
                [
                    p
                    for p, info in zip(preds, metadata, strict=False)
                    if "LeJEPA" in info.global_params["loss"]
                ],
                dim=0,
            ),
            "student_masks": torch.stack(
                [info.mask for info in metadata if "LeJEPA" in info.global_params["loss"]],
                dim=0,
            ).unsqueeze(1),
        }

    def _gather_targets(self, targets, metadata):
        """Gather teacher targets for LeJEPA (same pattern as JEPA)."""
        return {
            "teacher_patches": torch.stack(
                [
                    p
                    for p, info in zip(targets, metadata, strict=True)
                    if "LeJEPA" in info.global_params["loss"]
                ],
                dim=0,
            ),
            "teacher_masks": torch.stack(
                [info.mask for info in metadata if "LeJEPA" in info.global_params["loss"]],
                dim=0,
            ).unsqueeze(1),
        }
