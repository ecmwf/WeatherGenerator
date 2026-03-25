# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from typing import Any

import torch

from weathergen.model.ssl_target_processing import (
    DINOTargetProcessing,
    JEPATargetProcessing,
    iBOTPatchTargetProcessing,
)
from weathergen.train.target_and_aux_module_base import TargetAndAuxModuleBase, TargetAuxOutput


class EMATeacher(TargetAndAuxModuleBase):
    def __init__(self, model, ema_model, batch_size, training_cfg, **kwargs):
        # One of the issues is that the teacher model may have a different architecture
        # to the student, e.g. JEPA. So we need quite a flexible way to instantiate the
        # the teacher. Because of the device sharding etc that requires quite a bit of
        # massaging we assume that the teacher creates the EMA model correctly. However,
        # note that you cannot assume that model.state_dict equals ema_model.state_dict
        self.ema_model = ema_model
        self.batch_size = batch_size

        # is a dict of TargetProcessing classes as we may use several in parallel

        losses_cfg = [
            v.loss_fcts
            for k, v in training_cfg.losses.items()
            if v.type == "LossLatentSSLStudentTeacher"
        ]
        # TODO: support multiple LossLatentSSLStudentTeacher loss terms
        self.postprocess_targets = get_target_postprocessing(losses_cfg[0], training_cfg, **kwargs)

        self.reset()

    def reset(self, batch_size=None):
        self.ema_model.reset()
        if batch_size is not None:
            self.batch_size = batch_size

    def update_state_pre_backward(self, istep, batch, model, **kwargs) -> None:
        return

    def update_state_post_opt_step(self, istep, batch, model, **kwargs) -> None:
        if self.ema_model.is_model_sharded:
            self.ema_model.ema_model.reshard()
        self.ema_model.update(istep, self.batch_size)

    def compute(self, bidx, batch, model_params, model) -> tuple[Any, Any]:
        with torch.no_grad():
            outputs = self.ema_model.forward_eval(model_params, batch).get_latent_prediction(0)
            targets = {}
            for loss_name, target_module in self.postprocess_targets.items():
                targets[loss_name] = target_module(outputs[loss_name])

            # collect target meta-information for selected samples
            aux_outputs = [list(sample.meta_info.values())[0] for sample in batch.get_samples()]

            targets_out = TargetAuxOutput(batch.get_output_len(), batch.get_output_idxs())
            targets_out.latent = targets
            targets_out.aux_outputs = aux_outputs

            return targets_out

    def to_device(self, device) -> EMATeacher:
        for _, module in self.postprocess_targets.items():
            module.to(device)
        return self

    def get_current_beta(self, cur_step: int) -> float:
        beta = self.ema_model.get_current_beta(cur_step)
        return beta


def get_target_postprocessing(target_losses: list[str], training_cfg, **kwargs):
    return_dict = {}
    for loss_name, conf in target_losses.items():
        if loss_name == "iBOT":
            return_dict[loss_name] = iBOTPatchTargetProcessing(
                patch_out_dim=conf["out_dim"],
                center_momentum=conf["center_momentum"],
                student_temp=conf["loss_extra_args"]["student_temp"],
                teacher_temp=conf["teacher_temp"],
                teacher_style=conf["teacher_style"],
            )
        elif loss_name == "DINO":
            return_dict[loss_name] = DINOTargetProcessing(
                out_dim=conf["out_dim"],
                center_momentum=conf["center_momentum"],
                student_temp=conf["loss_extra_args"]["student_temp"],
                teacher_style=conf["teacher_style"],
            )
        elif loss_name == "JEPA":
            return_dict[loss_name] = JEPATargetProcessing()
        else:
            # We skip losses that are not handled by the EMATeacher
            continue
    return return_dict


class SelfTeacher(TargetAndAuxModuleBase):
    """Target calculator that runs the same model on the target batch with gradients.

    Unlike EMATeacher, this does not maintain a separate model copy. Both views
    are processed by the same encoder, and gradients flow through both. This is
    the setup required for LeJEPA (no EMA, no stop-gradient).
    """

    def __init__(self, model, training_cfg, **kwargs):
        pass

    def compute(self, bidx, batch, model_params, model) -> TargetAuxOutput:
        # Run the same model on the target batch WITH gradients
        outputs = model(model_params, batch)
        latent_dict = outputs.get_latent_prediction(0)

        # Use raw encoder latent (patch_tokens) as the target representation.
        # The student's MLP predictor head learns to predict this.
        latent_state = latent_dict["latent_state"]
        targets: dict[str, Any] = {"JEPA": latent_state.patch_tokens}

        # Collect target meta-information for selected samples
        aux_outputs = [list(sample.meta_info.values())[0] for sample in batch.get_samples()]

        targets_out = TargetAuxOutput(batch.get_output_len(), batch.get_output_idxs())
        targets_out.latent = targets
        targets_out.aux_outputs = aux_outputs

        return targets_out
