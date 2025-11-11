from typing import Any

import torch

from weathergen.train.target_and_aux_module_base import TargetAndAuxModuleBase
from weathergen.train.ssl_losses_utils import (
    iBOTPatchTargetProcessing,
    DINOTargetProcessing,
    JEPATargetProcessing,
)


class EMATeacher(TargetAndAuxModuleBase):
    def __init__(self, model, rng, ema_model, batch_size, **kwargs):
        # One of the issues is that the teacher model may have a different architecture
        # to the student, e.g. JEPA. So we need quite a flexible way to instantiate the
        # the teacher. Because of the device sharding etc that requires quite a bit of
        # massaging we assume that the teacher creates the EMA model correctly. However,
        # note that you cannot assume that model.state_dict equals ema_model.state_dict
        self.ema_model = ema_model
        self.batch_size = batch_size

        # is a dict of TargetProcessing classes as we may use several in parallel
        self.postprocess_targets = get_target_postprocessing(
            kwargs["losses"]["LossLatentSSLStudentTeacher"], **kwargs
        )

        self.reset()

    def reset(self, batch_size=None):
        self.ema_model.reset()
        if batch_size is not None:
            self.batch_size = batch_size

    def update_state_pre_backward(self, istep, batch, model, **kwargs) -> None:
        return

    def update_state_post_opt_step(self, istep, batch, model, **kwargs) -> None:
        self.ema_model.update(istep, self.batch_size)

    def compute(
        self, bidx, batch, model_params, model, forecast_offset, forecast_steps
    ) -> tuple[Any, Any]:
        """
        Likely will gain in complexity as we actually implement things as different losses
        DINO, iBOT, JEPA will have different heads, which then probably should be computed
        in the postprocess_targets modules, which are nn.Modules
        """
        targets = self.ema_model.forward_eval(model_params, batch, forecast_offset, forecast_steps)
        targets = {}
        for loss_name, target_module in self.postprocess_targets.items():
            with torch.no_grad():
                targets[loss_name] = None  # target_module(targets["loss_name"])
        return targets, None


def get_target_postprocessing(target_losses: list[str], **kwargs):
    return_dict = {}
    for loss_name, conf in target_losses.items():
        if loss_name == "iBOT":
            return_dict[loss_name] = iBOTPatchTargetProcessing(
                patch_out_dim=conf["ibot_patch_out_dim"],
                center_momentum=conf["center_momentum"],
                student_temp=conf["student_temp"],
                teacher_temp=conf["teacher_temp"],
                teacher_style=conf["teacher_style"],
            )
        elif loss_name == "DINO":
            return_dict[loss_name] = DINOTargetProcessing(
                out_dim=conf["dino_out_dim"],
                center_momentum=conf["center_momentum"],
                student_temp=conf["student_temp"],
                teacher_style=conf["teacher_style"],
            )
        elif loss_name == "JEPA":
            return_dict[loss_name] = JEPATargetProcessing()
        else:
            # We skip losses that are not handled by the EMATeacher
            continue
    return return_dict
