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
from typing import Any

import torch

from weathergen.common.config import Config, load_run_config, merge_configs
from weathergen.model.model import ModelParams
from weathergen.model.model_interface import get_model
from weathergen.model.ssl_target_processing import (
    DINOTargetProcessing,
    JEPATargetProcessing,
    iBOTPatchTargetProcessing,
)
from weathergen.train.target_and_aux_module_base import TargetAndAuxModuleBase, TargetAuxOutput
from weathergen.train.teacher_utils import (
    load_encoder_from_checkpoint,
    prepare_encoder_teacher,
)

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class EncoderTeacher(TargetAndAuxModuleBase):
    """Base class for SSL teacher models.

    Handles shared logic: SSL loss extraction, target postprocessing, compute loop.
    Subclasses must implement forward_teacher().
    """

    def __init__(self, teacher_model, training_cfg, **kwargs):
        self.teacher_model = teacher_model

        # Extract SSL loss configs
        losses_cfg = [
            v.loss_fcts
            for k, v in training_cfg.losses.items()
            if v.type == "LossLatentSSLStudentTeacher"
        ]
        # TODO: support multiple LossLatentSSLStudentTeacher loss terms
        self.postprocess_targets = get_target_postprocessing(losses_cfg[0], training_cfg, **kwargs)

    def forward_teacher(self, model_params, batch) -> Any:
        raise NotImplementedError("Subclasses must implement forward_teacher()")

    def compute(self, bidx, batch, model_params, model) -> TargetAuxOutput:
        with torch.no_grad():
            outputs = self.forward_teacher(model_params, batch).get_latent_prediction(0)
            targets = {}
            for loss_name, target_module in self.postprocess_targets.items():
                targets[loss_name] = target_module(outputs[loss_name])

            # collect target meta-information for selected samples
            aux_outputs = [list(sample.meta_info.values())[0] for sample in batch.get_samples()]

            targets_out = TargetAuxOutput(batch.get_output_len(), batch.get_output_idxs())
            targets_out.latent = targets
            targets_out.aux_outputs = aux_outputs

            return targets_out

    def update_state_pre_backward(self, istep, batch, model, **kwargs) -> None:
        return

    def to_device(self, device) -> EncoderTeacher:
        for _, module in self.postprocess_targets.items():
            module.to(device)
        return self

    def get_current_beta(self, cur_step: int) -> float:
        beta = self.ema_model.get_current_beta(cur_step)
        return beta


class EMATeacher(EncoderTeacher):
    """SSL teacher using exponential moving average of student weights."""

    def __init__(self, model, ema_model, batch_size, training_cfg, **kwargs):
        super().__init__(model, training_cfg, **kwargs)
        self.ema_model = ema_model
        self.batch_size = batch_size
        self.reset()

    def forward_teacher(self, model_params, batch):
        return self.ema_model.forward_eval(model_params, batch)

    def reset(self, batch_size=None):
        self.ema_model.reset()
        if batch_size is not None:
            self.batch_size = batch_size

    def update_state_post_opt_step(self, istep, batch, model, **kwargs) -> None:
        if self.ema_model.is_model_sharded:
            self.ema_model.ema_model.reshard()
        self.ema_model.update(istep, self.batch_size)

    def get_current_beta(self, cur_step: int) -> float:
        """Return the current EMA interpolation beta for monitoring."""
        return self.ema_model.get_current_beta(cur_step)


class FrozenTeacher(EncoderTeacher):
    """SSL teacher using a frozen pre-trained encoder.

    The encoder is loaded from a checkpoint and never updated. Non-encoder
    parts are discarded; latent heads are created fresh based on the student's
    SSL loss config.
    """

    def __init__(self, teacher_model, training_cfg, teacher_model_params=None):
        super().__init__(teacher_model, training_cfg)
        self.teacher_model_params = teacher_model_params

        # Freeze all parameters
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()

    @classmethod
    def from_pretrained(cls, cf: Config, dataset, device, params: dict) -> FrozenTeacher:
        """Create a FrozenTeacher from a pre-trained checkpoint.

        Args:
            cf: Full training config
            dataset: Dataset for model creation
            device: Target device
            params: Dict with 'teacher_run_id' and optional 'teacher_mini_epoch'
        """

        teacher_run_id = params["teacher_run_id"]
        teacher_mini_epoch = params.get("teacher_mini_epoch", -1)

        # Load teacher's config, create model with teacher's architecture
        teacher_config = load_run_config(teacher_run_id, teacher_mini_epoch, model_path=None)
        teacher_config = merge_configs(teacher_config, {"with_ddp": False, "with_fsdp": False})

        teacher_model = get_model(teacher_config, "student", dataset, {})

        # Load only encoder weights
        load_encoder_from_checkpoint(teacher_model, cf, teacher_run_id, teacher_mini_epoch, device)

        # Strip to encoder + create fresh heads
        prepare_encoder_teacher(teacher_model, cf.training_config, teacher_config)

        # Create model params matching teacher's architecture
        teacher_model_params = ModelParams(teacher_config).create(teacher_config).to(device)

        return cls(teacher_model, cf.training_config, teacher_model_params)

    def forward_teacher(self, model_params, batch):
        params = (
            self.teacher_model_params if self.teacher_model_params is not None else model_params
        )
        return self.teacher_model(params, batch)

    def reset(self, batch_size=None):
        pass

    def update_state_post_opt_step(self, istep, batch, model, **kwargs) -> None:
        pass


def get_target_postprocessing(
    target_losses: dict[str, Any], training_cfg, **kwargs
) -> dict[str, torch.nn.Module]:
    """Create target postprocessing modules for each SSL loss type.

    Args:
        target_losses: Dict mapping loss name → loss config
        training_cfg: Training configuration

    Returns:
        Dict mapping loss name → target processing module
    """
    return_dict = {}
    for loss_name, conf in target_losses.items():
        if loss_name == "iBOT":
            for key in ("out_dim", "center_momentum", "teacher_temp", "teacher_style"):
                if key not in conf:
                    raise KeyError(f"iBOT config missing required key {key!r}")
            return_dict[loss_name] = iBOTPatchTargetProcessing(
                patch_out_dim=conf["out_dim"],
                center_momentum=conf["center_momentum"],
                student_temp=conf["loss_extra_args"]["student_temp"],
                teacher_temp=conf["teacher_temp"],
                teacher_style=conf["teacher_style"],
            )
        elif loss_name == "DINO":
            for key in ("out_dim", "center_momentum", "teacher_style"):
                if key not in conf:
                    raise KeyError(f"DINO config missing required key {key!r}")
            return_dict[loss_name] = DINOTargetProcessing(
                out_dim=conf["out_dim"],
                center_momentum=conf["center_momentum"],
                student_temp=conf["loss_extra_args"]["student_temp"],
                teacher_style=conf["teacher_style"],
            )
        elif loss_name == "JEPA":
            return_dict[loss_name] = JEPATargetProcessing()
        else:
            # We skip losses that are not handled by the teacher
            logger.debug(f"Skipping unknown loss type {loss_name!r} in target postprocessing")
            continue
    return return_dict
