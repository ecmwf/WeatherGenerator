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
from typing import TYPE_CHECKING

import torch

from weathergen.model.ssl_target_processing import (
    DINOTargetProcessing,
    JEPATargetProcessing,
    iBOTPatchTargetProcessing,
)
from weathergen.train.target_and_aux_module_base import TargetAndAuxModuleBase, TargetAuxOutput

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class EncoderTeacher(TargetAndAuxModuleBase):
    """Abstract base class for SSL teachers that use an encoder to generate targets.

    This class provides the common functionality for teacher models in student-teacher
    SSL training setups. Subclasses must implement `_forward_teacher()` to define
    how the teacher model generates outputs.

    Attributes:
        teacher_model: The teacher model used to generate target representations.
        postprocess_targets: Dict of postprocessing modules for each loss type.
    """

    def __init__(self, teacher_model, training_cfg, **kwargs):
        """Initialize the EncoderTeacher.

        Args:
            teacher_model: The teacher model (can be EMA model wrapper or frozen model).
            training_cfg: Training configuration containing loss specifications.
            **kwargs: Additional arguments passed to postprocessing setup.
        """
        self.teacher_model = teacher_model

        # Parse SSL losses from config to set up target postprocessing
        losses_cfg = [
            v.loss_fcts
            for k, v in training_cfg.losses.items()
            if v.type == "LossLatentSSLStudentTeacher"
        ]
        # TODO: support multiple LossLatentSSLStudentTeacher loss terms
        self.postprocess_targets = get_target_postprocessing(losses_cfg[0], training_cfg, **kwargs)

    def _forward_teacher(self, model_params, batch):
        """Execute forward pass on the teacher model.

        Subclasses must implement this method to define their specific forward behavior.

        Args:
            model_params: Model parameters for the forward pass.
            batch: Input batch.

        Returns:
            Model output with get_latent_prediction() method.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement _forward_teacher()")

    def compute(self, istep, batch, model_params, model) -> TargetAuxOutput:
        """Compute target representations from the teacher model.

        Args:
            istep: Training step index.
            batch: Input batch.
            model_params: Model parameters.
            model: Student model (not used, but part of interface).

        Returns:
            TargetAuxOutput containing latent targets and auxiliary outputs.
        """
        with torch.no_grad():
            outputs = self._forward_teacher(model_params, batch).get_latent_prediction(0)
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
        """Update state before backward pass. Default is no-op."""
        return

    def to_device(self, device) -> EncoderTeacher:
        """Move postprocessors to the specified device.

        Args:
            device: Target device.

        Returns:
            Self for method chaining.
        """
        for _, module in self.postprocess_targets.items():
            module.to(device)
        return self


class EMATeacher(EncoderTeacher):
    """Teacher using Exponential Moving Average of student weights.

    This teacher maintains an EMA of the student model's weights and uses it
    to generate target representations for SSL training.
    """

    def __init__(self, model, ema_model, batch_size, training_cfg, **kwargs):
        """Initialize the EMATeacher.

        Args:
            model: The student model (used for reference, weights copied to EMA).
            ema_model: The EMA model wrapper that maintains averaged weights.
            batch_size: Global batch size for EMA update scheduling.
            training_cfg: Training configuration.
            **kwargs: Additional arguments passed to parent.

        Note:
            The teacher model may have a different architecture to the student,
            e.g. for JEPA. The ema_model handles weight copying appropriately.
            You cannot assume model.state_dict equals ema_model.state_dict.
        """
        self.ema_model = ema_model
        self.batch_size = batch_size
        super().__init__(ema_model, training_cfg, **kwargs)
        self.reset()

    def _forward_teacher(self, model_params, batch):
        """Execute forward pass using EMA model's forward_eval method."""
        return self.ema_model.forward_eval(model_params, batch)

    def reset(self, batch_size=None):
        """Reset EMA model weights to match current student weights.

        Args:
            batch_size: Optional new batch size to use for EMA updates.
        """
        self.ema_model.reset()
        if batch_size is not None:
            self.batch_size = batch_size

    def update_state_post_opt_step(self, istep, batch, model, **kwargs) -> None:
        """Update EMA weights after optimizer step.

        Args:
            istep: Current training step.
            batch: Current batch (unused).
            model: Student model (unused, EMA model tracks it internally).
            **kwargs: Additional arguments (unused).
        """
        if self.ema_model.is_model_sharded:
            self.ema_model.ema_model.reshard()
        self.ema_model.update(istep, self.batch_size)


class FrozenTeacher(EncoderTeacher):
    """Teacher loaded from a pre-trained checkpoint with frozen weights.

    This teacher uses a model loaded from a previous training run. The weights
    are frozen and never updated during training. This is useful for distillation
    from a pre-trained model as described in arXiv:2509.24317.
    """

    def __init__(self, teacher_model, training_cfg, **kwargs):
        """Initialize the FrozenTeacher.

        Args:
            teacher_model: Pre-trained model to use as teacher.
            training_cfg: Training configuration.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(teacher_model, training_cfg, **kwargs)

        # Ensure all parameters are frozen
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # Set to eval mode permanently
        self.teacher_model.eval()

    @classmethod
    def from_pretrained(cls, cf, dataset, device, params: dict) -> FrozenTeacher:
        """Create a FrozenTeacher from a pre-trained checkpoint.

        Args:
            cf: Current training configuration.
            dataset: Dataset for model creation.
            device: Target device.
            params: FrozenTeacher parameters from config, including:
                - teacher_run_id (required): Run ID of the pre-trained teacher model.
                - teacher_mini_epoch (optional): Mini-epoch to load. Default -1 (latest).

        Returns:
            FrozenTeacher instance with loaded and frozen weights.

        Raises:
            ValueError: If teacher_run_id is not provided.
        """
        # Lazy imports to avoid circular dependency with model_interface
        from weathergen.common.config import load_run_config, merge_configs
        from weathergen.model.model_interface import get_model, load_model
        from weathergen.utils.distributed import is_root

        teacher_run_id = params.get("teacher_run_id")
        teacher_mini_epoch = params.get("teacher_mini_epoch", -1)

        if teacher_run_id is None:
            raise ValueError("FrozenTeacher requires 'teacher_run_id' in config")

        if is_root():
            logger.info(
                f"Loading FrozenTeacher from run_id={teacher_run_id}, "
                f"mini_epoch={teacher_mini_epoch}"
            )

        # Load teacher's config (contains full architecture)
        teacher_config = load_run_config(teacher_run_id, teacher_mini_epoch, cf.get("model_path"))

        # Disable FSDP/DDP for frozen teacher - it's loaded as a simple non-sharded model
        teacher_config = merge_configs(teacher_config, {"with_ddp": False, "with_fsdp": False})

        # Create model with teacher's architecture
        teacher_model = get_model(teacher_config, "student", dataset, {})

        # Load weights
        teacher_model = load_model(
            teacher_config, teacher_model, device, teacher_run_id, teacher_mini_epoch
        )

        return cls(teacher_model, cf.training_config)

    def _forward_teacher(self, model_params, batch):
        """Execute forward pass on the frozen teacher model."""
        return self.teacher_model(model_params, batch)

    def reset(self, batch_size=None):
        """No-op: frozen teacher weights don't change."""
        pass

    def update_state_post_opt_step(self, istep, batch, model, **kwargs) -> None:
        """No-op: frozen teacher weights don't change."""
        pass


def get_target_postprocessing(target_losses: list[str], training_cfg, **kwargs):
    """Create postprocessing modules for each SSL loss type.

    Args:
        target_losses: Dict of loss configurations keyed by loss name.
        training_cfg: Training configuration.
        **kwargs: Additional arguments (unused).

    Returns:
        Dict mapping loss names to their postprocessing modules.
    """
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
            # We skip losses that are not handled by the EncoderTeacher
            continue
    return return_dict
