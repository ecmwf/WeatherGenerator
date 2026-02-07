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
import torch.nn as nn

from weathergen.model.engines import LatentPredictionHeadIdentity
from weathergen.model.ssl_target_processing import (
    DINOTargetProcessing,
    JEPATargetProcessing,
    iBOTPatchTargetProcessing,
)
from weathergen.train.target_and_aux_module_base import TargetAndAuxModuleBase, TargetAuxOutput

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from weathergen.common.config import Config

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

    def __init__(self, teacher_model, training_cfg: DictConfig, **kwargs):
        """Initialize the EncoderTeacher.

        Args:
            teacher_model: The teacher model (can be EMA model wrapper or frozen model).
            training_cfg: Training configuration containing loss specifications.
                Must have `losses` attribute with at least one LossLatentSSLStudentTeacher.
            **kwargs: Additional arguments passed to postprocessing setup.

        Raises:
            ValueError: If training_cfg has no LossLatentSSLStudentTeacher losses.
        """
        self.teacher_model = teacher_model

        # Parse SSL losses from config to set up target postprocessing
        assert hasattr(training_cfg, "losses"), (
            f"EncoderTeacher requires training_cfg with 'losses' attribute, "
            f"got {type(training_cfg).__name__}"
        )

        losses_cfg = [
            v.loss_fcts
            for k, v in training_cfg.losses.items()
            if v.type == "LossLatentSSLStudentTeacher"
        ]

        if not losses_cfg:
            raise ValueError(
                "EncoderTeacher requires at least one 'LossLatentSSLStudentTeacher' loss "
                "in training_config.losses. Found loss types: "
                f"{[v.type for v in training_cfg.losses.values()]}"
            )

        # TODO: support multiple LossLatentSSLStudentTeacher loss terms
        if len(losses_cfg) > 1:
            logger.warning(
                f"Found {len(losses_cfg)} LossLatentSSLStudentTeacher losses, "
                "but only the first one is used for target postprocessing."
            )

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

    def compute(self, istep: int, batch, model_params, model) -> TargetAuxOutput:
        """Compute target representations from the teacher model.

        Args:
            istep: Training step index.
            batch: Input batch with get_samples(), get_output_len(), get_output_idxs() methods.
            model_params: Model parameters for the forward pass.
            model: Student model (not used, but part of interface).

        Returns:
            TargetAuxOutput containing latent targets and auxiliary outputs.

        Raises:
            KeyError: If teacher model doesn't output a required loss type.
        """
        with torch.no_grad():
            model_output = self._forward_teacher(model_params, batch)
            outputs = model_output.get_latent_prediction(0)

            targets = {}
            for loss_name, target_module in self.postprocess_targets.items():
                if loss_name not in outputs:
                    available_keys = list(outputs.keys()) if hasattr(outputs, "keys") else "N/A"
                    raise KeyError(
                        f"Teacher model output missing key '{loss_name}'. "
                        f"Available keys: {available_keys}. "
                        f"Ensure teacher model has latent head for '{loss_name}'."
                    )
                targets[loss_name] = target_module(outputs[loss_name])

            # Collect target meta-information for selected samples
            samples = batch.get_samples()
            aux_outputs = []
            for sample in samples:
                if sample.meta_info:
                    aux_outputs.append(list(sample.meta_info.values())[0])
                else:
                    aux_outputs.append(None)

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

    def get_current_beta(self, cur_step: int) -> float:
        beta = self.ema_model.get_current_beta(cur_step)
        return beta


class EMATeacher(EncoderTeacher):
    """Teacher using Exponential Moving Average of student weights.

    This teacher maintains an EMA of the student model's weights and uses it
    to generate target representations for SSL training.
    """

    def __init__(self, model, ema_model, batch_size: int, training_cfg: DictConfig, **kwargs):
        """Initialize the EMATeacher.

        Args:
            model: The student model (used for reference, weights copied to EMA).
            ema_model: The EMA model wrapper that maintains averaged weights.
                Must have reset(), update(), forward_eval() methods.
            batch_size: Global batch size for EMA update scheduling. Must be positive.
            training_cfg: Training configuration with SSL loss specifications.
            **kwargs: Additional arguments passed to parent.

        Note:
            The teacher model may have a different architecture to the student,
            e.g. for JEPA. The ema_model handles weight copying appropriately.
            You cannot assume model.state_dict equals ema_model.state_dict.

        Raises:
            ValueError: If batch_size is not positive.
            AssertionError: If ema_model lacks required methods.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        # Validate ema_model interface
        assert hasattr(ema_model, "reset"), "ema_model must have reset() method"
        assert hasattr(ema_model, "update"), "ema_model must have update() method"
        assert hasattr(ema_model, "forward_eval"), "ema_model must have forward_eval() method"

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

    The teacher model may have been pre-trained with any method (forecasting, MAE, etc.)
    and doesn't need to have SSL latent heads. Identity heads are added automatically
    for any SSL losses the student needs.

    Note:
        This class intentionally does NOT call super().__init__() because:
        1. It sets up identity postprocessing (JEPATargetProcessing) for ALL losses,
           regardless of what the student config specifies for DINO/iBOT
        2. The parent class would try to parse the teacher's training config for SSL losses,
           but the teacher may have been trained without SSL (e.g., forecasting only)

    Warning:
        This class modifies the teacher_model in-place by adding latent_heads if missing.
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        training_cfg: DictConfig | None,
        teacher_model_params=None,
        **kwargs,
    ):
        """Initialize the FrozenTeacher.

        Args:
            teacher_model: Pre-trained model to use as teacher. Will be modified in-place
                to add identity latent heads if they don't exist.
            training_cfg: Current training configuration containing the student's SSL losses.
                Used to determine which identity heads to add to the teacher.
                If None, defaults to adding a JEPA head.
            teacher_model_params: Model parameters matching the teacher's architecture
                (positional embeddings, q_cells, etc.). If None, will use the student's
                model_params which may cause dimension mismatch if architectures differ.
            **kwargs: Additional arguments (unused, for interface compatibility).
        """
        # Note: We intentionally don't call super().__init__() - see class docstring
        self.teacher_model = teacher_model
        self.teacher_model_params = teacher_model_params

        # Get required SSL loss names from current training config
        required_heads = self._get_required_ssl_heads(training_cfg)
        assert len(required_heads) > 0, "No SSL heads required - this should never happen"

        # Add identity heads to teacher if it doesn't have them (modifies model in-place)
        self._ensure_identity_heads(teacher_model, required_heads)

        # Set up identity postprocessing for all SSL losses
        # FrozenTeacher always uses identity (JEPATargetProcessing) regardless of loss type
        self.postprocess_targets = {name: JEPATargetProcessing() for name in required_heads}

        # Freeze all parameters
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # Set to eval mode permanently (affects BatchNorm, Dropout, etc.)
        self.teacher_model.eval()

    def _get_required_ssl_heads(self, training_cfg: DictConfig | None) -> set[str]:
        """Extract SSL loss names from training config.

        Args:
            training_cfg: Training configuration containing losses specification.
                If None, defaults to {"JEPA"}.

        Returns:
            Set of SSL loss names (e.g., {"JEPA", "DINO"}). Never empty.
        """
        if training_cfg is None:
            logger.debug("FrozenTeacher: No training_cfg provided, defaulting to JEPA head")
            return {"JEPA"}

        if not hasattr(training_cfg, "losses"):
            logger.warning(
                "FrozenTeacher: training_cfg has no 'losses' attribute, defaulting to JEPA head"
            )
            return {"JEPA"}

        required_heads = set()
        for loss_name, loss_cfg in training_cfg.losses.items():
            if not hasattr(loss_cfg, "type"):
                continue
            if loss_cfg.type == "LossLatentSSLStudentTeacher":
                if hasattr(loss_cfg, "loss_fcts"):
                    required_heads.update(loss_cfg.loss_fcts.keys())
                else:
                    logger.warning(
                        f"FrozenTeacher: Loss '{loss_name}' has type LossLatentSSLStudentTeacher "
                        "but no loss_fcts, skipping"
                    )

        if not required_heads:
            logger.debug(
                "FrozenTeacher: No LossLatentSSLStudentTeacher losses found in config, "
                "defaulting to JEPA head"
            )
            return {"JEPA"}

        logger.debug(f"FrozenTeacher: Required SSL heads from config: {required_heads}")
        return required_heads

    def _ensure_identity_heads(self, teacher_model: nn.Module, required_heads: set[str]) -> None:
        """Add identity latent heads to teacher model if they don't exist.

        The teacher may have been pre-trained without SSL losses (e.g., forecasting).
        We add identity heads so that `get_latent_prediction()` returns the raw
        encoder representations (specifically, patch_tokens from LatentState) for
        the student's SSL losses.

        Warning:
            This method modifies teacher_model IN-PLACE by adding to its latent_heads.

        Args:
            teacher_model: The teacher model to modify. Will have latent_heads added/modified.
            required_heads: Set of head names that must exist (e.g., {"JEPA", "DINO"}).
        """
        # Ensure latent_heads ModuleDict exists
        if not hasattr(teacher_model, "latent_heads") or teacher_model.latent_heads is None:
            logger.info("FrozenTeacher: Teacher model has no latent_heads, creating ModuleDict")
            teacher_model.latent_heads = nn.ModuleDict()

        # Add missing identity heads
        for head_name in sorted(required_heads):  # sorted for deterministic logging
            if head_name not in teacher_model.latent_heads:
                logger.info(
                    f"FrozenTeacher: Adding identity head '{head_name}' to teacher model "
                    f"(teacher was likely pre-trained without SSL losses)"
                )
                teacher_model.latent_heads[head_name] = LatentPredictionHeadIdentity()

    @classmethod
    def from_pretrained(cls, cf: Config, dataset, device, params: dict) -> FrozenTeacher:
        """Create a FrozenTeacher from a pre-trained checkpoint.

        This factory method:
        1. Loads the teacher's config from the checkpoint
        2. Creates a model with the teacher's architecture
        3. Loads the pre-trained weights
        4. Creates ModelParams matching the teacher's architecture
        5. Returns a FrozenTeacher instance

        Args:
            cf: Current training configuration. Used for:
                - model_path: Where to find saved models
                - training_config: To determine which SSL heads are needed
            dataset: Dataset for model creation (provides input/output dimensions).
            device: Target device (e.g., "cuda:0", "cpu").
            params: FrozenTeacher parameters from config, including:
                - teacher_run_id (required): 8-character run ID of the pre-trained teacher.
                - teacher_mini_epoch (optional): Mini-epoch to load. Default -1 (latest).

        Returns:
            FrozenTeacher instance with loaded and frozen weights.

        Raises:
            ValueError: If teacher_run_id is not provided or invalid.
            FileNotFoundError: If checkpoint doesn't exist (from load_run_config/load_model).
        """
        # Lazy imports to avoid circular dependency with model_interface
        from weathergen.common.config import load_run_config, merge_configs
        from weathergen.model.model import ModelParams
        from weathergen.model.model_interface import get_model, load_model
        from weathergen.utils.distributed import is_root

        teacher_run_id = params.get("teacher_run_id")
        teacher_mini_epoch = params.get("teacher_mini_epoch", -1)

        # Validate teacher_run_id
        if teacher_run_id is None:
            raise ValueError(
                "FrozenTeacher requires 'teacher_run_id' in config. "
                "Example config:\n"
                "  target_and_aux_calc:\n"
                "    FrozenTeacher:\n"
                "      teacher_run_id: 'a1b2c3d4'"
            )

        if not isinstance(teacher_run_id, str) or len(teacher_run_id) == 0:
            raise ValueError(
                f"teacher_run_id must be a non-empty string, got {type(teacher_run_id).__name__}: "
                f"{teacher_run_id!r}"
            )

        if is_root():
            logger.info(
                f"Loading FrozenTeacher from run_id={teacher_run_id}, "
                f"mini_epoch={teacher_mini_epoch}"
            )

        # Load teacher's config (contains full architecture)
        model_path = cf.get("model_path")
        assert model_path is not None, "cf.model_path is required to load FrozenTeacher checkpoint"

        teacher_config = load_run_config(teacher_run_id, teacher_mini_epoch, model_path)

        # Disable FSDP/DDP for frozen teacher - it's loaded as a simple non-sharded model
        # This avoids complications with distributed training for the teacher
        teacher_config = merge_configs(teacher_config, {"with_ddp": False, "with_fsdp": False})

        # Create model with teacher's architecture
        teacher_model = get_model(teacher_config, "student", dataset, {})

        # Load weights from checkpoint
        teacher_model = load_model(
            teacher_config, teacher_model, device, teacher_run_id, teacher_mini_epoch
        )

        # Create model params matching teacher's architecture
        # This includes positional embeddings, q_cells, etc. that depend on architecture
        teacher_model_params = ModelParams(teacher_config).create(teacher_config)
        teacher_model_params = teacher_model_params.to(device)

        if is_root():
            num_params = sum(p.numel() for p in teacher_model.parameters())
            logger.info(f"FrozenTeacher loaded with {num_params:,} parameters")

        # Pass current training config so FrozenTeacher knows which SSL heads to add
        return cls(
            teacher_model,
            training_cfg=cf.training_config,
            teacher_model_params=teacher_model_params,
        )

    def _forward_teacher(self, model_params, batch):
        """Execute forward pass on the frozen teacher model.

        Uses the teacher's own model_params instead of the student's to ensure
        dimension compatibility.
        """
        # Use teacher's model params if available, otherwise fall back to passed-in params
        params_to_use = (
            self.teacher_model_params if self.teacher_model_params is not None else model_params
        )
        return self.teacher_model(params_to_use, batch)

    def reset(self, batch_size=None):
        """No-op: frozen teacher weights don't change."""
        pass

    def update_state_post_opt_step(self, istep, batch, model, **kwargs) -> None:
        """No-op: frozen teacher weights don't change."""
        pass


def get_target_postprocessing(
    target_losses: dict[str, DictConfig], training_cfg: DictConfig, **kwargs
) -> dict[str, nn.Module]:
    """Create postprocessing modules for each SSL loss type.

    This function creates the appropriate postprocessing module for each SSL loss
    based on its configuration. The postprocessing is applied to teacher outputs
    before computing the student-teacher loss.

    - JEPA: Identity (no postprocessing)
    - DINO: Centering and temperature sharpening
    - iBOT: Patch-level centering and temperature sharpening

    Args:
        target_losses: Dict of loss configurations keyed by loss name (e.g., "JEPA", "DINO").
            Each value should have the required config keys for that loss type.
        training_cfg: Training configuration (currently unused, reserved for future use).
        **kwargs: Additional arguments (currently unused).

    Returns:
        Dict mapping loss names to their postprocessing nn.Module instances.

    Raises:
        KeyError: If a loss config is missing required keys (e.g., out_dim for DINO).

    Example:
        >>> target_losses = {"JEPA": {"head": "identity"}, "DINO": {"out_dim": 256, ...}}
        >>> postprocessors = get_target_postprocessing(target_losses, training_cfg)
        >>> postprocessors["JEPA"](teacher_output)  # Identity transform
    """
    return_dict = {}
    for loss_name, conf in target_losses.items():
        if loss_name == "iBOT":
            # Validate required keys
            required_keys = [
                "out_dim",
                "center_momentum",
                "loss_extra_args",
                "teacher_temp",
                "teacher_style",
            ]
            missing = [k for k in required_keys if k not in conf]
            if missing:
                raise KeyError(f"iBOT loss config missing required keys: {missing}")

            return_dict[loss_name] = iBOTPatchTargetProcessing(
                patch_out_dim=conf["out_dim"],
                center_momentum=conf["center_momentum"],
                student_temp=conf["loss_extra_args"]["student_temp"],
                teacher_temp=conf["teacher_temp"],
                teacher_style=conf["teacher_style"],
            )
        elif loss_name == "DINO":
            # Validate required keys
            required_keys = ["out_dim", "center_momentum", "loss_extra_args", "teacher_style"]
            missing = [k for k in required_keys if k not in conf]
            if missing:
                raise KeyError(f"DINO loss config missing required keys: {missing}")

            return_dict[loss_name] = DINOTargetProcessing(
                out_dim=conf["out_dim"],
                center_momentum=conf["center_momentum"],
                student_temp=conf["loss_extra_args"]["student_temp"],
                teacher_style=conf["teacher_style"],
            )
        elif loss_name == "JEPA":
            return_dict[loss_name] = JEPATargetProcessing()
        else:
            # Skip losses that are not handled by the EncoderTeacher
            logger.debug(f"get_target_postprocessing: Skipping unknown loss type '{loss_name}'")
            continue

    return return_dict
