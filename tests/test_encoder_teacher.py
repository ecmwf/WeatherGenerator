# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for EncoderTeacher class hierarchy (EMATeacher and FrozenTeacher)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

# Mock flash_attn before importing weathergen modules
sys.modules["flash_attn"] = MagicMock()

from weathergen.train.target_and_aux_module_base import TargetAuxOutput  # noqa: E402


# =============================================================================
# Fixtures for mock objects
# =============================================================================


class MockLatentState:
    """Mock latent state that get_latent_prediction returns."""

    def __init__(self, data: dict):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]


class MockModelOutput:
    """Mock model output with get_latent_prediction method."""

    def __init__(self, latent_data: dict):
        self._latent_data = latent_data

    def get_latent_prediction(self, idx: int):
        return self._latent_data


class MockSample:
    """Mock sample with meta_info."""

    def __init__(self):
        self.meta_info = {"key": "value"}


class MockBatch:
    """Mock batch for testing compute()."""

    def __init__(self, num_samples: int = 2):
        self._samples = [MockSample() for _ in range(num_samples)]

    def get_samples(self):
        return self._samples

    def get_output_len(self):
        return 1

    def get_output_idxs(self):
        return [0]


class MockEMAModel:
    """Mock EMA model for testing EMATeacher."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.ema_model = model
        self.is_model_sharded = False
        self._reset_called = False
        self._update_called = False
        self._update_args = None

    def reset(self):
        self._reset_called = True
        # Copy weights from model to ema_model (simulating real behavior)
        with torch.no_grad():
            for p_ema, p_model in zip(
                self.ema_model.parameters(), self.model.parameters()
            ):
                p_ema.copy_(p_model)

    def update(self, istep: int, batch_size: int):
        self._update_called = True
        self._update_args = (istep, batch_size)
        # Simulate EMA update by slightly modifying weights
        with torch.no_grad():
            for p in self.ema_model.parameters():
                p.mul_(0.999).add_(torch.randn_like(p) * 0.001)

    def forward_eval(self, model_params, batch):
        return self.ema_model(model_params, batch)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 5))
    return model


@pytest.fixture
def mock_training_cfg():
    """Create mock training config with JEPA loss."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "losses": {
                "ssl_loss": {
                    "type": "LossLatentSSLStudentTeacher",
                    "loss_fcts": {"JEPA": {"head": "identity", "out_dim": 256}},
                }
            }
        }
    )
    return cfg


@pytest.fixture
def mock_ema_model(simple_model):
    """Create mock EMA model wrapping simple_model."""
    return MockEMAModel(simple_model)


@pytest.fixture
def model_with_latent_heads():
    """Create a model with latent_heads attribute for FrozenTeacher testing."""
    model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 5))
    # Add latent_heads attribute to mimic real model structure
    model.latent_heads = nn.ModuleDict({"JEPA": nn.Identity()})
    return model


# =============================================================================
# Interface Tests - Both EMATeacher and FrozenTeacher must pass these
# =============================================================================


class TestEncoderTeacherInterface:
    """Tests for the shared interface of EncoderTeacher subclasses."""

    def test_ema_teacher_has_required_methods(self):
        """Verify EMATeacher has all required interface methods."""
        from weathergen.train.target_and_aux_ssl_teacher import EMATeacher

        required_methods = [
            "reset",
            "update_state_pre_backward",
            "update_state_post_opt_step",
            "compute",
            "to_device",
        ]
        for method in required_methods:
            assert hasattr(EMATeacher, method), f"EMATeacher missing method: {method}"
            assert callable(
                getattr(EMATeacher, method)
            ), f"EMATeacher.{method} is not callable"

    def test_frozen_teacher_has_required_methods(self):
        """Verify FrozenTeacher has all required interface methods."""
        from weathergen.train.target_and_aux_ssl_teacher import FrozenTeacher

        required_methods = [
            "reset",
            "update_state_pre_backward",
            "update_state_post_opt_step",
            "compute",
            "to_device",
        ]
        for method in required_methods:
            assert hasattr(
                FrozenTeacher, method
            ), f"FrozenTeacher missing method: {method}"
            assert callable(
                getattr(FrozenTeacher, method)
            ), f"FrozenTeacher.{method} is not callable"

    def test_ema_teacher_update_state_pre_backward_is_noop(
        self, simple_model, mock_ema_model, mock_training_cfg
    ):
        """Verify update_state_pre_backward returns None (no-op)."""
        from weathergen.train.target_and_aux_ssl_teacher import EMATeacher

        teacher = EMATeacher(
            simple_model, mock_ema_model, batch_size=8, training_cfg=mock_training_cfg
        )
        result = teacher.update_state_pre_backward(
            istep=0, batch=MockBatch(), model=simple_model
        )
        assert result is None

    def test_frozen_teacher_update_state_pre_backward_is_noop(
        self, simple_model, model_with_latent_heads
    ):
        """Verify FrozenTeacher.update_state_pre_backward returns None (no-op)."""
        from weathergen.train.target_and_aux_ssl_teacher import FrozenTeacher

        teacher = FrozenTeacher(model_with_latent_heads, training_cfg=None)
        result = teacher.update_state_pre_backward(
            istep=0, batch=MockBatch(), model=simple_model
        )
        assert result is None

    def test_ema_teacher_to_device_moves_postprocessors(
        self, simple_model, mock_ema_model, mock_training_cfg
    ):
        """Verify to_device moves postprocessors to specified device."""
        from weathergen.train.target_and_aux_ssl_teacher import EMATeacher

        teacher = EMATeacher(
            simple_model, mock_ema_model, batch_size=8, training_cfg=mock_training_cfg
        )

        # Track if .to() was called on postprocessors
        for name, module in teacher.postprocess_targets.items():
            module.to = MagicMock(return_value=module)

        teacher.to_device("cpu")

        for name, module in teacher.postprocess_targets.items():
            module.to.assert_called_once_with("cpu")

    def test_frozen_teacher_to_device_moves_postprocessors(
        self, model_with_latent_heads
    ):
        """Verify FrozenTeacher.to_device moves postprocessors."""
        from weathergen.train.target_and_aux_ssl_teacher import FrozenTeacher

        teacher = FrozenTeacher(model_with_latent_heads, training_cfg=None)

        for name, module in teacher.postprocess_targets.items():
            module.to = MagicMock(return_value=module)

        teacher.to_device("cpu")

        for name, module in teacher.postprocess_targets.items():
            module.to.assert_called_once_with("cpu")


# =============================================================================
# EMATeacher-specific Tests
# =============================================================================


class TestEMATeacher:
    """Tests specific to EMATeacher behavior."""

    def test_ema_reset_calls_ema_model_reset(
        self, simple_model, mock_ema_model, mock_training_cfg
    ):
        """After reset, EMA model's reset method should be called."""
        from weathergen.train.target_and_aux_ssl_teacher import EMATeacher

        teacher = EMATeacher(
            simple_model, mock_ema_model, batch_size=8, training_cfg=mock_training_cfg
        )

        # Reset is called in __init__, so reset the flag first
        mock_ema_model._reset_called = False

        teacher.reset()
        assert mock_ema_model._reset_called

    def test_ema_reset_can_update_batch_size(
        self, simple_model, mock_ema_model, mock_training_cfg
    ):
        """Reset can optionally update batch size."""
        from weathergen.train.target_and_aux_ssl_teacher import EMATeacher

        teacher = EMATeacher(
            simple_model, mock_ema_model, batch_size=8, training_cfg=mock_training_cfg
        )
        assert teacher.batch_size == 8

        teacher.reset(batch_size=16)
        assert teacher.batch_size == 16

    def test_ema_update_post_opt_step_calls_ema_update(
        self, simple_model, mock_ema_model, mock_training_cfg
    ):
        """update_state_post_opt_step should call ema_model.update()."""
        from weathergen.train.target_and_aux_ssl_teacher import EMATeacher

        teacher = EMATeacher(
            simple_model, mock_ema_model, batch_size=8, training_cfg=mock_training_cfg
        )

        teacher.update_state_post_opt_step(
            istep=10, batch=MockBatch(), model=simple_model
        )

        assert mock_ema_model._update_called
        assert mock_ema_model._update_args == (10, 8)


# =============================================================================
# FrozenTeacher-specific Tests
# =============================================================================


class TestFrozenTeacher:
    """Tests specific to FrozenTeacher behavior."""

    def test_frozen_teacher_init_freezes_parameters(self, model_with_latent_heads):
        """FrozenTeacher should freeze all model parameters on init."""
        from weathergen.train.target_and_aux_ssl_teacher import FrozenTeacher

        # Verify model starts with requires_grad=True
        assert all(p.requires_grad for p in model_with_latent_heads.parameters())

        teacher = FrozenTeacher(model_with_latent_heads, training_cfg=None)

        # All parameters should be frozen
        assert all(not p.requires_grad for p in teacher.teacher_model.parameters())

    def test_frozen_teacher_init_sets_eval_mode(self, model_with_latent_heads):
        """FrozenTeacher should set model to eval mode."""
        from weathergen.train.target_and_aux_ssl_teacher import FrozenTeacher

        model_with_latent_heads.train()
        assert model_with_latent_heads.training

        teacher = FrozenTeacher(model_with_latent_heads, training_cfg=None)

        assert not teacher.teacher_model.training

    def test_frozen_reset_is_noop(self, model_with_latent_heads):
        """FrozenTeacher.reset() should not change weights."""
        from weathergen.train.target_and_aux_ssl_teacher import FrozenTeacher

        teacher = FrozenTeacher(model_with_latent_heads, training_cfg=None)

        # Get weights before reset
        weights_before = {
            k: v.clone() for k, v in teacher.teacher_model.state_dict().items()
        }

        teacher.reset()

        # Weights should be unchanged
        weights_after = teacher.teacher_model.state_dict()
        for key in weights_before:
            assert torch.equal(weights_before[key], weights_after[key])

    def test_frozen_update_is_noop(self, model_with_latent_heads):
        """FrozenTeacher.update_state_post_opt_step() should not change weights."""
        from weathergen.train.target_and_aux_ssl_teacher import FrozenTeacher

        teacher = FrozenTeacher(model_with_latent_heads, training_cfg=None)

        # Get weights before update
        weights_before = {
            k: v.clone() for k, v in teacher.teacher_model.state_dict().items()
        }

        teacher.update_state_post_opt_step(
            istep=10, batch=MockBatch(), model=MagicMock()
        )

        # Weights should be unchanged
        weights_after = teacher.teacher_model.state_dict()
        for key in weights_before:
            assert torch.equal(weights_before[key], weights_after[key])

    def test_frozen_weights_require_no_grad(self, model_with_latent_heads):
        """All FrozenTeacher parameters should have requires_grad=False."""
        from weathergen.train.target_and_aux_ssl_teacher import FrozenTeacher

        teacher = FrozenTeacher(model_with_latent_heads, training_cfg=None)

        for name, param in teacher.teacher_model.named_parameters():
            assert not param.requires_grad, f"Parameter {name} should have requires_grad=False"

    def test_frozen_model_in_eval_mode(self):
        """FrozenTeacher model should always be in eval mode."""
        from weathergen.train.target_and_aux_ssl_teacher import FrozenTeacher

        model = nn.Sequential(
            nn.Linear(10, 10), nn.BatchNorm1d(10), nn.Linear(10, 5)
        )
        # Add latent_heads to model
        model.latent_heads = nn.ModuleDict({"JEPA": nn.Identity()})
        model.train()  # Start in train mode

        teacher = FrozenTeacher(model, training_cfg=None)

        # Model should be in eval mode
        assert not teacher.teacher_model.training
        # All submodules should be in eval mode
        for module in teacher.teacher_model.modules():
            assert not module.training


# =============================================================================
# EncoderTeacher Base Class Tests
# =============================================================================


class TestEncoderTeacherBaseClass:
    """Tests for EncoderTeacher base class functionality."""

    def test_encoder_teacher_exists(self):
        """Verify EncoderTeacher base class exists."""
        from weathergen.train.target_and_aux_ssl_teacher import EncoderTeacher

        assert EncoderTeacher is not None

    def test_ema_teacher_inherits_from_encoder_teacher(self):
        """Verify EMATeacher inherits from EncoderTeacher."""
        from weathergen.train.target_and_aux_ssl_teacher import (
            EMATeacher,
            EncoderTeacher,
        )

        assert issubclass(EMATeacher, EncoderTeacher)

    def test_frozen_teacher_inherits_from_encoder_teacher(self):
        """Verify FrozenTeacher inherits from EncoderTeacher."""
        from weathergen.train.target_and_aux_ssl_teacher import (
            EncoderTeacher,
            FrozenTeacher,
        )

        assert issubclass(FrozenTeacher, EncoderTeacher)

    def test_encoder_teacher_has_forward_teacher_method(self):
        """Verify EncoderTeacher has _forward_teacher method."""
        from weathergen.train.target_and_aux_ssl_teacher import EncoderTeacher

        assert hasattr(EncoderTeacher, "_forward_teacher")
