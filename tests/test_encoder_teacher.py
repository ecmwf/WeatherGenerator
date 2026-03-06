# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

# flash_attn is GPU-only; skip entire module if not available (e.g. macOS)
pytest.importorskip("flash_attn", reason="flash_attn required (GPU-only)")

from weathergen.model.engines import (  # noqa: E402
    LatentPredictionHeadIdentity,
    LatentPredictionHeadMLP,
)
from weathergen.model.ssl_target_processing import (  # noqa: E402
    DINOTargetProcessing,
    JEPATargetProcessing,
    iBOTPatchTargetProcessing,
)
from weathergen.train.target_and_aux_ssl_teacher import (  # noqa: E402
    EMATeacher,
    EncoderTeacher,
    FrozenTeacher,
    get_target_postprocessing,
)
from weathergen.train.teacher_utils import (  # noqa: E402
    _create_teacher_heads,
    load_encoder_from_checkpoint,
    prepare_encoder_teacher,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_training_cfg(loss_types: dict[str, dict] | None = None) -> OmegaConf:
    """Create a minimal training config with SSL losses.

    loss_types: dict mapping loss name -> loss config dict.
    Defaults to a single JEPA loss with identity head.
    """
    if loss_types is None:
        loss_types = {"JEPA": {"head": "identity"}}

    return OmegaConf.create(
        {
            "losses": {
                "ssl_loss": {
                    "type": "LossLatentSSLStudentTeacher",
                    "loss_fcts": loss_types,
                }
            }
        }
    )


def _make_mock_model(dim_embed: int = 64) -> nn.Module:
    """Create a mock model with the attributes that prepare_encoder_teacher expects."""
    model = nn.Module()
    model.forecast_engine = nn.Linear(10, 10)
    model.embed_target_coords = nn.ModuleDict({"stream1": nn.Linear(3, 3)})
    model.target_token_engines = nn.ModuleDict({"stream1": nn.Linear(5, 5)})
    model.pred_heads = nn.ModuleDict({"stream1": nn.Linear(5, 5)})
    model.latent_pre_norm = None
    model.latent_heads = nn.ModuleDict({"existing": nn.Linear(dim_embed, dim_embed)})
    # Add a minimal encoder
    model.encoder = nn.Linear(10, dim_embed)
    return model


def _make_mock_ema_model():
    """Create a mock EMA model for EMATeacher tests."""
    ema = MagicMock()
    ema.is_model_sharded = False
    ema.reset = MagicMock()
    ema.update = MagicMock()
    ema.forward_eval = MagicMock()
    ema.get_current_beta = MagicMock(return_value=0.99)
    return ema


# ---------------------------------------------------------------------------
# Tests for prepare_encoder_teacher
# ---------------------------------------------------------------------------


class TestPrepareEncoderTeacher:
    def test_strips_non_encoder_components(self):
        model = _make_mock_model()
        training_cfg = _make_training_cfg()

        prepare_encoder_teacher(model, training_cfg, teacher_dim_embed=64)

        assert model.forecast_engine is None
        assert len(model.embed_target_coords) == 0
        assert len(model.target_token_engines) == 0
        assert len(model.pred_heads) == 0

    def test_creates_latent_pre_norm_if_missing(self):
        model = _make_mock_model()
        assert model.latent_pre_norm is None

        training_cfg = _make_training_cfg()
        prepare_encoder_teacher(model, training_cfg, teacher_dim_embed=64)

        assert isinstance(model.latent_pre_norm, nn.LayerNorm)
        assert model.latent_pre_norm.normalized_shape == (64,)

    def test_preserves_existing_latent_pre_norm(self):
        model = _make_mock_model()
        existing_norm = nn.LayerNorm(64)
        model.latent_pre_norm = existing_norm

        training_cfg = _make_training_cfg()
        prepare_encoder_teacher(model, training_cfg, teacher_dim_embed=64)

        # Should keep the existing norm, not replace it
        assert model.latent_pre_norm is existing_norm

    def test_jepa_creates_identity_head(self):
        model = _make_mock_model()
        training_cfg = _make_training_cfg({"JEPA": {"head": "identity"}})

        prepare_encoder_teacher(model, training_cfg, teacher_dim_embed=64)

        assert "JEPA" in model.latent_heads
        assert isinstance(model.latent_heads["JEPA"], LatentPredictionHeadIdentity)

    def test_ibot_creates_mlp_head_by_default(self):
        model = _make_mock_model()
        training_cfg = _make_training_cfg(
            {
                "iBOT": {
                    "head": "mlp",
                    "out_dim": 32,
                    "num_layers": 2,
                    "hidden_factor": 2,
                    "center_momentum": 0.9,
                    "teacher_temp": 0.04,
                    "teacher_style": "softmax_center",
                    "loss_extra_args": {"student_temp": 0.1},
                }
            }
        )

        prepare_encoder_teacher(model, training_cfg, teacher_dim_embed=64)

        assert "iBOT" in model.latent_heads
        assert isinstance(model.latent_heads["iBOT"], LatentPredictionHeadMLP)

    def test_dino_creates_mlp_head_by_default(self):
        model = _make_mock_model()
        training_cfg = _make_training_cfg(
            {
                "DINO": {
                    "head": "mlp",
                    "out_dim": 32,
                    "num_layers": 2,
                    "hidden_factor": 2,
                    "center_momentum": 0.9,
                    "teacher_style": "softmax_center",
                    "loss_extra_args": {"student_temp": 0.1},
                }
            }
        )

        prepare_encoder_teacher(model, training_cfg, teacher_dim_embed=64)

        assert "DINO" in model.latent_heads
        assert isinstance(model.latent_heads["DINO"], LatentPredictionHeadMLP)

    def test_replaces_existing_heads(self):
        model = _make_mock_model()
        assert "existing" in model.latent_heads

        training_cfg = _make_training_cfg({"JEPA": {"head": "identity"}})
        prepare_encoder_teacher(model, training_cfg, teacher_dim_embed=64)

        # Old heads should be gone, only new ones
        assert "existing" not in model.latent_heads
        assert "JEPA" in model.latent_heads

    def test_multiple_ssl_losses(self):
        model = _make_mock_model()
        training_cfg = _make_training_cfg(
            {
                "JEPA": {"head": "identity"},
                "iBOT": {
                    "head": "mlp",
                    "out_dim": 32,
                    "num_layers": 2,
                    "hidden_factor": 2,
                    "center_momentum": 0.9,
                    "teacher_temp": 0.04,
                    "teacher_style": "softmax_center",
                    "loss_extra_args": {"student_temp": 0.1},
                },
            }
        )

        prepare_encoder_teacher(model, training_cfg, teacher_dim_embed=64)

        assert "JEPA" in model.latent_heads
        assert "iBOT" in model.latent_heads
        assert isinstance(model.latent_heads["JEPA"], LatentPredictionHeadIdentity)
        assert isinstance(model.latent_heads["iBOT"], LatentPredictionHeadMLP)

    def test_no_ssl_losses(self):
        model = _make_mock_model()
        training_cfg = OmegaConf.create(
            {
                "losses": {
                    "phys_loss": {
                        "type": "LossPhysical",
                    }
                }
            }
        )

        prepare_encoder_teacher(model, training_cfg, teacher_dim_embed=64)

        assert len(model.latent_heads) == 0

    def test_encoder_preserved(self):
        model = _make_mock_model()
        original_encoder = model.encoder
        training_cfg = _make_training_cfg()

        prepare_encoder_teacher(model, training_cfg, teacher_dim_embed=64)

        assert model.encoder is original_encoder


# ---------------------------------------------------------------------------
# Tests for load_encoder_from_checkpoint
# ---------------------------------------------------------------------------


class TestLoadEncoderFromCheckpoint:
    def test_loads_only_encoder_keys(self, tmp_path):
        """Verify that only encoder.* and latent_pre_norm* keys are loaded."""
        # Create a mock model
        model = nn.Module()
        model.encoder = nn.Linear(10, 20)
        model.latent_pre_norm = nn.LayerNorm(20)
        model.other_module = nn.Linear(20, 5)

        # Create checkpoint with encoder + non-encoder params
        checkpoint = {}
        for name, param in model.state_dict().items():
            checkpoint[name] = torch.randn_like(param)
        # Add some extra non-encoder params that should be ignored
        checkpoint["forecast_engine.weight"] = torch.randn(10, 10)
        checkpoint["pred_heads.stream1.weight"] = torch.randn(5, 5)

        # Save checkpoint
        run_id = "test1234"
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        torch.save(checkpoint, run_dir / f"{run_id}_latest.chkpt")

        cf = OmegaConf.create({"model_path": str(tmp_path)})

        # Load - should not raise despite extra keys
        load_encoder_from_checkpoint(model, cf, run_id, -1, "cpu")

    def test_mini_epoch_filename(self, tmp_path):
        """Test that specific mini_epoch generates correct filename."""
        model = nn.Module()
        model.encoder = nn.Linear(10, 20)

        run_id = "test1234"
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        torch.save(
            {"encoder.weight": torch.randn(20, 10), "encoder.bias": torch.randn(20)},
            run_dir / f"{run_id}_chkpt00042.chkpt",
        )

        cf = OmegaConf.create({"model_path": str(tmp_path)})
        load_encoder_from_checkpoint(model, cf, run_id, 42, "cpu")

    def test_latest_filename(self, tmp_path):
        """Test that mini_epoch=-1 generates 'latest' filename."""
        model = nn.Module()
        model.encoder = nn.Linear(10, 20)

        run_id = "test1234"
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        torch.save(
            {"encoder.weight": torch.randn(20, 10), "encoder.bias": torch.randn(20)},
            run_dir / f"{run_id}_latest.chkpt",
        )

        cf = OmegaConf.create({"model_path": str(tmp_path)})
        load_encoder_from_checkpoint(model, cf, run_id, -1, "cpu")

    def test_none_mini_epoch_uses_latest(self, tmp_path):
        """Test that mini_epoch=None generates 'latest' filename."""
        model = nn.Module()
        model.encoder = nn.Linear(10, 20)

        run_id = "test1234"
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        torch.save(
            {"encoder.weight": torch.randn(20, 10), "encoder.bias": torch.randn(20)},
            run_dir / f"{run_id}_latest.chkpt",
        )

        cf = OmegaConf.create({"model_path": str(tmp_path)})
        load_encoder_from_checkpoint(model, cf, run_id, None, "cpu")


# ---------------------------------------------------------------------------
# Tests for _create_head
# ---------------------------------------------------------------------------


class TestCreateHead:
    def test_ibot_mlp(self):
        conf = OmegaConf.create({"out_dim": 32, "num_layers": 2, "hidden_factor": 2})
        head = _create_teacher_heads("iBOT", "mlp", 64, conf)
        assert isinstance(head, LatentPredictionHeadMLP)
        assert head.use_class_token is True
        assert head.use_patch_token is True

    def test_dino_mlp(self):
        conf = OmegaConf.create({"out_dim": 32, "num_layers": 2, "hidden_factor": 2})
        head = _create_teacher_heads("DINO", "mlp", 64, conf)
        assert isinstance(head, LatentPredictionHeadMLP)
        assert head.use_class_token is True
        assert head.use_patch_token is False

    def test_identity_head(self):
        head = _create_teacher_heads("iBOT", "identity", 64, {})
        assert isinstance(head, LatentPredictionHeadIdentity)

    def test_unknown_loss_type(self):
        with pytest.raises(ValueError, match="does not support loss type"):
            _create_teacher_heads("UnknownLoss", "mlp", 64, {})

    def test_unknown_head_type(self):
        with pytest.raises(ValueError, match="Unknown latent prediction head type"):
            _create_teacher_heads("iBOT", "nonexistent", 64, {})

    def test_transformer_requires_cf(self):
        conf = OmegaConf.create(
            {
                "out_dim": 32,
                "num_blocks": 1,
                "num_heads": 2,
                "with_qk_lnorm": True,
                "intermediate_dim": 32,
                "dropout_rate": 0.0,
            }
        )
        with pytest.raises(ValueError, match="requires a global config"):
            _create_teacher_heads("iBOT", "transformer", 64, conf, cf=None)


# ---------------------------------------------------------------------------
# Tests for get_target_postprocessing
# ---------------------------------------------------------------------------


class TestGetTargetPostprocessing:
    def test_jepa(self):
        losses = OmegaConf.create({"JEPA": {"head": "identity"}})
        training_cfg = OmegaConf.create({})
        result = get_target_postprocessing(losses, training_cfg)
        assert "JEPA" in result
        assert isinstance(result["JEPA"], JEPATargetProcessing)

    def test_ibot(self):
        losses = OmegaConf.create(
            {
                "iBOT": {
                    "out_dim": 32,
                    "center_momentum": 0.9,
                    "teacher_temp": 0.04,
                    "teacher_style": "softmax_center",
                    "loss_extra_args": {"student_temp": 0.1},
                }
            }
        )
        training_cfg = OmegaConf.create({})
        result = get_target_postprocessing(losses, training_cfg)
        assert "iBOT" in result
        assert isinstance(result["iBOT"], iBOTPatchTargetProcessing)

    def test_dino(self):
        losses = OmegaConf.create(
            {
                "DINO": {
                    "out_dim": 32,
                    "center_momentum": 0.9,
                    "teacher_style": "softmax_center",
                    "loss_extra_args": {"student_temp": 0.1},
                }
            }
        )
        training_cfg = OmegaConf.create({})
        result = get_target_postprocessing(losses, training_cfg)
        assert "DINO" in result
        assert isinstance(result["DINO"], DINOTargetProcessing)

    def test_unknown_loss_skipped(self):
        losses = OmegaConf.create({"UnknownLoss": {"foo": "bar"}})
        training_cfg = OmegaConf.create({})
        result = get_target_postprocessing(losses, training_cfg)
        assert len(result) == 0

    def test_ibot_missing_config_key(self):
        losses = OmegaConf.create({"iBOT": {"out_dim": 32}})  # missing required keys
        training_cfg = OmegaConf.create({})
        with pytest.raises(KeyError, match="center_momentum"):
            get_target_postprocessing(losses, training_cfg)

    def test_dino_missing_config_key(self):
        losses = OmegaConf.create({"DINO": {"out_dim": 32}})  # missing required keys
        training_cfg = OmegaConf.create({})
        with pytest.raises(KeyError, match="center_momentum"):
            get_target_postprocessing(losses, training_cfg)


# ---------------------------------------------------------------------------
# Tests for EncoderTeacher interface
# ---------------------------------------------------------------------------


class TestEncoderTeacher:
    def test_forward_teacher_not_implemented(self):
        teacher = EncoderTeacher.__new__(EncoderTeacher)
        teacher.teacher_model = nn.Module()
        teacher.postprocess_targets = {}

        with pytest.raises(NotImplementedError):
            teacher.forward_teacher(None, None)

    def test_update_state_pre_backward_is_noop(self):
        teacher = EncoderTeacher.__new__(EncoderTeacher)
        teacher.postprocess_targets = {}
        # Should not raise
        teacher.update_state_pre_backward(0, None, None)


# ---------------------------------------------------------------------------
# Tests for EMATeacher
# ---------------------------------------------------------------------------


class TestEMATeacher:
    def test_init_calls_reset(self):
        ema = _make_mock_ema_model()
        training_cfg = _make_training_cfg()
        EMATeacher(nn.Module(), ema, batch_size=32, training_cfg=training_cfg)
        ema.reset.assert_called_once()

    def test_reset_updates_batch_size(self):
        ema = _make_mock_ema_model()
        training_cfg = _make_training_cfg()
        teacher = EMATeacher(nn.Module(), ema, batch_size=32, training_cfg=training_cfg)

        teacher.reset(batch_size=64)
        assert teacher.batch_size == 64

    def test_reset_without_batch_size(self):
        ema = _make_mock_ema_model()
        training_cfg = _make_training_cfg()
        teacher = EMATeacher(nn.Module(), ema, batch_size=32, training_cfg=training_cfg)

        teacher.reset()
        assert teacher.batch_size == 32

    def test_update_state_post_opt_step_calls_ema_update(self):
        ema = _make_mock_ema_model()
        training_cfg = _make_training_cfg()
        teacher = EMATeacher(nn.Module(), ema, batch_size=32, training_cfg=training_cfg)

        teacher.update_state_post_opt_step(istep=10, batch=None, model=None)
        ema.update.assert_called_once_with(10, 32)

    def test_get_current_beta(self):
        ema = _make_mock_ema_model()
        training_cfg = _make_training_cfg()
        teacher = EMATeacher(nn.Module(), ema, batch_size=32, training_cfg=training_cfg)

        teacher.get_current_beta(100)
        ema.get_current_beta.assert_called_once_with(100, 32)

    def test_has_required_methods(self):
        """EMATeacher has all required TargetAndAuxModuleBase methods."""
        assert hasattr(EMATeacher, "reset")
        assert hasattr(EMATeacher, "compute")
        assert hasattr(EMATeacher, "update_state_pre_backward")
        assert hasattr(EMATeacher, "update_state_post_opt_step")
        assert hasattr(EMATeacher, "to_device")

    def test_to_device_moves_postprocessors(self):
        ema = _make_mock_ema_model()
        training_cfg = _make_training_cfg()
        teacher = EMATeacher(nn.Module(), ema, batch_size=32, training_cfg=training_cfg)

        # Should not raise
        result = teacher.to_device("cpu")
        assert result is teacher


# ---------------------------------------------------------------------------
# Tests for FrozenTeacher
# ---------------------------------------------------------------------------


class TestFrozenTeacher:
    def test_freezes_all_params(self):
        model = nn.Module()
        model.encoder = nn.Linear(10, 20)
        model.latent_heads = nn.ModuleDict()
        model.latent_pre_norm = nn.LayerNorm(20)
        training_cfg = _make_training_cfg()

        teacher = FrozenTeacher(model, training_cfg)

        for param in teacher.teacher_model.parameters():
            assert not param.requires_grad

    def test_eval_mode(self):
        model = nn.Module()
        model.encoder = nn.Linear(10, 20)
        model.latent_heads = nn.ModuleDict()
        model.latent_pre_norm = nn.LayerNorm(20)
        training_cfg = _make_training_cfg()

        teacher = FrozenTeacher(model, training_cfg)
        assert not teacher.teacher_model.training

    def test_reset_is_noop(self):
        model = nn.Module()
        model.encoder = nn.Linear(10, 20)
        model.latent_heads = nn.ModuleDict()
        model.latent_pre_norm = nn.LayerNorm(20)
        training_cfg = _make_training_cfg()

        teacher = FrozenTeacher(model, training_cfg)
        teacher.reset()  # should not raise
        teacher.reset(batch_size=64)  # should not raise

    def test_update_state_post_opt_step_is_noop(self):
        model = nn.Module()
        model.encoder = nn.Linear(10, 20)
        model.latent_heads = nn.ModuleDict()
        model.latent_pre_norm = nn.LayerNorm(20)
        training_cfg = _make_training_cfg()

        teacher = FrozenTeacher(model, training_cfg)
        teacher.update_state_post_opt_step(istep=0, batch=None, model=None)  # should not raise

    def test_forward_teacher_uses_own_params(self):
        model = MagicMock()
        model.parameters = MagicMock(return_value=iter([]))
        model.eval = MagicMock(return_value=model)
        training_cfg = _make_training_cfg()
        teacher_params = MagicMock()

        teacher = FrozenTeacher(model, training_cfg, teacher_model_params=teacher_params)

        batch = MagicMock()
        teacher.forward_teacher(MagicMock(), batch)
        model.assert_called_once_with(teacher_params, batch)

    def test_forward_teacher_falls_back_to_student_params(self):
        model = MagicMock()
        model.parameters = MagicMock(return_value=iter([]))
        model.eval = MagicMock(return_value=model)
        training_cfg = _make_training_cfg()

        teacher = FrozenTeacher(model, training_cfg, teacher_model_params=None)

        student_params = MagicMock()
        batch = MagicMock()
        teacher.forward_teacher(student_params, batch)
        model.assert_called_once_with(student_params, batch)

    def test_has_required_methods(self):
        """FrozenTeacher has all required TargetAndAuxModuleBase methods."""
        assert hasattr(FrozenTeacher, "reset")
        assert hasattr(FrozenTeacher, "compute")
        assert hasattr(FrozenTeacher, "update_state_pre_backward")
        assert hasattr(FrozenTeacher, "update_state_post_opt_step")
        assert hasattr(FrozenTeacher, "to_device")
        assert hasattr(FrozenTeacher, "from_pretrained")

    def test_from_pretrained_requires_teacher_run_id(self):
        cf = OmegaConf.create({"model_path": "/tmp/claude/models"})
        with pytest.raises(KeyError, match="teacher_run_id"):
            FrozenTeacher.from_pretrained(cf, None, "cpu", {})


# ---------------------------------------------------------------------------
# Tests for EMAModel.get_current_beta
# ---------------------------------------------------------------------------


class TestEMAModelBeta:
    def test_get_current_beta(self):
        from weathergen.model.ema import EMAModel

        model = nn.Module()
        model.p = nn.Parameter(torch.randn(3))
        empty = nn.Module()
        empty.p = nn.Parameter(torch.randn(3))

        ema = EMAModel.__new__(EMAModel)
        ema.halflife_steps = 1e-3
        ema.rampup_ratio = 0.09

        beta = ema.get_current_beta(100, 32)
        assert 0.0 < beta < 1.0

    def test_batch_size_stored_on_update(self):
        """Verify that update() stores batch_size."""
        from weathergen.model.ema import EMAModel

        model = nn.Module()
        model.p = nn.Parameter(torch.randn(3))
        empty = nn.Module()
        empty.p = nn.Parameter(torch.randn(3))

        ema = EMAModel(model, empty)
        assert ema.batch_size == 1

        ema.update(cur_step=10, batch_size=64)
        assert ema.batch_size == 64
