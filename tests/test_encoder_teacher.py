# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from test_multi_resolution_data import make_model_and_batch, make_model_config

# flash_attn is GPU-only; skip entire module if not available (e.g. macOS)
pytest.importorskip("flash_attn", reason="flash_attn required (GPU-only)")

from weathergen.datasets.batch import MultiBatch  # noqa: E402
from weathergen.model.ema import EMAModel  # noqa: E402
from weathergen.model.engines import (  # noqa: E402
    IdentityEngine,
    LatentPredictionHeadIdentity,
    LatentPredictionHeadMLP,
)
from weathergen.model.model_interface import load_model  # noqa: E402
from weathergen.model.ssl_target_processing import (  # noqa: E402
    DINOTargetProcessing,
    JEPATargetProcessing,
    iBOTPatchTargetProcessing,
)
from weathergen.train.target_and_aux_ssl_teacher import (  # noqa: E402
    EMATeacher,
    FrozenTeacher,
    get_target_postprocessing,
)
from weathergen.train.teacher_utils import (  # noqa: E402
    _create_teacher_heads,
    check_teacher_grid_compatibility,
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


@pytest.fixture
def teacher_setup():
    torch.manual_seed(42)
    cf = make_model_config()
    cf.training_config.losses = _make_training_cfg().losses
    # Construct real forecast weights, then disable them through teacher preparation.
    cf.fe_num_blocks = 1
    model, params, batch = make_model_and_batch(cf, input_steps=2)
    prepare_encoder_teacher(model, cf.training_config, override_cfg=cf)
    return cf, model, params, batch


def _target_view(model, batch, indices):
    return MultiBatch(
        batch.batch.as_target_input(indices),
        {name: list(config.streams) for name, config in model.encoder_configs.items()},
    )


def _common_reference(model, params, target_batch):
    """Independently sum native states after time reduction and nested child pooling."""
    native, _ = model.encoder(params, target_batch)
    common = []
    for name, tokens in native.items():
        samples = target_batch.batches[name].get_source_samples()
        tokens = tokens.reshape(
            samples.get_num_source_steps(), len(samples), *tokens.shape[1:]
        ).sum(0)
        spatial = tokens[:, 2:].reshape(len(samples), 12, -1, 32).mean(2)
        common.append(torch.cat((tokens[:, :2], spatial), dim=1))
    return sum(common)


@pytest.mark.parametrize("indices", [[1], [1, 0]])
def test_prepared_teacher_common_latents_and_metadata(teacher_setup, indices):
    cf, model, params, batch = teacher_setup
    assert set(model.forecast_engines) == {"0", "1"}
    assert all(isinstance(engine, IdentityEngine) for engine in model.forecast_engines.values())
    assert not model.pred_heads
    teacher = FrozenTeacher(model, cf.training_config, params)
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())

    # First-metadata semantics must project fine masks without mutating native masks.
    fine_name = next(
        name for name, stream in cf.streams.items() if stream.encoder == "observations"
    )
    originals = []
    for index, sample in enumerate(batch.batch.target_samples.samples):
        fine_meta = sample.meta_info.pop(fine_name)
        fine_meta.mask = torch.arange(48) % (5 + index) == 0
        fine_meta.global_params = {"sample": index}
        sample.meta_info = {fine_name: fine_meta, **sample.meta_info}
        originals.append(fine_meta.mask.clone())

    selected = _target_view(model, batch, indices)
    with torch.no_grad():
        common = _common_reference(model, params, selected)
        output = teacher.forward_teacher(None, batch, indices)
        result = teacher.compute(0, batch, params, None, indices)
    for step in (0, 1):
        torch.testing.assert_close(
            output.latent[step]["latent_state"].z_pre_norm, common, rtol=1e-5, atol=1e-4
        )
    expected = model.latent_pre_norm(common)[:, 2:]
    torch.testing.assert_close(result.latent["JEPA"], expected)
    assert expected.shape == (len(indices), 12, 32)
    for metadata, index in zip(result.aux_outputs, indices, strict=True):
        torch.testing.assert_close(metadata.mask, originals[index].reshape(12, 4).any(1))
        assert metadata.global_params == {"sample": index}
        original = batch.batch.target_samples.samples[index].meta_info[fine_name]
        assert metadata.params is original.params
        torch.testing.assert_close(original.mask, originals[index])


def test_frozen_teacher_uses_own_parameters(teacher_setup):
    cf, _, student_params, batch = teacher_setup
    teacher_cf = deepcopy(cf)
    teacher_cf.encoders.renamed = teacher_cf.encoders.pop("observations")
    for stream in teacher_cf.streams.values():
        if stream.encoder == "observations":
            stream.encoder = "renamed"
    model, own_params, _ = make_model_and_batch(teacher_cf, input_steps=2)
    prepare_encoder_teacher(model, cf.training_config, override_cfg=teacher_cf)
    teacher = FrozenTeacher(model, cf.training_config, own_params)
    with torch.no_grad():
        expected = model(own_params, _target_view(model, batch, [1]))
        # Student params lack the renamed encoder; only the teacher's own params can work.
        actual = teacher.forward_teacher(student_params, batch, [1])
    torch.testing.assert_close(actual.latent[0]["JEPA"], expected.latent[0]["JEPA"])


def test_frozen_teacher_falls_back_to_student_parameters(teacher_setup):
    cf, model, params, batch = teacher_setup
    teacher = FrozenTeacher(model, cf.training_config)
    with torch.no_grad():
        expected = model(params, _target_view(model, batch, [1, 0]))
        actual = teacher.forward_teacher(params, batch, [1, 0])
    torch.testing.assert_close(actual.latent[0]["JEPA"], expected.latent[0]["JEPA"])


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
        original_other = {
            name: value.clone() for name, value in model.other_module.state_dict().items()
        }

        # Create checkpoint with encoder + non-encoder params
        checkpoint = {}
        for name, param in model.state_dict().items():
            checkpoint[name] = torch.randn_like(param)
        # Add some extra non-encoder params that should be ignored
        checkpoint["forecast_engines.0.weight"] = torch.randn(10, 10)
        checkpoint["decoders.physical.pred_heads.stream1.weight"] = torch.randn(5, 5)

        # Save checkpoint
        run_id = "test1234"
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        torch.save(checkpoint, run_dir / f"{run_id}_latest.chkpt")

        cf = OmegaConf.create({"model_path": str(tmp_path)})

        # Load - should not raise despite extra keys
        load_encoder_from_checkpoint(model, cf, run_id, -1, "cpu")
        for name, value in model.encoder.state_dict().items():
            torch.testing.assert_close(value, checkpoint[f"encoder.{name}"])
        for name, value in model.latent_pre_norm.state_dict().items():
            torch.testing.assert_close(value, checkpoint[f"latent_pre_norm.{name}"])
        for name, value in model.other_module.state_dict().items():
            torch.testing.assert_close(value, original_other[name])

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


@pytest.mark.parametrize("prefix", ["", "module."])
def test_full_checkpoint_rejects_single_forecaster(tmp_path, prefix):
    run_id = "legacy"
    path = tmp_path / f"{run_id}_latest.chkpt"
    torch.save({f"{prefix}forecast_engine.fe_blocks.weight": torch.ones(1)}, path)
    cf = OmegaConf.create(
        {
            "full_model_path": str(tmp_path),
            "with_ddp": False,
            "with_fsdp": False,
        }
    )
    with pytest.raises(ValueError) as error:
        load_model(cf, nn.Linear(1, 1), "cpu", run_id)
    assert str(error.value) == (
        "Checkpoint uses a single forecast_engine; load encoder-only weights or use a "
        "checkpoint with forecast_engines."
    )


@pytest.mark.parametrize("mismatch", ["stream", "common"])
def test_teacher_grid_compatibility(mismatch):
    cf = make_model_config()
    teacher_cf = deepcopy(cf)
    if mismatch == "stream":
        teacher_cf.encoders.observations.healpix_level = 2
    else:
        # Keep stream grids unchanged; an unused coarser encoder still sets the common grid.
        cf.encoders.default.healpix_level = 1
        teacher_cf = deepcopy(cf)
        teacher_cf.encoders.unused = {"healpix_level": 0}
    with pytest.raises(ValueError) as error:
        check_teacher_grid_compatibility(cf, teacher_cf)
    assert str(error.value) == (
        "Teacher and student must use matching per-stream and common HEALPix levels."
    )


def test_teacher_grid_compatibility_allows_encoder_regrouping():
    cf = make_model_config()
    teacher_cf = deepcopy(cf)
    teacher_cf.encoders.renamed = teacher_cf.encoders.pop("observations")
    for stream in teacher_cf.streams.values():
        if stream.encoder == "observations":
            stream.encoder = "renamed"
    check_teacher_grid_compatibility(cf, teacher_cf)


def test_ema_construction_rejects_changed_input_grid():
    from weathergen.train.target_and_aux_utils import get_target_aux_calculator

    cf = make_model_config()
    loss_cfg = OmegaConf.create(
        {
            "target_and_aux_calc": {
                "EMATeacher": {
                    "model_param_overrides": {"encoders": {"observations": {"healpix_level": 2}}},
                }
            }
        }
    )
    with pytest.raises(ValueError, match="matching per-stream and common HEALPix levels"):
        get_target_aux_calculator(cf, loss_cfg, None, None, "cpu", 2)


def test_frozen_construction_rejects_changed_input_grid(monkeypatch):
    cf = make_model_config()
    teacher_cf = deepcopy(cf)
    teacher_cf.encoders.observations.healpix_level = 2
    monkeypatch.setattr(
        "weathergen.train.target_and_aux_ssl_teacher.load_run_config",
        lambda *args, **kwargs: teacher_cf,
    )
    with pytest.raises(ValueError, match="matching per-stream and common HEALPix levels"):
        FrozenTeacher.from_pretrained(cf, None, "cpu", {"teacher_run_id": "different_grid"})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="EMA lifecycle requires CUDA")
def test_real_ema_teacher_targets_and_update():
    cf = make_model_config()
    cf.training_config.losses = _make_training_cfg().losses
    student, params, batch = make_model_and_batch(cf)
    student, params, batch = student.cuda(), params.cuda(), batch.to_device("cuda")
    teacher_model = deepcopy(student)
    prepare_encoder_teacher(teacher_model, cf.training_config, override_cfg=cf)
    ema = EMAModel(student, teacher_model, halflife_steps=100, rampup_ratio=None)
    teacher = EMATeacher(student, ema, batch_size=2, training_cfg=cf.training_config)
    output = teacher.compute(0, batch, params, student, [1])
    with torch.no_grad():
        expected = student(params, _target_view(student, batch, [1]))
    torch.testing.assert_close(output.latent["JEPA"], expected.latent[0]["JEPA"])
    name, target_parameter = next(
        (name, value)
        for name, value in teacher_model.named_parameters()
        if "q_cells" not in name and "identity" not in name.lower()
    )
    original = target_parameter.clone()
    with torch.no_grad():
        dict(student.named_parameters())[name].add_(1)
    teacher.update_state_post_opt_step(10, batch, student)
    beta = teacher.get_current_beta(10)
    torch.testing.assert_close(target_parameter, original + (1 - beta))
    assert 0 < beta < 1


def test_ema_beta_uses_stored_batch_size():
    # No model allocation or CUDA is needed to exercise the schedule.
    ema = EMAModel.__new__(EMAModel)
    ema.halflife_steps = 100
    ema.rampup_ratio = None
    ema.batch_size = 32
    assert ema.get_current_beta(100) == pytest.approx(0.5 ** (32 / 100))
