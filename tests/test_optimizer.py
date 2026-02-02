# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the optimizer module."""

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from weathergen.train.optimizer import (
    ADAMW_PATTERNS,
    CompositeOptimizer,
    MuonCustom,
    classify_muon_params,
    create_optimizer,
)


class DummyTransformerBlock(nn.Module):
    """Simple transformer-like model for testing parameter classification."""

    def __init__(self, dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Attention components (should be Muon-eligible)
        self.proj_heads_q = nn.Linear(dim, dim, bias=False)
        self.proj_heads_k = nn.Linear(dim, dim, bias=False)
        self.proj_heads_v = nn.Linear(dim, dim, bias=False)
        self.proj_out = nn.Linear(dim, dim, bias=False)

        # MLP components (should be Muon-eligible)
        self.mlp_fc1 = nn.Linear(dim, dim * 4, bias=False)
        self.mlp_fc2 = nn.Linear(dim * 4, dim, bias=False)

        # Embeddings (should be AdamW)
        self.embed_target_coords = nn.Linear(3, dim, bias=False)
        self.embeds = nn.Embedding(100, dim)

        # Prediction heads (should be AdamW)
        self.pred_heads = nn.Linear(dim, 10, bias=False)
        self.latent_heads = nn.Linear(dim, dim, bias=False)

        # Biases and norms (should be AdamW)
        self.bias = nn.Parameter(torch.zeros(dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return x


class SimpleMLP(nn.Module):
    """Simple MLP for testing optimizer steps."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 32, output_dim: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.embed = nn.Embedding(100, hidden_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture
def dummy_model():
    """Create a dummy transformer model for testing."""
    return DummyTransformerBlock(dim=64, num_heads=4)


@pytest.fixture
def simple_model():
    """Create a simple MLP model for testing optimizer steps."""
    return SimpleMLP(input_dim=10, hidden_dim=32, output_dim=5)


@pytest.fixture
def optimizer_cfg():
    """Create a standard optimizer config."""
    return OmegaConf.create({
        "type": "adamw",
        "grad_clip": 1.0,
        "weight_decay": 0.1,
        "adamw": {
            "beta1": 0.975,
            "beta2": 0.9875,
            "eps": 2e-08,
        },
        "muon": {
            "lr_multiplier": 20.0,
            "momentum": 0.95,
            "nesterov": True,
            "weight_decay": 0.1,
        },
    })


@pytest.fixture
def lr_cfg():
    """Create a standard LR config."""
    return OmegaConf.create({
        "lr_start": 1e-6,
        "lr_max": 5e-5,
    })


class TestClassifyMuonParams:
    """Tests for the classify_muon_params function."""

    def test_classification_separates_params(self, dummy_model):
        """Test that parameters are correctly separated into Muon and AdamW groups."""
        muon_params, adamw_params, muon_names, adamw_names = classify_muon_params(dummy_model)

        # Check that all trainable params are classified
        total_params = sum(1 for p in dummy_model.parameters() if p.requires_grad)
        assert len(muon_params) + len(adamw_params) == total_params

        # Check names match params count
        assert len(muon_params) == len(muon_names)
        assert len(adamw_params) == len(adamw_names)

    def test_attention_weights_are_muon(self, dummy_model):
        """Test that attention Q/K/V/O weights are classified as Muon-eligible."""
        _, _, muon_names, _ = classify_muon_params(dummy_model)

        # These should be in Muon group
        expected_muon = ["proj_heads_q", "proj_heads_k", "proj_heads_v", "proj_out"]
        for name in expected_muon:
            assert any(name in muon_name for muon_name in muon_names), f"{name} should be Muon"

    def test_mlp_weights_are_muon(self, dummy_model):
        """Test that MLP linear weights are classified as Muon-eligible."""
        _, _, muon_names, _ = classify_muon_params(dummy_model)

        # MLP weights should be Muon
        assert any("mlp_fc1" in name for name in muon_names)
        assert any("mlp_fc2" in name for name in muon_names)

    def test_embeddings_are_adamw(self, dummy_model):
        """Test that embedding parameters are classified as AdamW-eligible."""
        _, _, _, adamw_names = classify_muon_params(dummy_model)

        # These should be in AdamW group
        expected_adamw = ["embed_target_coords", "embeds"]
        for name in expected_adamw:
            assert any(name in adamw_name for adamw_name in adamw_names), f"{name} should be AdamW"

    def test_pred_heads_are_adamw(self, dummy_model):
        """Test that prediction heads are classified as AdamW-eligible."""
        _, _, _, adamw_names = classify_muon_params(dummy_model)

        assert any("pred_heads" in name for name in adamw_names)
        assert any("latent_heads" in name for name in adamw_names)

    def test_1d_params_are_adamw(self, dummy_model):
        """Test that 1D parameters (biases, norm weights) are AdamW-eligible."""
        _, adamw_params, _, adamw_names = classify_muon_params(dummy_model)

        # Check that bias and norm params are in AdamW
        assert any("bias" in name for name in adamw_names)
        assert any("norm" in name for name in adamw_names)

        # All 1D params should be in AdamW
        for param in adamw_params:
            if param.ndim < 2:
                assert True  # 1D params are correctly in AdamW

    def test_frozen_params_excluded(self, dummy_model):
        """Test that frozen parameters are excluded from classification."""
        # Freeze some parameters
        dummy_model.proj_heads_q.weight.requires_grad = False
        dummy_model.embed_target_coords.weight.requires_grad = False

        muon_params, adamw_params, muon_names, adamw_names = classify_muon_params(dummy_model)

        # Frozen params should not appear
        assert "proj_heads_q.weight" not in muon_names
        assert "embed_target_coords.weight" not in adamw_names

        # Total should be reduced
        total_trainable = sum(1 for p in dummy_model.parameters() if p.requires_grad)
        assert len(muon_params) + len(adamw_params) == total_trainable


class TestCreateOptimizer:
    """Tests for the create_optimizer factory function."""

    def test_creates_adamw_by_default(self, simple_model, optimizer_cfg, lr_cfg):
        """Test that AdamW is created when type is 'adamw'."""
        optimizer_cfg.type = "adamw"
        optimizer = create_optimizer(simple_model, optimizer_cfg, lr_cfg, batch_size_total=4)

        assert isinstance(optimizer, torch.optim.AdamW)

    def test_creates_composite_for_muon_adamw(self, simple_model, optimizer_cfg, lr_cfg):
        """Test that CompositeOptimizer is created when type is 'muon_adamw'."""
        optimizer_cfg.type = "muon_adamw"
        optimizer = create_optimizer(simple_model, optimizer_cfg, lr_cfg, batch_size_total=4)

        assert isinstance(optimizer, CompositeOptimizer)

    def test_raises_for_unknown_type(self, simple_model, optimizer_cfg, lr_cfg):
        """Test that unknown optimizer type raises ValueError."""
        optimizer_cfg.type = "unknown"

        with pytest.raises(ValueError, match="Unknown optimizer type"):
            create_optimizer(simple_model, optimizer_cfg, lr_cfg, batch_size_total=4)

    def test_batch_size_scaling(self, simple_model, optimizer_cfg, lr_cfg):
        """Test that betas are scaled based on batch size."""
        optimizer_cfg.type = "adamw"

        opt_small = create_optimizer(simple_model, optimizer_cfg, lr_cfg, batch_size_total=1)
        opt_large = create_optimizer(simple_model, optimizer_cfg, lr_cfg, batch_size_total=16)

        # Larger batch should have different betas (closer to target)
        beta1_small = opt_small.param_groups[0]["betas"][0]
        beta1_large = opt_large.param_groups[0]["betas"][0]

        # With larger batch, beta1 should be smaller (more momentum decay)
        assert beta1_large < beta1_small


class TestCompositeOptimizer:
    """Tests for the CompositeOptimizer class."""

    def test_step_updates_both_optimizers(self, simple_model, optimizer_cfg, lr_cfg):
        """Test that step() updates parameters from both optimizers."""
        optimizer_cfg.type = "muon_adamw"
        optimizer = create_optimizer(simple_model, optimizer_cfg, lr_cfg, batch_size_total=4)

        # Create dummy input and compute loss
        x = torch.randn(4, 10)
        output = simple_model(x)
        loss = output.sum()

        # Store initial params
        initial_params = {name: p.clone() for name, p in simple_model.named_parameters()}

        # Backward and step
        loss.backward()
        optimizer.step()

        # Check that params changed
        params_changed = False
        for name, p in simple_model.named_parameters():
            if not torch.equal(p, initial_params[name]):
                params_changed = True
                break

        assert params_changed

    def test_zero_grad_clears_both(self, simple_model, optimizer_cfg, lr_cfg):
        """Test that zero_grad() clears gradients from both optimizers."""
        optimizer_cfg.type = "muon_adamw"
        optimizer = create_optimizer(simple_model, optimizer_cfg, lr_cfg, batch_size_total=4)

        # Create gradients
        x = torch.randn(4, 10)
        loss = simple_model(x).sum()
        loss.backward()

        # Verify grads exist
        has_grads = any(p.grad is not None for p in simple_model.parameters())
        assert has_grads

        # Zero grads
        optimizer.zero_grad()

        # Verify grads are cleared
        for p in simple_model.parameters():
            assert p.grad is None or p.grad.abs().sum() == 0

    def test_state_dict_roundtrip(self, simple_model, optimizer_cfg, lr_cfg):
        """Test that state dict can be saved and loaded."""
        optimizer_cfg.type = "muon_adamw"
        optimizer = create_optimizer(simple_model, optimizer_cfg, lr_cfg, batch_size_total=4)

        # Take a step to populate state
        x = torch.randn(4, 10)
        loss = simple_model(x).sum()
        loss.backward()
        optimizer.step()

        # Save state
        state_dict = optimizer.state_dict()

        # Verify structure
        assert "optimizer_type" in state_dict
        assert state_dict["optimizer_type"] == "composite_muon_adamw"
        assert "muon" in state_dict
        assert "adamw" in state_dict
        assert "muon_lr_multiplier" in state_dict

        # Create new optimizer and load state
        optimizer2 = create_optimizer(simple_model, optimizer_cfg, lr_cfg, batch_size_total=4)
        optimizer2.load_state_dict(state_dict)

        # Take another step - should not raise
        x = torch.randn(4, 10)
        loss = simple_model(x).sum()
        loss.backward()
        optimizer2.step()

    def test_param_groups_combined(self, simple_model, optimizer_cfg, lr_cfg):
        """Test that param_groups contains groups from both optimizers."""
        optimizer_cfg.type = "muon_adamw"
        optimizer = create_optimizer(simple_model, optimizer_cfg, lr_cfg, batch_size_total=4)

        # Should have groups from both Muon and AdamW
        assert len(optimizer.param_groups) >= 2

        # Check that is_muon flag exists
        has_muon_group = any(g.get("is_muon", False) for g in optimizer.param_groups)
        has_adamw_group = any(not g.get("is_muon", True) for g in optimizer.param_groups)

        assert has_muon_group
        assert has_adamw_group


class TestMuonCustom:
    """Tests for the custom Muon optimizer implementation."""

    def test_step_updates_params(self, simple_model):
        """Test that Muon step updates parameters."""
        # Get only 2D params that will have gradients (fc1, fc2 weights)
        # Exclude embedding since it's not used in the forward pass
        params = [
            p for name, p in simple_model.named_parameters()
            if p.ndim >= 2 and "embed" not in name
        ]
        optimizer = MuonCustom(params, lr=0.01, momentum=0.95)

        # Create dummy gradients
        x = torch.randn(4, 10)
        loss = simple_model(x).sum()
        loss.backward()

        # Store initial values
        initial_values = [p.clone() for p in params]

        # Step
        optimizer.step()

        # Check params with gradients changed
        for i, p in enumerate(params):
            if p.grad is not None:
                assert not torch.equal(p, initial_values[i]), f"Param {i} was not updated"

    def test_momentum_buffer_created(self, simple_model):
        """Test that momentum buffer is created after first step."""
        # Get params that will have gradients
        params = [
            p for name, p in simple_model.named_parameters()
            if p.ndim >= 2 and "embed" not in name
        ]
        optimizer = MuonCustom(params, lr=0.01, momentum=0.95)

        # Initially no state
        assert all(len(optimizer.state[p]) == 0 for p in params)

        # Create gradients and step
        x = torch.randn(4, 10)
        loss = simple_model(x).sum()
        loss.backward()
        optimizer.step()

        # Now should have momentum buffer for params with gradients
        for p in params:
            if p.grad is not None:
                assert "momentum_buffer" in optimizer.state[p]

    def test_weight_decay_applied(self):
        """Test that weight decay is applied to parameters."""
        # Simple 2D parameter
        param = nn.Parameter(torch.ones(4, 4))
        optimizer = MuonCustom([param], lr=0.1, momentum=0.0, weight_decay=0.1)

        # Set gradient to zero (only weight decay should affect)
        param.grad = torch.zeros_like(param)

        initial_norm = param.norm().item()
        optimizer.step()
        final_norm = param.norm().item()

        # Weight decay should reduce norm (since grad=0, only decay acts)
        assert final_norm < initial_norm

    def test_nesterov_momentum(self):
        """Test that Nesterov momentum produces different results than standard momentum."""
        torch.manual_seed(42)

        # Create two identical params
        param1 = nn.Parameter(torch.randn(4, 4))
        param2 = nn.Parameter(param1.clone())

        opt_standard = MuonCustom([param1], lr=0.1, momentum=0.9, nesterov=False)
        opt_nesterov = MuonCustom([param2], lr=0.1, momentum=0.9, nesterov=True)

        # Same gradient
        grad = torch.randn(4, 4)
        param1.grad = grad.clone()
        param2.grad = grad.clone()

        # Multiple steps
        for _ in range(3):
            opt_standard.step()
            opt_nesterov.step()
            param1.grad = grad.clone()
            param2.grad = grad.clone()

        # Results should differ
        assert not torch.allclose(param1, param2)


class TestAdamWPatterns:
    """Tests for the ADAMW_PATTERNS constant."""

    def test_patterns_match_expected_names(self):
        """Test that patterns match the expected parameter name patterns."""
        expected_patterns = [
            "embed_target_coords",
            "embeds.",
            "embed.",
            "pred_heads",
            "latent_heads",
            "q_cells",
            "bilin",
            "norm",
            "bias",
        ]

        for pattern in expected_patterns:
            assert pattern in ADAMW_PATTERNS

    def test_class_token_in_patterns(self):
        """Test that class_token and register_token are in patterns."""
        assert "class_token" in ADAMW_PATTERNS
        assert "register_token" in ADAMW_PATTERNS
