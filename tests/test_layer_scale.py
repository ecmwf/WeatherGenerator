# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for LayerScale and StochasticDepth modules."""

import pytest
import torch

from weathergen.model.layers import LayerScale, MLP, StochasticDepth


class TestLayerScale:
    """Tests for the LayerScale module."""

    def test_init_value(self):
        """Test that gamma is initialized to the specified value."""
        dim = 64
        init_value = 1e-5
        layer_scale = LayerScale(dim, init_value)

        assert layer_scale.gamma.shape == (dim,)
        assert torch.allclose(layer_scale.gamma, torch.full((dim,), init_value))

    def test_init_value_rezero(self):
        """Test ReZero initialization (init_value=0)."""
        dim = 64
        layer_scale = LayerScale(dim, init_value=0.0)

        assert torch.allclose(layer_scale.gamma, torch.zeros(dim))

    def test_forward_scaling(self):
        """Test that forward applies per-channel scaling."""
        dim = 64
        batch_size = 8
        seq_len = 16
        init_value = 0.5

        layer_scale = LayerScale(dim, init_value)
        x = torch.randn(batch_size, seq_len, dim)

        out = layer_scale(x)

        expected = x * init_value
        assert torch.allclose(out, expected)

    def test_forward_with_learned_gamma(self):
        """Test forward with modified gamma values."""
        dim = 64
        layer_scale = LayerScale(dim, init_value=1.0)

        # Modify gamma
        with torch.no_grad():
            layer_scale.gamma.fill_(2.0)

        x = torch.randn(8, 16, dim)
        out = layer_scale(x)

        expected = x * 2.0
        assert torch.allclose(out, expected)

    def test_gradient_flow(self):
        """Test that gradients flow through LayerScale."""
        dim = 64
        layer_scale = LayerScale(dim, init_value=1e-5)
        x = torch.randn(8, 16, dim, requires_grad=True)

        out = layer_scale(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert layer_scale.gamma.grad is not None

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        dim = 64
        layer_scale = LayerScale(dim, init_value=1e-5)

        for shape in [(8, dim), (8, 16, dim), (8, 16, 32, dim)]:
            x = torch.randn(*shape)
            out = layer_scale(x)
            assert out.shape == x.shape


class TestStochasticDepth:
    """Tests for the StochasticDepth module."""

    def test_init(self):
        """Test initialization with drop probability."""
        drop_prob = 0.1
        sd = StochasticDepth(drop_prob)
        assert sd.drop_prob == drop_prob

    def test_eval_mode_no_drop(self):
        """Test that eval mode never drops (identity)."""
        drop_prob = 0.9  # High drop prob
        sd = StochasticDepth(drop_prob)
        sd.eval()

        x = torch.randn(8, 16, 64)
        out = sd(x)

        assert torch.equal(out, x)

    def test_train_mode_zero_prob(self):
        """Test that zero drop probability is identity in train mode."""
        sd = StochasticDepth(drop_prob=0.0)
        sd.train()

        x = torch.randn(8, 16, 64)
        out = sd(x)

        assert torch.equal(out, x)

    def test_train_mode_high_prob(self):
        """Test that very high drop probability drops most samples in train mode."""
        sd = StochasticDepth(drop_prob=0.99)
        sd.train()

        torch.manual_seed(42)
        x = torch.ones(100, 16, 64)
        out = sd(x)

        # With 99% drop, most samples should be zero
        zero_samples = (out.sum(dim=(1, 2)) == 0).sum().item()
        assert zero_samples > 90  # At least 90 out of 100 should be dropped

    def test_expected_value_preservation(self):
        """Test that expected value is preserved during training."""
        drop_prob = 0.3
        sd = StochasticDepth(drop_prob)
        sd.train()

        torch.manual_seed(42)
        x = torch.ones(1000, 16, 64)

        # Run many times to average
        outputs = []
        for _ in range(1000):
            outputs.append(sd(x).mean().item())

        mean_output = sum(outputs) / len(outputs)
        # Expected value should be approximately 1.0 (the input value)
        assert abs(mean_output - 1.0) < 0.1  # Allow 10% tolerance

    def test_per_sample_dropping(self):
        """Test that dropping is per-sample in batch dimension."""
        drop_prob = 0.5
        sd = StochasticDepth(drop_prob)
        sd.train()

        torch.manual_seed(42)
        batch_size = 100
        x = torch.ones(batch_size, 16, 64)

        out = sd(x)

        # Check that samples are either scaled or zero
        sample_sums = out.sum(dim=(1, 2))
        expected_sum_scaled = 16 * 64 / (1 - drop_prob)

        for s in sample_sums:
            # Each sample should be either 0 or scaled
            assert s.item() == 0.0 or abs(s.item() - expected_sum_scaled) < 1e-4

    def test_gradient_flow(self):
        """Test that gradients flow through StochasticDepth."""
        sd = StochasticDepth(drop_prob=0.5)
        sd.train()

        torch.manual_seed(42)  # Ensure some samples are kept
        x = torch.randn(8, 16, 64, requires_grad=True)

        out = sd(x)
        loss = out.sum()
        loss.backward()

        # Gradient should exist for kept samples
        assert x.grad is not None

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        sd = StochasticDepth(drop_prob=0.5)
        sd.train()

        for shape in [(8, 64), (8, 16, 64), (8, 16, 32, 64)]:
            x = torch.randn(*shape)
            out = sd(x)
            assert out.shape == x.shape


class TestMLPWithLayerScaleAndStochasticDepth:
    """Integration tests for MLP with LayerScale and StochasticDepth."""

    def test_mlp_with_layer_scale(self):
        """Test MLP with LayerScale enabled."""
        mlp = MLP(
            dim_in=64,
            dim_out=64,
            with_residual=True,
            layer_scale_init=1e-5,
        )

        assert mlp.layer_scale is not None
        assert isinstance(mlp.layer_scale, LayerScale)

        x = torch.randn(8, 16, 64)
        out = mlp(x)

        assert out.shape == x.shape

    def test_mlp_with_stochastic_depth(self):
        """Test MLP with StochasticDepth enabled."""
        mlp = MLP(
            dim_in=64,
            dim_out=64,
            with_residual=True,
            stochastic_depth_rate=0.1,
        )

        assert mlp.drop_path is not None
        assert isinstance(mlp.drop_path, StochasticDepth)

        mlp.train()
        x = torch.randn(8, 16, 64)
        out = mlp(x)

        assert out.shape == x.shape

    def test_mlp_with_both(self):
        """Test MLP with both LayerScale and StochasticDepth."""
        mlp = MLP(
            dim_in=64,
            dim_out=64,
            with_residual=True,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.1,
        )

        assert mlp.layer_scale is not None
        assert mlp.drop_path is not None

        mlp.train()
        x = torch.randn(8, 16, 64)
        out = mlp(x)

        assert out.shape == x.shape

    def test_mlp_without_features(self):
        """Test MLP with neither feature (backward compatibility)."""
        mlp = MLP(
            dim_in=64,
            dim_out=64,
            with_residual=True,
        )

        assert mlp.layer_scale is None
        assert mlp.drop_path is None

        x = torch.randn(8, 16, 64)
        out = mlp(x)

        assert out.shape == x.shape

    def test_mlp_layer_scale_in_state_dict(self):
        """Test that LayerScale parameters appear in state_dict."""
        mlp = MLP(
            dim_in=64,
            dim_out=64,
            with_residual=True,
            layer_scale_init=1e-5,
        )

        state_dict = mlp.state_dict()
        assert "layer_scale.gamma" in state_dict

    def test_mlp_gradient_flow_with_features(self):
        """Test gradient flow through MLP with LayerScale and StochasticDepth."""
        mlp = MLP(
            dim_in=64,
            dim_out=64,
            with_residual=True,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.1,
        )
        mlp.train()

        torch.manual_seed(42)
        x = torch.randn(8, 16, 64, requires_grad=True)

        out = mlp(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert mlp.layer_scale.gamma.grad is not None


class TestReZero:
    """Tests specifically for ReZero initialization (layer_scale_init=0)."""

    def test_rezero_initial_output(self):
        """Test that ReZero initially outputs just the residual."""
        mlp = MLP(
            dim_in=64,
            dim_out=64,
            with_residual=True,
            layer_scale_init=0.0,  # ReZero
        )

        x = torch.randn(8, 16, 64)
        out = mlp(x)

        # With ReZero, initial output should be approximately equal to input
        # (since layer_scale starts at 0, the layer contribution is 0)
        assert torch.allclose(out, x, atol=1e-5)

    def test_rezero_gradual_learning(self):
        """Test that ReZero allows gradual learning of layer scale."""
        mlp = MLP(
            dim_in=64,
            dim_out=64,
            with_residual=True,
            layer_scale_init=0.0,
        )

        # Initially gamma is 0
        assert torch.allclose(mlp.layer_scale.gamma, torch.zeros(64))

        # After gradient update, gamma should change
        x = torch.randn(8, 16, 64)
        target = torch.randn(8, 16, 64)

        optimizer = torch.optim.SGD(mlp.parameters(), lr=0.1)

        for _ in range(10):
            optimizer.zero_grad()
            out = mlp(x)
            loss = ((out - target) ** 2).mean()
            loss.backward()
            optimizer.step()

        # Gamma should now be non-zero
        assert not torch.allclose(mlp.layer_scale.gamma, torch.zeros(64))
