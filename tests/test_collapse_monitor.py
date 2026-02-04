# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for collapse monitoring metrics."""

import pytest
import torch

from weathergen.train.collapse_monitor import CollapseMonitor


@pytest.fixture
def default_config():
    """Default enabled config for collapse monitoring."""
    return {
        "enabled": True,
        "compute_frequency": 100,
        "log_frequency": 100,
        "metrics": {
            "effective_rank": {
                "enabled": True,
                "tensor_source": "both",
                "sample_size": 2048,
            },
            "singular_values": {
                "enabled": True,
                "top_k": 10,
                "tensor_source": "both",
                "sample_size": 2048,
            },
            "dimension_variance": {
                "enabled": True,
                "tensor_source": "both",
            },
            "prototype_entropy": {
                "enabled": True,
            },
            "ema_beta": {
                "enabled": True,
            },
        },
    }


@pytest.fixture
def monitor(default_config):
    """Create a collapse monitor with default config."""
    device = torch.device("cpu")
    return CollapseMonitor(default_config, device)


class TestCollapseMonitorInitialization:
    """Test CollapseMonitor initialization."""

    def test_disabled_monitor(self):
        """Test that disabled monitor doesn't compute metrics."""
        config = {"enabled": False}
        monitor = CollapseMonitor(config, torch.device("cpu"))
        assert not monitor.enabled
        assert not monitor.should_compute(100)
        assert not monitor.should_log(100)

    def test_enabled_monitor(self, default_config):
        """Test that enabled monitor computes at correct intervals."""
        monitor = CollapseMonitor(default_config, torch.device("cpu"))
        assert monitor.enabled
        assert monitor.should_compute(0)
        assert monitor.should_compute(100)
        assert not monitor.should_compute(50)

    def test_frequency_settings(self):
        """Test custom frequency settings."""
        config = {
            "enabled": True,
            "compute_frequency": 50,
            "log_frequency": 200,
        }
        monitor = CollapseMonitor(config, torch.device("cpu"))
        assert monitor.should_compute(50)
        assert monitor.should_compute(100)  # 100 is a multiple of 50
        assert not monitor.should_compute(75)  # 75 is not a multiple of 50
        assert monitor.should_log(200)
        assert not monitor.should_log(100)


class TestEffectiveRank:
    """Test effective rank computation."""

    def test_full_rank_matrix(self, monitor):
        """Full rank random matrix should have effective rank close to min(N, D)."""
        torch.manual_seed(42)
        # Create a full-rank matrix with orthogonal columns
        dim = 64
        num_samples = 128
        z = torch.randn(num_samples, dim)
        # Make it more orthogonal via QR decomposition
        q, _ = torch.linalg.qr(z.T)
        z = q.T  # Now z is [dim, dim] with orthogonal rows
        z = torch.cat([z, torch.randn(num_samples - dim, dim)], dim=0)

        eff_rank = monitor._compute_effective_rank(z, sample_size=0)
        # For a full-rank matrix, effective rank should be significant portion of D
        assert eff_rank > dim * 0.3, f"Expected effective rank > {dim * 0.3}, got {eff_rank}"

    def test_low_rank_matrix(self, monitor):
        """Low rank matrix should have effective rank close to actual rank."""
        torch.manual_seed(42)
        # Create a rank-5 matrix
        actual_rank = 5
        num_samples, dim = 128, 64
        u_mat = torch.randn(num_samples, actual_rank)
        v_mat = torch.randn(actual_rank, dim)
        z = u_mat @ v_mat

        eff_rank = monitor._compute_effective_rank(z, sample_size=0)
        # Effective rank should be close to actual rank
        assert eff_rank < actual_rank * 2, (
            f"Expected effective rank < {actual_rank * 2}, got {eff_rank}"
        )
        assert eff_rank > actual_rank * 0.5, (
            f"Expected effective rank > {actual_rank * 0.5}, got {eff_rank}"
        )

    def test_collapsed_matrix(self, monitor):
        """Completely collapsed matrix should have effective rank ~1."""
        num_samples, dim = 128, 64
        # All rows are the same (rank 1)
        row = torch.randn(1, dim)
        z = row.expand(num_samples, dim).clone()

        eff_rank = monitor._compute_effective_rank(z, sample_size=0)
        # Effective rank should be very close to 1
        assert eff_rank < 2, f"Expected effective rank < 2, got {eff_rank}"

    def test_3d_tensor_flattening(self, monitor):
        """Test that [B, N, D] tensors are properly flattened."""
        torch.manual_seed(42)
        batch_size, num_patches, dim = 4, 32, 64
        z = torch.randn(batch_size, num_patches, dim)

        eff_rank = monitor._compute_effective_rank(z, sample_size=0)
        # Should compute without error and return reasonable value
        assert 1 <= eff_rank <= dim


class TestSingularValues:
    """Test singular value spectrum computation."""

    def test_top_k_singular_values(self, monitor):
        """Test that top-k singular values are correctly computed."""
        torch.manual_seed(42)
        num_samples, dim = 128, 64
        z = torch.randn(num_samples, dim)

        sv_metrics = monitor._compute_singular_values(z, top_k=5, sample_size=0)

        # Check that we got top-5 singular values
        assert "singular_value_0" in sv_metrics
        assert "singular_value_4" in sv_metrics
        assert "singular_value_5" not in sv_metrics

        # Singular values should be in descending order
        for i in range(4):
            assert sv_metrics[f"singular_value_{i}"] >= sv_metrics[f"singular_value_{i + 1}"]

    def test_concentration_ratio(self, monitor):
        """Test singular value concentration ratio."""
        torch.manual_seed(42)
        # Create a rank-1 matrix where first SV dominates
        num_samples, dim = 128, 64
        # Use outer product to create a truly rank-1 dominated matrix
        u_vec = torch.randn(num_samples, 1)
        v_vec = torch.randn(1, dim)
        z = u_vec @ v_vec * 10 + torch.randn(num_samples, dim) * 0.01  # Strong rank-1 component

        sv_metrics = monitor._compute_singular_values(z, top_k=5, sample_size=0)

        # Concentration should be high when one SV dominates
        assert "sv_concentration" in sv_metrics
        assert sv_metrics["sv_concentration"] > 0.8  # First SV dominates strongly

    def test_uniform_singular_values(self, monitor):
        """Test with approximately uniform singular values."""
        torch.manual_seed(42)
        # Create orthogonal matrix with equal singular values
        dim = 64
        q, _ = torch.linalg.qr(torch.randn(dim, dim))
        z = q * 10  # Scale uniformly

        sv_metrics = monitor._compute_singular_values(z, top_k=5, sample_size=0)

        # Concentration should be low (close to 1/D)
        assert sv_metrics["sv_concentration"] < 0.1


class TestDimensionVariance:
    """Test per-dimension variance computation."""

    def test_random_matrix_balanced_variance(self, monitor):
        """Random matrix should have balanced variance across dimensions."""
        torch.manual_seed(42)
        num_samples, dim = 1024, 64
        z = torch.randn(num_samples, dim)

        var_metrics = monitor._compute_dimension_variance(z)

        # All variances should be close to 1 for standard normal
        assert abs(var_metrics["var_mean"] - 1.0) < 0.2
        # Variance ratio should be small for random matrix
        var_ratio = var_metrics["var_max"] / (var_metrics["var_min"] + 1e-8)
        assert var_ratio < 5  # Balanced dimensions

    def test_dead_dimensions(self, monitor):
        """Test detection of dead (zero-variance) dimensions."""
        torch.manual_seed(42)
        num_samples, dim = 128, 64
        z = torch.randn(num_samples, dim)
        # Kill some dimensions (set to constant)
        z[:, :10] = 0.5

        var_metrics = monitor._compute_dimension_variance(z)

        # Minimum variance should be very close to 0 (dead dimensions)
        assert var_metrics["var_min"] < 1e-6

    def test_imbalanced_dimensions(self, monitor):
        """Test with highly imbalanced dimension variances."""
        torch.manual_seed(42)
        num_samples, dim = 128, 64
        z = torch.randn(num_samples, dim)
        # Scale some dimensions much more than others
        z[:, 0] *= 100
        z[:, 1:10] *= 0.01

        var_metrics = monitor._compute_dimension_variance(z)

        # Large variance ratio indicates imbalance
        var_ratio = var_metrics["var_max"] / (var_metrics["var_min"] + 1e-8)
        assert var_ratio > 1000


class TestPrototypeEntropy:
    """Test DINO prototype entropy computation."""

    def test_uniform_prototype_distribution(self, monitor):
        """Uniform prototype distribution should have entropy ~1."""
        batch_size, num_prototypes = 128, 64
        # Uniform distribution
        probs = torch.ones(batch_size, num_prototypes) / num_prototypes

        entropy = monitor._compute_prototype_entropy(probs)

        # Normalized entropy should be close to 1
        assert abs(entropy - 1.0) < 0.01

    def test_single_prototype_collapse(self, monitor):
        """Collapse to single prototype should have entropy ~0."""
        batch_size, num_prototypes = 128, 64
        # All mass on first prototype
        probs = torch.zeros(batch_size, num_prototypes)
        probs[:, 0] = 1.0

        entropy = monitor._compute_prototype_entropy(probs)

        # Normalized entropy should be close to 0
        assert entropy < 0.01

    def test_partial_collapse(self, monitor):
        """Partial collapse should have intermediate entropy."""
        batch_size, num_prototypes = 128, 64
        # Only 4 prototypes used uniformly (much stronger collapse)
        probs = torch.zeros(batch_size, num_prototypes)
        probs[:, :4] = 0.25  # Only 4 out of 64 prototypes

        entropy = monitor._compute_prototype_entropy(probs)

        # Entropy should be between 0 and 1 (log(4)/log(64) ≈ 0.33)
        assert 0.2 < entropy < 0.5


class TestMetricsCaching:
    """Test metrics caching and averaging."""

    def test_cache_accumulation(self, monitor):
        """Test that metrics are properly cached."""
        torch.manual_seed(42)
        z1 = torch.randn(64, 32)
        z2 = torch.randn(64, 32)

        # Compute metrics twice
        monitor.compute_metrics(student_latent=z1)
        monitor.compute_metrics(student_latent=z2)

        # Cache should contain averaged values
        cached = monitor.get_cached_metrics()
        assert "collapse.student.effective_rank" in cached

    def test_cache_clear(self, monitor):
        """Test that cache is cleared after get_cached_metrics."""
        torch.manual_seed(42)
        z = torch.randn(64, 32)

        monitor.compute_metrics(student_latent=z)
        _ = monitor.get_cached_metrics()

        # Second call should return empty
        cached = monitor.get_cached_metrics()
        assert len(cached) == 0


class TestIntegration:
    """Integration tests with both student and teacher tensors."""

    def test_full_metrics_computation(self, monitor):
        """Test computing all metrics with both student and teacher."""
        torch.manual_seed(42)
        batch_size, num_patches, dim = 4, 32, 64
        student = torch.randn(batch_size, num_patches, dim)
        teacher = torch.randn(batch_size, num_patches, dim)

        metrics = monitor.compute_metrics(
            student_latent=student,
            teacher_latent=teacher,
            ema_beta=0.999,
            loss_type="JEPA",
        )

        # Check that both student and teacher metrics are computed
        assert "collapse.student.effective_rank" in metrics
        assert "collapse.teacher.effective_rank" in metrics
        assert "collapse.student.var_min" in metrics
        assert "collapse.teacher.var_min" in metrics
        assert "collapse.ema_beta" in metrics
        assert metrics["collapse.ema_beta"] == 0.999

    def test_dino_prototype_entropy(self, monitor):
        """Test DINO prototype entropy computation."""
        torch.manual_seed(42)
        batch_size, num_patches, dim = 4, 32, 64
        num_prototypes = 128
        student = torch.randn(batch_size, num_patches, dim)
        probs = torch.softmax(torch.randn(batch_size, num_prototypes), dim=-1)

        metrics = monitor.compute_metrics(
            student_latent=student,
            prototype_probs=probs,
            loss_type="DINO",
        )

        assert "collapse.dino.prototype_entropy" in metrics
        assert 0 <= metrics["collapse.dino.prototype_entropy"] <= 1

    def test_disabled_metrics(self):
        """Test that disabled metrics are not computed."""
        config = {
            "enabled": True,
            "compute_frequency": 1,
            "log_frequency": 1,
            "metrics": {
                "effective_rank": {"enabled": False},
                "singular_values": {"enabled": False},
                "dimension_variance": {"enabled": True, "tensor_source": "student"},
                "prototype_entropy": {"enabled": False},
                "ema_beta": {"enabled": False},
            },
        }
        monitor = CollapseMonitor(config, torch.device("cpu"))

        torch.manual_seed(42)
        z = torch.randn(64, 32)
        metrics = monitor.compute_metrics(student_latent=z)

        # Only dimension variance should be computed
        assert "collapse.student.var_min" in metrics
        assert "collapse.student.effective_rank" not in metrics
        assert "collapse.student.singular_value_0" not in metrics


class TestSampling:
    """Test row sampling for SVD computations."""

    def test_sampling_reduces_computation(self, monitor):
        """Test that sampling works for large tensors."""
        torch.manual_seed(42)
        num_samples, dim = 10000, 64
        z = torch.randn(num_samples, dim)

        # With sampling
        eff_rank_sampled = monitor._compute_effective_rank(z, sample_size=1024)
        # Without sampling
        eff_rank_full = monitor._compute_effective_rank(z, sample_size=0)

        # Results should be in same ballpark
        assert abs(eff_rank_sampled - eff_rank_full) < eff_rank_full * 0.3

    def test_no_sampling_when_small(self, monitor):
        """Test that small tensors aren't sampled."""
        torch.manual_seed(42)
        num_samples, dim = 100, 64
        z = torch.randn(num_samples, dim)

        # Sample size larger than N
        sampled = monitor._sample_rows(z, sample_size=1024)
        assert sampled.shape[0] == num_samples  # No sampling occurred
