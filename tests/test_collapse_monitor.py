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

    def test_singular_value_statistics(self, monitor):
        """Test that singular value statistics are correctly computed."""
        torch.manual_seed(42)
        num_samples, dim = 128, 64
        z = torch.randn(num_samples, dim)

        sv_metrics = monitor._compute_singular_values(z, sample_size=0)

        # Check that we got min, max, mean statistics
        assert "sv_min" in sv_metrics
        assert "sv_max" in sv_metrics
        assert "sv_mean" in sv_metrics
        assert "sv_concentration" in sv_metrics

        # Max should be >= mean >= min
        assert sv_metrics["sv_max"] >= sv_metrics["sv_mean"]
        assert sv_metrics["sv_mean"] >= sv_metrics["sv_min"]

    def test_concentration_ratio(self, monitor):
        """Test singular value concentration ratio."""
        torch.manual_seed(42)
        # Create a rank-1 matrix where first SV dominates
        num_samples, dim = 128, 64
        # Use outer product to create a truly rank-1 dominated matrix
        u_vec = torch.randn(num_samples, 1)
        v_vec = torch.randn(1, dim)
        z = u_vec @ v_vec * 10 + torch.randn(num_samples, dim) * 0.01  # Strong rank-1 component

        sv_metrics = monitor._compute_singular_values(z, sample_size=0)

        # Concentration should be high when one SV dominates
        assert "sv_concentration" in sv_metrics
        assert sv_metrics["sv_concentration"] > 0.8  # First SV dominates strongly

        # Max should be much larger than min for rank-1 dominated matrix
        assert sv_metrics["sv_max"] > sv_metrics["sv_min"] * 10

    def test_uniform_singular_values(self, monitor):
        """Test with random matrix (spread singular values)."""
        torch.manual_seed(42)
        # Random matrix will have spread singular values
        num_samples, dim = 128, 64
        z = torch.randn(num_samples, dim)

        sv_metrics = monitor._compute_singular_values(z, sample_size=0)

        # Concentration should be relatively low for random matrix
        assert sv_metrics["sv_concentration"] < 0.2

        # All statistics should be positive
        assert sv_metrics["sv_min"] > 0
        assert sv_metrics["sv_max"] > 0
        assert sv_metrics["sv_mean"] > 0


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
        assert "collapse.student.sv_max" not in metrics


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


class TestConfigValidation:
    """Test configuration validation."""

    def test_invalid_compute_frequency(self):
        """Test that non-positive compute_frequency raises error."""
        config = {"enabled": True, "compute_frequency": 0}
        with pytest.raises(ValueError, match="compute_frequency must be positive"):
            CollapseMonitor(config, torch.device("cpu"))

        config = {"enabled": True, "compute_frequency": -1}
        with pytest.raises(ValueError, match="compute_frequency must be positive"):
            CollapseMonitor(config, torch.device("cpu"))

    def test_invalid_log_frequency(self):
        """Test that non-positive log_frequency raises error."""
        config = {"enabled": True, "log_frequency": 0}
        with pytest.raises(ValueError, match="log_frequency must be positive"):
            CollapseMonitor(config, torch.device("cpu"))

    def test_invalid_tensor_source(self):
        """Test that invalid tensor_source raises error."""
        config = {
            "enabled": True,
            "metrics": {
                "effective_rank": {"tensor_source": "invalid_source"},
            },
        }
        with pytest.raises(ValueError, match="Invalid tensor_source"):
            CollapseMonitor(config, torch.device("cpu"))

    def test_invalid_forecast_aggregation(self):
        """Test that invalid forecast_aggregation raises error."""
        config = {
            "enabled": True,
            "metrics": {
                "effective_rank": {"forecast_aggregation": "invalid_agg"},
            },
        }
        with pytest.raises(ValueError, match="Invalid forecast_aggregation"):
            CollapseMonitor(config, torch.device("cpu"))

    def test_valid_tensor_sources(self):
        """Test that valid tensor_source values are accepted."""
        for source in ["student", "teacher", "both"]:
            config = {
                "enabled": True,
                "metrics": {
                    "effective_rank": {"tensor_source": source},
                },
            }
            monitor = CollapseMonitor(config, torch.device("cpu"))
            assert monitor is not None

    def test_valid_forecast_aggregations(self):
        """Test that valid forecast_aggregation values are accepted."""
        for agg in ["all", "aggregate_only", "per_step_only"]:
            config = {
                "enabled": True,
                "metrics": {
                    "effective_rank": {"forecast_aggregation": agg},
                },
            }
            monitor = CollapseMonitor(config, torch.device("cpu"))
            assert monitor is not None


class TestDimensionVarianceValidation:
    """Test validation in _compute_dimension_variance."""

    def test_empty_tensor(self, monitor):
        """Test that empty tensor returns empty dict."""
        z = torch.empty(0, 64)
        result = monitor._compute_dimension_variance(z)
        assert result == {}

    def test_nan_tensor(self, monitor):
        """Test that tensor with NaN returns empty dict."""
        z = torch.randn(64, 32)
        z[0, 0] = float("nan")
        result = monitor._compute_dimension_variance(z)
        assert result == {}

    def test_inf_tensor(self, monitor):
        """Test that tensor with Inf returns empty dict."""
        z = torch.randn(64, 32)
        z[0, 0] = float("inf")
        result = monitor._compute_dimension_variance(z)
        assert result == {}

    def test_single_sample(self, monitor):
        """Test that single sample returns empty dict (can't compute variance)."""
        z = torch.randn(1, 32)
        result = monitor._compute_dimension_variance(z)
        assert result == {}

    def test_valid_tensor(self, monitor):
        """Test that valid tensor returns metrics."""
        torch.manual_seed(42)
        z = torch.randn(64, 32)
        result = monitor._compute_dimension_variance(z)
        assert "var_min" in result
        assert "var_mean" in result
        assert "var_max" in result


class TestForecastingSequences:
    """Test forecasting with sequences of latents."""

    @pytest.fixture
    def forecast_config(self):
        """Config for forecasting tests."""
        return {
            "enabled": True,
            "compute_frequency": 1,
            "log_frequency": 1,
            "metrics": {
                "effective_rank": {
                    "enabled": True,
                    "tensor_source": "student",
                    "sample_size": 0,
                    "forecast_aggregation": "all",
                },
                "singular_values": {
                    "enabled": True,
                    "tensor_source": "student",
                    "sample_size": 0,
                    "forecast_aggregation": "all",
                },
                "dimension_variance": {
                    "enabled": True,
                    "tensor_source": "student",
                    "forecast_aggregation": "all",
                },
                "prototype_entropy": {"enabled": False},
                "ema_beta": {"enabled": False},
            },
        }

    @pytest.fixture
    def forecast_monitor(self, forecast_config):
        """Create a monitor configured for forecasting."""
        return CollapseMonitor(forecast_config, torch.device("cpu"))

    def test_sequence_per_step_metrics(self, forecast_monitor):
        """Test that per-step metrics are computed for sequences."""
        torch.manual_seed(42)
        # Create 3 time steps of latents
        latents = [torch.randn(32, 64) for _ in range(3)]

        metrics = forecast_monitor.compute_metrics(student_latent=latents)

        # Check per-step effective rank metrics
        assert "collapse.student.effective_rank.step_0" in metrics
        assert "collapse.student.effective_rank.step_1" in metrics
        assert "collapse.student.effective_rank.step_2" in metrics

        # Check per-step variance metrics
        assert "collapse.student.var_min.step_0" in metrics
        assert "collapse.student.var_min.step_1" in metrics
        assert "collapse.student.var_min.step_2" in metrics

    def test_sequence_aggregate_metrics(self, forecast_monitor):
        """Test that aggregate metrics are computed for sequences."""
        torch.manual_seed(42)
        latents = [torch.randn(32, 64) for _ in range(3)]

        metrics = forecast_monitor.compute_metrics(student_latent=latents)

        # Check aggregate effective rank metrics
        assert "collapse.student.effective_rank.mean" in metrics
        assert "collapse.student.effective_rank.min" in metrics
        assert "collapse.student.effective_rank.max" in metrics
        assert "collapse.student.effective_rank.degradation" in metrics

        # Verify aggregates are consistent
        step_values = [
            metrics["collapse.student.effective_rank.step_0"],
            metrics["collapse.student.effective_rank.step_1"],
            metrics["collapse.student.effective_rank.step_2"],
        ]
        assert metrics["collapse.student.effective_rank.min"] == min(step_values)
        assert metrics["collapse.student.effective_rank.max"] == max(step_values)
        assert abs(metrics["collapse.student.effective_rank.mean"] - sum(step_values) / 3) < 1e-6

    def test_degradation_metric(self, forecast_monitor):
        """Test degradation metric (final/initial ratio)."""
        torch.manual_seed(42)
        # Create latents with controlled rank degradation
        dim = 64
        # Step 0: Full rank random matrix
        step_0 = torch.randn(128, dim)
        # Step 1: Lower rank (rank ~32)
        u1 = torch.randn(128, 32)
        v1 = torch.randn(32, dim)
        step_1 = u1 @ v1
        # Step 2: Even lower rank (rank ~8)
        u2 = torch.randn(128, 8)
        v2 = torch.randn(8, dim)
        step_2 = u2 @ v2

        latents = [step_0, step_1, step_2]
        metrics = forecast_monitor.compute_metrics(student_latent=latents)

        # Degradation should be < 1 since rank decreases
        degradation = metrics["collapse.student.effective_rank.degradation"]
        assert degradation < 1.0, f"Expected degradation < 1.0, got {degradation}"

        # Verify degradation is step_2 / step_0
        expected = (
            metrics["collapse.student.effective_rank.step_2"]
            / metrics["collapse.student.effective_rank.step_0"]
        )
        assert abs(degradation - expected) < 1e-6

    def test_aggregate_only_mode(self):
        """Test forecast_aggregation='aggregate_only' mode."""
        config = {
            "enabled": True,
            "compute_frequency": 1,
            "log_frequency": 1,
            "metrics": {
                "effective_rank": {
                    "enabled": True,
                    "tensor_source": "student",
                    "forecast_aggregation": "aggregate_only",
                },
                "singular_values": {"enabled": False},
                "dimension_variance": {"enabled": False},
                "prototype_entropy": {"enabled": False},
                "ema_beta": {"enabled": False},
            },
        }
        monitor = CollapseMonitor(config, torch.device("cpu"))

        torch.manual_seed(42)
        latents = [torch.randn(32, 64) for _ in range(3)]
        metrics = monitor.compute_metrics(student_latent=latents)

        # Should NOT have per-step metrics
        assert "collapse.student.effective_rank.step_0" not in metrics

        # Should have aggregate metrics
        assert "collapse.student.effective_rank.mean" in metrics
        assert "collapse.student.effective_rank.min" in metrics

    def test_per_step_only_mode(self):
        """Test forecast_aggregation='per_step_only' mode."""
        config = {
            "enabled": True,
            "compute_frequency": 1,
            "log_frequency": 1,
            "metrics": {
                "effective_rank": {
                    "enabled": True,
                    "tensor_source": "student",
                    "forecast_aggregation": "per_step_only",
                },
                "singular_values": {"enabled": False},
                "dimension_variance": {"enabled": False},
                "prototype_entropy": {"enabled": False},
                "ema_beta": {"enabled": False},
            },
        }
        monitor = CollapseMonitor(config, torch.device("cpu"))

        torch.manual_seed(42)
        latents = [torch.randn(32, 64) for _ in range(3)]
        metrics = monitor.compute_metrics(student_latent=latents)

        # Should have per-step metrics
        assert "collapse.student.effective_rank.step_0" in metrics
        assert "collapse.student.effective_rank.step_1" in metrics

        # Should NOT have aggregate metrics
        assert "collapse.student.effective_rank.mean" not in metrics
        assert "collapse.student.effective_rank.degradation" not in metrics

    def test_empty_sequence(self, forecast_monitor):
        """Test that empty sequence returns no metrics."""
        metrics = forecast_monitor.compute_metrics(student_latent=[])

        # No effective rank metrics should be present
        assert not any(key.startswith("collapse.student.effective_rank") for key in metrics)

    def test_single_step_sequence(self, forecast_monitor):
        """Test sequence with single step (no degradation possible)."""
        torch.manual_seed(42)
        latents = [torch.randn(32, 64)]

        metrics = forecast_monitor.compute_metrics(student_latent=latents)

        # Should have step_0
        assert "collapse.student.effective_rank.step_0" in metrics

        # Should have aggregates (single value)
        assert "collapse.student.effective_rank.mean" in metrics

        # Degradation should be 1.0 (same step)
        assert metrics["collapse.student.effective_rank.degradation"] == 1.0

    def test_mixed_single_and_sequence(self, forecast_monitor):
        """Test with single tensor for student and sequence for teacher."""
        # Need to enable teacher monitoring
        config = {
            "enabled": True,
            "compute_frequency": 1,
            "log_frequency": 1,
            "metrics": {
                "effective_rank": {
                    "enabled": True,
                    "tensor_source": "both",
                    "forecast_aggregation": "all",
                },
                "singular_values": {"enabled": False},
                "dimension_variance": {"enabled": False},
                "prototype_entropy": {"enabled": False},
                "ema_beta": {"enabled": False},
            },
        }
        monitor = CollapseMonitor(config, torch.device("cpu"))

        torch.manual_seed(42)
        student = torch.randn(32, 64)  # Single tensor
        teacher = [torch.randn(32, 64) for _ in range(3)]  # Sequence

        metrics = monitor.compute_metrics(student_latent=student, teacher_latent=teacher)

        # Student should have single metric
        assert "collapse.student.effective_rank" in metrics
        assert "collapse.student.effective_rank.step_0" not in metrics

        # Teacher should have sequence metrics
        assert "collapse.teacher.effective_rank.step_0" in metrics
        assert "collapse.teacher.effective_rank.mean" in metrics

    def test_3d_tensor_sequence(self, forecast_monitor):
        """Test sequence of 3D tensors [B, N, D]."""
        torch.manual_seed(42)
        batch_size, num_patches, dim = 4, 32, 64
        latents = [torch.randn(batch_size, num_patches, dim) for _ in range(3)]

        metrics = forecast_monitor.compute_metrics(student_latent=latents)

        # Should flatten each tensor and compute metrics
        assert "collapse.student.effective_rank.step_0" in metrics
        assert "collapse.student.effective_rank.mean" in metrics

        # Values should be reasonable (between 1 and dim)
        for i in range(3):
            value = metrics[f"collapse.student.effective_rank.step_{i}"]
            assert 1 <= value <= dim


class TestSequenceSingularValues:
    """Test singular value metrics for sequences."""

    @pytest.fixture
    def sv_monitor(self):
        """Monitor configured for singular value tests."""
        config = {
            "enabled": True,
            "compute_frequency": 1,
            "log_frequency": 1,
            "metrics": {
                "effective_rank": {"enabled": False},
                "singular_values": {
                    "enabled": True,
                    "tensor_source": "student",
                    "sample_size": 0,
                    "forecast_aggregation": "all",
                },
                "dimension_variance": {"enabled": False},
                "prototype_entropy": {"enabled": False},
                "ema_beta": {"enabled": False},
            },
        }
        return CollapseMonitor(config, torch.device("cpu"))

    def test_sv_sequence_metrics(self, sv_monitor):
        """Test singular value metrics for sequences."""
        torch.manual_seed(42)
        latents = [torch.randn(64, 32) for _ in range(3)]

        metrics = sv_monitor.compute_metrics(student_latent=latents)

        # Per-step metrics
        assert "collapse.student.sv_min.step_0" in metrics
        assert "collapse.student.sv_max.step_0" in metrics
        assert "collapse.student.sv_concentration.step_0" in metrics

        # Aggregate metrics
        assert "collapse.student.sv_min.mean" in metrics
        assert "collapse.student.sv_max.mean" in metrics
        assert "collapse.student.sv_concentration.mean" in metrics


class TestSequenceVariance:
    """Test dimension variance metrics for sequences."""

    @pytest.fixture
    def var_monitor(self):
        """Monitor configured for variance tests."""
        config = {
            "enabled": True,
            "compute_frequency": 1,
            "log_frequency": 1,
            "metrics": {
                "effective_rank": {"enabled": False},
                "singular_values": {"enabled": False},
                "dimension_variance": {
                    "enabled": True,
                    "tensor_source": "student",
                    "forecast_aggregation": "all",
                },
                "prototype_entropy": {"enabled": False},
                "ema_beta": {"enabled": False},
            },
        }
        return CollapseMonitor(config, torch.device("cpu"))

    def test_variance_sequence_metrics(self, var_monitor):
        """Test variance metrics for sequences."""
        torch.manual_seed(42)
        latents = [torch.randn(64, 32) for _ in range(3)]

        metrics = var_monitor.compute_metrics(student_latent=latents)

        # Per-step metrics
        assert "collapse.student.var_min.step_0" in metrics
        assert "collapse.student.var_mean.step_0" in metrics
        assert "collapse.student.var_max.step_0" in metrics

        # Aggregate metrics
        assert "collapse.student.var_min.mean" in metrics
        assert "collapse.student.var_mean.mean" in metrics
        assert "collapse.student.var_max.mean" in metrics

    def test_variance_detects_collapse_over_time(self, var_monitor):
        """Test that variance metrics can detect collapse over forecast steps."""
        torch.manual_seed(42)
        dim = 32

        # Step 0: Normal variance
        step_0 = torch.randn(128, dim)

        # Step 1: Some dimensions start dying
        step_1 = torch.randn(128, dim)
        step_1[:, :8] *= 0.1  # Reduce variance in 8 dims

        # Step 2: More dimensions dead
        step_2 = torch.randn(128, dim)
        step_2[:, :16] *= 0.01  # Almost dead in 16 dims

        latents = [step_0, step_1, step_2]
        metrics = var_monitor.compute_metrics(student_latent=latents)

        # var_min should decrease over steps (more dead dimensions)
        assert metrics["collapse.student.var_min.step_0"] > metrics["collapse.student.var_min.step_1"]
        assert metrics["collapse.student.var_min.step_1"] > metrics["collapse.student.var_min.step_2"]
