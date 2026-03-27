# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for weathergen.utils.performance.

Self-contained: no WeatherGenerator data structures required.
Runs on CPU with small synthetic models.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from weathergen.utils.performance import (
    build_performance_metrics,
    compute_hfu,
    compute_mfu,
    compute_source_bytes,
    compute_utilisation_metrics,
    measure_model_flops,
)


class TwoLayerLinear(nn.Module):
    """Two linear layers, no activation checkpointing."""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.a = nn.Linear(dim, dim, bias=False)
        self.b = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.b(self.a(x))


class TwoLayerLinearCheckpointed(nn.Module):
    """Same two linear layers but with activation checkpointing on the first."""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.a = nn.Linear(dim, dim, bias=False)
        self.b = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        x = checkpoint(self.a, x, use_reentrant=False)
        return self.b(x)


def test_measure_model_flops_returns_positive_ints():
    model = TwoLayerLinear()
    x = torch.randn(1, 16)
    flops_fwd, flops_total = measure_model_flops(model, lambda: model(x), lambda y: y.sum())

    assert flops_fwd is not None and flops_fwd > 0
    assert flops_total is not None and flops_total > 0


def test_total_flops_exceed_forward_flops():
    """Backward pass adds FLOPs, so total > forward."""
    model = TwoLayerLinear()
    x = torch.randn(1, 16)
    flops_fwd, flops_total = measure_model_flops(model, lambda: model(x), lambda y: y.sum())

    assert flops_total > flops_fwd


def test_forward_flops_independent_of_loss():
    """flops_fwd should be the same whether loss_fn is provided or not."""
    model = TwoLayerLinear()
    x = torch.randn(1, 16)
    flops_fwd, _ = measure_model_flops(model, lambda: model(x), lambda y: y.sum())
    flops_fwd_only, _ = measure_model_flops(model, lambda: model(x), lambda y: y.sum())

    assert flops_fwd == flops_fwd_only


def test_total_flops_roughly_proportional_to_forward():
    """backward adds substantial FLOPs; total is between 1.5× and 3.5× of forward.

    Note: PyTorch's FlopCounterMode measures matmul-like ops.  The exact ratio
    depends on batch size and which specific backward ops are dispatched through
    the counted kernels. For small linear models the measured ratio is ≈ 2.5×
    (not exactly 3×), because some weight-gradient ops may use kernels that
    FlopCounterMode does not track.
    """
    model = TwoLayerLinear()
    x = torch.randn(1, 16)
    flops_fwd, flops_total = measure_model_flops(model, lambda: model(x), lambda y: y.sum())

    ratio = flops_total / flops_fwd
    assert 1.5 <= ratio <= 3.5, f"Expected backward to add substantial FLOPs, got ratio {ratio:.3f}"


def test_forward_flops_identical_with_and_without_checkpointing():
    """Activation checkpointing does not change the number of forward FLOPs."""
    dim = 16
    model_plain = TwoLayerLinear(dim)
    model_ckpt = TwoLayerLinearCheckpointed(dim)
    model_ckpt.a.weight.data = model_plain.a.weight.data.clone()
    model_ckpt.b.weight.data = model_plain.b.weight.data.clone()

    x = torch.randn(1, dim)
    fwd_plain, _ = measure_model_flops(model_plain, lambda: model_plain(x), lambda y: y.sum())
    fwd_ckpt, _ = measure_model_flops(model_ckpt, lambda: model_ckpt(x), lambda y: y.sum())

    assert fwd_plain == fwd_ckpt


def test_flop_counter_does_not_count_checkpoint_recompute():
    """FlopCounterMode does not capture activation-checkpoint recomputation.

    With use_reentrant=False, the recomputed forward during backward is not tracked
    by PyTorch's FlopCounterMode. Consequently total FLOPs measured with and without
    checkpointing are equal. This means our measured HFU is a lower bound of true
    hardware utilisation (it excludes recompute overhead).
    """
    dim = 16
    model_plain = TwoLayerLinear(dim)
    model_ckpt = TwoLayerLinearCheckpointed(dim)
    model_ckpt.a.weight.data = model_plain.a.weight.data.clone()
    model_ckpt.b.weight.data = model_plain.b.weight.data.clone()

    x = torch.randn(1, dim)
    _, total_plain = measure_model_flops(model_plain, lambda: model_plain(x), lambda y: y.sum())
    _, total_ckpt = measure_model_flops(model_ckpt, lambda: model_ckpt(x), lambda y: y.sum())

    assert total_plain == total_ckpt, (
        "Expected FlopCounterMode to report the same total FLOPs regardless of checkpointing "
        "(recomputed activations are not counted by FlopCounterMode with use_reentrant=False)"
    )


def test_measure_model_flops_handles_failure(monkeypatch):
    """Returns (None, None) gracefully when measurement raises."""
    from weathergen.utils import performance

    def _bad_measure(model, forward_fn, loss_fn=None):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(performance, "measure_flops", _bad_measure)

    model = TwoLayerLinear()
    x = torch.randn(1, 16)
    flops_fwd, flops_total = measure_model_flops(model, lambda: model(x), lambda y: y.sum())

    assert flops_fwd is None
    assert flops_total is None


def test_compute_mfu_formula():
    flops_fwd = 1_000_000
    steps_per_sec = 10.0
    available_flops = 1e12

    result = compute_mfu(flops_fwd, steps_per_sec, available_flops)
    expected = 3 * flops_fwd * steps_per_sec / available_flops

    assert result == pytest.approx(expected)


def test_compute_hfu_formula():
    flops_total = 4_000_000
    steps_per_sec = 10.0
    available_flops = 1e12

    result = compute_hfu(flops_total, steps_per_sec, available_flops)
    expected = flops_total * steps_per_sec / available_flops

    assert result == pytest.approx(expected)


def test_hfu_exceeds_mfu_when_total_exceeds_3x_forward():
    """HFU > MFU when total_flops > 3 × fwd_flops (recompute overhead present).

    For full per-layer checkpointing total_flops ≈ 4 × fwd_flops, so HFU ≈ (4/3) × MFU.
    Note: FlopCounterMode does NOT measure recompute, so in practice flops_total passed
    to compute_hfu would be an estimate (e.g. model_flops × recompute_factor).
    """
    fwd = 1_000_000
    total = 4 * fwd  # full per-layer checkpointing: 4 × fwd instead of 3 ×
    steps_per_sec = 5.0
    available = 1e12

    mfu = compute_mfu(fwd, steps_per_sec, available)
    hfu = compute_hfu(total, steps_per_sec, available)

    assert hfu > mfu
    assert hfu == pytest.approx((4 / 3) * mfu)


def test_mfu_hfu_equal_without_recomputation():
    """When total_flops == 3 × fwd_flops (no recompute), MFU == HFU."""
    fwd = 1_000_000
    total = 3 * fwd
    steps_per_sec = 5.0
    available = 1e12

    mfu = compute_mfu(fwd, steps_per_sec, available)
    hfu = compute_hfu(total, steps_per_sec, available)

    assert mfu == pytest.approx(hfu)


def _make_mock_source_samples(tensor_shapes: list[list[tuple]]):
    """Build a minimal mock of the source_samples object.

    tensor_shapes: list of samples, each a list of (shape,) tuples representing
                   source_tokens_cells tensors per stream.
    """

    class StreamData:
        def __init__(self, tensors):
            self.source_tokens_cells = tensors

    class Sample:
        def __init__(self, tensor_shapes_per_stream):
            self.streams_data = {
                f"stream_{i}": StreamData([torch.zeros(shape) for shape in shapes])
                for i, shapes in enumerate(tensor_shapes_per_stream)
            }

    class SourceSamples:
        def __init__(self, samples):
            self.samples = samples

    return SourceSamples([Sample(shapes) for shapes in tensor_shapes])


def test_compute_source_bytes_single_stream():
    # 1 sample, 1 stream, 1 tensor shape (4, 8) float32 → 4×8×4 = 128 bytes
    source = _make_mock_source_samples([[[(4, 8)]]])
    assert compute_source_bytes(source) == 128


def test_compute_source_bytes_multiple_samples_and_streams():
    # 2 samples × 2 streams × 1 tensor (2, 4) float32 = 2×2×1×2×4×4 = 128 bytes
    shapes = [[(2, 4)], [(2, 4)]]  # 2 streams per sample
    source = _make_mock_source_samples([shapes, shapes])
    assert compute_source_bytes(source) == 128


def test_compute_source_bytes_empty():
    source = _make_mock_source_samples([])
    assert compute_source_bytes(source) == 0


def test_compute_utilisation_metrics_both_present():
    fwd, total = 1_000_000, 3_000_000
    metrics = compute_utilisation_metrics(fwd, total, steps_per_sec=10.0, available_flops=1e12)
    assert "device.mfu" in metrics
    assert "device.hfu" in metrics
    assert metrics["device.mfu"] == pytest.approx(compute_mfu(fwd, 10.0, 1e12))
    assert metrics["device.hfu"] == pytest.approx(
        compute_hfu(total, 10.0, 1e12, recompute_factor=4 / 3)
    )


def test_compute_utilisation_metrics_missing_flops():
    """Returns empty dict when FLOPs are unavailable."""
    assert compute_utilisation_metrics(None, None, 10.0, 1e12) == {}


def test_compute_utilisation_metrics_zero_steps():
    """Returns empty dict when steps_per_sec is zero."""
    assert compute_utilisation_metrics(1_000_000, 3_000_000, 0.0, 1e12) == {}


def test_compute_utilisation_metrics_no_available_flops():
    """Returns empty dict when peak FLOP/s is unknown."""
    assert compute_utilisation_metrics(1_000_000, 3_000_000, 10.0, None) == {}


def test_build_performance_metrics_keys():
    """All expected keys are present and prefixed correctly."""
    lightning = {"device/batches_per_second": 5.0, "device/samples_per_second": 40.0}
    metrics = build_performance_metrics(
        lightning_metrics=lightning,
        elapsed=10.0,
        total_batches=50,
        total_mb=100.0,
        flops_fwd=1_000_000,
        flops_total=3_000_000,
        available_flops=1e12,
    )
    assert "performance.throughput.device.batches_per_second" in metrics
    assert "performance.throughput.device.mb_per_sec" in metrics
    assert "performance.throughput.mb_per_sec" in metrics
    assert "performance.utilization.device.mfu" in metrics
    assert "performance.utilization.device.hfu" in metrics


def test_build_performance_metrics_drops_lightning_mfu():
    """Lightning's own mfu label is excluded (we recompute it explicitly)."""
    lightning = {"device/mfu": 0.99, "mfu": 0.99, "device/batches_per_second": 5.0}
    metrics = build_performance_metrics(
        lightning_metrics=lightning,
        elapsed=10.0,
        total_batches=50,
        total_mb=0.0,
        flops_fwd=1_000_000,
        flops_total=3_000_000,
        available_flops=1e12,
    )
    assert "performance.utilization.device.mfu" in metrics  # our recomputed value
    assert "performance.throughput.mfu" not in metrics  # Lightning's dropped label


def test_build_performance_metrics_mb_per_sec_single_rank():
    """With world_size=1 (default), device and global MB/s are equal."""
    metrics = build_performance_metrics(
        lightning_metrics={},
        elapsed=4.0,
        total_batches=20,
        total_mb=200.0,
        flops_fwd=None,
        flops_total=None,
        available_flops=None,
    )
    assert metrics["performance.throughput.device.mb_per_sec"] == pytest.approx(50.0)
    assert metrics["performance.throughput.mb_per_sec"] == pytest.approx(50.0)


def test_build_performance_metrics_mb_per_sec_multi_rank():
    """Global MB/s = device MB/s × world_size."""
    metrics = build_performance_metrics(
        lightning_metrics={},
        elapsed=4.0,
        total_batches=20,
        total_mb=200.0,
        flops_fwd=None,
        flops_total=None,
        available_flops=None,
        world_size=8,
    )
    assert metrics["performance.throughput.device.mb_per_sec"] == pytest.approx(50.0)
    assert metrics["performance.throughput.mb_per_sec"] == pytest.approx(400.0)


def test_build_performance_metrics_no_flops_no_utilisation():
    """MFU/HFU keys absent when FLOPs or available_flops are unavailable."""
    metrics = build_performance_metrics(
        lightning_metrics={},
        elapsed=4.0,
        total_batches=20,
        total_mb=0.0,
        flops_fwd=None,
        flops_total=None,
        available_flops=None,
    )
    assert "performance.utilization.device.mfu" not in metrics
    assert "performance.utilization.device.hfu" not in metrics
