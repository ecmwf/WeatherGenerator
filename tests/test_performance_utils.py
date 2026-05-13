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
Runs on CPU with small synthetic tensors.
"""

import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from weathergen.utils.performance import (
    ThroughputTracker,
    compute_source_bytes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_cuda_sync():
    """Disable cuda.synchronize globally — tests run on CPU."""
    with patch("weathergen.utils.performance.torch.cuda.synchronize"):
        yield


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
                f"stream_{i}": StreamData([torch.zeros(shape)])
                for i, shape in enumerate(tensor_shapes_per_stream)
            }

    class SourceSamples:
        def __init__(self, samples):
            self.samples = samples

    return SourceSamples([Sample(shapes) for shapes in tensor_shapes])


def _make_mock_batch(source_samples):
    """Create a mock batch whose get_source_samples() returns *source_samples*."""
    batch = MagicMock()
    batch.get_source_samples.return_value = source_samples
    return batch


# ---------------------------------------------------------------------------
# compute_source_bytes
# ---------------------------------------------------------------------------


def test_compute_source_bytes_single_stream():
    # 1 sample, 1 stream, 1 tensor shape (4, 8) float32 → 4×8×4 = 128 bytes
    source = _make_mock_source_samples([[(4, 8)]])
    assert compute_source_bytes(source) == 128


def test_compute_source_bytes_multiple_samples_and_streams():
    # 2 samples × 2 streams × 1 tensor (2, 4) float32 = 2×2×1×2×4×4 = 128 bytes
    shapes = [(2, 4), (2, 4)]  # 2 streams per sample
    source = _make_mock_source_samples([shapes, shapes])  # 2 samples
    assert compute_source_bytes(source) == 128


def test_compute_source_bytes_empty():
    source = _make_mock_source_samples([])
    assert compute_source_bytes(source) == 0


# ---------------------------------------------------------------------------
# ThroughputTracker
# ---------------------------------------------------------------------------


@pytest.fixture()
def tracker():
    """A tracker with warmup_steps=2 on CPU."""
    return ThroughputTracker(device=torch.device("cpu"), warmup_steps=2, batch_size_per_gpu=4)


def test_no_metrics_before_warmup(tracker):
    """compute_metrics returns None during the warmup phase."""
    tracker.update(istep=0, source_mb=1.0)
    assert tracker.compute_metrics() is None


def test_metrics_available_after_warmup(tracker):
    """After warmup_steps, metrics become available."""
    tracker.update(istep=1, source_mb=1.0)
    tracker.update(istep=2, source_mb=1.0)
    tracker._sync()
    metrics = tracker.compute_metrics()
    assert metrics is not None


def test_metrics_keys(tracker):
    """All expected metric keys are present."""
    tracker.update(istep=1, source_mb=1.0)
    tracker.update(istep=2, source_mb=2.0)
    tracker._sync()
    metrics = tracker.compute_metrics()

    expected_keys = [
        "performance.throughput.device.batches_per_sec",
        "performance.throughput.device.samples_per_sec",
        "performance.throughput.device.mb_per_sec",
        "performance.throughput.global.batches_per_sec",
        "performance.throughput.global.samples_per_sec",
        "performance.throughput.global.mb_per_sec",
    ]
    for key in expected_keys:
        assert key in metrics, f"Missing metric key: {key}"


def test_accumulates_batches_and_samples(tracker):
    """Counters accumulate correctly after warmup."""
    tracker.update(istep=1, source_mb=0.5)
    tracker.update(istep=2, source_mb=1.0)
    tracker.update(istep=3, source_mb=1.5)

    assert tracker._total_batches == 2
    assert tracker._total_samples == 8
    assert tracker._total_mb == pytest.approx(2.5)


def test_warmup_steps_not_counted():
    """Steps during warmup do not contribute to totals."""
    tracker = ThroughputTracker(device=torch.device("cpu"), warmup_steps=3, batch_size_per_gpu=4)
    for istep in range(3):
        tracker.update(istep=istep, source_mb=1.0)

    assert tracker._total_batches == 0
    assert tracker._total_samples == 0


def test_throughput_values_positive(tracker):
    """Throughput values are positive after real steps elapse."""
    tracker.update(istep=1, source_mb=1.0)
    tracker._t0 = time.time() - 0.1
    tracker.update(istep=2, source_mb=1.0)
    tracker._sync()
    metrics = tracker.compute_metrics()

    assert metrics is not None
    assert metrics["performance.throughput.device.batches_per_sec"] > 0
    assert metrics["performance.throughput.device.samples_per_sec"] > 0
    assert metrics["performance.throughput.device.mb_per_sec"] > 0


def test_step_calls_log_fn_on_root(tracker):
    """step() invokes log_fn with metrics on the root rank after warmup."""
    source = _make_mock_source_samples([[(2, 2)]])
    batch = _make_mock_batch(source)

    logged = {}

    def log_fn(m):
        logged.update(m)

    tracker.step(batch, istep=1, log_fn=log_fn)
    assert logged == {}

    tracker._t0 = time.time() - 0.1

    with patch("weathergen.utils.performance.is_root", return_value=True):
        tracker.step(batch, istep=2, log_fn=log_fn)

    assert "performance.throughput.device.batches_per_sec" in logged


def test_step_does_not_log_on_non_root(tracker):
    """step() does not invoke log_fn on non-root ranks."""
    source = _make_mock_source_samples([[(2, 2)]])
    batch = _make_mock_batch(source)

    logged = {}

    tracker.step(batch, istep=1, log_fn=lambda m: logged.update(m))
    tracker._t0 = time.time() - 0.1

    with patch("weathergen.utils.performance.is_root", return_value=False):
        tracker.step(batch, istep=2, log_fn=lambda m: logged.update(m))

    assert logged == {}
