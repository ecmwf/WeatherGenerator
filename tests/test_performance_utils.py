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

from unittest.mock import MagicMock, patch

import pytest
import torch

from weathergen.utils.performance import (
    MemoryTracker,
    ThroughputTracker,
    compute_source_bytes,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    """A throughput tracker on CPU with batch_size_per_gpu=4."""
    return ThroughputTracker(device=torch.device("cpu"), batch_size_per_gpu=4)


def test_no_metrics_before_any_step(tracker):
    """compute_metrics returns None until at least one step is recorded."""
    assert tracker.compute_metrics() is None


def test_metrics_available_after_step(tracker):
    """After a step is recorded, metrics become available."""
    tracker.update(source_mb=1.0)
    assert tracker.compute_metrics() is not None


def test_metrics_keys(tracker):
    """compute_metrics reports exactly the expected global count keys."""
    tracker.update(source_mb=1.0)
    tracker.update(source_mb=2.0)
    metrics = tracker.compute_metrics()

    assert set(metrics) == {
        "performance.throughput.global.batches",
        "performance.throughput.global.samples",
        "performance.throughput.global.mb",
    }


def test_accumulates_batches_samples_and_mb(tracker):
    """Each update adds one batch, batch_size_per_gpu samples, and its source mb."""
    tracker.update(source_mb=0.5)
    tracker.update(source_mb=1.0)
    tracker.update(source_mb=1.5)

    assert tracker._total_batches == 3
    assert tracker._total_samples == 12  # 3 batches × batch_size_per_gpu (4)
    assert tracker._total_mb == pytest.approx(3.0)


def test_step_accumulates_from_batch(tracker):
    """step() derives the source mb from the batch and accumulates counts."""
    # 1 sample × 1 stream × (2, 2) float32 = 16 bytes = 16 / 1e6 MB per step
    source = _make_mock_source_samples([[(2, 2)]])
    batch = _make_mock_batch(source)

    tracker.step(batch)
    tracker.step(batch)

    assert tracker._total_batches == 2
    assert tracker._total_samples == 8  # 2 batches × batch_size_per_gpu (4)
    assert tracker._total_mb == pytest.approx(2 * 16 / 1e6)


def test_metrics_report_cumulative_counts(tracker):
    """compute_metrics reports the accumulated global counts.

    Without a process group the global counts equal the local per-device totals.
    """
    tracker.update(source_mb=1.0)
    tracker.update(source_mb=2.0)
    metrics = tracker.compute_metrics()

    assert metrics["performance.throughput.global.batches"] == pytest.approx(2)
    assert metrics["performance.throughput.global.samples"] == pytest.approx(8)
    assert metrics["performance.throughput.global.mb"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# MemoryTracker
# ---------------------------------------------------------------------------

_GIB = 1024**3


def _patch_cuda_memory(allocated, reserved, calls: list | None = None):
    """Patch the CUDA peak counters to return successive values — tests run on CPU."""
    allocated, reserved = iter(allocated), iter(reserved)

    def read(it, name):
        def _read(device):
            if calls is not None:
                calls.append(name)
            return next(it)

        return _read

    def reset(device):
        if calls is not None:
            calls.append("reset")

    return (
        patch("torch.cuda.max_memory_allocated", side_effect=read(allocated, "read_allocated")),
        patch("torch.cuda.max_memory_reserved", side_effect=read(reserved, "read_reserved")),
        patch("torch.cuda.reset_peak_memory_stats", side_effect=reset),
    )


def test_memory_tracker_returns_peaks_in_gib():
    a, r, z = _patch_cuda_memory([2 * _GIB], [3 * _GIB])
    with a, r, z:
        metrics = MemoryTracker(device=torch.device("cpu")).collect("train")

    assert metrics == {
        "performance.memory.train.max_allocated_gib": pytest.approx(2.0),
        "performance.memory.train.max_reserved_gib": pytest.approx(3.0),
    }


def test_memory_tracker_reads_peaks_before_resetting():
    """Resetting first would zero the high-water marks; each read must precede its reset."""
    calls = []
    a, r, z = _patch_cuda_memory([0, 0], [0, 0], calls)
    with a, r, z:
        tracker = MemoryTracker(device=torch.device("cpu"))
        tracker.collect("train")
        tracker.collect("train")

    read = ["read_allocated", "read_reserved", "reset"]
    assert calls == ["reset", *read, *read]


def test_memory_tracker_closed_windows_keep_their_own_peak():
    """A closed window keeps the max of its parts; other windows do not see its peak."""
    a, r, z = _patch_cuda_memory([5 * _GIB, 9 * _GIB, 1 * _GIB], [6 * _GIB, 9 * _GIB, 2 * _GIB])
    with a, r, z:
        tracker = MemoryTracker(device=torch.device("cpu"))
        tracker.close_window("train")  # training steps before a checkpoint
        save = tracker.collect("save_model")  # the checkpoint itself
        train = tracker.collect("train")  # the rest of the training steps

    assert save["performance.memory.save_model.max_allocated_gib"] == pytest.approx(9.0)
    assert train["performance.memory.train.max_allocated_gib"] == pytest.approx(5.0)
    assert train["performance.memory.train.max_reserved_gib"] == pytest.approx(6.0)
