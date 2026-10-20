# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from weathergen.utils.performance import MemoryTracker, ThroughputTracker, nvtx_batches
from weathergen.utils.profiling import ProfilingConfig

_DEVICE = torch.device("cpu")


def test_throughput_tracker_from_config():
    cfg = OmegaConf.create({"performance_logging": {"throughput": {}}})

    assert not ProfilingConfig.from_config(cfg).enabled
    assert not ThroughputTracker.is_enabled(cfg), "throughput is off by default"
    assert ThroughputTracker.from_config(cfg, _DEVICE, batch_size_per_gpu=1) is None

    cfg.performance_logging.throughput.enabled = True
    tracker = ThroughputTracker.from_config(cfg, _DEVICE, batch_size_per_gpu=3)

    assert tracker is not None
    assert tracker.batch_size_per_gpu == 3


def test_throughput_tracker_without_the_section():
    assert not ThroughputTracker.is_enabled(OmegaConf.create({}))


def test_nvtx_range_encloses_the_work_on_each_batch():
    calls = []
    with (
        patch("torch.cuda.nvtx.range_push", side_effect=lambda n: calls.append(f"push {n}")),
        patch("torch.cuda.nvtx.range_pop", side_effect=lambda: calls.append("pop")),
    ):
        for bidx, _batch in nvtx_batches(enumerate(["a", "b"])):
            calls.append(f"step_{bidx}")

    assert calls == ["push batch_0", "step_0", "pop", "push batch_1", "step_1", "pop"]


def test_memory_tracker_from_config():
    cfg = OmegaConf.create({"performance_logging": {"memory": {}}})
    assert not MemoryTracker.is_enabled(cfg), "peak-memory logging is off by default"
    assert MemoryTracker.from_config(cfg, _DEVICE) is None

    cfg.performance_logging.memory.enabled = True
    with patch("torch.cuda.is_available", return_value=False):
        assert MemoryTracker.from_config(cfg, _DEVICE) is None, "needs CUDA"

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.reset_peak_memory_stats"),
    ):
        assert MemoryTracker.from_config(cfg, _DEVICE) is not None
