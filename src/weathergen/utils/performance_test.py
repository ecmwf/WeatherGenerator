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

from weathergen.utils.performance import (
    MemoryTracker,
    ThroughputTracker,
    nvtx_annotator,
    nvtx_batches,
    nvtx_next,
    nvtx_patched,
    nvtx_wrap,
)
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


def test_nvtx_annotator_is_a_noop_unless_enabled():
    calls = []
    with (
        patch("torch.cuda.nvtx.range_push", side_effect=lambda n: calls.append(n)),
        patch("torch.cuda.nvtx.range_pop", side_effect=lambda: calls.append("pop")),
    ):
        with nvtx_annotator(False)("off"):
            pass
        with nvtx_annotator(True)("on"):
            pass

    assert calls == ["on", "pop"]


def test_nvtx_next_wraps_each_fetch_but_not_the_loop_body():
    calls = []
    with (
        patch("torch.cuda.nvtx.range_push", side_effect=lambda n: calls.append(f"push {n}")),
        patch("torch.cuda.nvtx.range_pop", side_effect=lambda: calls.append("pop")),
    ):
        for item in nvtx_next([1, 2], "next"):
            calls.append(f"body {item}")

    # the final fetch, which ends the iteration, is closed too
    fetch = ["push next", "pop"]
    assert calls == [*fetch, "body 1", *fetch, "body 2", *fetch]


def test_nvtx_wrap_patches_only_the_instance():
    class Optimizer:
        def zero_grad(self, set_to_none: bool = True) -> str:
            return f"zeroed {set_to_none}"

    calls = []
    wrapped, untouched = Optimizer(), Optimizer()
    with (
        patch("torch.cuda.nvtx.range_push", side_effect=lambda n: calls.append(f"push {n}")),
        patch("torch.cuda.nvtx.range_pop", side_effect=lambda: calls.append("pop")),
    ):
        nvtx_wrap(wrapped, "zero_grad", "zero_grad")
        assert wrapped.zero_grad(set_to_none=False) == "zeroed False"
        assert untouched.zero_grad() == "zeroed True"

    assert calls == ["push zero_grad", "pop"]


def test_nvtx_patched_restores_the_original():
    original = torch.nn.utils.clip_grad_norm_
    with (
        patch("torch.cuda.nvtx.range_push"),
        patch("torch.cuda.nvtx.range_pop"),
        nvtx_patched(torch.nn.utils, "clip_grad_norm_", "clip_grad_norm"),
    ):
        assert torch.nn.utils.clip_grad_norm_ is not original

    assert torch.nn.utils.clip_grad_norm_ is original
