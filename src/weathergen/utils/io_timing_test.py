# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import os

import pytest

from weathergen.utils import io_timing


@pytest.fixture
def nvtx_calls(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(io_timing, "_nvtx_push", lambda name: calls.append(f"push {name}"))
    monkeypatch.setattr(io_timing, "_nvtx_pop", lambda: calls.append("pop"))
    return calls


def test_disabled_is_noop(monkeypatch, nvtx_calls):
    monkeypatch.setattr(io_timing, "NVTX_ENABLED", False)
    monkeypatch.setattr(io_timing, "MEMORY_ENABLED", False)
    with io_timing.io_timer("ERA5", "read"):
        pass
    assert nvtx_calls == []


def test_nvtx_nested_and_balanced(monkeypatch, nvtx_calls):
    monkeypatch.setattr(io_timing, "NVTX_ENABLED", True)
    monkeypatch.setattr(io_timing, "MEMORY_ENABLED", False)
    with io_timing.io_timer("ERA5", "outer"):
        with io_timing.io_timer("ERA5", "inner"):
            pass
    assert nvtx_calls == ["push io:ERA5 : outer", "push io:ERA5 : inner", "pop", "pop"]


def test_nvtx_popped_on_exception(monkeypatch, nvtx_calls):
    monkeypatch.setattr(io_timing, "NVTX_ENABLED", True)
    monkeypatch.setattr(io_timing, "MEMORY_ENABLED", False)
    with pytest.raises(RuntimeError), io_timing.io_timer("ERA5", "boom"):
        raise RuntimeError
    assert nvtx_calls == ["push io:ERA5 : boom", "pop"]


def test_memory_sampler_started_by_io_timer(monkeypatch, nvtx_calls):
    started: list[str] = []
    monkeypatch.setattr(io_timing, "NVTX_ENABLED", False)
    monkeypatch.setattr(io_timing, "MEMORY_ENABLED", True)
    monkeypatch.setattr(io_timing, "_ensure_memory_sampler", lambda: started.append("start"))
    with io_timing.io_timer("ERA5", "read"):
        pass
    assert started == ["start"]
    assert nvtx_calls == []


def test_io_event_only_while_memory_sampling(monkeypatch, tmp_path):
    monkeypatch.setattr(io_timing, "_OUT_DIR", tmp_path)
    monkeypatch.setattr(io_timing, "_out", None)
    monkeypatch.setattr(io_timing, "MEMORY_ENABLED", False)
    io_timing.io_event("nsys-window", "startup open")
    assert list(tmp_path.glob("io_timing_*.jsonl")) == []
    monkeypatch.setattr(io_timing, "MEMORY_ENABLED", True)
    io_timing.io_event("nsys-window", "startup open")
    io_timing._out.close()
    monkeypatch.setattr(io_timing, "_out", None)
    (f,) = tmp_path.glob("io_timing_*.jsonl")
    (rec,) = [json.loads(line) for line in f.read_text().splitlines()]
    assert rec["stream"] == "nsys-window" and rec["op"] == "startup open" and rec["dt"] == 0.0


def test_lock_is_released_in_forked_child():
    """A lock held by another thread (the sampler) at fork time must not block the child."""
    io_timing._out_lock.acquire()
    try:
        pid = os.fork()
        if pid == 0:  # child: the module lock must be a fresh, free one
            free = io_timing._out_lock.acquire(blocking=False)
            os._exit(0 if free else 1)
        _, status = os.waitpid(pid, 0)
    finally:
        io_timing._out_lock.release()
    assert os.WEXITSTATUS(status) == 0
