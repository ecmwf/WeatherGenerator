# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json

from weathergen.utils import io_timing, memory_sampler

SMAPS = """\
00400000-7fff00000000 ---p 00000000 00:00 0                [rollup]
Rss:              204800 kB
Pss:              102400 kB
Pss_Anon:          51200 kB
Pss_File:          25600 kB
Pss_Shmem:         25600 kB
Shared_Clean:      10 kB
"""


def test_read_host_memory_smaps(tmp_path):
    (tmp_path / "smaps_rollup").write_text(SMAPS)
    assert memory_sampler.read_host_memory(tmp_path) == {
        "rss_mib": 200.0,
        "pss_mib": 100.0,
        "pss_anon_mib": 50.0,
        "pss_file_mib": 25.0,
        "pss_shmem_mib": 25.0,
    }


def test_read_host_memory_falls_back_to_status(tmp_path):
    (tmp_path / "status").write_text("Name:\tpython\nVmRSS:\t   2048 kB\n")
    assert memory_sampler.read_host_memory(tmp_path) == {"rss_mib": 2.0}


def test_read_host_memory_without_proc(tmp_path):
    assert memory_sampler.read_host_memory(tmp_path / "missing") is None


def test_sample_once_writes_record(monkeypatch, tmp_path):
    monkeypatch.setattr(io_timing, "_OUT_DIR", tmp_path)
    monkeypatch.setattr(io_timing, "_out", None)
    monkeypatch.setattr(
        memory_sampler, "read_host_memory", lambda: {"rss_mib": 1.5, "pss_mib": 1.0}
    )
    monkeypatch.setattr(memory_sampler, "read_gpu_memory", lambda: {"gpu_alloc_mib": 7.0})
    assert memory_sampler.sample_once()
    io_timing._out.close()
    monkeypatch.setattr(io_timing, "_out", None)
    (f,) = tmp_path.glob("io_timing_*.jsonl")
    (rec,) = [json.loads(line) for line in f.read_text().splitlines()]
    assert rec["stream"] == "memory" and rec["role"] == "main"
    assert rec["rss_mib"] == 1.5 and rec["pss_mib"] == 1.0 and rec["gpu_alloc_mib"] == 7.0
    assert {"t_start", "pid", "ppid", "host", "dt"} <= rec.keys()


def test_sample_once_without_proc_returns_false(monkeypatch):
    monkeypatch.setattr(memory_sampler, "read_host_memory", lambda: None)
    assert not memory_sampler.sample_once()


def test_ensure_started_once_per_process(monkeypatch):
    started: list[object] = []
    monkeypatch.setattr(memory_sampler, "_started_pid", None)
    monkeypatch.setattr(
        memory_sampler.threading,
        "Thread",
        lambda **kw: type("T", (), {"start": lambda self: started.append(kw["name"])})(),
    )
    memory_sampler.ensure_started()
    memory_sampler.ensure_started()
    assert started == ["io-timing-memory"]
