"""Tests for the memory part of scripts/analyze_io_memory.py (standalone script, loaded by path)."""

import importlib.util
import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

_SPEC = importlib.util.spec_from_file_location(
    "analyze_io_memory", Path(__file__).parents[1] / "scripts" / "analyze_io_memory.py"
)
ana = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ana)


def mem_rec(t, pid, role, pss, ppid=100, **extra):
    return {
        "stream": "memory",
        "op": "sample",
        "path": "sample",
        "t_start": 1000.0 + t,
        "dt": 0.0,
        "pid": pid,
        "ppid": ppid if role != "main" else 1,
        "host": "h",
        "role": role,
        "rss_mib": pss * 2,
        "pss_mib": pss,
        "pss_anon_mib": pss / 2,
        "pss_file_mib": pss / 4,
        "pss_shmem_mib": pss / 4,
        **extra,
    }


# main (pid 100) lives 0..6 s; the worker (pid 200) is sampled at 1 and 2 s, then exits
RECORDS = [
    mem_rec(0, 100, "main", 1000, gpu_alloc_mib=10, gpu_reserved_mib=20, gpu_max_alloc_mib=30),
    mem_rec(1, 100, "main", 1100, gpu_alloc_mib=50, gpu_reserved_mib=60, gpu_max_alloc_mib=70),
    mem_rec(1, 200, "worker0", 300),
    mem_rec(2, 100, "main", 1100, gpu_alloc_mib=5, gpu_reserved_mib=60, gpu_max_alloc_mib=70),
    mem_rec(2, 200, "worker0", 500),
    mem_rec(3, 100, "main", 1050),
    mem_rec(4, 100, "main", 1000),
    mem_rec(5, 100, "main", 1000),
    mem_rec(6, 100, "main", 1000),
]


@pytest.fixture
def mem():
    df = pd.DataFrame.from_records(RECORDS)
    df["t_rel"] = df["t_start"] - df["t_start"].min()
    return ana.prepare_memory(df)


def test_workers_grouped_with_their_rank(mem):
    assert set(mem.loc[mem["role"] == "worker0", "group"]) == {100}
    assert set(mem.loc[mem["role"] == "main", "group"]) == {100}


def test_process_peaks(mem):
    peaks = ana.process_peaks(mem).set_index("role")
    assert peaks.loc["main", "pss_mib"] == 1100
    assert peaks.loc["worker0", "pss_mib"] == 500
    assert peaks.loc["worker0", "t_peak_s"] == pytest.approx(2.0)


def test_timeline_sums_pss_and_stops_counting_exited_workers(mem):
    tl = ana.memory_timeline(mem, 1.0).set_index("t_s")
    assert tl.loc[0, "total_mib"] == 1000  # worker not started yet
    assert tl.loc[1, "total_mib"] == 1100 + 300
    assert tl.loc[2, "total_mib"] == 1100 + 500
    # the exited worker is carried for 1.5 sampling intervals (2 bins), then dropped
    assert tl.loc[4, "total_mib"] == 1000 + 500
    assert tl.loc[5, "total_mib"] == 1000
    assert tl.loc[2, "workers_mib"] == 500 and tl.loc[2, "main_mib"] == 1100
    assert tl["total_mib"].idxmax() == 2


def test_process_without_sample_in_window_is_carried_forward(mem):
    inside = mem[mem["t_rel"].between(0.5, 2.0)]  # main has no sample at 2.0 only by window edge
    inside = inside[~((inside["role"] == "main") & (inside["t_rel"] > 1))]
    tl = ana.memory_timeline(inside, 1.0).set_index("t_s")
    assert tl.loc[2, "total_mib"] == 1100 + 500  # main carried from t=1, worker sampled at 2


def test_timeline_gpu_columns_are_max_over_main_processes(mem):
    tl = ana.memory_timeline(mem, 1.0)
    assert tl["gpu_alloc_mib"].max() == 50
    assert tl["gpu_max_alloc_mib"].max() == 70


def test_rss_used_where_pss_is_missing():
    df = pd.DataFrame.from_records([mem_rec(0, 100, "main", 10)]).drop(columns=["pss_mib"])
    df["t_rel"] = 0.0
    assert ana.prepare_memory(df)["mem_mib"].iloc[0] == 20  # rss = 2 * 10


def test_end_to_end_cli(tmp_path, capsys, monkeypatch):
    d = tmp_path / "run"
    d.mkdir()
    (d / "io_timing_h_100.jsonl").write_text("\n".join(json.dumps(r) for r in RECORDS))
    csv = tmp_path / "memory.csv"
    monkeypatch.setattr(
        "sys.argv", ["analyze_io_memory.py", str(d), "--memory-csv", str(csv), "--memory-bin", "1"]
    )
    ana.main()
    out = capsys.readouterr().out
    assert "peak 1600 MiB" in out and "worker0" in out
    assert csv.exists() and "total_mib" in csv.read_text()


def test_plot_writes_image_with_and_without_gpu(mem, tmp_path):
    pytest.importorskip("matplotlib")
    timeline = ana.memory_timeline(mem, 1.0)
    spans = {"startup": (0.5, 2.5)}
    with_gpu = tmp_path / "with_gpu.png"
    ana.plot_memory(timeline, spans, with_gpu, title="t")
    assert with_gpu.stat().st_size > 1000
    no_gpu = tmp_path / "no_gpu.png"
    ana.plot_memory(timeline.assign(gpu_reserved_mib=float("nan")), {}, no_gpu)
    assert no_gpu.stat().st_size > 1000


def test_window_bounds_relative_to_run_start():
    df = pd.DataFrame.from_records(
        [
            {"stream": "nsys-window", "op": "startup open", "t_start": 1010.0},
            {"stream": "nsys-window", "op": "startup close", "t_start": 1030.0},
            {"stream": "memory", "op": "sample", "t_start": 1000.0},
        ]
    )
    assert ana.window_bounds(df, 1000.0) == {"startup": (10.0, 30.0)}
