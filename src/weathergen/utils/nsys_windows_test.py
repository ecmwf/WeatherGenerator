# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from weathergen.utils import nsys_windows
from weathergen.utils.nsys_windows import NsysWindows


@pytest.fixture
def calls(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(nsys_windows.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(nsys_windows.torch.cuda.profiler, "start", lambda: calls.append("start"))
    monkeypatch.setattr(nsys_windows.torch.cuda.profiler, "stop", lambda: calls.append("stop"))
    return calls


def run_loop(w: NsysWindows, steps: int, log: list[str]) -> None:
    w.before_data_iter()
    for step in range(steps):
        w.before_step(step)
        w.after_forward(step)
        w.after_step(step)
        log.append(f"step{step}:{w._open}")


def test_disabled_without_config(monkeypatch):
    monkeypatch.delenv(nsys_windows.ENV_VAR, raising=False)
    assert NsysWindows.from_config({}) is None
    assert NsysWindows.from_config({"profiling": {"nsys_windows": {"enabled": False}}}) is None
    # profiling.enabled is the master switch of the section
    assert NsysWindows.from_config({"profiling": {"nsys_windows": {"enabled": True}}}) is None
    assert NsysWindows.from_config(
        {"profiling": {"enabled": True, "nsys_windows": {"enabled": True}}}
    )


def test_two_windows_in_order(calls):
    log: list[str] = []
    run_loop(NsysWindows(warmup_steps=2, steady_steps=4, stop_after_capture=False), 10, log)
    assert calls == ["start", "stop", "start", "stop"]
    # startup closes after the first forward; steady covers steps 2..5
    assert [e for e in log if e.endswith("None")] == [
        "step0:None",
        "step1:None",
        "step5:None",
        "step6:None",
        "step7:None",
        "step8:None",
        "step9:None",
    ]
    assert log[2] == "step2:steady" and log[4] == "step4:steady"


def test_windows_open_once(calls):
    w = NsysWindows(warmup_steps=1, steady_steps=1, stop_after_capture=False)
    run_loop(w, 3, [])
    run_loop(w, 3, [])  # second mini-epoch
    assert calls == ["start", "stop", "start", "stop"]


def test_only_steady(calls):
    run_loop(
        NsysWindows(startup=False, warmup_steps=1, steady_steps=2, stop_after_capture=False), 5, []
    )
    assert calls == ["start", "stop"]


def test_env_overrides_config(monkeypatch):
    cf = {"profiling": {"nsys_windows": {"enabled": False, "startup": True, "warmup_steps": 3}}}
    monkeypatch.setenv(nsys_windows.ENV_VAR, "steady")
    w = NsysWindows.from_config(cf)
    assert w is not None and not w.startup and w.steady and w.warmup_steps == 3
    for value in ("startup,steady", "startup:steady"):
        monkeypatch.setenv(nsys_windows.ENV_VAR, value)
        w = NsysWindows.from_config({})
        assert w is not None and w.startup and w.steady


def test_env_unknown_window(monkeypatch):
    monkeypatch.setenv(nsys_windows.ENV_VAR, "startup,bogus")
    with pytest.raises(ValueError):
        NsysWindows.from_config({})


def test_stop_after_capture(calls):
    w = NsysWindows(warmup_steps=1, steady_steps=2, stop_after_capture=True)
    reached: list[int] = []
    w.before_data_iter()
    for step in range(10):
        w.before_step(step)
        w.after_forward(step)
        w.after_step(step)
        reached.append(step)
        if w.ends_run:
            break
    # the run ends after the step closing the last window (steady: steps 1..2), never earlier
    assert reached == [0, 1, 2] and w.last_step == 2
    assert calls == ["start", "stop", "start", "stop"]


def test_stop_after_startup_only(calls):
    w = NsysWindows(steady=False, stop_after_capture=True)
    w.before_data_iter()
    assert not w.ends_run
    w.after_forward(0)
    assert w.ends_run and w.last_step == 0


def test_no_stop_without_stop_after_capture(calls):
    w = NsysWindows(steady=False, stop_after_capture=False)
    w.before_data_iter()
    w.after_forward(0)
    assert w.done and not w.ends_run


def test_stop_default_on_and_env_override(monkeypatch):
    cf = {"profiling": {"enabled": True, "nsys_windows": {"enabled": True}}}
    monkeypatch.delenv(nsys_windows.ENV_VAR_STOP, raising=False)
    w = NsysWindows.from_config(cf)
    assert w is not None and w.stop_after_capture
    monkeypatch.setenv(nsys_windows.ENV_VAR_STOP, "0")
    w = NsysWindows.from_config(cf)
    assert w is not None and not w.stop_after_capture
    # the environment also overrides an explicit config value, in both directions
    cf["profiling"]["nsys_windows"]["stop_after_capture"] = False
    monkeypatch.setenv(nsys_windows.ENV_VAR_STOP, "1")
    w = NsysWindows.from_config(cf)
    assert w is not None and w.stop_after_capture
