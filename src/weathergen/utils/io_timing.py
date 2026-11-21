# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Annotations of data reading and preprocessing for profiling.

``io_timer(stream, op)`` marks a block of the data pipeline with an NVTX range
``io:<stream> : <op>`` for Nsight Systems; nested timers give nested ranges. The ranges are on
in nsys runs (``launch-slurm.py --nsys-profiling``, which sets ``WEATHERGEN_NSYS_PROFILING=1``)
and with ``WEATHERGEN_IO_TIMING_NVTX=1``; otherwise ``io_timer`` is a no-op.

Environment variables (read once at import, so set them before the process and its data
loader workers start; on the clusters ``launch-slurm.py`` sets them per job):

- ``WEATHERGEN_IO_TIMING_MEMORY=1`` (``launch-slurm.py --io-timing-memory``) starts a
  background thread in every process that uses ``io_timer`` (rank, data loader workers) which
  appends host-memory samples to JSON Lines files; see ``weathergen.utils.memory_sampler``.
  ``WEATHERGEN_IO_TIMING_MEMORY_INTERVAL`` is the sampling period in seconds (default 1). The
  capture-window boundaries of ``weathergen.utils.nsys_windows`` are recorded as events in the
  same files while this is on.
- ``WEATHERGEN_IO_TIMING_DIR`` is the directory of those files (default ``./io_timing``; the
  launcher uses ``logs/<run-id>/profiling/io_timing``). Every process writes its own file
  ``io_timing_<host>_<pid>.jsonl``.

Analyze the memory samples with ``scripts/analyze_io_memory.py``.
"""

import json
import os
import pathlib
import socket
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import TextIO


def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes")


NVTX_ENABLED = _env_flag("WEATHERGEN_IO_TIMING_NVTX", "0") or _env_flag(
    "WEATHERGEN_NSYS_PROFILING", "0"
)
MEMORY_ENABLED = _env_flag("WEATHERGEN_IO_TIMING_MEMORY", "0")
_OUT_DIR = pathlib.Path(os.environ.get("WEATHERGEN_IO_TIMING_DIR", "io_timing"))
_HOST = socket.gethostname()

# Output file of the current process; reopened after a fork (data loader workers).
_out: TextIO | None = None
_out_pid: int | None = None
_out_lock = threading.Lock()


def _reinit_after_fork() -> None:
    """Fresh lock and output file in a forked child (data loader worker).

    The memory sampler thread may hold the lock while the parent forks; the child would then
    inherit a lock that nobody can release.
    """
    global _out, _out_lock
    _out_lock = threading.Lock()
    _out = None


os.register_at_fork(after_in_child=_reinit_after_fork)


def write_record(record: dict) -> None:
    """Append ``record`` to this process's JSON Lines file."""
    global _out, _out_pid
    with _out_lock:
        pid = os.getpid()
        if _out is None or _out_pid != pid:
            _OUT_DIR.mkdir(parents=True, exist_ok=True)
            fname = _OUT_DIR / f"io_timing_{_HOST}_{pid}.jsonl"
            # line buffered: workers can be terminated without running exit handlers
            _out = open(fname, "a", buffering=1)  # noqa: SIM115
            _out_pid = pid
        _out.write(json.dumps(record) + "\n")


@contextmanager
def io_timer(stream: str, op: str) -> Generator[None]:
    """Mark the ``with`` block as ``op`` of ``stream`` with an NVTX range, if enabled."""
    if not (NVTX_ENABLED or MEMORY_ENABLED):
        yield
        return
    if MEMORY_ENABLED:
        _ensure_memory_sampler()
    if NVTX_ENABLED:
        _nvtx_push(f"io:{stream} : {op}")
    try:
        yield
    finally:
        if NVTX_ENABLED:
            _nvtx_pop()


def _ensure_memory_sampler() -> None:
    from weathergen.utils import memory_sampler  # noqa: PLC0415 (circular import)

    memory_sampler.ensure_started()


def _nvtx_push(name: str) -> None:
    import torch  # noqa: PLC0415 (lazy: keep this module importable without torch)

    torch.cuda.nvtx.range_push(name)


def _nvtx_pop() -> None:
    import torch  # noqa: PLC0415

    torch.cuda.nvtx.range_pop()


def io_event(stream: str, op: str) -> None:
    """Record an instantaneous marker, e.g. the boundaries of a capture window.

    Only while memory sampling is on: the markers split the memory samples by window.
    """
    if not MEMORY_ENABLED:
        return
    write_record(
        {
            "stream": stream,
            "op": op,
            "path": op,
            "t_start": time.time(),
            "dt": 0.0,
            "pid": os.getpid(),
            "host": _HOST,
        }
    )
