# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Opt-in sampling of host (and rank-process GPU) memory into the io_timing records.

Enabled with ``WEATHERGEN_IO_TIMING_MEMORY=1`` (period: ``WEATHERGEN_IO_TIMING_MEMORY_INTERVAL``
seconds, default 1; see ``weathergen.utils.io_timing``). The first ``io_timer`` call in a
process starts a daemon thread there; in data loader workers that happens in the worker itself,
because threads do not survive ``fork``. Each sample is one record of the process's JSON Lines
file with ``stream`` = ``memory`` and ``op`` = ``sample``:

- ``role``: ``main`` or ``worker<id>``; ``ppid``: parent process id (a worker's ``ppid`` is the
  pid of its rank's main process).
- ``rss_mib``, ``pss_mib``, ``pss_anon_mib``, ``pss_file_mib``, ``pss_shmem_mib``: from
  ``/proc/self/smaps_rollup``. PSS divides shared pages among the processes sharing them, so
  summing ``pss_mib`` over all processes of a node gives its real use; summing RSS counts shared
  pages (fork-inherited memory, shared memory used for batches) once per process. Without
  ``smaps_rollup`` only ``rss_mib`` (from ``/proc/self/status``) is recorded.
- ``gpu_alloc_mib``, ``gpu_reserved_mib``, ``gpu_max_alloc_mib``: PyTorch allocator state, only
  in a main process whose CUDA is initialized (never touched from workers, which are forks of a
  CUDA-initialized process).

Only Linux has ``/proc``; elsewhere nothing is sampled.
"""

import logging
import os
import threading
import time
from pathlib import Path

from weathergen.utils.io_timing import _HOST, write_record

_logger = logging.getLogger("weathergen.io_timing")

STREAM = "memory"
OP = "sample"
_KIB = 1024.0

_SMAPS_FIELDS = {
    "Rss": "rss_mib",
    "Pss": "pss_mib",
    "Pss_Anon": "pss_anon_mib",
    "Pss_File": "pss_file_mib",
    "Pss_Shmem": "pss_shmem_mib",
}

_started_pid: int | None = None
_start_lock = threading.Lock()


def interval() -> float:
    """Sampling period in seconds."""
    try:
        return max(float(os.environ.get("WEATHERGEN_IO_TIMING_MEMORY_INTERVAL", "1")), 0.01)
    except ValueError:
        return 1.0


def _parse_kib_lines(text: str, fields: dict[str, str]) -> dict[str, float]:
    """Values (MiB) of the ``Name:   123 kB`` lines of /proc files whose name is in ``fields``."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        if name in fields:
            out[fields[name]] = float(rest.split()[0]) / _KIB
    return out


def read_host_memory(proc: Path = Path("/proc/self")) -> dict[str, float] | None:
    """Host memory of this process in MiB, or None where /proc is unavailable."""
    try:
        text = (proc / "smaps_rollup").read_text()
        values = _parse_kib_lines(text, _SMAPS_FIELDS)
        if values:
            return values
    except OSError:
        pass
    try:
        return _parse_kib_lines((proc / "status").read_text(), {"VmRSS": "rss_mib"})
    except OSError:
        return None


def _role() -> str:
    import torch  # noqa: PLC0415 (lazy: only in processes that already use torch)

    info = torch.utils.data.get_worker_info()
    return "main" if info is None else f"worker{info.id}"


def read_gpu_memory() -> dict[str, float]:
    """PyTorch allocator state of the current device; empty unless CUDA is initialized.

    Must not be called from a data loader worker (a fork of a CUDA-initialized process).
    """
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return {}
    mib = 1024.0**2
    return {
        "gpu_alloc_mib": torch.cuda.memory_allocated() / mib,
        "gpu_reserved_mib": torch.cuda.memory_reserved() / mib,
        "gpu_max_alloc_mib": torch.cuda.max_memory_allocated() / mib,
    }


def sample_once() -> bool:
    """Write one sample record for this process; False if host memory is not readable."""
    host = read_host_memory()
    if host is None:
        return False
    role = _role()
    record = {
        "stream": STREAM,
        "op": OP,
        "path": OP,
        "t_start": time.time(),
        "dt": 0.0,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "host": _HOST,
        "role": role,
        **{k: round(v, 3) for k, v in host.items()},
    }
    if role == "main":
        record.update({k: round(v, 3) for k, v in read_gpu_memory().items()})
    write_record(record)
    return True


def _loop(period: float) -> None:
    while True:
        try:
            if not sample_once():
                return  # no /proc here; nothing will ever be readable
        except Exception:  # a failing sample must never take the process down
            _logger.exception("memory sample failed")
            return
        time.sleep(period)


def _reinit_after_fork() -> None:
    global _start_lock
    _start_lock = threading.Lock()


os.register_at_fork(after_in_child=_reinit_after_fork)


def ensure_started() -> None:
    """Start the sampler thread of this process unless it is already running (fork-safe)."""
    global _started_pid
    pid = os.getpid()
    if _started_pid == pid:
        return
    with _start_lock:
        if _started_pid == pid:
            return
        _started_pid = pid
        threading.Thread(
            target=_loop, args=(interval(),), name="io-timing-memory", daemon=True
        ).start()
