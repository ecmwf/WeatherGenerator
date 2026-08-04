# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Configuration and helpers for measuring a training run.

The `profiling` and `performance_logging` sections of a run config are parsed here; the
trainer side of both lives in `weathergen.train.profiling_trainer`. Profiling traces a
bounded stretch of training (PyTorch profiler, CUDA memory snapshot) and is expensive;
performance logging measures the run as a whole (throughput, later peak memory) and is not.
"""

import contextlib
import dataclasses
import logging
import platform
from collections.abc import Callable, Iterator
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Protocol

import torch
from torch.profiler import ProfilerActivity, profile, record_function

import weathergen.common.config as config
from weathergen.common.config import Config
from weathergen.utils.distributed import get_rank, is_root

logger: logging.Logger = logging.getLogger(__name__)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000


class BatchTracker(Protocol):
    """
    What `ProfilingTrainer` expects of a per-step measurement tool.

    `step` is called once per training step, after that step has completed, on every rank
    (`ThroughputTracker` and anything else that syncs across ranks relies on that). It is
    given the batch that was just trained on, the step index it was trained at, and a
    `log_fn` that writes a metrics dict to the train logger at that step.
    """

    def step(
        self, batch, istep: int, log_fn: Callable[[dict[str, float]], None] | None = None
    ) -> None: ...


@dataclasses.dataclass(frozen=True)
class ProfilingSchedule:
    """
    The wait/warmup/active/repeat cycle of the profiled stretch, in training steps.

    It belongs to the profiling section as a whole, not to one collector: the PyTorch
    profiler steps through it, the memory snapshot records from the first active step
    onwards, and `num_steps` is how long a `stop_after_profiling` run lasts.
    """

    wait: int = 1
    warmup: int = 1
    active: int = 1
    repeat: int = 1

    @classmethod
    def from_config(cls, profiling_cfg: Config | dict) -> "ProfilingSchedule":
        """Read the schedule from the `profiling` section of a run config."""
        defaults = cls()
        return cls(
            wait=profiling_cfg.get("wait_iteration", defaults.wait),
            warmup=profiling_cfg.get("warmup_iteration", defaults.warmup),
            active=profiling_cfg.get("active_iteration", defaults.active),
            repeat=profiling_cfg.get("repeat", defaults.repeat),
        )

    @property
    def num_steps(self) -> int:
        """Number of training steps needed to walk the full schedule."""
        return (self.wait + self.warmup + self.active) * self.repeat

    @property
    def steps_before_active(self) -> int:
        """Steps run before the first active window, i.e. what collectors should skip."""
        return self.wait + self.warmup

    def to_torch(self) -> Callable[[int], torch.profiler.ProfilerAction]:
        return torch.profiler.schedule(
            wait=self.wait, warmup=self.warmup, active=self.active, repeat=self.repeat
        )


@dataclasses.dataclass(frozen=True)
class ProfilingConfig:
    """
    The `profiling` section of a run config: tracing a bounded stretch of training.

    The section is deliberately absent from `config/default_config.yml` — the defaults below
    are the only ones, so a run config that predates a key (e.g. when continuing an older
    run) needs no migration. `config/config_performance.yml` documents the keys.
    """

    # collect traces, which requires the ProfilingTrainer
    enabled: bool = False
    # end the run once the profiled stretch is done, instead of training as configured
    stop_after_profiling: bool = False
    # how long the profiled stretch is, and how it is split into wait/warmup/active
    schedule: ProfilingSchedule = ProfilingSchedule()
    # collectors, independent of each other; each one is opted into explicitly
    pytorch_profiler: bool = False
    memory_snapshot: bool = False
    nvtx_annotate: bool = False

    @classmethod
    def from_config(cls, cf: Config) -> "ProfilingConfig":
        cfg = cf.get("profiling") or {}
        pytorch_profiler_cfg = cfg.get("pytorch_profiler") or {}
        memory_snapshot_cfg = cfg.get("memory_snapshot") or {}
        defaults = cls()

        return cls(
            enabled=cfg.get("enabled", defaults.enabled),
            stop_after_profiling=cfg.get("stop_after_profiling", defaults.stop_after_profiling),
            schedule=ProfilingSchedule.from_config(cfg),
            pytorch_profiler=pytorch_profiler_cfg.get("enabled", defaults.pytorch_profiler),
            memory_snapshot=memory_snapshot_cfg.get("enabled", defaults.memory_snapshot),
            nvtx_annotate=cfg.get("nvtx_annotate", defaults.nvtx_annotate),
        )

    @property
    def collects_traces(self) -> bool:
        """Whether the profiled stretch writes anything to the traces directory."""
        return self.enabled and (self.pytorch_profiler or self.memory_snapshot)


@dataclasses.dataclass(frozen=True)
class PerformanceLoggingConfig:
    """
    The `performance_logging` section of a run config: how the run itself performs.

    Unlike profiling, these metrics are cheap, cover the whole run and are logged next to
    the training metrics rather than written to a trace. Throughput is the only one so far;
    peak memory is meant to join it. As with `ProfilingConfig`, the defaults below are the
    only ones; the section is not in `config/default_config.yml`.
    """

    throughput: bool = False
    throughput_warmup_steps: int = 2

    @classmethod
    def from_config(cls, cf: Config) -> "PerformanceLoggingConfig":
        cfg = cf.get("performance_logging") or {}
        throughput_cfg = cfg.get("throughput") or {}
        defaults = cls()

        return cls(
            throughput=throughput_cfg.get("enabled", defaults.throughput),
            throughput_warmup_steps=throughput_cfg.get(
                "warmup_steps", defaults.throughput_warmup_steps
            ),
        )

    @property
    def enabled(self) -> bool:
        """Whether anything is logged, i.e. whether the run needs the ProfilingTrainer."""
        return self.throughput


@contextlib.contextmanager
def pytorch_profiler_session(cf: Config, schedule: ProfilingSchedule) -> Iterator[profile | None]:
    """
    Run the enclosed block under the PyTorch profiler, on the root rank only.

    Yields the profiler on the root rank (call `.step()` on it once per training step) and
    None everywhere else. Each cycle of the schedule writes a chrome trace and a memory
    timeline to `config.get_path_profiling_traces(cf)`; a summary is logged at the end.
    """
    if not is_root():
        yield None
        return

    traces_path = _prepare_traces_path(cf)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True,
        with_flops=True,
        schedule=schedule.to_torch(),
        on_trace_ready=partial(trace_handler, cf),
    ) as prof:
        yield prof

    log_profiler_summary(prof)
    logger.info(f"PyTorch profiler traces written to {traces_path}")


@contextlib.contextmanager
def memory_snapshot_session(cf: Config) -> Iterator[None]:
    """
    Record the CUDA memory history over the enclosed block, on the root rank only.

    The snapshot is dumped to `config.get_path_profiling_traces(cf)` on exit and can be
    viewed at https://pytorch.org/memory_viz. Independent of the PyTorch profiler; the
    caller enters this once the schedule's wait and warmup steps are done.
    """
    if not is_root() or not _cuda_available():
        yield
        return

    traces_path = _prepare_traces_path(cf)
    logger.info("Starting snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT)
    try:
        yield
        _export_memory_snapshot(cf)
        logger.info(f"Memory snapshot written to {traces_path}")
    finally:
        logger.info("Stopping snapshot record_memory_history")
        torch.cuda.memory._record_memory_history(enabled=None)


def log_profiler_summary(prof: profile) -> None:
    """Log the aggregated profiler tables (FLOPs, time per module, memory)."""
    logger.info("\n" + "=" * 80 + "\nPROFILING SUMMARY\n" + "=" * 80)

    logger.info("\n--- Top Operations by FLOPs ---")
    logger.info(
        prof.key_averages().table(sort_by="flops", row_limit=20, top_level_events_only=False)
    )

    logger.info("\n--- Operations Grouped by Module ---")
    logger.info(
        prof.key_averages(group_by_stack_n=5).table(sort_by="cuda_time_total", row_limit=30)
    )

    logger.info("\n--- Memory Usage ---")
    logger.info(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))


def trace_handler(cf: Config, prof: profile) -> None:
    """Write the chrome trace and the memory timeline for one profiler cycle."""
    file_prefix = _trace_file_prefix(cf)

    prof.export_chrome_trace(f"{file_prefix}.json.gz")

    # the memory timeline relies on kineto functionality unavailable on aarch64
    if platform.machine() == "aarch64":
        logger.info("[profiler] Memory distribution timeline skipped on aarch64")
    else:
        prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")


def _export_memory_snapshot(cf: Config) -> None:
    file_prefix = _trace_file_prefix(cf)
    try:
        logger.info(f"Saving snapshot to local file: {file_prefix}.pickle")
        torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
    except Exception as e:
        logger.error(f"Failed to capture memory snapshot {e}")


def _prepare_traces_path(cf: Config) -> Path:
    traces_path = config.get_path_profiling_traces(cf)
    traces_path.mkdir(exist_ok=True, parents=True)
    return traces_path


def _trace_file_prefix(cf: Config) -> Path:
    """Timestamped, rank-specific path prefix shared by all profiling artifacts."""
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    return config.get_path_profiling_traces(cf) / f"{timestamp}_rank_{get_rank()}"


def _cuda_available() -> bool:
    if torch.cuda.is_available():
        return True

    logger.info("CUDA unavailable. Not recording memory history")
    return False


def wrap_module_forward_with_profiling(model: torch.nn.Module, prefix: str = "") -> None:
    """
    Recursively annotate the forward of every custom submodule with `record_function`.

    This makes the trace readable in terms of WeatherGenerator modules instead of bare aten
    ops. It patches `forward` on the module instances, so only use it on a model that is
    about to be profiled and then thrown away.
    """
    for name, module in model.named_children():
        module_name = f"{prefix}.{name}" if prefix else name

        # standard PyTorch modules are already traced, but their children may not be
        if not type(module).__module__.startswith("torch.nn.modules"):
            module.forward = _profiled_forward(module_name, module.forward)

        wrap_module_forward_with_profiling(module, module_name)


def _profiled_forward(module_name: str, forward: Callable) -> Callable:
    def profiled_forward(*args, **kwargs):
        with record_function(f"nn.Module: {module_name}"):
            return forward(*args, **kwargs)

    return profiled_forward
