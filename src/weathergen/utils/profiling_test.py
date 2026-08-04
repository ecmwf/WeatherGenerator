# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from omegaconf import OmegaConf

from weathergen.common.config import _DEFAULT_CONFIG_PTH
from weathergen.utils.profiling import (
    PerformanceLoggingConfig,
    ProfilingConfig,
    ProfilingSchedule,
)

_PERFORMANCE_CONFIG_PTH = _DEFAULT_CONFIG_PTH.parent / "config_performance.yml"


def test_schedule_from_config():
    cfg = OmegaConf.create(
        {"wait_iteration": 2, "warmup_iteration": 3, "active_iteration": 4, "repeat": 5}
    )
    schedule = ProfilingSchedule.from_config(cfg)

    assert schedule == ProfilingSchedule(wait=2, warmup=3, active=4, repeat=5)
    assert schedule.num_steps == (2 + 3 + 4) * 5
    assert schedule.steps_before_active == 2 + 3


def test_schedule_is_shared_by_the_collectors():
    """The schedule sits on `profiling`, not on `pytorch_profiler`."""
    cfg = OmegaConf.create(
        {
            "profiling": {
                "active_iteration": 4,
                "pytorch_profiler": {"enabled": False},
                "memory_snapshot": {"enabled": True},
            }
        }
    )
    profiling_cfg = ProfilingConfig.from_config(cfg)

    assert profiling_cfg.schedule == ProfilingSchedule(active=4)
    assert not profiling_cfg.pytorch_profiler
    assert profiling_cfg.memory_snapshot


def test_collecting_traces_needs_profiling_enabled():
    cfg = OmegaConf.create({"profiling": {"enabled": False, "memory_snapshot": {"enabled": True}}})

    assert not ProfilingConfig.from_config(cfg).collects_traces


def test_performance_logging_is_independent_of_profiling():
    cfg = OmegaConf.create({"performance_logging": {"throughput": {"warmup_steps": 5}}})
    performance_cfg = PerformanceLoggingConfig.from_config(cfg)

    assert not ProfilingConfig.from_config(cfg).enabled
    assert not performance_cfg.enabled, "throughput is off by default"
    assert performance_cfg.throughput_warmup_steps == 5

    cfg.performance_logging.throughput.enabled = True
    assert PerformanceLoggingConfig.from_config(cfg).enabled


def test_config_without_the_sections():
    """Run configs from before the sections existed (e.g. when continuing) fall back."""
    cfg = OmegaConf.create({})

    assert ProfilingConfig.from_config(cfg) == ProfilingConfig()
    assert PerformanceLoggingConfig.from_config(cfg) == PerformanceLoggingConfig()


def test_config_defaults_match_default_config():
    """The dataclass defaults are a copy of the config defaults; keep them in sync."""
    default_cfg = OmegaConf.load(_DEFAULT_CONFIG_PTH)

    assert ProfilingConfig.from_config(default_cfg) == ProfilingConfig()
    assert PerformanceLoggingConfig.from_config(default_cfg) == PerformanceLoggingConfig()


def test_performance_config_measures_everything():
    cfg = OmegaConf.merge(
        OmegaConf.load(_DEFAULT_CONFIG_PTH), OmegaConf.load(_PERFORMANCE_CONFIG_PTH)
    )
    profiling_cfg = ProfilingConfig.from_config(cfg)

    assert profiling_cfg.enabled
    assert profiling_cfg.stop_after_profiling
    assert profiling_cfg.collects_traces
    assert profiling_cfg.schedule.num_steps == 5
    assert PerformanceLoggingConfig.from_config(cfg).enabled
