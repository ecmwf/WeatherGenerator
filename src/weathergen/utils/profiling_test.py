# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from weathergen.common.config import _DEFAULT_CONFIG_PTH
from weathergen.utils import profiling
from weathergen.utils.performance import ThroughputTracker
from weathergen.utils.profiling import ProfilingConfig, ProfilingSchedule

_PERFORMANCE_CONFIG_PTH = _DEFAULT_CONFIG_PTH.parent / "config_performance.yml"


def test_schedule_from_section():
    cfg = OmegaConf.create(
        {"wait_iteration": 2, "warmup_iteration": 3, "active_iteration": 4, "repeat": 5}
    )
    schedule = ProfilingSchedule.from_section(cfg)

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


def test_nvtx_annotation_needs_profiling_enabled():
    cfg = OmegaConf.create({"profiling": {"enabled": False, "nvtx_annotate": True}})
    assert not ProfilingConfig.from_config(cfg).annotates_nvtx

    cfg.profiling.enabled = True
    assert ProfilingConfig.from_config(cfg).annotates_nvtx


def test_memory_snapshot_failure_is_not_reported_as_written(tmp_path: Path, caplog):
    with (
        patch.object(profiling, "_trace_file_prefix", return_value=tmp_path / "snapshot"),
        patch.object(profiling.torch.cuda.memory, "_dump_snapshot", side_effect=RuntimeError),
        caplog.at_level(logging.INFO, logger=profiling.__name__),
    ):
        profiling._export_memory_snapshot(OmegaConf.create({}))

    assert "Failed to capture memory snapshot" in caplog.text
    assert "written" not in caplog.text


def test_config_without_the_sections():
    """
    Neither section is in default_config.yml, so every run config may be missing them.

    That includes configs of runs that predate a key and are continued, which is why the
    dataclass defaults are the only defaults.
    """
    default_cfg = OmegaConf.load(_DEFAULT_CONFIG_PTH)

    assert "profiling" not in default_cfg
    assert "performance_logging" not in default_cfg
    assert ProfilingConfig.from_config(default_cfg) == ProfilingConfig()
    assert not ThroughputTracker.is_enabled(default_cfg)
    assert ProfilingConfig.from_config(OmegaConf.create({})) == ProfilingConfig()


def test_performance_config_measures_everything():
    cfg = OmegaConf.merge(
        OmegaConf.load(_DEFAULT_CONFIG_PTH), OmegaConf.load(_PERFORMANCE_CONFIG_PTH)
    )
    profiling_cfg = ProfilingConfig.from_config(cfg)

    assert profiling_cfg.enabled
    assert profiling_cfg.stop_after_profiling
    assert profiling_cfg.collects_traces
    assert profiling_cfg.schedule.num_steps == 5
    assert ThroughputTracker.is_enabled(cfg)
