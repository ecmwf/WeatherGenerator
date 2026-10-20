# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
from omegaconf import OmegaConf

from weathergen.utils.performance import ThroughputTracker
from weathergen.utils.profiling import ProfilingConfig

_DEVICE = torch.device("cpu")


def test_throughput_tracker_from_config():
    cfg = OmegaConf.create({"performance_logging": {"throughput": {"warmup_steps": 5}}})

    assert not ProfilingConfig.from_config(cfg).enabled
    assert not ThroughputTracker.is_enabled(cfg), "throughput is off by default"
    assert ThroughputTracker.from_config(cfg, _DEVICE, batch_size_per_gpu=1) is None

    cfg.performance_logging.throughput.enabled = True
    tracker = ThroughputTracker.from_config(cfg, _DEVICE, batch_size_per_gpu=1)

    assert tracker is not None
    assert tracker._warmup_steps == 5


def test_throughput_tracker_without_the_section():
    tracker_cfg = OmegaConf.create({"performance_logging": {"throughput": {"enabled": True}}})
    tracker = ThroughputTracker.from_config(tracker_cfg, _DEVICE, batch_size_per_gpu=1)

    assert tracker is not None
    assert tracker._warmup_steps == ThroughputTracker.DEFAULT_WARMUP_STEPS
    assert not ThroughputTracker.is_enabled(OmegaConf.create({}))
