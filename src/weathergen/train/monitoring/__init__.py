# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Training-time monitoring of grads, weights, updates, activations and attention.

Configured by `train_logging.monitoring` (merged with `MONITORING_DEFAULTS`); metrics are logged
as `mon/<monitor>/<group>/<stat>` through `TrainLogger.log_metrics("train", ...)`.
"""

from __future__ import annotations

import logging

from omegaconf import DictConfig, OmegaConf

# import monitor modules so that they register themselves
from weathergen.train.monitoring import activations, attention, grads, weights  # noqa: F401
from weathergen.train.monitoring.base import Monitor, StatBuffer, get_registry, register_monitor
from weathergen.train.monitoring.manager import MonitorManager, NullMonitorManager
from weathergen.train.monitoring.selection import Selection

logger = logging.getLogger(__name__)

__all__ = [
    "MONITORING_DEFAULTS",
    "Monitor",
    "MonitorManager",
    "NullMonitorManager",
    "StatBuffer",
    "build_monitor_manager",
    "register_monitor",
]

# Per-monitor `every`, `include`, `group_by` override the global ones; `exclude` is added to
# the global list. `sample_layers` (activations / attention) keeps, per block container
# (e.g. `fe_blocks`), the given positions: ints (negative from the end) or "mid".
MONITORING_DEFAULTS = {
    "enabled": False,
    "every": 100,  # istep cadence
    "group_by": "block",  # param | block | module_type
    "include": ["*"],  # FQN globs
    "exclude": ["*target_token_engines*", "*ema*"],
    "monitors": {
        "grads": {"enabled": True, "every": 20},
        "weights": {"enabled": True},
        "updates": {"enabled": True, "eps": 1e-12},
        "activations": {
            "enabled": True,
            "include": ["*fe_blocks*", "*ae_global_engine*"],
            "sample_layers": [0, "mid", -1],
        },
        "attention": {"enabled": True, "every": 500, "num_queries": 128},
    },
}


def _monitor_cfg(cfg: DictConfig, name: str) -> DictConfig:
    node = cfg.monitors[name]
    mcfg = OmegaConf.to_container(node, resolve=True) if node is not None else {}
    for key in ("every", "group_by", "include"):
        if mcfg.get(key) is None:
            mcfg[key] = OmegaConf.to_container(cfg[key]) if key == "include" else cfg[key]
    mcfg["exclude"] = list(cfg.exclude or []) + list(mcfg.get("exclude") or [])
    return OmegaConf.create(mcfg)


def build_monitor_manager(cfg, model, optimizers, train_logger):
    """Build the manager from the `train_logging.monitoring` config section (dict or
    DictConfig). Returns a `NullMonitorManager` when monitoring is disabled.

    `optimizers` is unused for now; it is reserved for optimizer-state monitors (Adam √v).
    """
    cfg = OmegaConf.merge(OmegaConf.create(MONITORING_DEFAULTS), cfg or {})
    if not cfg.enabled:
        return NullMonitorManager()

    registry = get_registry()
    monitors = []
    for name in cfg.monitors:
        node = cfg.monitors[name]
        if node is not None and not node.get("enabled", True):
            continue
        if name not in registry:
            raise ValueError(
                f"Unknown monitor '{name}' in train_logging.monitoring.monitors; "
                f"available: {sorted(registry)}"
            )
        mcfg = _monitor_cfg(cfg, name)
        selection = Selection(
            mcfg.include, mcfg.exclude, mcfg.group_by, mcfg.get("sample_layers", None)
        )
        monitors.append(registry[name](mcfg, model, selection))

    if not monitors:
        return NullMonitorManager()
    logger.info(
        "Training monitoring enabled: %s",
        ", ".join(f"{m.name} (every {m.every})" for m in monitors),
    )
    return MonitorManager(model, monitors, train_logger)
