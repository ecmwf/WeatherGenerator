# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import dataclasses
from json import dumps, loads
from logging import getLogger

from omegaconf import OmegaConf

import weathergen.common.config as config

_logger = getLogger(__name__)


@dataclasses.dataclass
class _HistoryItem:
    run_id: str
    istep: int


@dataclasses.dataclass
class RunState:
    world_size: int
    rank: int
    local_rank: int
    with_fsdp: dataclasses.InitVar[bool] # only use during initialization
    # set/modified when loading additional RunState from file
    istep: int = 0
    world_size_original: int | None = None
    run_history: list[_HistoryItem] = dataclasses.field(default_factory=list)

    def __post_init__(self, with_fsdp):
        self.with_ddp = self.world_size > 1
        self.is_sharded = self.with_ddp and with_fsdp

    @staticmethod
    def _get_file_name(run_id: str, mini_epoch: int):
        if mini_epoch == -1:
            mini_epoch_str = "latest"
        else:
            mini_epoch_str = f"chkpt{mini_epoch:05d}"
        return f"runstate_{run_id}_{mini_epoch_str}.json"

    def save(self, run_id: str, mini_epoch: int):
        model_path = config.get_path_model(run_id=run_id)
        assert model_path.is_dir(), f"Missing model directory for runstate at: {model_path}"

        runstate = OmegaConf.structured(self)
        json_str = dumps(OmegaConf.to_container(runstate)) + "\n"
        with open(model_path / self._get_file_name(run_id, mini_epoch), "w") as f:
            f.write(json_str)

    def load(self, run_id: str, mini_epoch: int):
        model_path = config.get_path_model(run_id=run_id)
        filename = model_path / self._get_file_name(run_id, mini_epoch)
        assert filename.is_file(), (
            f"Cannot load runstate for id: \
            {run_id} and mini epoch: {mini_epoch}: file not found at {filename}"
        )

        with filename.open() as f:
            json_str = f.read()

        deserialized_runstate = OmegaConf.create(loads(json_str))
        typed_runstate = OmegaConf.merge(OmegaConf.structured(RunState),deserialized_runstate)
        runstate = OmegaConf.to_object(typed_runstate)
        assert isinstance(runstate, RunState)  # for type checker

        # bookkeeping
        self.istep = runstate.istep
        # remember world size from first ever run
        self.world_size_original = runstate.world_size_original
        self.run_history.append(_HistoryItem(run_id, runstate.istep))
