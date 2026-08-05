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
from pathlib import Path

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
    # set/modified when loading additional RunState from file
    with_ddp: bool | None = None
    is_sharded: bool | None = None
    istep: int = 0
    world_size_original: int | None = None
    run_history: list[_HistoryItem] = dataclasses.field(default_factory=list)

    @staticmethod
    def _get_file_name(run_id: str, mini_epoch: int):
        if mini_epoch == -1:
            mini_epoch_str = "latest"
        else:
            mini_epoch_str = f"chkpt{mini_epoch:05d}"
        return f"runstate_{run_id}_{mini_epoch_str}.json"

    def save(self, run_id: str, mini_epoch: int):
        """
        Save runstate artifact as json in model directory.

        The artifact is either suffixed with a particular mini-epoch or with '_latest'
        """
        model_path = config.get_path_model(run_id=run_id)
        assert model_path.is_dir(), f"Missing model directory for runstate at: {model_path}"

        runstate = OmegaConf.structured(self)
        json_str = dumps(OmegaConf.to_container(runstate)) + "\n"
        with open(model_path / self._get_file_name(run_id, mini_epoch), "w") as f:
            f.write(json_str)

    def load(self, run_id: str, mini_epoch: int):
        """
        Load required information from previous run into current runstate.

        Loaded from previous run:
          - optimization step counts
          - run history (previous run gets added)
          - world size of first ever run in history to consistently calculate mini-epochs.
        """
        model_path = config.get_path_model(run_id=run_id)
        filename = model_path / self._get_file_name(run_id, mini_epoch)

        # TODO: remove after transition period
        self._apply_fix(run_id, mini_epoch, model_path)

        assert filename.is_file(), (
            f"Cannot load runstate for id: \
            {run_id} and mini epoch: {mini_epoch}: file not found at {filename}"
        )

        with filename.open() as f:
            json_str = f.read()

        deserialized_runstate = OmegaConf.create(loads(json_str))
        typed_runstate = OmegaConf.merge(OmegaConf.structured(RunState), deserialized_runstate)
        runstate = OmegaConf.to_object(typed_runstate)
        assert isinstance(runstate, RunState)  # for type checker

        self.istep = runstate.istep
        self.world_size_original = runstate.world_size_original
        self.run_history.append(_HistoryItem(run_id, runstate.istep))

    # TODO remove after transition period
    def _apply_fix(self, run_id: str, mini_epoch: int, model_path: Path):
        """
        Best effort backward compatibility.

        Detect if previous run has not implemented yet config/runstate split.
        Create required runstate file.
        """

        filename = model_path / self._get_file_name(run_id, mini_epoch)

        if not filename.is_file():
            _logger.info("Missing RunState, try to obtain RunState from config")
            _logger.info(f"config file: {filename}")

            cf = config.load_run_config(run_id=run_id, mini_epoch=mini_epoch, model_path=model_path)

            try:
                runstate = RunState(
                    world_size=cf.general.world_size,
                    rank=cf.general.rank,
                    local_rank=cf.general.local_rank,
                    istep=cf.general.istep,
                    world_size_original=cf.general.world_size_original,
                    run_history=[
                        _HistoryItem(run_id, istep) for run_id, istep in cf.general.run_history
                    ],
                )
                runstate.save(run_id, mini_epoch)
            except AttributeError:
                _logger.warning("Cannot construct RunState from config.")
