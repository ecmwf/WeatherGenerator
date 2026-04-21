# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from omegaconf import OmegaConf, DictConfig

import logging
import json
from pathlib import Path

from weathergen.common.config import Config, get_path_model

RunState = DictConfig

_logger = logging.getLogger(__name__)

def init_runstate() -> RunState:

    startdict = {"istep": 0,
                 "world_size": None,
                 "world_size_original": None,
                 "rank": None,
                 "local_rank": None,
                 "with_ddp": None,
                 "is_sharded": None,
                 "run_history": []}

    runstate = OmegaConf.create(startdict)
    print("startdict: ", type(startdict))
    print("runstate:  ", type(runstate))
    assert isinstance(runstate, RunState)
    return runstate


def _get_runstate_file_write_name(run_id: str, mini_epoch: int | None):
    """Generate the filename for writing a model run state file."""
    if mini_epoch is None:
        mini_epoch_str = ""
    elif mini_epoch == -1:
        mini_epoch_str = "_latest"
    else:
        mini_epoch_str = f"_chkpt{mini_epoch:05d}"

    return f"runstate_{run_id}{mini_epoch_str}.json"


def save_runstate(runstate: RunState, config: Config, mini_epoch: int | None):
    """
    Save runstate
    """

    dirname = get_path_model(config)
    dirname.mkdir(exist_ok=True, parents=True)

    fname = _get_runstate_file_write_name(config.general.run_id, mini_epoch)

    json_str = json.dumps(OmegaConf.to_container(runstate)) + '\n'

    with (dirname/f"{fname}").open("w") as f:
        f.write(json_str)


def load_runstate(run_id: str, mini_epoch: int | None, model_path: str | None) -> RunState:
    """
    Load runstate
    """

    # Loading path
    if Path(run_id).exists():  # load from the full path if a full path is provided
        fname = Path(run_id)
        _logger.info(f"Loading run_state from provided full run_id path: {fname}")

    else:
        # Load model runstate here. In case model_path is not provided, get it from private conf
        if model_path is None:
            path = get_path_model(run_id=run_id)
        else:
            path = Path(model_path) / run_id

        runstate_path_with_epoch = path / _get_runstate_file_write_name(run_id, mini_epoch)
        runstate_path_without_epoch = path / _get_runstate_file_write_name(run_id, None)

        if runstate_path_with_epoch.exists():
            fname = runstate_path_with_epoch
            _logger.info(f"Loading runstate from specified run_id and mini_epoch: {fname}")
        elif runstate_path_without_epoch.exists():
            fname = runstate_path_without_epoch
            _logger.info(
                f"Runstate for mini_epoch {mini_epoch} not found. "
                f"Falling back to runstate without mini_epoch: {fname}"
            )

        else:
            raise FileNotFoundError(
                f"Could not find model runstate for run_id '{run_id}' "
                f"(mini_epoch={mini_epoch}) in '{path}'. "
                f"Tried: '{runstate_path_with_epoch.name}' and '{runstate_path_without_epoch.name}'. "
                f"Please check run_id and mini_epoch."
            )

    with fname.open() as f:
        json_str = f.read()

    runstate = OmegaConf.create(json.loads(json_str))

    return runstate

