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
    """Generate the filename for writing a model config file."""
    if mini_epoch is None:
        mini_epoch_str = ""
    elif mini_epoch == -1:
        mini_epoch_str = "_latest"
    else:
        mini_epoch_str = f"_chkpt{mini_epoch:05d}"

    return f"runstate_{run_id}{mini_epoch_str}.json"


def save_runstate(runstate: RunState, config: Config, mini_epoch: int | None):

    import json
    from weathergen.utils.distributed import is_root

    dirname = get_path_model(config)
    dirname.mkdir(exist_ok=True, parents=True)

    fname = _get_runstate_file_write_name(config.general.run_id, mini_epoch)

    json_str = json.dumps(OmegaConf.to_container(runstate)) + '\n'

    with (dirname/f"{fname}").open("w") as f:
        f.write(json_str)

