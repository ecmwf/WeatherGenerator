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


def save_runstate(runstate: RunState, config: Config, mini_epoch: int):

    import json
    from weathergen.utils.distributed import is_root

    if is_root():
        dirname = get_path_model(config)
        dirname.mkdir(exist_ok=True, parents=True)

        json_str = json.dumps(OmegaConf.to_container(runstate)) + '\n'

        with (dirname/f"wololo_{mini_epoch}.json").open("w") as f:
            f.write(json_str)

