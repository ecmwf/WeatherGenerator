# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import os
import yaml
from typing import Any
from pathlib import Path


###########################################
class Config:
    def __init__(self):
        pass

    def print(self):
        self_dict = self.__dict__
        for key, value in self_dict.items():
            if key != "streams":
                print(f"{key} : {value}")
            else:
                for rt in value:
                    for k, v in rt.items():
                        print(
                            "{}{} : {}".format("" if k == "reportypes" else "  ", k, v)
                        )

    def save(self, epoch=None):
        # save in directory with model files
        dirname = f"./models/{self.run_id}"
        # if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
        dirname = f"./models/{self.run_id}"
        # if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

        fname = f"./models/{self.run_id}/model_{self.run_id}"
        epoch_str = ""
        if epoch is not None:
            epoch_str = "_latest" if epoch == -1 else f"_epoch{epoch:05d}"
        fname += f"{epoch_str}.json"

        json_str = json.dumps(self.__dict__)
        with open(fname, "w") as f:
            f.write(json_str)

    @staticmethod
    def load(run_id, epoch=None):
        if "/" in run_id:  # assumed to be full path instead of just id
            fname = run_id
        else:
            fname = f"./models/{run_id}/model_{run_id}"
            epoch_str = ""
            if epoch is not None:
                epoch_str = "_latest" if epoch == -1 else f"_epoch{epoch:05d}"
            fname += f"{epoch_str}.json"

        with open(fname) as f:
            json_str = f.readlines()

        cf = Config()
        cf.__dict__ = json.loads(json_str[0])

        return cf


# Function that checks if WEATHERGEN_PRIVATE_HOME is set and returns it:
def private_conf() -> Any:
    if "WEATHERGEN_PRIVATE_CONF" in os.environ:
        private_home = Path(os.environ["WEATHERGEN_PRIVATE_CONF"])
        private_conf = yaml.safe_load(private_home.read_text())
        return private_conf
    else:
        raise ValueError("WEATHERGEN_PRIVATE_CONF is not set.")
