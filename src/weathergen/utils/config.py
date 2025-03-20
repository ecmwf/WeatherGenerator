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
from pathlib import Path

import yaml


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
                        print("{}{} : {}".format("" if k == "reportypes" else "  ", k, v))

    def save(self, epoch: str = None) -> None:
        # save in directory with model files
        dirname = Path(self.model_path) / self.run_id
        dirname.mkdir(parents=True, exist_ok=True)

        fname = dirname / f"model_{self.run_id}"
        epoch_str = ""
        if epoch is not None:
            epoch_str = "_latest" if epoch == -1 else f"_epoch{epoch:05d}"
        fname = fname.with_name(fname.name + f"{epoch_str}.json")

        json_str = json.dumps(self.__dict__)
        with fname.open("w") as f:
            f.write(json_str)

    @staticmethod
    def load(run_id: str, epoch: int = None, model_path: str = "./models") -> "Config":
        """
        Load a configuration file from a given run_id and epoch.
        If run_id is a full path, loads it from the full path.
        """
        if Path(run_id).exists():  # load from the full path if a full path is provided
            fname = Path(run_id)
            print(f"Loading config from provided full run_id path: {fname}")
        else:
            fname = Path(model_path) / run_id / f"model_{run_id}"

            # append also the epoch to the file name
            epoch_str = ""
            if epoch is not None:
                epoch_str = "_latest" if epoch == -1 else f"_epoch{epoch:05d}"
            fname = fname.with_name(fname.name + f"{epoch_str}.json")

            print(f"Loading config from specified run_id and epoch: {fname}")

        # open the file and read into a config object
        with fname.open() as f:
            json_str = f.readlines()

        cf = Config()
        cf.__dict__ = json.loads(json_str[0])

        return cf


def load_overwrite_conf(pth: str) -> dict:
    "Return the overwrite configuration."
    "If path is None, return an empty dictionary."
    if not pth:
        return {}
    else:
        overwrite_path = Path(pth)
        overwrite_conf = yaml.safe_load(overwrite_path.read_text())
        return overwrite_conf


def load_private_conf(pth: str) -> dict:
    "Return the private configuration."
    "If none, take it from the environment variable WEATHERGEN_PRIVATE_CONF."

    if not pth:
        if "WEATHERGEN_PRIVATE_CONF" in os.environ:
            private_home = Path(os.environ["WEATHERGEN_PRIVATE_CONF"])
            private_conf = yaml.safe_load(private_home.read_text())
            return private_conf
        else:
            raise ValueError(
                "No private config path is provided in the command line and WEATHERGEN_PRIVATE_CONF is not set."
            )
    else:
        private_home = Path(pth)
        private_conf = yaml.safe_load(private_home.read_text())
        return private_conf
