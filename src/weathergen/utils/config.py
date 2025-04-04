# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import itertools
import json
import logging
import os
from pathlib import Path

import yaml
from omegaconf import OmegaConf

from weathergen.train.utils import get_run_id

_REPO_ROOT = Path(__file__).parent.parent.parent.parent  # TODO use importlib for resources
DEFAULT_CONFIG_PTH = _REPO_ROOT / "config" / "default_config.yml"

_logger = logging.getLogger(__name__)


Config = OmegaConf


def print_cf(config: Config):
    for key, value in config.items():
        if key != "streams":
            print(f"{key} : {value}")
        else:
            for rt in value:
                for k, v in rt.items():
                    print("{}{} : {}".format("" if k == "reportypes" else "  ", k, v))


def save(config: Config, epoch: int | None = None):
    path_models = Path(config.model_path)
    # save in directory with model files
    dirname = path_models / config.run_id
    dirname.mkdir(exist_ok=True, parents=True)

    epoch_str = ""
    if epoch is not None:
        epoch_str = "_latest" if epoch == -1 else f"_epoch{epoch:05d}"
    fname = dirname / f"model_{config.run_id}{epoch_str}.json"

    json_str = json.dumps(OmegaConf.to_container(config))
    with fname.open("w") as f:
        f.write(json_str)


def load_model_config(
    run_id: str, epoch: int | None = None, model_path: str = "./models"
) -> Config:
    """
    Load a configuration file from a given run_id and epoch.
    If run_id is a full path, loads it from the full path.
    """
    if Path(run_id).exists():  # load from the full path if a full path is provided
        fname = Path(run_id)
        _logger.info(f"Loading config from provided full run_id path: {fname}")
    else:
        path_models = Path(model_path)
        epoch_str = ""
        if epoch is not None:
            epoch_str = "_latest" if epoch == -1 else f"_epoch{epoch:05d}"
        fname = path_models / run_id / f"model_{run_id}{epoch_str}.json"

    _logger.info(f"Loading config from specified run_id and epoch: {fname}")

    with fname.open() as f:
        json_str = f.read()
        print(json_str)

    return OmegaConf.create(json.loads(json_str))


def load_config(
    private_home: Path | None = None,
    run_id: str | None = None,
    epoch: int | None = None,
    overwrite_path: Path | None = None,
) -> Config:
    private_config = load_private_conf(private_home)
    overwrite_config = load_overwrite_conf(overwrite_path)

    if run_id is None:
        base_config = load_default_conf()
        base_config.run_id = get_run_id()
    else:
        base_config = load_model_config(run_id, epoch, private_config["model_path"])

    # use OmegaConf.unsafe_merge if too slow
    return OmegaConf.merge(base_config, private_config, overwrite_config)


def load_overwrite_conf(overwrite_path: Path | None = None) -> Config:
    "Return the overwrite configuration."

    "If path is None, return an empty dictionary."
    if overwrite_path is None or not overwrite_path.is_file():
        return {}
    else:
        return OmegaConf.load(overwrite_path)


def create_empty() -> Config:
    return OmegaConf.create({})


def load_private_conf(private_home: Path | None = None) -> dict:
    "Return the private configuration."
    "If none, take it from the environment variable WEATHERGEN_PRIVATE_CONF."

    if private_home is None or not private_home.is_file():
        if "WEATHERGEN_PRIVATE_CONF" in os.environ:
            private_home = Path(os.environ["WEATHERGEN_PRIVATE_CONF"])
        else:
            raise ValueError(
                "No private config path is provided in the command line and WEATHERGEN_PRIVATE_CONF is not set."
            )

    private_cf = OmegaConf.load(private_home)
    private_cf["model_path"] = (
        private_cf["model_path"] if "model_path" in private_cf.keys() else "./models"
    )
    return private_cf


def load_default_conf() -> Config:
    """Deserialize default configuration."""
    return OmegaConf.load(DEFAULT_CONFIG_PTH)


def load_streams(streams_directory: Path) -> list[Config]:
    if not streams_directory.is_dir():
        _logger.warning(f"Streams directory {streams_directory} does not exist.")

    # read all reportypes from directory, append to existing ones
    streams_directory = streams_directory.absolute()
    _logger.info(f"Reading streams from {streams_directory}")

    # append streams to existing (only relevant for evaluation)
    streams = []
    for config_file in sorted(streams_directory.rglob("*.yml")):
        try:
            stream_name, stream_config = [*OmegaConf.load(config_file).items()][0]
        except yaml.scanner.ScannerError:
            _logger.warning(f"Invalid yaml file: {config_file}")
            continue

        stream_config.name = stream_name
        streams.append(stream_config)
        _logger.info(f"loaded stream config: {stream_name}")

    # sanity checking (at some point, the dict should be parsed into a class)
    # check if all filenames accross all streams are unique
    rts = [rt["filenames"] for rt in streams]
    rts = list(itertools.chain.from_iterable(rts))
    if len(rts) != len(set(rts)):
        _logger.warning("Duplicate reportypes specified.")

    # return OmegaConf.create({"streams": streams})
    return streams
