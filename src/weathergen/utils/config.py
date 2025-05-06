# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import logging
import os
import subprocess
from pathlib import Path

import yaml
from omegaconf import OmegaConf

from weathergen.train.utils import get_run_id

_REPO_ROOT = Path(__file__).parent.parent.parent.parent  # TODO use importlib for resources
_DEFAULT_CONFIG_PTH = _REPO_ROOT / "config" / "default_config.yml"
_DEFAULT_MODEL_PATH = "./models"

_logger = logging.getLogger(__name__)


Config = OmegaConf


def print_cf(config: Config):
    """Print formatted the contents of the configuration."""
    for key, value in config.items():
        if key != "streams":
            print(f"{key} : {value}")
        else:
            for rt in value:
                for k, v in rt.items():
                    print("{}{} : {}".format("" if k == "reportypes" else "  ", k, v))


def save(config: Config, epoch: int | None):
    """Save current config into the current runs model directory."""
    path_models = Path(config.model_path)
    # save in directory with model files
    dirname = path_models / config.run_id
    dirname.mkdir(exist_ok=True, parents=True)

    if epoch is None:
        epoch_str = ""
    elif epoch == -1:
        epoch_str = "_latest"
    else:
        epoch_str = f"_epoch{epoch:05d}"
    fname = dirname / f"model_{config.run_id}{epoch_str}.json"

    json_str = json.dumps(OmegaConf.to_container(config))
    with fname.open("w") as f:
        f.write(json_str)


def load_model_config(run_id: str, epoch: int | None, model_path: str | None) -> OmegaConf:
    """
    Load a configuration file from a given run_id and epoch.
    If run_id is a full path, loads it from the full path.
    """

    if Path(run_id).exists():  # load from the full path if a full path is provided
        fname = Path(run_id)
        _logger.info(f"Loading config from provided full run_id path: {fname}")
    else:
        path_models = Path(model_path or _DEFAULT_MODEL_PATH)
        epoch_str = ""
        if epoch is not None:
            epoch_str = "_latest" if epoch == -1 else f"_epoch{epoch:05d}"
        fname = path_models / run_id / f"model_{run_id}{epoch_str}.json"

    _logger.info(f"Loading config from specified run_id and epoch: {fname}")

    with fname.open() as f:
        json_str = f.read()

    return OmegaConf.create(json.loads(json_str))


def load_config(
    private_home: Path | None,
    run_id: str | None,
    epoch: int | None,
    overwrite_path: Path | None,
) -> Config:
    private_config = _load_private_conf(private_home)
    overwrite_config = _load_overwrite_conf(overwrite_path)

    if run_id is None:
        base_config = _load_default_conf()
        base_config.run_id = get_run_id()
    else:
        base_config = load_model_config(run_id, epoch, private_config["model_path"])

    # use OmegaConf.unsafe_merge if too slow
    return OmegaConf.merge(base_config, private_config, overwrite_config)


def _load_overwrite_conf(overwrite_path: Path | None) -> OmegaConf:
    "Return the overwrite configuration."

    "If path is None, return an empty dictionary."
    if overwrite_path is None:
        return {}
    else:
        _logger.info(f"Loading overwrite config from {overwrite_path}.")
        return OmegaConf.load(overwrite_path)


def _load_private_conf(private_home: Path | None) -> OmegaConf:
    "Return the private configuration."
    "If none, take it from the environment variable WEATHERGEN_PRIVATE_CONF."

    env_script_path = _REPO_ROOT.parent / "WeatherGenerator-private" / "hpc" / "platform-env.py"

    if private_home is not None and private_home.is_file():
        _logger.info(f"Loading private config from {private_home}.")

    elif "WEATHERGEN_PRIVATE_CONF" in os.environ:
        private_home = Path(os.environ["WEATHERGEN_PRIVATE_CONF"])
        _logger.info(f"Loading private config fromWEATHERGEN_PRIVATE_CONF:{private_home}.")

    elif env_script_path.is_file():
        result = subprocess.run(
            [str(env_script_path), "hpc-config"], capture_output=True, text=True
        )
        private_home = Path(result.stdout.strip())
        _logger.info(f"Loading private config from platform-env.py output: {private_home}.")
    else:
        _logger.info(f"Could not find platform script at {env_script_path}")
        raise FileNotFoundError(
            "Could not find private config. Please set the environment variable WEATHERGEN_PRIVATE_CONF or provide a path."
        )
    private_cf = OmegaConf.load(private_home)
    private_cf["model_path"] = (
        private_cf["model_path"] if "model_path" in private_cf.keys() else "./models"
    )
    return private_cf


def _load_default_conf() -> OmegaConf:
    """Deserialize default configuration."""
    return OmegaConf.load(_DEFAULT_CONFIG_PTH)


def load_streams(streams_directory: Path) -> list[Config]:
    if not streams_directory.is_dir():
        msg = f"Streams directory {streams_directory} does not exist."
        raise FileNotFoundError(msg)

    # read all reportypes from directory, append to existing ones
    streams_directory = streams_directory.absolute()
    _logger.info(f"Reading streams from {streams_directory}")

    # append streams to existing (only relevant for evaluation)
    streams = []
    for config_file in sorted(streams_directory.rglob("*.yml")):
        try:
            # Stream config schema is {stream_name: stream_config} where stream_config
            # itself is a dict containing the actual options. stream_name needs to be
            # added to this dict since only stream_config will be further processed.
            stream_name, stream_config = [*OmegaConf.load(config_file).items()][0]
        except yaml.scanner.ScannerError as e:
            msg = f"Invalid yaml file while parsing stream configs: {config_file}"
            raise RuntimeError(msg) from e
        except IndexError:
            # support commenting out entire stream files to avoid loading them.
            _logger.warning(f"Parsed stream configuration file is empty: {config_file}")
            continue

        stream_config.name = stream_name
        streams.append(stream_config)
        _logger.info(f"Loaded stream config: {stream_name}")

    return streams
