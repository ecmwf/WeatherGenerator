# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import io
import json
import logging
import os
import subprocess
from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf

from weathergen.train.utils import get_run_id

_REPO_ROOT = Path(__file__).parent.parent.parent.parent  # TODO use importlib for resources
_DEFAULT_CONFIG_PTH = _REPO_ROOT / "config" / "default_config.yml"
_DEFAULT_MODEL_PATH = "./models"

_logger = logging.getLogger(__name__)


Config = DictConfig


def format_cf(config: Config) -> str:
    stream = io.StringIO()
    for key, value in config.items():
        match key:
            case "streams":
                for rt in value:
                    for k, v in rt.items():
                        whitespace = "" if k == "reportypes" else "  "
                        stream.write(f"{whitespace}{k} : {v}")
            case _:
                stream.write(f"{key} : {value}\n")

    return stream.getvalue()


def save(config: Config, epoch: int | None):
    """Save current config into the current runs model directory."""
    path_models = Path(config.model_path)
    # save in directory with model files
    dirname = path_models / config.run_id
    dirname.mkdir(exist_ok=True, parents=True)

    fname = dirname / _get_model_config_file_name(config.run_id, epoch)

    json_str = json.dumps(OmegaConf.to_container(config))
    with fname.open("w") as f:
        f.write(json_str)


def load_model_config(run_id: str, epoch: int | None, model_path: str | None) -> Config:
    """
    Load a configuration file from a given run_id and epoch.
    If run_id is a full path, loads it from the full path.
    """

    if Path(run_id).exists():  # load from the full path if a full path is provided
        fname = Path(run_id)
        _logger.info(f"Loading config from provided full run_id path: {fname}")
    else:
        path_models = Path(model_path or _DEFAULT_MODEL_PATH)
        fname = path_models / run_id / _get_model_config_file_name(run_id, epoch)

    _logger.info(f"Loading config from specified run_id and epoch: {fname}")

    with fname.open() as f:
        json_str = f.read()

    return OmegaConf.create(json.loads(json_str))


def _get_model_config_file_name(run_id: str, epoch: int | None):
    if epoch is None:
        epoch_str = ""
    elif epoch == -1:
        epoch_str = "_latest"
    else:
        epoch_str = f"_epoch{epoch:05d}"
    return f"model_{run_id}{epoch_str}.json"


def load_config(
    private_home: Path | None,
    from_run_id: str | None,
    epoch: int | None,
    *overwrites: Path | dict | Config,
) -> Config:
    """
    Merge config information from multiple sources into one run_config. Anything in the
    private configs "secrets" section will be discarted.

    Args:
        private_home: Configuration file containing platform dependent information and secretes
        from_run_id: Run/model id of pretrained WeatherGenerator model to continue training or evaluate
        epoch: epoch of the checkpoint to load. -1 indicates last checkpoint available.
        *overwrites: Additional overwrites from different sources

    Note: The order of precendence for merging the final config is in ascending order:
        - base config (either default config or loaded from previous run)
        - private config
        - overwrites (also in ascending order)
    """
    private_config = _load_private_conf(private_home)
    overwrite_configs = [_load_overwrite_conf(overwrite) for overwrite in overwrites]
    
    if from_run_id is None:
        base_config = _load_default_conf()
    else:
        base_config = load_model_config(from_run_id, epoch, private_config["model_path"])
    

    # use OmegaConf.unsafe_merge if too slow
    return OmegaConf.merge(base_config, private_config, *overwrite_configs)

def set_run_id(config: Config, run_id: str | None, reuse_run_id: bool):
    """
    Determine run_id of current run.
    
    Args:
        config: Base configuration loaded from previous run or default.
        run_id: Id assigned to this run. If None a new one will be generated.
        reuse_run_id: Reuse run_id from base configuration instead.
    """
    if not reuse_run_id:
        if run_id is None:
            run_id = get_run_id()

        config.run_id = run_id
    
    assert config.run_id is not None
    

def from_cli_arglist(arg_list: list[str]) -> Config:
    """
    Parse a Config instance from cli arguments.

    This enables convenient collecting of arguments into an overwrite.

    Args:
        arg_list: items in this list should be of the form: parent_obj.nested_obj=value
    """
    return OmegaConf.from_cli(arg_list)


def _load_overwrite_conf(overwrite: Path | dict | DictConfig) -> DictConfig:
    """
    Convert different sources into configs that can be used as overwrites.

    raises: ValueError if argument cannot be turned into DictConfig.
    """

    match overwrite:  # match the type
        case Path():
            _logger.info(f"Loading overwrite config from file: {overwrite}.")
            overwrite_config = OmegaConf.load(overwrite)
        case dict():
            _logger.info(f"Loading overwrite config from dict: {overwrite}.")
            overwrite_config = OmegaConf.create(overwrite)
        case DictConfig():
            _logger.info(f"Using existing config as overwrite: {overwrite}.")
            overwrite_config = overwrite
        case _:
            msg = f"Cannot build config from overwrite: {overwrite}, with type {type(overwrite)}"
            raise ValueError(msg)

    return overwrite_config


def _load_private_conf(private_home: Path | None) -> DictConfig:
    "Return the private configuration."
    "If none, take it from the environment variable WEATHERGEN_PRIVATE_CONF."

    env_script_path = _REPO_ROOT.parent / "WeatherGenerator-private" / "hpc" / "platform-env.py"

    if private_home is not None and private_home.is_file():
        _logger.info(f"Loading private config from {private_home}.")

    elif "WEATHERGEN_PRIVATE_CONF" in os.environ:
        private_home = Path(os.environ["WEATHERGEN_PRIVATE_CONF"])
        _logger.info(f"Loading private config fromWEATHERGEN_PRIVATE_CONF:{private_home}.")

    elif env_script_path.is_file():
        _logger.info(f"Loading private config from platform-env.py: {env_script_path}.")
        # This code does many checks to ensure that any error message is surfaced. Since it is a process call,
        # it can be hard to diagnose the error.
        # TODO: eventually, put all this wrapper code in a separate function
        try:
            result_hpc = subprocess.run(
                [str(env_script_path), "hpc"], capture_output=True, text=True, check=True
            )
        except subprocess.CalledProcessError as e:
            _logger.error(
                f"Error while running platform-env.py: {e} {e.stderr} {e.stdout} {e.output} {e.returncode}"
            )
            raise
        if result_hpc.returncode != 0:
            _logger.error(f"Error while running platform-env.py: {result_hpc.stderr.strip()}")
            raise RuntimeError(f"Error while running platform-env.py: {result_hpc.stderr.strip()}")
        _logger.info(f"Detected HPC: {result_hpc.stdout.strip()}.")

        result = subprocess.run(
            [str(env_script_path), "hpc-config"], capture_output=True, text=True, check=True
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
    del private_cf["secrets"]
    return private_cf


def _load_default_conf() -> Config:
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
    # exclude temp files starting with "." or "#" (eg. emacs, vim, macos savefiles)
    stream_files = sorted(streams_directory.rglob("[!.#]*.yml"))
    _logger.info(f"discover stream configs: {stream_files}")
    for config_file in stream_files:
        try:
            # Stream config schema is {stream_name: stream_config} where stream_config
            # itself is a dict containing the actual options. stream_name needs to be
            # added to this dict since only stream_config will be further processed.
            stream_name, stream_config = [*OmegaConf.load(config_file).items()][0]
            stream_config.name = stream_name
        except yaml.scanner.ScannerError as e:
            msg = f"Invalid yaml file while parsing stream configs: {config_file}"
            raise RuntimeError(msg) from e
        except AttributeError as e:
            msg = f"Invalid yaml file while parsing stream configs: {config_file}"
            raise RuntimeError(msg) from e
        except IndexError:
            # support commenting out entire stream files to avoid loading them.
            _logger.warning(f"Parsed stream configuration file is empty: {config_file}")
            continue

        streams.append(stream_config)
        _logger.info(f"Loaded stream config: {stream_name}")

    return streams
