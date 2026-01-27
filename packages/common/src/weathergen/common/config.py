# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import functools
import io
import json
import logging
import os
import random
import string
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yaml.constructor
import yaml.scanner
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.omegaconf import open_dict

from weathergen.common.io import StoreType

_REPO_ROOT = Path(
    __file__
).parent.parent.parent.parent.parent.parent  # TODO use importlib for resources
_DEFAULT_CONFIG_PTH = _REPO_ROOT / "config" / "default_config.yml"

_DATETIME_TYPE_NAME = "datetime"  # Names for custom resolvers used in Omegaconf
_TIMEDELTA_TYPE_NAME = "timedelta"


_logger = logging.getLogger(__name__)


Config = DictConfig


def parse_timedelta(val: str | int | float | np.timedelta64) -> np.timedelta64:
    """
    Parse a value into a numpy timedelta64[ms].
    Integers and floats are interpreted as hours.
    Strings are parsed using pandas.to_timedelta.
    """
    if isinstance(val, int | float | np.number):
        return np.timedelta64(pd.to_timedelta(val, unit="s")).astype("timedelta64[ms]")
    return np.timedelta64(pd.to_timedelta(val)).astype("timedelta64[ms]")


def timedelta_to_str(val: np.timedelta64 | pd.Timedelta) -> str:
    """
    Put timedelta into string in format HH:MM:SS
    """
    dt = pd.to_timedelta(val)
    total_seconds = int(dt.total_seconds())

    # Calculate HH:MM:SS
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Format string (e.g., "06:00:00" or "24:00:00")
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def str_to_datetime64(s: str | int | np.datetime64) -> np.datetime64:
    """
    Convert a string to a numpy datetime64 object.
    """
    if isinstance(s, np.datetime64):
        return s

    # Convert to string to handle YAML integers (e.g. 20001010000000)
    s = str(s)
    return pd.to_datetime(s).to_datetime64()


OmegaConf.register_new_resolver(_TIMEDELTA_TYPE_NAME, parse_timedelta)
OmegaConf.register_new_resolver(_DATETIME_TYPE_NAME, str_to_datetime64)


def _sanitize_start_end_time_keys(sub_conf):
    """Convert start_date and end_date keys to datetime resolvers."""
    time_keys = ["start_date", "end_date"]
    for key in time_keys:
        if key in sub_conf:
            raw_key = f"_{key}"
            sub_conf[raw_key] = f"${{{key}}}"
            sub_conf[key] = f"${{{_DATETIME_TYPE_NAME}:{sub_conf[key]}}}"


def _sanitize_delta_time_keys(sub_conf):
    """Convert time delta keys to timedelta resolvers."""
    delta_keys = ["time_window_step", "time_window_len"]
    for key in delta_keys:
        if key in sub_conf:
            raw_key = f"_{key}"
            sub_conf[raw_key] = f"${{{key}}}"
            sub_conf[key] = f"${{{_TIMEDELTA_TYPE_NAME}:{sub_conf[key]}}}"

    if sub_conf.get("forecast") is not None:
        key = "time_step"
        if key in sub_conf.forecast:
            raw_key = f"_{key}"
            sub_conf.forecast[raw_key] = f"${{{key}}}"
            sub_conf.forecast[key] = f"${{{_TIMEDELTA_TYPE_NAME}:{sub_conf.forecast[key]}}}"


def _sanitize_time_keys(conf: Config) -> Config:
    """
    Convert time keys into a time format supported by OmegaConf

    Create an alias using interpolation syntax "${_keyname}"
    This stores a string instead of the resolved timedelta object.
    """

    conf = conf.copy()

    if conf.get("training_config") is not None:
        _sanitize_delta_time_keys(conf.training_config)
        _sanitize_start_end_time_keys(conf.training_config)

    if conf.get("validation_config") is not None:
        _sanitize_delta_time_keys(conf.validation_config)
        _sanitize_start_end_time_keys(conf.validation_config)

    if conf.get("test_config") is not None:
        _sanitize_delta_time_keys(conf.test_config)
        _sanitize_start_end_time_keys(conf.test_config)

    return conf


def _strip_interpolation(conf: Config) -> Config:
    """Remove OmegaConf interpolations and convert timedelta/datetime objects to strings."""
    stripped = OmegaConf.create()
    for key in list(conf.keys()):
        if key.startswith("_"):
            # Skip hidden/backup keys
            continue
        elif OmegaConf.is_interpolation(conf, key):
            raw_key = f"_{key}"
            if raw_key in conf:
                # Retrieve the value from the backup key (resolves interpolation)
                val = conf[raw_key]
            else:
                # Fallback to the original key
                val = conf[key]
        else:
            # Standard key retrieval
            val = conf[key]

        # Convert unsupported types (timedelta/datetime) to strings
        if isinstance(val, np.timedelta64 | pd.Timedelta):
            val = timedelta_to_str(val)
        elif isinstance(val, np.datetime64 | pd.Timestamp):
            dt = pd.to_datetime(val)
            # Format: Standard ISO without microseconds
            val = dt.strftime("%Y-%m-%dT%H:%M:%S")

        stripped[key] = val

    return stripped


def get_run_id():
    """Generate a random 8-character run ID."""
    s1 = string.ascii_lowercase
    s2 = string.ascii_lowercase + string.digits
    return "".join(random.sample(s1, 1)) + "".join(random.sample(s2, 7))


def get_run_id_from_config(config: Config) -> str:
    general_cfg = config.get("general", None)
    return general_cfg.run_id if general_cfg else config.run_id


def format_cf(config: Config) -> str:
    """Format config as a human-readable string."""
    stream = io.StringIO()
    clean_cf = _strip_interpolation(config)
    for key, value in clean_cf.items():
        match key:
            case "streams":
                for rt in value:
                    for k, v in rt.items():
                        whitespace = "" if k == "reportypes" else "  "
                        stream.write(f"{whitespace}{k} : {v}")
            case _:
                stream.write(f"{key} : {value}\n")

    return stream.getvalue()


def save(config: Config, mini_epoch: int | None):
    """Save current config into the current runs model directory."""
    # save in directory with model files
    dirname = get_path_model(config)
    dirname.mkdir(exist_ok=True, parents=True)

    fname = _get_model_config_file_write_name(get_run_id_from_config(config), mini_epoch)

    json_str = json.dumps(OmegaConf.to_container(_strip_interpolation(config)))
    with (dirname / fname).open("w") as f:
        f.write(json_str)


def load_run_config(run_id: str, mini_epoch: int | None, model_path: str | None) -> Config:
    """
    Load a configuration file from a given run_id and mini_epoch.
    If run_id is a full path, loads it from the full path.

    Args:
        run_id: Run ID of the pretrained WeatherGenerator model
        mini_epoch: Mini_epoch of the checkpoint to load. -1 indicates last checkpoint available.
        model_path: Path to the model directory. If None, uses the model_path from private config.

    Returns:
        Configuration object loaded from the specified run and mini_epoch.
    """
    # Loading path
    if Path(run_id).exists():  # load from the full path if a full path is provided
        fname = Path(run_id)
        _logger.info(f"Loading config from provided full run_id path: {fname}")
    else:
        # Load model config here. In case model_path is not provided, get it from private conf
        if model_path is None:
            path = get_path_model(run_id=run_id)
        else:
            path = Path(model_path) / run_id

        fname = path / _get_model_config_file_read_name(run_id, mini_epoch)
        if not fname.exists():
            # Fallback for old naming convention
            # TODO remove compatibility
            fname = path / _get_model_config_file_read_name(run_id, mini_epoch, use_old_name=True)
        assert fname.exists(), (
            "The fallback path to the model does not exist. Please provide a `model_path`.",
            fname,
        )
        _logger.info(f"Loading config from specified run_id and mini_epoch: {fname}")

    with fname.open() as f:
        json_str = f.read()

    config = OmegaConf.create(json.loads(json_str))
    config = _sanitize_time_keys(config)

    return _apply_fixes(config)


def _get_model_config_file_write_name(run_id: str, mini_epoch: int | None):
    """Generate the filename for writing a model config file."""
    if mini_epoch is None:
        mini_epoch_str = ""
    elif mini_epoch == -1:
        mini_epoch_str = "_latest"
    else:
        mini_epoch_str = f"_chkpt{mini_epoch:05d}"

    return f"model_{run_id}{mini_epoch_str}.json"


def _get_model_config_file_read_name(run_id: str, mini_epoch: int | None, use_old_name=False):
    """Generate the filename for reading a model config file."""
    if mini_epoch is None:
        mini_epoch_str = ""
    elif mini_epoch == -1:
        mini_epoch_str = "_latest"
    elif use_old_name:
        mini_epoch_str = f"_epoch{mini_epoch:05d}"  # TODO remove compatibility
    else:
        mini_epoch_str = f"_chkpt{mini_epoch:05d}"

    return f"model_{run_id}{mini_epoch_str}.json"


def get_model_results(run_id: str, mini_epoch: int, rank: int) -> Path:
    """
    Get the path to the model results zarr store from a given run_id and mini_epoch.
    """
    run_results = Path(_load_private_conf(None)["path_shared_working_dir"]) / f"results/{run_id}"

    for ext in StoreType.extensions():
        zarr_path_new = run_results / f"validation_chkpt{mini_epoch:05d}_rank{rank:04d}.{ext}"
        zarr_path_old = run_results / f"validation_epoch{mini_epoch:05d}_rank{rank:04d}.{ext}"

        if zarr_path_new.exists() or zarr_path_new.is_dir():
            return zarr_path_new
        elif zarr_path_old.exists() or zarr_path_old.is_dir():
            return zarr_path_old
    raise FileNotFoundError(
        f"Zarr file with run_id {run_id}, mini_epoch {mini_epoch} and rank {rank} does not "
        f"exist or is not a directory."
    )


def _apply_fixes(config: Config) -> Config:
    """
    Apply fixes to maintain a best effort backward combatibility.

    This method should act as a central hook to implement config backward
    compatibility fixes. This is needed to run inference/continuing from
    "outdatet" run configurations. The fixes in this function should be
    eventually removed.
    """
    config = _check_logging(config)
    return config


def _check_logging(config: Config) -> Config:
    """
    Apply fixes to log frequency config.
    """
    config = config.copy()
    if config.get("train_log_freq") is None:  # TODO remove this for next version
        config.train_log_freq = OmegaConf.create(
            {"checkpoint": 250, "terminal": 10, "metrics": config.train_log.log_interval}
        )

    return config


def merge_configs(base_config: Config, update_config: Config):
    """
    Merge two configs using OmegaConf's default strategy
    """
    return OmegaConf.merge(base_config, update_config)


def load_merge_configs(
    private_home: Path | None = None,
    from_run_id: str | None = None,
    mini_epoch: int | None = None,
    base: Path | Config | None = None,
    *overwrites: Path | dict | Config,
) -> Config:
    """
    Merge config information from multiple sources into one run_config. Anything in the
    private configs "secrets" section will be discarded.

    Args:
        private_home: Configuration file containing platform dependent information and secrets
        from_run_id: Run id of the pretrained WeatherGenerator model
        to continue training or inference
        mini_epoch: Mini_epoch of the checkpoint to load. -1 indicates last checkpoint available.
        base: Path to the base configuration file. Uses default configuration if None.
        *overwrites: Additional overwrites from different sources

    Note: The order of precedence for merging the final config is in ascending order:
        - base config (either default config or loaded from previous run)
        - private config
        - overwrites (also in ascending order)

    Returns:
        Merged configuration object.
    """
    private_config = _load_private_conf(private_home)
    overwrite_configs: list[Config] = []
    for overwrite in overwrites:
        if isinstance(overwrite, (str | Path)):
            # Because of the way we pass extra configs through slurm,
            # all the paths may be concatenated with ":"
            p = str(overwrite).split(":")
            for path in p:
                c = _load_overwrite_conf(Path(path))
                c = _load_streams_in_config(c)
                overwrite_configs.append(c)
        else:
            # If it is a dict or DictConfig, we can directly use it
            c = _load_overwrite_conf(overwrite)
            c = _load_streams_in_config(c)
            overwrite_configs.append(c)

    if from_run_id is None:
        base_config = _load_base_conf(base)
    else:
        base_config = load_run_config(from_run_id, mini_epoch, None)
        from_run_id = get_run_id_from_config(base_config)
    with open_dict(base_config):
        base_config.from_run_id = from_run_id
    # use OmegaConf.unsafe_merge if too slow
    c = OmegaConf.merge(base_config, private_config, *overwrite_configs)
    assert isinstance(c, Config)
    c = _sanitize_time_keys(c)

    # Ensure the config has mini-epoch notation
    if hasattr(c, "samples_per_epoch"):
        c.samples_per_mini_epoch = c.samples_per_epoch
        c.num_mini_epochs = c.num_epochs
    return c


def _load_streams_in_config(config: Config) -> Config:
    """If the config contains a streams_directory, loads the streams and returns the config with
    the streams set."""
    streams_directory = config.get("streams_directory", None)
    config = config.copy()
    if streams_directory is not None:
        streams_directory = Path(streams_directory)
        config.streams = load_streams(streams_directory)
    return config


def set_run_id(config: Config, run_id: str | None, reuse_run_id: bool) -> Config:
    """
    Determine and set run_id of current run.

    Determining the run id should follow the following logic:

    1. (default case): run train, train_continue or inference without any flags
        => generate a new run_id for this run.
    2. (assign run_id): run train, train_continue or inference with --run_id <RUNID> flag
        => assign a run_id manually to this run.
        This is intend for outside tooling and should not be used manually.
    3. (reuse run_id -> only for train_continue and inference):
        reuse the run_id from the run specified by --from_run_id <RUNID>.
        Since the run_id correct run_id is already loaded in the config nothing has to be assigned.
        This case will happen if --reuse_run_id is specified.


    Args:
        config: Base configuration loaded from previous run or default.
        run_id: Id assigned to this run. If None a new one will be generated.
        reuse_run_id: Reuse run_id from base configuration instead.

    Returns:
        config object with the run_id attribute properly set.
    """
    config = config.copy()
    if reuse_run_id:
        assert get_run_id_from_config(config) is not None, "Loaded run_id should not be None."
        _logger.info(f"reusing run_id from previous run: {get_run_id_from_config(config)}")
    else:
        if run_id is None:
            # generate new id if run_id is None
            config.general.run_id = run_id or get_run_id()
            _logger.info(f"Using generated run_id: {config.general.run_id}")
        else:
            config.general.run_id = run_id
            _logger.info(
                f"Using assigned run_id: {config.general.run_id}."
                f" If you manually selected this run_id, this is an error."
            )

    return config


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

    assert isinstance(overwrite_config, DictConfig)
    return overwrite_config


def _load_private_conf(private_home: Path | None = None) -> DictConfig:
    """
    Return the private configuration from file or environment variable WEATHERGEN_PRIVATE_CONF.
    """
    env_script_path = _REPO_ROOT.parent / "WeatherGenerator-private" / "hpc" / "platform-env.py"

    if private_home is not None and private_home.is_file():
        _logger.info(f"Loading private config from {private_home}.")

    elif "WEATHERGEN_PRIVATE_CONF" in os.environ:
        private_home = Path(os.environ["WEATHERGEN_PRIVATE_CONF"])
        _logger.info(f"Loading private config from WEATHERGEN_PRIVATE_CONF:{private_home}.")

    elif env_script_path.is_file():
        _logger.info(f"Loading private config from platform-env.py: {env_script_path}.")
        # This code does many checks to ensure that any error message is surfaced.
        # Since it is a process call, it can be hard to diagnose the error.
        # TODO: eventually, put all this wrapper code in a separate function
        try:
            result_hpc = subprocess.run(
                [str(env_script_path), "hpc"], capture_output=True, text=True, check=True
            )
        except subprocess.CalledProcessError as e:
            _logger.error(
                (
                    "Error while running platform-env.py:",
                    f" {e} {e.stderr} {e.stdout} {e.output} {e.returncode}",
                )
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
            "Could not find private config. Please set the environment variable "
            "WEATHERGEN_PRIVATE_CONF or provide a path."
        )
    private_cf = OmegaConf.load(private_home)

    if "secrets" in private_cf:
        del private_cf["secrets"]

    assert isinstance(private_cf, DictConfig)
    return private_cf


def _load_base_conf(base: Path | Config | None) -> Config:
    """Return the base configuration"""
    match base:
        case Path():
            _logger.info(f"Loading specified base config from file: {base}.")
            conf = OmegaConf.load(base)
        case Config():
            _logger.info(f"Using existing config as base: {base}.")
            conf = base
        case _:
            _logger.info("Deserialize default configuration.")
            conf = OmegaConf.load(_DEFAULT_CONFIG_PTH)
    assert isinstance(conf, Config)
    return conf


def load_streams(streams_directory: Path) -> list[Config]:
    """Load all stream configurations from a directory."""
    # TODO: might want to put this into config later instead of hardcoding it here...
    streams_history = {
        "streams_anemoi": "era5_1deg",
        "streams_mixed": "era5_nppatms_synop",
        "streams_ocean": "fesom",
        "streams_icon": "icon",
        "streams_mixed_experimental": "cerra_seviri",
    }
    if not streams_directory.is_dir():
        streams_directory_config = streams_directory
        dirs = [streams_directory]
        while streams_directory.name in streams_history and not streams_directory.is_dir():
            streams_directory = streams_directory.with_name(streams_history[streams_directory.name])
            dirs.append(streams_directory)
        if not streams_directory.is_dir():
            msg = f"Could not find stream directory, nor its history: {[str(dir) for dir in dirs]}"
            raise FileNotFoundError(msg)
        _logger.info(
            f"Streams directory {streams_directory} found in "
            f"history for {streams_directory_config}. "
            "Note: This change will not be reflected in the config. "
            "Please update the 'streams_directory' variable manually."
        )

    # read all reportypes from directory, append to existing ones
    streams_directory = streams_directory.absolute()
    _logger.info(f"Reading streams from {streams_directory}")

    # append streams to existing (only relevant for evaluation)
    streams = {}
    # exclude temp files starting with "." or "#" (eg. emacs, vim, macos savefiles)
    stream_files = sorted(streams_directory.rglob("[!.#]*.yml"))
    _logger.info(f"Discover stream configs: {', '.join(map(str, stream_files))}")
    for config_file in stream_files:
        try:
            config = OmegaConf.load(config_file)
            for stream_name, stream_config in config.items():
                # Stream config schema is {stream_name: stream_config}
                # where stream_config itself is a dict containing the actual options.
                # stream_name needs to be added to this dict since only stream_config
                # will be further processed.
                stream_config.name = stream_name
                if stream_name in streams:
                    msg = f"Duplicate stream name found: {stream_name}."
                    "Please ensure all stream names are unique."
                    raise ValueError(msg)
                else:
                    streams[stream_name] = stream_config
                    _logger.info(f"Loaded stream config: {stream_name} from file {config_file}")

        except (yaml.scanner.ScannerError, yaml.constructor.ConstructorError) as e:
            msg = f"Invalid yaml file while parsing stream configs: {config_file}"
            raise ValueError(msg) from e
        except AttributeError as e:
            msg = f"Invalid yaml file while parsing stream configs: {config_file}"
            raise ValueError(msg) from e
        except IndexError:
            # support commenting out entire stream files to avoid loading them.
            _logger.warning(f"Parsed stream configuration file is empty: {config_file}")
            continue

    return list(streams.values())


def get_path_run(config: Config) -> Path:
    """Get the current runs results_path for storing run results and logs."""
    return _get_shared_wg_path() / "results" / get_run_id_from_config(config)


def get_path_model(config: Config | None = None, run_id: str | None = None) -> Path:
    """Get the current runs model_path for storing model checkpoints."""
    if config or run_id:
        run_id = run_id if run_id else get_run_id_from_config(config)
    else:
        msg = f"Missing run_id and cannot infer it from config: {config}"
        raise ValueError(msg)
    return _get_shared_wg_path() / "models" / run_id


def get_path_results(config: Config, mini_epoch: int) -> Path:
    """Get the path to validation results for a specific mini_epoch and rank."""
    ext = StoreType(config.zarr_store).value  # validate extension
    base_path = get_path_run(config)
    fname = f"validation_chkpt{mini_epoch:05d}_rank{config.rank:04d}.{ext}"

    return base_path / fname


@functools.cache
def _get_shared_wg_path() -> Path:
    """Get the shared working directory for WeatherGenerator."""
    private_config = _load_private_conf()
    return Path(private_config.get("path_shared_working_dir"))


def validate_forecast_policy_and_steps(forecast_cfg: OmegaConf, mode: str):
    """
    Validates the forecast policy, steps and offset within a configuration object.

    This method enforces specific rules for the `forecast.num_steps` attribute, which can be
    either a single integer or a list of integers, ensuring consistency with the
    `forecast.policy` attribute. Furthermore `forecast.offset` is enforeced to be either 0 or 1.

    The validation logic is as follows:
    - `forecast.offset` must either be 0 or 1.
    - If `forecast.offset` is 0, `forecast.num_steps` must be an integer and can either be 0 (e.g.
      used for SSL training without forecast engine) or 1 (e.g. used for diffusion in which the
      forecast engine is used to denoise the current time window).
    - If `forecast.offset` is 1, a `forecast.policy` must be specified and `forecast.num_steps` can
      either be a single integer greater than 1 or a non-empty list and all of its elements must be
      integers greater than 1.

    Args:
        mode_cfg (OmegaConf): The training/validation/test configuration object containing the
                             `forecast.num_steps` and `forecast.policy` attributes.
        mode (str): the training mode, i.e. training_config, validation_config, or test_config

    Raises:
        TypeError: If `forecast.offset` is not an integer of value 0 or 1.
        TypeError: If `forecast.num_steps` is not an integer or a non-empty list.
        AssertionError: If a `forecast.policy` is required but not provided, or
                        if `forecast_step` is negative while `forecast.policy` is provided, or
                        if any of the forecast steps in a list are negative.
    """

    if len(forecast_cfg) == 0:
        return

    provide_forecast_policy = (
        f"'{mode}.forecast.policy' must be specified when '{mode}.forecast.num_steps' is not zero "
        f"and '{mode}.forecast.offset' is 1. "
    )
    valid_forecast_policies = (
        "Valid values for '{mode}.forecast.policy' are, e.g., 'fixed' when using constant number "
        "of forecast steps throughout the training, or 'sequential' when varying the number of "
        "forecast steps over mini_epochs, such as, e.g., 'forecast.num_steps: [2, 2, 4, 4]'. "
    )
    valid_forecast_offset = f"'{mode}.forecast.offset' must be an integer of either value 0 or 1. "
    valid_forecast_steps_offset0 = (
        f"For '{mode}.forecast.offset: 0', '{mode}.forecast.num_steps' must be an integer of value "
        f"either 0 or 1. "
    )
    valid_forecast_steps_offset1 = (
        f"For '{mode}.forecast.offset: 0', '{mode}.forecast.num_steps' must be an integer greater "
        "than 1 or a non-empty list and all of its elements must be integers greater than 1."
    )

    # get output_offset or set default to 0 as in multi_stream_data_sampler.py
    output_offset = forecast_cfg.get("offset", 0)
    assert isinstance(output_offset, int), TypeError(valid_forecast_offset)
    if output_offset == 0:
        if isinstance(forecast_cfg.num_steps, int):
            assert forecast_cfg.num_steps in [0, 1], valid_forecast_steps_offset0
        else:
            raise TypeError(valid_forecast_steps_offset0)
    elif output_offset == 1:
        assert forecast_cfg.policy, (provide_forecast_policy, valid_forecast_policies)
        if isinstance(forecast_cfg.num_steps, int):
            assert forecast_cfg.num_steps > 0, valid_forecast_steps_offset1
        elif isinstance(forecast_cfg.num_steps, ListConfig) and len(forecast_cfg.num_steps) > 0:
            assert all(step > 0 for step in forecast_cfg.num_steps), valid_forecast_steps_offset1
        else:
            raise TypeError(valid_forecast_steps_offset1)
    else:
        raise TypeError(valid_forecast_offset)
