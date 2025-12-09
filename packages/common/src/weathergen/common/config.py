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
import random
import string
import subprocess
from pathlib import Path

import yaml
import yaml.constructor
import yaml.scanner
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.omegaconf import open_dict

_REPO_ROOT = Path(
    __file__
).parent.parent.parent.parent.parent.parent  # TODO use importlib for resources
_DEFAULT_CONFIG_PTH = _REPO_ROOT / "config" / "default_config.yml"

_logger = logging.getLogger(__name__)


Config = DictConfig


def get_run_id():
    s1 = string.ascii_lowercase
    s2 = string.ascii_lowercase + string.digits
    return "".join(random.sample(s1, 1)) + "".join(random.sample(s2, 7))


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


def save(config: Config, mini_epoch: int | None):
    """Save current config into the current runs model directory."""
    path_models = Path(config.model_path)
    # save in directory with model files
    dirname = path_models / config.run_id
    dirname.mkdir(exist_ok=True, parents=True)

    fname = _get_model_config_file_write_name(path_models, config.run_id, mini_epoch)

    json_str = json.dumps(OmegaConf.to_container(config))
    with fname.open("w") as f:
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
    if Path(run_id).exists():  # load from the full path if a full path is provided
        fname = Path(run_id)
        _logger.info(f"Loading config from provided full run_id path: {fname}")
    else:
        # Load model config here. In case model_path is not provided, get it from private conf
        if model_path is None:
            pconf = _load_private_conf()
            model_path = _get_config_attribute(
                config=pconf, attribute_name="model_path", fallback="models"
            )
        path = Path(model_path)
        fname = _get_model_config_file_read_name(path, run_id, mini_epoch)
        assert fname.exists(), (
            "The fallback path to the model does not exist. Please provide a `model_path`.",
            fname,
        )

    _logger.info(f"Loading config from specified run_id and mini_epoch: {fname}")

    with fname.open() as f:
        json_str = f.read()

    config = OmegaConf.create(json.loads(json_str))

    return _apply_fixes(config)


def _get_model_config_file_write_name(path: Path, run_id: str, mini_epoch: int | None):
    if mini_epoch is None:
        mini_epoch_str = ""
    elif mini_epoch == -1:
        mini_epoch_str = "_latest"
    else:
        mini_epoch_str = f"_chkpt{mini_epoch:05d}"

    return path / run_id / f"model_{run_id}{mini_epoch_str}.json"


def _get_model_config_file_read_name(path: Path, run_id: str, mini_epoch: int | None):
    if mini_epoch is None:
        mini_epoch_str = ""
    elif mini_epoch == -1:
        mini_epoch_str = "_latest"
    elif (path / run_id / f"model_{run_id}_epoch{mini_epoch:05d}.json").exists():
        mini_epoch_str = f"_epoch{mini_epoch:05d}"
    else:
        mini_epoch_str = f"_chkpt{mini_epoch:05d}"

    return path / run_id / f"model_{run_id}{mini_epoch_str}.json"


def get_model_results(run_id: str, mini_epoch: int, rank: int) -> Path:
    """
    Get the path to the model results zarr store from a given run_id and mini_epoch.
    """
    run_results = Path(_load_private_conf(None)["path_shared_working_dir"]) / f"results/{run_id}"

    zarr_path_new = run_results / f"validation_chkpt{mini_epoch:05d}_rank{rank:04d}.zarr"
    zarr_path_old = run_results / f"validation_epoch{mini_epoch:05d}_rank{rank:04d}.zarr"

    if zarr_path_new.exists() or zarr_path_new.is_dir():
        zarr_path = zarr_path_new
    elif zarr_path_old.exists() or zarr_path_old.is_dir():
        zarr_path = zarr_path_old
    else:
        raise FileNotFoundError(
            f"Zarr file with run_id {run_id}, mini_epoch {mini_epoch} and rank {rank} does not "
            f"exist or is not a directory."
        )

    return zarr_path


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


def load_merge_configs(
    private_home: Path | None = None,
    from_run_id: str | None = None,
    mini_epoch: int | None = None,
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

    private_config = set_paths(private_config)

    if from_run_id is None:
        base_config = _load_default_conf()
    else:
        base_config = load_run_config(
            from_run_id, mini_epoch, private_config.get("model_path", None)
        )
        from_run_id = base_config.run_id
    with open_dict(base_config):
        base_config.from_run_id = from_run_id
    # use OmegaConf.unsafe_merge if too slow
    c = OmegaConf.merge(base_config, private_config, *overwrite_configs)
    assert isinstance(c, Config)

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
        assert config.run_id is not None, "run_id loaded from previous run should not be None."
        _logger.info(f"reusing run_id from previous run: {config.run_id}")
    else:
        if run_id is None:
            # generate new id if run_id is None
            config.run_id = run_id or get_run_id()
            _logger.info(f"using generated run_id: {config.run_id}")
        else:
            config.run_id = run_id
            _logger.info(
                f"using assigned run_id: {config.run_id}."
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
    "Return the private configuration."
    "If none, take it from the environment variable WEATHERGEN_PRIVATE_CONF."

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


def _load_default_conf() -> Config:
    """Deserialize default configuration."""
    c = OmegaConf.load(_DEFAULT_CONFIG_PTH)
    assert isinstance(c, Config)
    return c


def load_streams(streams_directory: Path) -> list[Config]:
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


def set_paths(config: Config) -> Config:
    """Set the configs run_path model_path attributes to default values if not present."""
    config = config.copy()
    config.run_path = _get_config_attribute(
        config=config, attribute_name="run_path", fallback="results"
    )
    config.model_path = _get_config_attribute(
        config=config, attribute_name="model_path", fallback="models"
    )

    return config


def _get_config_attribute(config: Config, attribute_name: str, fallback: str) -> str:
    """Get an attribute from a Config. If not available, fall back to path_shared_working_dir
    concatenated with the desired fallback path. Raise an error if neither the attribute nor a
    fallback is specified."""
    attribute = OmegaConf.select(config, attribute_name)
    fallback_root = OmegaConf.select(config, "path_shared_working_dir")
    assert attribute is not None or fallback_root is not None, (
        f"Must specify `{attribute_name}` in config if `path_shared_working_dir` is None in config"
    )
    attribute = attribute if attribute else fallback_root + fallback
    return attribute


def get_path_run(config: Config) -> Path:
    """Get the current runs run_path for storing run results and logs."""
    return Path(config.run_path) / config.run_id


def get_path_model(config: Config) -> Path:
    """Get the current runs model_path for storing model checkpoints."""
    return Path(config.model_path) / config.run_id


def get_path_output(config: Config, mini_epoch: int) -> Path:
    base_path = get_path_run(config)
    fname = f"validation_chkpt{mini_epoch:05d}_rank{config.rank:04d}.zarr"

    return base_path / fname


def get_shared_wg_path(local_path: str | Path) -> Path:
    """
    Resolves a local, relative path to an absolute path within the configured shared working
    directory.

    This utility function retrieves the base path defined for the shared WeatherGenerator (WG)
    working directory from the private configuration and appends the provided local path segment.

    Parameters
    ----------
    local_path : str or Path
        The local or relative path segment (e.g., 'results', 'models', 'output') that needs
        to be located within the shared working directory structure.

    Returns
    -------
    Path
        The absolute pathlib.Path object pointing to the specified location
        within the shared working directory.

    Notes
    -----
    The shared working directory base is retrieved from the 'path_shared_working_dir'
    key found in the private configuration loaded by `_load_private_conf()`.
    """
    pcfg = _load_private_conf()
    return Path(pcfg.get("path_shared_working_dir")) / local_path


def validate_forecast_policy_and_steps(cf: OmegaConf):
    """
    Validates the forecast policy and steps within a configuration object.

    This method enforces specific rules for the `forecast_steps` attribute, which can be
    either a single integer or a list of integers, ensuring consistency with the
    `forecast_policy` attribute.

    The validation logic is as follows:
    - If `cf.forecast_steps` is a single integer, a `forecast_policy` must be defined
    (i.e., not None or empty) only if `forecast_steps` is unequal to 0.
    - If `cf.forecast_steps` is a list, it must be non-empty, and all of its elements
    must be non-negative integers. Additionally, a `forecast_policy` must be
    defined if any of the forecast steps in the list are greater than 0.

    Args:
        cf (OmegaConf): The configuration object containing the `forecast_steps`
                        and `forecast_policy` attributes.

    Raises:
        TypeError: If `cf.forecast_steps` is not an integer or a non-empty list.
        AssertionError: If a `forecast_policy` is required but not provided, or
                        if `forecast_step` is negative while `forecast_policy` is provided, or
                        if any of the forecast steps in a list are negative.
    """
    provide_forecast_policy = (
        "A 'forecast_policy' must be specified when 'forecast_steps' is not zero. "
    )
    valid_forecast_policies = (
        "Valid values for 'forecast_policy' are, e.g., 'fixed' when using constant "
        "forecast steps throughout the training, or 'sequential' when varying the forecast "
        "steps over mini_epochs, such as, e.g., 'forecast_steps: [2, 2, 4, 4]'. "
    )
    valid_forecast_steps = (
        "'forecast_steps' must be a positive integer or a non-empty list of positive integers. "
    )
    if isinstance(cf.forecast_steps, int):
        assert cf.forecast_policy and cf.forecast_steps > 0 if cf.forecast_steps != 0 else True, (
            provide_forecast_policy + valid_forecast_policies + valid_forecast_steps
        )
    elif isinstance(cf.forecast_steps, ListConfig) and len(cf.forecast_steps) > 0:
        assert (
            cf.forecast_policy and all(step >= 0 for step in cf.forecast_steps)
            if any(n > 0 for n in cf.forecast_steps)
            else True
        ), provide_forecast_policy + valid_forecast_policies + valid_forecast_steps
    else:
        raise TypeError(valid_forecast_steps)
