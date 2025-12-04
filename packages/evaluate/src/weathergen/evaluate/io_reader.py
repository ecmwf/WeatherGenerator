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
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import omegaconf as oc
import pandas as pd
import xarray as xr
from tqdm import tqdm

from weathergen.common.config import get_shared_wg_path, load_config, load_model_config
from weathergen.common.io import ZarrIO
from weathergen.evaluate.derived_channels import DeriveChannels
from weathergen.evaluate.score_utils import to_list

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


@dataclass
class ReaderOutput:
    """
    Dataclass to hold the output of the Reader.get_data method.
    Attributes
    ----------
    target : dict[str, xr.Dataset]
        Dictionary of xarray Datasets for targets, indexed by forecast step.
    prediction : dict[str, xr.Dataset]
        Dictionary of xarray Datasets for predictions, indexed by forecast step.
    points_per_sample : xr.DataArray | None
        xarray DataArray containing the number of points per sample, if `return_counts` is True
    """

    target: dict[str, xr.Dataset]
    prediction: dict[str, xr.Dataset]
    points_per_sample: xr.DataArray | None


@dataclass
class DataAvailability:
    """
    Dataclass to hold information about data availability in the input files.
    Attributes
    ----------
    score_availability: bool
        True if the metric file contains the requested combination.
    channels: list[str]
        List of channels requested
    fsteps: list[int]
        List of forecast steps requested
    samples: list[int]
        List of samples requested
    """

    score_availability: bool
    channels: list[str] | None
    fsteps: list[int] | None
    samples: list[int] | None
    ensemble: list[str] | None = None


class Reader:
    def __init__(self, eval_cfg: dict, run_id: str, private_paths: dict[str, str] | None = None):
        """
        Generic data reader class.

        Parameters
        ----------
        eval_cfg : dir
            config with plotting and evaluation options for that run id
        run_id : str
            run id of the model
        private_paths: dict[srt, str]
            dictionary of private paths for the supported HPC
        """
        self.eval_cfg = eval_cfg
        self.run_id = run_id
        self.private_paths = private_paths
        self.streams = eval_cfg.streams.keys()
        # TODO: propagate it to the other functions using global plotting opts
        self.global_plotting_options = eval_cfg.get("global_plotting_options", {})

        # If results_base_dir and model_base_dir are not provided, default paths are used
        self.model_base_dir = self.eval_cfg.get("model_base_dir", None)

        self.results_base_dir = self.eval_cfg.get(
            "results_base_dir", None
        )  # base directory where results will be stored

    def get_stream(self, stream: str):
        """
        returns the dictionary associated to a particular stream

        Parameters
        ----------
        stream: str
            the stream name

        Returns
        -------
        dict
            the config dictionary associated to that stream
        """
        return self.eval_cfg.streams.get(stream, {})

    def get_samples(self) -> set[int]:
        """Placeholder implementation of sample getter. Override in subclass."""
        return set()

    def get_forecast_steps(self) -> set[int]:
        """Placeholder implementation forecast step getter. Override in subclass."""
        return set()

    # TODO: get this from config
    def get_channels(self, stream: str | None = None) -> list[str]:
        """Placeholder implementation channel names getter. Override in subclass."""
        return list()

    def get_ensemble(self, stream: str | None = None) -> list[str]:
        """Placeholder implementation ensemble member names getter. Override in subclass."""
        return list()

    def is_regular(self, stream: str) -> bool:
        """
        Placeholder implementation to check if lat/lon are regularly spaced.
        Override in subclass.
        """
        return True

    def load_scores(self, stream: str, region: str, metric: str) -> xr.DataArray:
        """Placeholder to load pre-computed scores for a given run, stream, metric"""
        return None

    def check_availability(
        self,
        stream: str,
        available_data: dict | None = None,
        mode: str = "",
    ) -> DataAvailability:
        """
        Check if requested channels, forecast steps and samples are
        i) available in the previously saved metric file if specified (return False otherwise)
        ii) available in the source file (e.g. the Zarr file, return error otherwise)
        Additionally, if channels, forecast steps or samples is None/'all', it will
        i) set the variable to all available vars in source file
        ii) return True only if the respective variable contains the same indeces in metric file
            and source file (return False otherwise)

        Parameters
        ----------
        stream : str
            The stream considered.
        available_data : dict, optional
            The available data loaded from metric file.
        Returns
        -------
        DataAvailability
            A dataclass containing:
            - channels: list of channels or None if 'all'
            - fsteps: list of forecast steps or None if 'all'
            - samples: list of samples or None if 'all'
        """

        # fill info for requested channels, fsteps, samples
        requested_data = self._get_channels_fsteps_samples(stream, mode)

        channels = requested_data.channels
        fsteps = requested_data.fsteps
        samples = requested_data.samples
        ensemble = requested_data.ensemble
        requested = {
            "channel": set(channels) if channels is not None else None,
            "fstep": set(fsteps) if fsteps is not None else None,
            "sample": set(samples) if samples is not None else None,
            "ensemble": set(ensemble) if ensemble is not None else None,
        }

        # fill info from available metric file (if provided)
        available = {
            "channel": (
                set(available_data["channel"].values.ravel())
                if available_data is not None
                else set()
            ),
            "fstep": (
                set(available_data["forecast_step"].values.ravel())
                if available_data is not None
                else set()
            ),
            "sample": (
                set(available_data.coords["sample"].values.ravel())
                if available_data is not None
                else set()
            ),
            "ensemble": (
                set(available_data["ens"].values.ravel())
                if available_data is not None and "ens" in available_data.coords
                else set()
            ),
        }

        # fill info from reader
        reader_data = {
            "fstep": set(int(f) for f in self.get_forecast_steps()),
            "sample": set(int(s) for s in self.get_samples()),
            "channel": set(self.get_channels(stream)),
            "ensemble": set(self.get_ensemble(stream)),
        }

        check_score = True
        corrected = False
        for name in ["channel", "fstep", "sample", "ensemble"]:
            if requested[name] is None:
                # Default to all in Zarr
                requested[name] = reader_data[name]
                # If file with metrics exists, must exactly match
                if available_data is not None and reader_data[name] != available[name]:
                    _logger.info(
                        f"Requested all {name}s for {mode}, but previous config was a "
                        "strict subset. Recomputing."
                    )
                    check_score = False

            # Must be subset of Zarr
            if not requested[name] <= reader_data[name]:
                missing = requested[name] - reader_data[name]

                if name == "ensemble" and "mean" in missing:
                    missing.remove("mean")
                if missing:
                    _logger.info(
                        f"Requested {name}(s) {missing} do(es) not exist in Zarr. "
                        f"Removing missing {name}(s) for {mode}."
                    )
                    requested[name] = requested[name] & reader_data[name]
                    corrected = True

            # Must be a subset of available_data (if provided)
            if available_data is not None and not requested[name] <= available[name]:
                missing = requested[name] - available[name]
                _logger.info(
                    f"{name.capitalize()}(s) {missing} missing in previous evaluation. Recomputing."
                )
                check_score = False

        if check_score and not corrected:
            scope = "metric file" if available_data is not None else "Zarr file"
            _logger.info(
                f"All checks passed – All channels, samples, fsteps requested for {mode} are "
                f"present in {scope}..."
            )

        return DataAvailability(
            score_availability=check_score,
            channels=sorted(list(requested["channel"])),
            fsteps=sorted(list(requested["fstep"])),
            samples=sorted(list(requested["sample"])),
            ensemble=sorted(list(requested["ensemble"])),
        )

    def _get_channels_fsteps_samples(self, stream: str, mode: str) -> DataAvailability:
        """
        Get channels, fsteps and samples for a given run and stream from the config.
        Replace 'all' with None.

        Parameters
        ----------
        stream: str
            The stream considered.
        mode: str
            if plotting or evaluation mode

        Returns
        -------
        DataAvailability
            A dataclass containing:
            - channels: list of channels or None if 'all'
            - fsteps: list of forecast steps or None if 'all'
            - samples: list of samples or None if 'all'
        """
        assert mode == "plotting" or mode == "evaluation", (
            "get_channels_fsteps_samples:: Mode should be either 'plotting' or 'evaluation'"
        )

        stream_cfg = self.get_stream(stream)
        assert stream_cfg.get(mode, False), "Mode does not exist in stream config. Please add it."

        samples = stream_cfg[mode].get("sample", None)
        fsteps = stream_cfg[mode].get("forecast_step", None)
        channels = stream_cfg.get("channels", None)
        ensemble = stream_cfg[mode].get("ensemble", None)
        if ensemble == "mean":
            ensemble = ["mean"]

        if isinstance(fsteps, str) and fsteps != "all":
            assert re.match(r"^\d+-\d+$", fsteps), (
                "String format for forecast_step in config must be 'digit-digit' or 'all'"
            )
            fsteps = list(range(int(fsteps.split("-")[0]), int(fsteps.split("-")[1]) + 1))
        if isinstance(samples, str) and samples != "all":
            assert re.match(r"^\d+-\d+$", samples), (
                "String format for sample in config must be 'digit-digit' or 'all'"
            )
            samples = list(range(int(samples.split("-")[0]), int(samples.split("-")[1]) + 1))

        return DataAvailability(
            score_availability=True,
            channels=None if (channels == "all" or channels is None) else list(channels),
            fsteps=None if (fsteps == "all" or fsteps is None) else list(fsteps),
            samples=None if (samples == "all" or samples is None) else list(samples),
            ensemble=None if (ensemble == "all" or ensemble is None) else list(ensemble),
        )


##### Helper function for CSVReader ####
def _rename_channels(data) -> pd.DataFrame:
    """
    The scores downloaded from Quaver have a different convention. Need renaming.
    Rename channel names to include underscore between letters and digits.
    E.g., 'z500' -> 'z_500', 't850' -> 't_850', '2t' -> '2t', '10ff' -> '10ff'

    Parameters
    ----------
    name : str
        Original channel name.

    Returns
    -------
    pd.DataFrame
        Dataset with renamed channel names.
    """
    for name in list(data.index):
        # If it starts with digits (surface vars like 2t, 10ff) → leave unchanged
        if re.match(r"^\d", name):
            continue

        # Otherwise, insert underscore between letters and digits
        data = data.rename(index={name: re.sub(r"([a-zA-Z])(\d+)", r"\1_\2", name)})

    return data


class CsvReader(Reader):
    """
    Reader class to read evaluation data from CSV files and convert to xarray DataArray.
    """

    def __init__(self, eval_cfg: dict, run_id: str, private_paths: dict | None = None):
        """
        Initialize the CsvReader.

        Parameters
        ----------
        eval_cfg : dir
            config with plotting and evaluation options for that run id
        run_id : str
            run id of the model
        private_paths: lists
            list of private paths for the supported HPC
        """

        super().__init__(eval_cfg, run_id, private_paths)
        self.csv_path = eval_cfg.get("csv_path")
        assert self.csv_path is not None, "CSV path must be provided in the config."

        pd_data = pd.read_csv(self.csv_path, index_col=0)

        self.data = _rename_channels(pd_data)
        self.metrics_base_dir = Path(self.csv_path).parent
        # for backward compatibility allow metric_dir to be specified in the run config
        self.metrics_dir = Path(
            self.eval_cfg.get("metrics_dir", self.metrics_base_dir / self.run_id / "evaluation")
        )

        assert len(eval_cfg.streams.keys()) == 1, "CsvReader only supports one stream."
        self.stream = list(eval_cfg.streams.keys())[0]
        self.channels = self.data.index.tolist()
        self.samples = [0]
        self.forecast_steps = [int(col.split()[0]) for col in self.data.columns]
        self.npoints_per_sample = [0]
        self.epoch = eval_cfg.get("epoch", 0)
        self.metric = eval_cfg.get("metric")
        self.region = eval_cfg.get("region")

    def get_samples(self) -> set[int]:
        """get set of samples for the retrieved scores (initialisation times)"""
        return set(self.samples)  # Placeholder implementation

    def get_forecast_steps(self) -> set[int]:
        """get set of forecast steps"""
        return set(self.forecast_steps)  # Placeholder implementation

    # TODO: get this from config
    def get_channels(self, stream: str | None = None) -> list[str]:
        """get set of channels"""
        assert stream == self.stream, "streams do not match in CSVReader."
        return list(self.channels)  # Placeholder implementation

    def get_values(self) -> xr.DataArray:
        """get score values in the right format"""
        return self.data.values[np.newaxis, :, :, np.newaxis].T

    def load_scores(self, stream: str, region: str, metric: str) -> xr.DataArray:
        """
        Load the existing scores for a given run, stream and metric.

        Parameters
        ----------
        reader :
            Reader object containing all info for a specific run_id
        stream :
            Stream name.
        region :
            Region name.
        metric :
            Metric name.

        Returns
        -------
        xr.DataArray
            The metric DataArray.
        """

        available_data = self.check_availability(stream, mode="evaluation")

        # fill it only for matching metric
        if metric == self.metric and region == self.region and stream == self.stream:
            data = self.get_values()
        else:
            data = np.full(
                (
                    len(available_data.samples),
                    len(available_data.fsteps),
                    len(available_data.channels),
                    1,
                ),
                np.nan,
            )

        da = xr.DataArray(
            data.astype(np.float32),
            dims=("sample", "forecast_step", "channel", "metric"),
            coords={
                "sample": available_data.samples,
                "forecast_step": available_data.fsteps,
                "channel": available_data.channels,
                "metric": [metric],
            },
            attrs={"npoints_per_sample": self.npoints_per_sample},
        )

        return da


class WeatherGenReader(Reader):
    def __init__(self, eval_cfg: dict, run_id: str, private_paths: dict | None = None):
        """Data reader class for WeatherGenerator model outputs stored in Zarr format."""

        super().__init__(eval_cfg, run_id, private_paths)

        # TODO: remove backwards compatibility to "epoch" in Feb. 2026
        self.mini_epoch = getattr(eval_cfg, "mini_epoch", getattr(eval_cfg, "epoch", -1))
        self.rank = eval_cfg.rank

        # Load model configuration and set (run-id specific) directories
        self.inference_cfg = self.get_inference_config()

        if not self.results_base_dir:
            self.results_base_dir = Path(get_shared_wg_path("results"))
            _logger.info(f"Results directory obtained from private config: {self.results_base_dir}")
        else:
            _logger.info(f"Results directory parsed: {self.results_base_dir}")

        self.runplot_base_dir = Path(
            self.eval_cfg.get("runplot_base_dir", self.results_base_dir)
        )  # base directory where map plots and histograms will be stored

        self.metrics_base_dir = Path(
            self.eval_cfg.get("metrics_base_dir", self.results_base_dir)
        )  # base directory where score files will be stored

        self.results_dir, self.runplot_dir = (
            Path(self.results_base_dir) / self.run_id,
            Path(self.runplot_base_dir) / self.run_id,
        )
        # for backward compatibility allow metric_dir to be specified in the run config
        self.metrics_dir = Path(
            self.eval_cfg.get("metrics_dir", self.metrics_base_dir / self.run_id / "evaluation")
        )

        fname_zarr_new = self.results_dir.joinpath(
            f"validation_chkpt{self.mini_epoch:05d}_rank{self.rank:04d}.zarr"
        )
        fname_zarr_old = self.results_dir.joinpath(
            f"validation_epoch{self.mini_epoch:05d}_rank{self.rank:04d}.zarr"
        )

        if fname_zarr_new.exists() or fname_zarr_new.is_dir():
            self.fname_zarr = fname_zarr_new
        else:
            self.fname_zarr = fname_zarr_old

        if not self.fname_zarr.exists() or not self.fname_zarr.is_dir():
            _logger.error(f"Zarr file {self.fname_zarr} does not exist.")
            raise FileNotFoundError(
                f"Zarr file {self.fname_zarr} does not exist or is not a directory."
            )

    def get_inference_config(self):
        """
        load the config associated to the inference run (different from the eval_cfg which
        contains plot and evaluaiton options.)

        Returns
        -------
        dict
            configuration file from the inference run
        """
        if self.private_paths:
            _logger.info(
                f"Loading config for run {self.run_id} from private paths: {self.private_paths}"
            )
            config = load_config(self.private_paths, self.run_id, self.mini_epoch)
        else:
            _logger.info(
                f"Loading config for run {self.run_id} from model directory: {self.model_base_dir}"
            )
            config = load_model_config(self.run_id, self.mini_epoch, self.model_base_dir)

        if type(config) not in [dict, oc.DictConfig]:
            _logger.warning("Model config not found. inference config will be empty.")
            config = {}

        return config

    def get_data(
        self,
        stream: str,
        samples: list[int] | None = None,
        fsteps: list[str] | None = None,
        channels: list[str] | None = None,
        ensemble: list[str] | None = None,
        return_counts: bool = False,
    ) -> ReaderOutput:
        """
        Retrieve prediction and target data for a given run from the Zarr store.

        Parameters
        ----------
        cfg :
            Configuration dictionary containing all information for the evaluation.
        results_dir : Path
            Directory where the inference results are stored.
            Expected scheme `<results_base_dir>/<run_id>`.
        stream :
            Stream name to retrieve data for.
        samples :
            List of sample indices to retrieve. If None, all samples are retrieved.
        fsteps :
            List of forecast steps to retrieve. If None, all forecast steps are retrieved.
        channels :
            List of channel names to retrieve. If None, all channels are retrieved.
        return_counts :
            If True, also return the number of points per sample.

        Returns
        -------
        ReaderOutput
            A dataclass containing:
            - target: Dictionary of xarray DataArrays for targets, indexed by forecast step.
            - prediction: Dictionary of xarray DataArrays for predictions, indexed by forecast step.
            - points_per_sample: xarray DataArray containing the number of points per sample,
              if `return_counts` is True.
        """

        with ZarrIO(self.fname_zarr) as zio:
            stream_cfg = self.get_stream(stream)
            all_channels = self.get_channels(stream)
            _logger.info(f"RUN {self.run_id}: Processing stream {stream}...")

            fsteps = self.get_forecast_steps() if fsteps is None else fsteps

            # TODO: Avoid conversion of fsteps and sample to integers (as obtained from the ZarrIO)
            fsteps = sorted([int(fstep) for fstep in fsteps])
            samples = samples or sorted([int(sample) for sample in self.get_samples()])
            channels = channels or stream_cfg.get("channels", all_channels)
            channels = to_list(channels)

            ensemble = ensemble or self.get_ensemble(stream)
            ensemble = to_list(ensemble)

            dc = DeriveChannels(
                all_channels,
                channels,
                stream_cfg,
            )

            da_tars, da_preds = [], []

            if return_counts:
                points_per_sample = xr.DataArray(
                    np.full((len(fsteps), len(samples)), np.nan),
                    coords={"forecast_step": fsteps, "sample": samples},
                    dims=("forecast_step", "sample"),
                    name=f"points_per_sample_{stream}",
                )
            else:
                points_per_sample = None

            fsteps_final = []

            for fstep in fsteps:
                _logger.info(f"RUN {self.run_id} - {stream}: Processing fstep {fstep}...")
                da_tars_fs, da_preds_fs, pps = [], [], []

                for sample in tqdm(samples, desc=f"Processing {self.run_id} - {stream} - {fstep}"):
                    
                    out = zio.get_data(sample, stream, fstep)
                    
                    if out.target is None or out.prediction is None:
                        _logger.info(
                            f"Skipping {stream} sample {sample} forecast step: {fstep}. "
                            "No data found."
                        )
                        continue

                    target, pred = out.target.as_xarray(), out.prediction.as_xarray()

                    npoints = len(target.ipoint)
                    pps.append(npoints)

                    if npoints == 0:
                        _logger.info(
                            f"Skipping {stream} sample {sample} forecast step: {fstep}. "
                            "Dataset is empty."
                        )
                        continue

                    if ensemble == ["mean"]:
                        _logger.debug("Averaging over ensemble members.")
                        pred = pred.mean("ens", keepdims=True)
                    else:
                        _logger.debug(f"Selecting ensemble members {ensemble}.")
                        pred = pred.sel(ens=ensemble)

                    da_tars_fs.append(target.squeeze())
                    da_preds_fs.append(pred.squeeze())

                if not da_tars_fs:
                    _logger.info(
                        f"[{self.run_id} - {stream}] No valid data found for fstep {fstep}."
                    )
                    continue

                fsteps_final.append(fstep)

                _logger.debug(
                    f"Concatenating targets and predictions for stream {stream}, "
                    f"forecast_step {fstep}..."
                )

                # faster processing
                if self.is_regular(stream):
                    # Efficient concatenation for regular grid
                    da_preds_fs = _force_consistent_grids(da_preds_fs)
                    da_tars_fs = _force_consistent_grids(da_tars_fs)

                else:
                    # Irregular (scatter) case. concatenate over ipoint
                    da_tars_fs = xr.concat(da_tars_fs, dim="ipoint")
                    da_preds_fs = xr.concat(da_preds_fs, dim="ipoint")

                if len(samples) == 1:
                    _logger.debug("Repeating sample coordinate for single-sample case.")
                    for da in (da_tars_fs, da_preds_fs):
                        da.assign_coords(
                            sample=(
                                "ipoint",
                                np.repeat(da.sample.values, da.sizes["ipoint"]),
                            )
                        )

                if set(channels) != set(all_channels):
                    _logger.debug(
                        f"Restricting targets and predictions to channels {channels} "
                        f"for stream {stream}..."
                    )

                    da_tars_fs, da_preds_fs, channels = dc.get_derived_channels(
                        da_tars_fs, da_preds_fs
                    )

                    da_tars_fs = da_tars_fs.sel(channel=channels)
                    da_preds_fs = da_preds_fs.sel(channel=channels)

                da_tars.append(da_tars_fs)
                da_preds.append(da_preds_fs)
                if return_counts:
                    points_per_sample.loc[{"forecast_step": fstep}] = np.array(pps)

            # Safer than a list
            da_tars = {fstep: da for fstep, da in zip(fsteps_final, da_tars, strict=True)}
            da_preds = {fstep: da for fstep, da in zip(fsteps_final, da_preds, strict=True)}

            return ReaderOutput(
                target=da_tars, prediction=da_preds, points_per_sample=points_per_sample
            )

    ######## reader utils ########

    def get_climatology_filename(self, stream: str) -> str | None:
        """
        Get the climatology filename for a given stream from the inference configuration.
        Parameters
        ----------
        stream : str
            Name of the data stream.
        Returns
        -------
        str or None
            Climatology filename if specified, otherwise None.
        """

        stream_dict = self.get_stream(stream)

        clim_data_path = stream_dict.get("climatology_path", None)
        if not clim_data_path:
            clim_base_dir = self.inference_cfg.get("data_path_aux", None)

            clim_fn = next(
                (
                    item.get("climatology_filename")
                    for item in self.inference_cfg["streams"]
                    if item.get("name") == stream
                ),
                None,
            )

            if clim_base_dir and clim_fn:
                clim_data_path = Path(clim_base_dir).join(clim_fn)
            else:
                _logger.warning(
                    f"No climatology path specified for stream {stream}. Setting climatology to "
                    "NaN. Add 'climatology_path' to evaluation config to use metrics like ACC."
                )

        return clim_data_path

    def get_stream(self, stream: str):
        """
        returns the dictionary associated to a particular stream.
        Returns an empty dictionary if the stream does not exist in the Zarr file.

        Parameters
        ----------
        stream: str
            the stream name

        Returns
        -------
        dict
            the config dictionary associated to that stream
        """
        stream_dict = {}
        with ZarrIO(self.fname_zarr) as zio:
            if stream in zio.streams:
                stream_dict = self.eval_cfg.streams.get(stream, {})
        return stream_dict

    def get_samples(self) -> set[int]:
        """Get the set of sample indices from the Zarr file."""
        with ZarrIO(self.fname_zarr) as zio:
            return set(int(s) for s in zio.samples)

    def get_forecast_steps(self) -> set[int]:
        """Get the set of forecast steps from the Zarr file."""
        with ZarrIO(self.fname_zarr) as zio:
            return set(int(f) for f in zio.forecast_steps)

    def get_channels(self, stream: str) -> list[str]:
        """
        Get the list of channels for a given stream from the config.

        Parameters
        ----------
        stream : str
            The name of the stream to get channels for.

        Returns
        -------
        list[str]
            A list of channel names.
        """
        _logger.debug(f"Getting channels for stream {stream}...")
        all_channels = self.get_inference_stream_attr(stream, "val_target_channels")
        _logger.debug(f"Channels found in config: {all_channels}")
        return all_channels

    def get_ensemble(self, stream: str | None = None) -> list[str]:
        """Get the list of ensemble member names for a given stream from the config.
        Parameters
        ----------
        stream : str
            The name of the stream to get channels for.

        Returns
        -------
        list[str]
            A list of ensemble members.
        """
        _logger.debug(f"Getting ensembles for stream {stream}...")

        # TODO: improve this to get ensemble from io class
        with ZarrIO(self.fname_zarr) as zio:
            dummy = zio.get_data(0, stream, zio.forecast_steps[0])
        return list(dummy.prediction.as_xarray().coords["ens"].values)

    # TODO: improve this
    def is_regular(self, stream: str) -> bool:
        """Check if the latitude and longitude coordinates are regularly spaced for a given stream.
        Parameters
        ----------
        stream : str
            The name of the stream to get channels for.

        Returns
        -------
        bool
            True if the stream is regularly spaced. False otherwise.
        """
        _logger.debug(f"Checking regular spacing for stream {stream}...")

        with ZarrIO(self.fname_zarr) as zio:
            dummy = zio.get_data(0, stream, zio.forecast_steps[0])

            sample_idx = zio.samples[1] if len(zio.samples) > 1 else zio.samples[0]
            fstep_idx = (
                zio.forecast_steps[1] if len(zio.forecast_steps) > 1 else zio.forecast_steps[0]
            )
            dummy1 = zio.get_data(sample_idx, stream, fstep_idx)

        da = dummy.prediction.as_xarray()
        da1 = dummy1.prediction.as_xarray()

        if (
            da["lat"].shape != da1["lat"].shape
            or da["lon"].shape != da1["lon"].shape
            or not (
                np.allclose(sorted(da["lat"].values), sorted(da1["lat"].values))
                and np.allclose(sorted(da["lon"].values), sorted(da1["lon"].values))
            )
        ):
            _logger.debug("Latitude and/or longitude coordinates are not regularly spaced.")
            return False

        _logger.debug("Latitude and longitude coordinates are regularly spaced.")
        return True

    def load_scores(self, stream: str, region: str, metric: str) -> xr.DataArray | None:
        """
        Load the pre-computed scores for a given run, stream and metric and epoch.

        Parameters
        ----------
        reader :
            Reader object containing all info for a specific run_id
        stream :
            Stream name.
        region :
            Region name.
        metric :
            Metric name.

        Returns
        -------
        xr.DataArray
            The metric DataArray or None if the file does not exist.
        """
        score_path = (
            Path(self.metrics_dir)
            / f"{self.run_id}_{stream}_{region}_{metric}_chkpt{self.mini_epoch:05d}.json"
        )
        _logger.debug(f"Looking for: {score_path}")

        if score_path.exists():
            with open(score_path) as f:
                data_dict = json.load(f)
                return xr.DataArray.from_dict(data_dict)
        else:
            return None

    def get_inference_stream_attr(self, stream_name: str, key: str, default=None):
        """
        Get the value of a key for a specific stream from the a model config.

        Parameters:
        ------------
            config: dict
                The full configuration dictionary.
            stream_name: str
                The name of the stream (e.g. 'ERA5').
            key: str
                The key to look up (e.g. 'tokenize_spacetime').
            default: Optional
                Value to return if not found (default: None).

        Returns:
            The parameter value if found, otherwise the default.
        """
        for stream in self.inference_cfg.get("streams", []):
            if stream.get("name") == stream_name:
                return stream.get(key, default)
        return default


################### Helper functions ########################


def _force_consistent_grids(ref: list[xr.DataArray]) -> xr.DataArray:
    """
    Force all samples to share the same ipoint order.

    Parameters
    ----------
    ref:
       Input dataset
    Returns
    -------
        xr.DataArray
        Returns a Dataset where all samples have the same lat lon and ipoint ordering
    """

    # Pick first sample as reference
    ref_lat = ref[0].lat
    ref_lon = ref[0].lon

    sort_idx = np.lexsort((ref_lon.values, ref_lat.values))
    npoints = sort_idx.size
    aligned = []
    for a in ref:
        a_sorted = a.isel(ipoint=sort_idx)

        a_sorted = a_sorted.assign_coords(
            ipoint=np.arange(npoints),
            lat=("ipoint", ref_lat.values[sort_idx]),
            lon=("ipoint", ref_lon.values[sort_idx]),
        )
        aligned.append(a_sorted)

    return xr.concat(aligned, dim="sample")
