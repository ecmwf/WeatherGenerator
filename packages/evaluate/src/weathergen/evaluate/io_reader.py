# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import omegaconf as oc
import xarray as xr
from tqdm import tqdm
from weathergen.common.config import _load_private_conf, load_config, load_model_config
from weathergen.common.io import ZarrIO
from weathergen.evaluate.derived_channels import DeriveChannels
from weathergen.evaluate.score_utils import RegionBoundingBox, to_list

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


class Reader:
    def __init__(self, eval_cfg: dict, run_id: str, private_paths: dict | None = None):
        """
        Generic data reader class.

        Parameters
        ----------
        eval_cfg : dir
            config with plotting and evaluation options for that run id
        run_id : str
            run id of the model
        private_paths: dict
            dictionary of private paths for the supported HPC
        """
        self.eval_cfg = eval_cfg
        self.run_id = run_id
        self.private_paths = private_paths

        self.streams = eval_cfg.streams.keys()

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
        ii) return True only if the respective variable contains the same indeces in metric file and source file (return False otherwise)

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

        requested = {
            "channel": set(channels) if channels is not None else None,
            "fstep": set(fsteps) if fsteps is not None else None,
            "sample": set(samples) if samples is not None else None,
        }

        # fill info from available metric file (if provided)
        available = {
            "channel": set(available_data["channel"].values.ravel())
            if available_data is not None
            else {},
            "fstep": set(available_data["forecast_step"].values.ravel())
            if available_data is not None
            else {},
            "sample": set(available_data.coords["sample"].values.ravel())
            if available_data is not None
            else {},
        }

        # fill info from reader
        reader_data = {
            "fstep": set(int(f) for f in self.get_forecast_steps()),
            "sample": set(int(s) for s in self.get_samples()),
            "channel": set(self.get_channels(stream)),
        }

        check_score = True
        corrected = False
        for name in ["channel", "fstep", "sample"]:
            if requested[name] is None:
                # Default to all in Zarr
                requested[name] = reader_data[name]
                # If file with metrics exists, must exactly match
                if available_data is not None and reader_data[name] != available[name]:
                    _logger.info(
                        f"Requested all {name}s for {mode}, but previous config was a strict subset. Recomputing."
                    )
                    check_score = False

            # Must be subset of Zarr
            if not requested[name] <= reader_data[name]:
                missing = requested[name] - reader_data[name]
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
                f"All checks passed – All channels, samples, fsteps requested for {mode} are present in {scope}..."
            )

        return DataAvailability(
            score_availability=check_score,
            channels=sorted(list(requested["channel"])),
            fsteps=sorted(list(requested["fstep"])),
            samples=sorted(list(requested["sample"])),
        )

    def _get_channels_fsteps_samples(self, stream: str, mode: str) -> DataAvailability:
        """
        Get channels, fsteps and samples for a given run and stream from the config. Replace 'all' with None.

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
        assert stream_cfg.get(mode, False), (
            "Mode does not exist in stream config. Please add it."
        )

        samples = stream_cfg[mode].get("sample", None)
        fsteps = stream_cfg[mode].get("forecast_step", None)
        channels = stream_cfg.get("channels", None)

        return DataAvailability(
            score_availability=True,
            channels=None
            if (channels == "all" or channels is None)
            else list(channels),
            fsteps=None if (fsteps == "all" or fsteps is None) else list(fsteps),
            samples=None if (samples == "all" or samples is None) else list(samples),
        )


class WeatherGenReader(Reader):
    def __init__(self, eval_cfg: dict, run_id: str, private_paths: dict | None = None):
        """Data reader class for WeatherGenerator model outputs stored in Zarr format."""

        super().__init__(eval_cfg, run_id, private_paths)

        self.epoch = eval_cfg.epoch
        self.rank = eval_cfg.rank

        # Load model configuration and set (run-id specific) directories
        self.inference_cfg = self.get_inference_config()

        if not self.results_base_dir:
            pcfg = _load_private_conf()
            self.results_base_dir = (
                Path(pcfg.get("path_shared_working_dir")) / "results"
            )
            _logger.info(
                f"Results directory obtained from private config: {self.results_base_dir}"
            )
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
            self.eval_cfg.get(
                "metrics_dir", self.metrics_base_dir / self.run_id / "evaluation"
            )
        )

        self.fname_zarr = self.results_dir.joinpath(
            f"validation_epoch{self.epoch:05d}_rank{self.rank:04d}.zarr"
        )

        if not self.fname_zarr.exists() or not self.fname_zarr.is_dir():
            _logger.error(
                f"Zarr file {self.fname_zarr} does not exist or is not a directory."
            )
            raise FileNotFoundError(
                f"Zarr file {self.fname_zarr} does not exist or is not a directory."
            )

    def get_inference_config(self):
        """
        load the config associated to the inference run (different from the eval_cfg which contains plot and evaluaiton options.)

        Returns
        -------
        dict
            configuration file from the inference run
        """
        if self.private_paths:
            _logger.info(
                f"Loading config for run {self.run_id} from private paths: {self.private_paths}"
            )
            config = load_config(self.private_paths, self.run_id, self.epoch)
        else:
            _logger.info(
                f"Loading config for run {self.run_id} from model directory: {self.model_base_dir}"
            )
            config = load_model_config(self.run_id, self.epoch, self.model_base_dir)

        if type(config) not in [dict, oc.DictConfig]:
            _logger.warning("Model config not found. inference config will be empty.")
            config = {}

        return config

    def get_data(
        self,
        stream: str,
        region: str = "global",
        samples: list[int] | None = None,
        fsteps: list[str] | None = None,
        channels: list[str] | None = None,
        return_counts: bool = False,
    ) -> ReaderOutput:
        """
        Retrieve prediction and target data for a given run from the Zarr store.

        Parameters
        ----------
        cfg :
            Configuration dictionary containing all information for the evaluation.
        results_dir : Path
            Directory where the inference results are stored. Expected scheme `<results_base_dir>/<run_id>`.
        stream :
            Stream name to retrieve data for.
        region :
            Region name to retrieve data for. Possible values: "global", "shem", "nhem", "tropics"
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
            - points_per_sample: xarray DataArray containing the number of points per sample, if `return_counts` is True.
        """

        bbox = RegionBoundingBox.from_region_name(region)

        with ZarrIO(self.fname_zarr) as zio:
            stream_cfg = self.get_stream(stream)
            all_channels = self.get_channels(stream)
            _logger.info(f"RUN {self.run_id}: Processing stream {stream}...")

            fsteps = self.get_forecast_steps() if fsteps is None else fsteps

            # TODO: Avoid conversion of fsteps and sample to integers (as obtained from the ZarrIO)
            fsteps = sorted([int(fstep) for fstep in fsteps])
            samples = sorted(
                [int(sample) for sample in self.get_samples()]
                if samples is None
                else samples
            )
            channels = channels or stream_cfg.get("channels", all_channels)
            channels = to_list(channels)

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
                _logger.info(
                    f"RUN {self.run_id} - {stream}: Processing fstep {fstep}..."
                )
                da_tars_fs, da_preds_fs = [], []
                pps = []

                for sample in tqdm(
                    samples, desc=f"Processing {self.run_id} - {stream} - {fstep}"
                ):
                    out = zio.get_data(sample, stream, fstep)
                    target, pred = out.target.as_xarray(), out.prediction.as_xarray()

                    if region != "global":
                        _logger.debug(
                            f"Applying bounding box mask for region '{region}' to targets and predictions..."
                        )
                        target = bbox.apply_mask(target)
                        pred = bbox.apply_mask(pred)

                    npoints = len(target.ipoint)
                    if npoints == 0:
                        _logger.info(
                            f"Skipping {stream} sample {sample} forecast step: {fstep}. Dataset is empty."
                        )
                        continue

                    da_tars_fs.append(target.squeeze())
                    da_preds_fs.append(pred.squeeze())
                    pps.append(npoints)

                if len(da_tars_fs) > 0:
                    fsteps_final.append(fstep)

                _logger.debug(
                    f"Concatenating targets and predictions for stream {stream}, forecast_step {fstep}..."
                )

                if da_tars_fs:
                    da_tars_fs = xr.concat(da_tars_fs, dim="ipoint")
                    da_preds_fs = xr.concat(da_preds_fs, dim="ipoint")
                    if len(samples) == 1:
                        # Ensure sample coordinate is repeated along ipoint even if only one sample
                        da_tars_fs = da_tars_fs.assign_coords(
                            sample=(
                                "ipoint",
                                np.repeat(
                                    da_tars_fs.sample.values, len(da_tars_fs.ipoint)
                                ),
                            )
                        )
                        da_preds_fs = da_preds_fs.assign_coords(
                            sample=(
                                "ipoint",
                                np.repeat(
                                    da_preds_fs.sample.values, len(da_preds_fs.ipoint)
                                ),
                            )
                        )

                    if set(channels) != set(all_channels):
                        _logger.debug(
                            f"Restricting targets and predictions to channels {channels} for stream {stream}..."
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
            da_tars = {
                fstep: da for fstep, da in zip(fsteps_final, da_tars, strict=True)
            }
            da_preds = {
                fstep: da for fstep, da in zip(fsteps_final, da_preds, strict=True)
            }

            return ReaderOutput(
                target=da_tars, prediction=da_preds, points_per_sample=points_per_sample
            )

    ######## reader utils ########

    def get_samples(self) -> set[int]:
        with ZarrIO(self.fname_zarr) as zio:
            return set(int(s) for s in zio.samples)

    def get_forecast_steps(self) -> set[int]:
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
