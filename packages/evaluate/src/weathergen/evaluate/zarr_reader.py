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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import omegaconf as oc
import xarray as xr
from tqdm import tqdm

from weathergen.common.io import ZarrIO
from weathergen.evaluate.plot_utils import (
    plot_metric_region,
)
from weathergen.evaluate.plotter import LinePlots, Plotter
from weathergen.evaluate.score import VerifiedData, get_score
from weathergen.evaluate.score_utils import RegionBoundingBox, to_list
from weathergen.utils.config import _REPO_ROOT, load_config, load_model_config
from weathergen.evaluate.io_reader import Reader, WeatherGeneratorOutput

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

class ZarrReader(Reader):
    """Specialized Reader for runs that include a Zarr output file."""

    def __init__(self, eval_cfg: dict, run_id: str, private_paths: dict = None):
        super().__init__(eval_cfg, run_id, private_paths)

        # Load model configuration and set (run-id specific) directories
        self.inference_cfg = self._get_inference_config() 

        self.results_base_dir = self.eval_cfg.get(
                "results_base_dir", None
            )  # base directory where results will be stored

        if not self.results_base_dir:
            self.results_base_dir = Path(self.inference_cfg["run_path"])
            logging.info(
                f"Results directory obtained from model config: {self.results_base_dir}"
            )
        else:
            logging.info(f"Results directory parsed: {self.results_base_dir}")

        # Add zarr-specific handling
        self.fname_zarr = self.results_dir.joinpath(
            f"validation_epoch{self.epoch:05d}_rank{self.rank:04d}.zarr"
        )

        if not self.fname_zarr.exists() or not self.fname_zarr.is_dir():
            _logger.error(f"Zarr file {self.fname_zarr} does not exist or is not a directory.")
            raise FileNotFoundError(
                f"Zarr file {self.fname_zarr} does not exist or is not a directory."
            )

    def _get_inference_config(self):
        """
        load the config associated to the inference run (different from the eval_cfg which contains plot and evaluaiton options.)

        Returns
        -------
        dict
            configuration file from the inference run
        """
        try: 
            if self.private_paths:
                _logger.info(
                    f"Loading config for run {self.run_id} from private paths: {self.private_paths}"
                )
                return load_config(self.private_paths, self.run_id, self.epoch)
            else:
                _logger.info(
                    f"Loading config for run {self.run_id} from model directory: {self.model_base_dir}"
                )
                return load_model_config(self.run_id, self.epoch, self.model_base_dir)
        except AssertionError:
            _logger.warning("Model config not found. inference config will be empty.")
            return {}

    def get_data(self, 
        stream: str,
        region: str = "global",
        samples: list[int] = None,
        fsteps: list[str] = None,
        channels: list[str] = None,
        return_counts: bool = False,
    ) -> WeatherGeneratorOutput:
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
            Region name to retrieve data for.
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
        WeatherGeneratorOutput
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
                [int(sample) for sample in self.get_samples()] if samples is None else samples
            )
            channels = channels or stream_cfg.get("channels", all_channels)
            channels = to_list(channels)

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

                    if set(channels) != set(all_channels):
                        _logger.debug(
                            f"Restricting targets and predictions to channels {channels} for stream {stream}..."
                        )
                        available_channels = da_tars_fs.channel.values
                        existing_channels = [
                            ch for ch in channels if ch in available_channels
                        ]
                        if len(existing_channels) < len(channels):
                            _logger.warning(
                                f"The following channels were not found: {list(set(channels) - set(existing_channels))}. Skipping them."
                            )

                        da_tars_fs = da_tars_fs.sel(channel=existing_channels)
                        da_preds_fs = da_preds_fs.sel(channel=existing_channels)

                    da_tars.append(da_tars_fs)
                    da_preds.append(da_preds_fs)
                if return_counts:
                    points_per_sample.loc[{"forecast_step": fstep}] = np.array(pps)

            # Safer than a list
            da_tars = {fstep: da for fstep, da in zip(fsteps_final, da_tars, strict=False)}
            da_preds = {
                fstep: da for fstep, da in zip(fsteps_final, da_preds, strict=False)
            }

            return WeatherGeneratorOutput(
                target=da_tars, prediction=da_preds, points_per_sample=points_per_sample
            )

    ######## reader utils ########

    def get_samples(self):
        with ZarrIO(self.fname_zarr) as zio:
            return set(int(s) for s in zio.samples)
    
    def get_forecast_steps(self):
        with ZarrIO(self.fname_zarr) as zio:
            return set(int(f) for f in zio.forecast_steps)

    #TODO: get this from config
    def get_channels(self, stream: str) -> list[str]:
        """
        Peek the channels of a target stream.

        Parameters
        ----------
        stream :
            The name of the tar stream to peek.
        fstep :
            The forecast step to peek. Default is 0.
        Returns
        -------
        channels :
            A list of channel names in the tar stream.
        """
        with ZarrIO(self.fname_zarr) as zio:
            dummy_out = zio.get_data(list(self.get_samples())[0], stream, list(self.get_forecast_steps())[0])
            channels = dummy_out.target.channels
            _logger.debug(f"Peeked channels for stream {stream}: {channels}")

        return channels
    
    def plot_subtimesteps(self, stream: str):
        """
        automatically retrieve if the run contains fixed timesteps in the window. 

        Parameters:
        ------------
            stream: str
                The name of the stream (e.g. 'ERA5').

        Returns:
        --------
        bool
            False if it should be plotted as accumulated values (e.g. NPP-ATMS)
            True if the stream contains multiple separate timesteps (e.g. ERA5 or CERRA)
        """
        return self._get_inference_stream_attr(stream, "tokenize_spacetime", False)

    def _get_inference_stream_attr(self, stream_name: str, key: str, default = None):
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

    def check_availability(self, 
        stream: str, 
        available_data: dict = None,
        mode: str = "",
    ):
        """
        Check if requested channels, forecast steps and samples are
        i) available in the previously saved json if metric data is specified (return False otherwise)
        ii) available in the Zarr file (return error otherwise)
        Additionally, if channels, forecast steps or samples is None/'all', it will
        i) set the variable to all available vars in Zarr file
        ii) return True only if the respective variable contains the same indeces in JSON and Zarr (return False otherwise)

        Parameters
        ----------
        stream : str
            The stream considered.
        available_data : dict, optional
            The available data loaded from JSON.
        Returns
        -------
        bool
            True/False depending on the above logic (True if metrics do not need recomputing)
        str
            channels
        str
            fsteps
        str
            samples
        """

        # fill info for requested channels, fsteps, samples
        channels, fsteps, samples = self._get_channels_fsteps_samples(stream, mode)

        requested = {
            "channel": set(channels) if channels is not None else None,
            "fstep": set(fsteps) if fsteps is not None else None,
            "sample": set(samples) if samples is not None else None,
        }

        # fill info from available json file (if provided)
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

        check = True
        corrected = False
        for name in ["channel", "fstep", "sample"]:
            if requested[name] is None:
                # Default to all in Zarr
                requested[name] = reader_data[name]
                # If JSON exists, must exactly match
                if available_data is not None and reader_data[name] != available[name]:
                    _logger.info(
                        f"Requested all {name}s for {mode}, but previous config was a strict subset. Recomputing."
                    )
                    check = False

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
                check = False

        if check and not corrected:
            scope = "metric file" if available_data is not None else "Zarr file"
            _logger.info(
                f"All checks passed – All channels, samples, fsteps requested for {mode} are present in {scope}..."
            )
        return check, (
            sorted(list(requested["channel"])),
            sorted(list(requested["fstep"])),
            sorted(list(requested["sample"])),
        )



