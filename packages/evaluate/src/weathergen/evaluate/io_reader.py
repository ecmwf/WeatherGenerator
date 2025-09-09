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

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


@dataclass
class WeatherGeneratorOutput:
    target: dict
    prediction: dict
    points_per_sample: xr.DataArray | None

class Reader(object):
    def __init__(self, eval_cfg: dict, run_id: str, private_paths: dict = None):
        """
        Generic data reader class.

        Parameters
        ----------
        eval_cfg : dir
            config with plotting and evaluation options for that run id
        run_id : str
            run id of the model
        private_paths: lists
            list of private paths for the supported HPC
        """
        self.eval_cfg = eval_cfg
        self.run_id = run_id
        self.private_paths = private_paths

        self.streams = eval_cfg.streams.keys()
        self.epoch = eval_cfg.epoch
        self.rank = eval_cfg.rank

        # If results_base_dir and model_base_dir are not provided, default paths are used
        self.model_base_dir = self.eval_cfg.get("model_base_dir", None)

        # Load model configuration and set (run-id specific) directories
        self.inference_cfg = self.get_inference_config() 

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
        
        self.fname_zarr = self.results_dir.joinpath(
            f"validation_epoch{self.epoch:05d}_rank{self.rank:04d}.zarr"
        )

        if not self.fname_zarr.exists() or not self.fname_zarr.is_dir():
            _logger.error(f"Zarr file {self.fname_zarr} does not exist or is not a directory.")
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
    
    def get_inference_stream_attr(self, stream_name: str, key: str, default = None):
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


