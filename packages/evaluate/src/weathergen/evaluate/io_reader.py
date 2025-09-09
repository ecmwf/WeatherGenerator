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
        self.run_type =  eval_cfg.run_type

        # If results_base_dir and model_base_dir are not provided, default paths are used
        self.model_base_dir = self.eval_cfg.get("model_base_dir", None)

        self.results_base_dir = self.eval_cfg.get(
                "results_base_dir", None
            )  # base directory where results will be stored

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

    def get_samples(self):
        return None 
    
    def get_forecast_steps(self):
        return None 

    def get_channels(self, stream: str):
        return None

    #TODO: improve this
    def plot_subtimesteps(self, stream: str):
        return False

    def get_data(self, stream: str, 
        region: str = "global",
        samples: list[int] = None,
        fsteps: list[str] = None,
        channels: list[str] = None,
        return_counts: bool = False,
    ):
        return WeatherGeneratorOutput({}, {})
    
    def _get_channels_fsteps_samples(self, stream: str, mode: str):
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
        list/None
            channels
        list/None
            fsteps
        list/None
            samples
        """
        assert mode == "plotting" or mode == "evaluation", (
            "get_channels_fsteps_samples:: Mode should be either 'plotting' or 'evaluation'"
        )

        stream_cfg = self.get_stream(stream)
        assert stream_cfg.get(mode, False), "Mode does not exist in stream config. Please add it."

        samples = stream_cfg[mode].get("sample", None)
        fsteps = stream_cfg[mode].get("forecast_step", None)
        channels = stream_cfg.get("channels", None)

        channels = None if (channels == "all" or channels is None) else list(channels)
        fsteps = None if (fsteps == "all" or fsteps is None) else list(fsteps)
        samples = None if (samples == "all" or samples is None) else list(samples)

        return channels, fsteps, samples


