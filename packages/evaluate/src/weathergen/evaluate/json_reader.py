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
from weathergen.evaluate.io_reader import Reader

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

class JsonReader(Reader):
    """Specialized Reader for runs that include a Zarr output file."""

    def __init__(self, eval_cfg: dict, run_id: str, private_paths: dict = None):
        super().__init__(eval_cfg, run_id, private_paths)

    def check_availability(self, 
        stream: str, 
        available_data: dict = None,
        mode: str = ""
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

        for name in ["channel", "fstep", "sample"]:
            if requested[name] is None:
                # Default to all in Zarr
                requested[name] = available[name]

            # Must be a subset of available_data (if provided)
            breakpoint()
            missing = requested[name] - available[name]
            assert requested[name] <= available[name], f"{name.capitalize()}(s) {missing} missing in evaluation. Adjust selection."

            _logger.info(
                f"All checks passed – All channels, samples, fsteps requested for {mode} are present in json file..."
            )

        return True, (
            sorted(list(requested["channel"])),
            sorted(list(requested["fstep"])),
            sorted(list(requested["sample"])),
        )

    #TODO: improve this
    def plot_subtimesteps(self, stream: str):
        return True


