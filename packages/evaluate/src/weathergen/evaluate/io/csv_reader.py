# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# Standard library
import logging
import re
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
import xarray as xr

# Local application / package
from weathergen.evaluate.io.io_reader import Reader

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class CsvReader(Reader):
    """
    Reader class to read evaluation data from CSV files and convert to xarray DataArray.
    """

    def __init__(self, eval_cfg: dict, run_id: str, private_paths: dict | None = None):
        """
        Initialize the CsvReader.

        Parameters
        ----------
        eval_cfg :
            config with plotting and evaluation options for that run id
        run_id : str
            run id of the model
        private_paths:
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


##### Helper function for CSVReader ####
def _rename_channels(data) -> pd.DataFrame:
    """
    The scores downloaded from Quaver have a different convention. Need renaming.
    Rename channel names to include underscore between letters and digits.
    E.g., 'z500' -> 'z_500', 't850' -> 't_850', '2t' -> '2t', '10ff' -> '10ff'

    Parameters
    ----------
    name :
        Original channel name.

    Returns
    -------
        Dataset with renamed channel names.
    """
    for name in list(data.index):
        # If it starts with digits (surface vars like 2t, 10ff) → leave unchanged
        if re.match(r"^\d", name):
            continue

        # Otherwise, insert underscore between letters and digits
        data = data.rename(index={name: re.sub(r"([a-zA-Z])(\d+)", r"\1_\2", name)})

    return data
