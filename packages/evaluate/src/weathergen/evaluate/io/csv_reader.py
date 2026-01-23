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
        self.metrics_dir = Path(self.eval_cfg.get("metrics_dir"))

        self.metrics_base_dir = self.metrics_dir
        # for backward compatibility allow metric_dir to be specified in the run config

        assert self.metrics_dir is not None, "metrics_dir folder must be provided in the config."

        self.stream = list(eval_cfg.streams.keys())
        assert self.stream is not None, "stream must be provided in the config."
        assert len(self.stream) == 1, "CsvReader only supports one stream."
        self.stream = self.stream[0]

        self.channels = eval_cfg.streams.get(self.stream).get("channels")
        assert self.channels is not None, "channels must be provided in the config."

        self.data = pd.DataFrame()

        # parameter,level,number,score,step,date,domain_name,value
        for channel_file in (self.metrics_dir / self.run_id).iterdir():
            data = pd.read_csv(channel_file)
            if data.empty:
                continue
            else:
                self.data = pd.concat([self.data, data], ignore_index=True)

        self.data["channel"] = (
            self.data["parameter"].astype(str) + "_" + self.data["level"].astype(str)
            if "level" in self.data.columns
            else self.data["parameter"].astype(str)
        )
        self.data["step"] = pd.to_timedelta(self.data["step"]) / np.timedelta64(1, "h")
        self.data["step"] = self.data["step"].astype(int)

        self.samples = [0]

        self.forecast_steps = sorted(self.data.step.dropna().unique().tolist())
        self.npoints_per_sample = [0]
        self.epoch = [0]

    def get_samples(self) -> set[int]:
        """get set of samples for the retrieved scores (initialisation times)"""
        return set(self.samples)  # Placeholder implementation

    def get_forecast_steps(self) -> set[int]:
        """get set of forecast steps"""
        return set(self.forecast_steps)  # Placeholder implementation

    # TODO: get this from config
    def get_channels(self, stream: str | None = None) -> list[str]:
        """get set of channels
        Parameters
        ----------
        stream :
            Stream name.
        Returns
        -------
            List of channels.
        """
        assert stream == self.stream, "streams do not match in CSVReader."
        return list(self.channels)  # Placeholder implementation

    def get_values(
        self, region: str, metric: str, forecast_steps: list[int], channels: list[str]
    ) -> xr.DataArray:
        """
        Get score values in the right format
        Parameters
        ----------
        region :
            Region name.
        metric :
            Metric name.
        forecast_steps :
            List of forecast steps.
        channels :
            List of channels.
        Returns
        -------
            The metric DataArray.
        """
        metric_name = _metric_quaver_convention(metric)
        region_name = _region_quaver_convention(region)

        data = self.data.loc[
            (self.data["score"] == metric_name)
            & (self.data["domain_name"] == region_name)
            & (self.data["step"].isin(forecast_steps))
            & (self.data["channel"].isin(channels))
        ]

        data = data.copy()
        data["sample"] = data["date"].astype("category").cat.codes
        data["forecast_step"] = data["step"].astype("category").cat.codes
        data = data.rename(columns={"step": "lead_time"})
        data = data.rename(columns={"score": "metric"})

        data = data[["sample", "forecast_step", "lead_time", "channel", "metric", "value"]]
        df = data.set_index(["sample", "forecast_step", "channel", "metric"])
        da = df["value"].to_xarray()

        lead_time_map = (
            data[["forecast_step", "lead_time"]]
            .drop_duplicates()
            .set_index("forecast_step")["lead_time"]
        )

        da = da.assign_coords(
            lead_time=("forecast_step", lead_time_map.loc[da.forecast_step.values].values)
        )

        da.attrs["npoints_per_sample"] = self.npoints_per_sample
        da["metric"] = [metric]

        return da

    def load_scores(self, stream: str, regions: str, metrics: str) -> xr.DataArray:
        """
        Load the existing scores for a given run, stream and metric.

        Parameters
        ----------
        reader :
            Reader object containing all info for a specific run_id
        stream :
            Stream name.
        regions :
            Regions name.
        metrics :
            Metrics name.

        Returns
        -------
            The metric DataArray.
        """

        available_data = self.check_availability(stream, mode="evaluation")
        channels = available_data.channels
        fsteps = available_data.fsteps
        samples = available_data.samples

        local_scores = {}

        for metric in metrics:
            for region in regions:
                # fill it only for matching metric
                data = self.get_values(
                    region=region, metric=metric, forecast_steps=fsteps, channels=channels
                )

                if data.size == 0:
                    data = xr.DataArray(
                        np.full(
                            (len(samples), len(fsteps), len(channels), 1),
                            np.nan,
                            dtype=np.float32,
                        ),
                        dims=("sample", "forecast_step", "channel", "metric"),
                        coords={
                            "sample": samples,
                            "lead_time": fsteps,
                            "forecast_step": range(len(fsteps)),
                            "channel": channels,
                            "metric": [metric],
                        },
                        attrs={"npoints_per_sample": self.npoints_per_sample},
                    )

                local_scores.setdefault(metric, {}).setdefault(region, {}).setdefault(stream, {})[
                    self.run_id
                ] = data

        return local_scores, None


def _metric_quaver_convention(metric: str) -> str:
    """
    Convert metric name to Quaver convention if needed.

    Parameters
    ----------
    metric :
        Original metric name.
    Returns
    -------
        Metric name in Quaver convention.
    """
    metric_mapping = {
        "rmse": "rmsef",
        "mae": "maef",
        "fact": "sdaf",
        "tact": "sdav",
        "acc": "ccaf",
        # Add more mappings as needed
    }
    return metric_mapping.get(metric, metric)


def _region_quaver_convention(region: str) -> str:
    """
    Convert region name to Quaver convention if needed.
    Parameters
    ----------
    region :
        Original region name.
    Returns
    -------
        Region name in Quaver convention.
    """
    region_mapping = {
        "nhem": "n.hem",
        "shem": "s.hem",
        # Add more mappings as needed
    }
    return region_mapping.get(region, region)


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
