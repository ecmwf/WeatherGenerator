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
        eval_cfg : dict
            Configuration containing plotting and evaluation options for the given run ID.
        run_id : str
            Run identifier of the model.
        private_paths : dict or None, optional
            Dictionary of private paths for the target HPC system. Defaults to None.
        """

        super().__init__(eval_cfg, run_id, private_paths)
        self.metrics_dir = Path(self.eval_cfg.get("metrics_dir"))

        self.metrics_base_dir = self.metrics_dir
        # for backward compatibility allow metric_dir to be specified
        # in the run config

        assert self.metrics_dir is not None, "metrics_dir folder must be provided in the config."

        self.stream = list(eval_cfg.streams.keys())

        assert self.stream is not None, "stream must be provided in the config."
        assert len(self.stream) == 1, "CsvReader only supports one stream."

        self.stream = self.stream[0]

        self.channels = eval_cfg.streams.get(self.stream).get("channels")
        assert self.channels is not None, "channels must be provided in the config."

        self.data = pd.DataFrame()

        # parameter,level,number,score,step,date,domain_name,value
        metrics_run_dir = self.metrics_dir / self.run_id
        for channel_file in metrics_run_dir.iterdir():
            data = pd.read_csv(channel_file)
            if not data.empty:
                self.data = pd.concat([self.data, data], ignore_index=True)

        self.data = self.data.dropna(subset=["step", "level"])
        self.data["level"] = self.data["level"].astype(int)

        self.data["channel"] = (
            self.data["parameter"].astype(str) + "_" + self.data["level"].astype(str)
            if "level" in self.data.columns
            else self.data["parameter"].astype(str)
        )
        self.data["step"] = (pd.to_timedelta(self.data["step"]) / np.timedelta64(1, "h")).astype(
            int
        )
        self.samples = [0]
        self.forecast_steps = sorted(self.data["step"].dropna().unique().tolist())
        self.npoints_per_sample = [0]
        self.epoch = [0]

    def get_samples(self) -> set[int]:
        """
        Get set of samples for the retrieved scores (initialisation times).

        Returns
        -------
        samples: set[int]
            A set containing the sample indices.
        """
        return set(self.samples)  # Placeholder implementation

    def get_forecast_steps(self) -> set[int]:
        """
        Get set of forecast steps.

        Returns
        -------
        fsteps: set[int]
            A set containing the forecast step values.
        """
        return set(self.forecast_steps)  # Placeholder implementation

    # TODO: get this from config
    def get_channels(self, stream: str | None = None) -> list[str]:
        """Get the list of available channels for a given stream.

        Parameters
        ----------
        stream : str
            The name of the stream for which to retrieve channels.

        Returns
        -------
        list[str]
            A list of channels available in the stream.
        """
        assert stream == self.stream, "streams do not match in CSVReader."
        return list(self.channels)  # Placeholder implementation

    def get_values(
        self, region: str, metric: str, forecast_steps: list[int], channels: list[str]
    ) -> xr.DataArray | None:
        """
        Retrieve metric values for the specified region, metric, forecast steps and channels.


        Parameters
        ----------
        region : str
            The name of the region to filter by.
        metric : str
            The name of the metric to filter by.
        forecast_steps : list[int]
            A list of forecast step values to include.
        channels : list[str]
            A list of channel names to include.

        Returns
        -------
        da: xr.DataArray or None
            An xarray DataArray containing the metric values with dimensions for sample,
            forecast_step, lead_time, channel, and metric. The DataArray includes attributes
            ``npoints_per_sample`` and the metric name as a coordinate.
            If no data was found for the specified region, metric, forecast steps, and channels,
            None is returned instead.
        """
        metric_name = _metric_quaver_convention(metric)
        region_name = _region_quaver_convention(region)

        data = self.data.loc[
            (self.data["score"] == metric_name)
            & (self.data["domain_name"] == region_name)
            & (self.data["step"].isin(forecast_steps))
            & (self.data["channel"].isin(channels))
        ]

        if data.empty:
            _logger.warning(
                f"No values were found for region '{region}', metric '{metric}', "
                f"forecast steps '{forecast_steps}', and channels '{channels}'"
            )
            return None

        # convert to DataArray
        data = data.copy()
        data["sample"] = data["date"].astype("category").cat.codes
        data["forecast_step"] = data["step"].astype("category").cat.codes
        data = data.rename(columns={"step": "lead_time", "score": "metric"})
        cols = ["sample", "forecast_step", "lead_time", "channel", "metric", "value"]
        df = data[cols].set_index(["sample", "forecast_step", "channel", "metric"])
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

    def load_scores(self, stream: str, regions: list[str], metrics: list[str]) -> tuple[dict, None]:
        """
        Load the existing scores for a given run, stream and metric.

        Parameters
        ----------
        stream : str
            Stream name.
        regions : list[str]
            List of region names.
        metrics : list[str]
            List of metric names.

        Returns
        -------
        scores: tuple[dict,None]
            Dictionary of local scores keyed by metric/region/stream/run_id.
        """
        available_data = self.check_availability(stream, mode="evaluation")
        channels = available_data.channels
        fsteps = available_data.fsteps
        samples = available_data.samples

        local_scores = {}

        for metric in metrics:
            local_scores[metric] = {}
            for region in regions:
                data = self.get_values(
                    region=region, metric=metric, forecast_steps=fsteps, channels=channels
                )
                if data is None:
                    data = xr.DataArray(
                        np.full(
                            (len(samples), len(fsteps), len(channels), 1),
                            np.nan,
                            dtype=np.float32,
                        ),
                        dims=("sample", "forecast_step", "channel", "metric"),
                        coords={
                            "sample": samples,
                            "lead_time": ("forecast_step", fsteps),
                            "forecast_step": range(len(fsteps)),
                            "channel": channels,
                            "metric": [metric],
                        },
                        attrs={"npoints_per_sample": self.npoints_per_sample},
                    )

                local_scores[metric].setdefault(region, {})[stream] = {
                    self.run_id: data,
                }
        return local_scores, None


def _metric_quaver_convention(metric: str) -> str:
    """
    Convert metric name to Quaver convention if needed.

    Parameters
    ----------
    metric : str
        Original metric name.

    Returns
    -------
    metric: str
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
    region : str
        Original region name.

    Returns
    -------
    region: str
        Region name in Quaver convention.
    """
    region_mapping = {
        "nhem": "n.hem",
        "shem": "s.hem",
        # Add more mappings as needed
    }
    return region_mapping.get(region, region)
