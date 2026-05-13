"""Bar plot classes for the evaluation plotting subpackage."""

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from weathergen.evaluate.plotting.plot_utils import (
    calculate_average_over_dim,
    lower_is_better,
)

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class BarPlots:
    """
    Initialize the BarPlots class.

    Parameters
    ----------
    plotter_cfg:
        Configuration dictionary containing basic information for plotting.
        Expected keys are:
            - image_format: Format of the saved images (e.g., 'png', 'pdf', etc.)
            - improvement: Size of the figure (width, height) in inches
    output_basedir:
        Base directory under which the score cards will be saved.
    """

    def __init__(self, plotter_cfg: dict, output_basedir: str | Path) -> None:
        self.image_format = plotter_cfg.get("image_format")
        self.dpi_val = plotter_cfg.get("dpi_val")
        self.cmap = plotter_cfg.get("cmap", "bwr")
        self.out_plot_dir = Path(output_basedir) / "bar_plots"
        self.baseline = plotter_cfg.get("baseline")
        if not os.path.exists(self.out_plot_dir):
            _logger.info(f"Creating dir {self.out_plot_dir}")
            os.makedirs(self.out_plot_dir, exist_ok=True)

    def plot(
        self,
        data: list[xr.DataArray],
        runs: list[str],
        metric: str,
        channels: list[str],
        tag: str,
    ) -> None:
        """
        Plot (ratio) bar plots comparing performance between different run_ids over channels of
        interest.

        Parameters
        ----------
        data:
            List of (xarray) DataArrays with the scores (stream, region and metric specific)
        runs:
            List containing runs (in str format) to be compared (provided in the config)
        metric:
            Metric name
        channels:
            List containing channels (in str format) of interest (provided in the config)
        tag:
            Tag to be added to the plot title and filename
        """

        fig, ax = plt.subplots(
            1,
            len(runs) - 1 if len(runs) > 1 else 1,
            figsize=(5 * len(runs), 2 * len(channels)),
            dpi=self.dpi_val,
            squeeze=False,
        )
        ax = ax.flatten()
        if self.baseline and self.baseline in runs:
            baseline_idx = runs.index(self.baseline)
            runs = [runs[baseline_idx]] + runs[:baseline_idx] + runs[baseline_idx + 1 :]
            data = [data[baseline_idx]] + data[:baseline_idx] + data[baseline_idx + 1 :]
        elif len(runs) < 2:
            _logger.warning(
                "BarPlots:: Less than two runs provided. Generating bar plot against ones."
            )
            ones_array = xr.full_like(data[0], 1.0)
            runs = [""] + runs
            data = [ones_array] + data

        for run_index in range(1, len(runs)):
            score, channels_per_comparison = self.calc_ratio_per_run_id(data, channels, run_index)
            if len(score) > 0:
                ax[run_index - 1].barh(
                    np.arange(len(score)),
                    score,
                    color=self.colors(score, metric),
                    align="center",
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax[run_index - 1].set_yticks(np.arange(len(score)), labels=channels_per_comparison)
                ax[run_index - 1].invert_yaxis()

                xlabel = (
                    f"Relative {data[0].coords['metric'].item().upper()}: "
                    f"Target Model ({runs[run_index]}) / Reference Model ({runs[0]})"
                )

                if len(runs) == 2 and runs[0] == "":
                    xlabel = xlabel.replace("Relative ", "")
                    xlabel = xlabel.replace(
                        f"Target Model ({runs[run_index]}) / Reference Model ({runs[0]})",
                        f"Model ({runs[run_index]})",
                    )

                ax[run_index - 1].set_xlabel(xlabel)
            else:
                ax[run_index - 1].set_visible(False)  # or annotate as missing
                # Or show a message:
                ax[run_index - 1].text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    transform=ax[run_index - 1].transAxes,
                )

        _logger.info(f"Saving bar plots to: {self.out_plot_dir}")
        parts = ["bar_plot", tag] + runs
        name = "_".join(filter(None, parts))
        plt.savefig(
            f"{self.out_plot_dir.joinpath(name)}.{self.image_format}",
            bbox_inches="tight",
            dpi=self.dpi_val,
        )
        plt.close(fig)

    def calc_ratio_per_run_id(
        self,
        data: list[xr.DataArray],
        channels: list[str],
        run_index: int,
        x_dim="channel",
    ) -> tuple[np.array, str]:
        """
        This function calculates the ratio per comparison model for each channel.

        Parameters
        ----------
        data: list[xr.DataArray]
            List of all scores for each model in xarrays format.
        channels: list[str]
            All the available channels.
        run_index: int
            The order index over the run_ids.
        xdim: str
            The dimension for which an average will not be calculated.

        Returns
        ----------
        ratio_score: np.array
            The (ratio) skill over each channel for a specific model
        channels_per_comparison: str
            The common channels over which the baseline and the other model will be compared.

        """
        ratio_score = []
        channels_per_comparison = []

        for _, var in enumerate(channels):
            if var not in data[0].channel.values or var not in data[run_index].channel.values:
                continue
            baseline_var = data[0].sel({"channel": var})
            data_var = data[run_index].sel({"channel": var})
            channels_per_comparison.append(var)

            baseline_score, model_score = calculate_average_over_dim(x_dim, baseline_var, data_var)

            ratio_score.append(model_score / baseline_score)

        if np.allclose(baseline_score, 1.0, atol=1e-6):
            ratio_score = np.array(ratio_score)
        else:
            ratio_score = np.array(ratio_score) - 1

        return ratio_score, channels_per_comparison

    def colors(self, ratio_score: np.array, metric: str) -> list[tuple]:
        """
        This function calculates colormaps based on the skill scores. From negative value blue
        color variations should be given otherwise red color variations should be given.

        Parameters
        ----------
        ratio_score: np.array
            The (ratio) skill for a specific model
        metric: str
            The metric of interest
        Returns
        ----------
        colors: list[tuple]
            The color magnitude (blue to red) of the bars in the plots
        """
        max_val = np.abs(ratio_score).max()
        if lower_is_better(metric):
            cmap = plt.get_cmap("bwr")
        else:
            cmap = plt.get_cmap("bwr_r")
        colors = [cmap(0.5 + v / (2 * max_val)) for v in ratio_score]
        return colors
