"""Score card classes for the evaluation plotting subpackage."""

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon

from weathergen.evaluate.plotting.plot_utils import (
    calculate_average_over_dim,
    lower_is_better,
)

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class ScoreCards:
    """
    Initialize the ScoreCards class.

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
        self.improvement = plotter_cfg.get("improvement_scale", 0.2)
        self.out_plot_dir = Path(output_basedir) / "score_cards"
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
        Plot score cards comparing performance between run_ids against a baseline over channels
        of interest.

        Parameters
        ----------
        data:
            List of (xarray) DataArrays with the scores (stream, region and metric specific)
        runs:
            List containing runs (in str format) to be compared (provided in the config)
        metric:
            Metric for which we are plotting
        channels:
            List containing channels (in str format) of interest (provided in the config)
        tag:
            Tag to be added to the plot title and filename
        """
        n_runs = len(runs)

        if self.baseline and self.baseline in runs:
            baseline_idx = runs.index(self.baseline)
            runs = [runs[baseline_idx]] + runs[:baseline_idx] + runs[baseline_idx + 1 :]
            data = [data[baseline_idx]] + data[:baseline_idx] + data[baseline_idx + 1 :]

        common_channels, n_common_channels = self.extract_common_channels(data, channels, n_runs)

        fig, ax = plt.subplots(figsize=(2 * n_runs, 1.2 * n_common_channels))

        baseline = data[0]
        skill_models = []
        for run_index in range(1, n_runs):
            skill_model = 0.0
            data0_channels = [str(x) for x in np.atleast_1d(data[0].channel.values)]
            data_idx_channels = [str(x) for x in np.atleast_1d(data[run_index].channel.values)]
            for var_index, var in enumerate(common_channels):
                if var not in data0_channels or var not in data_idx_channels:
                    continue
                diff, avg_diff, avg_skill = self.compare_models(
                    data, baseline, run_index, var, metric
                )
                skill_model += avg_skill.values

                # Get symbols based on difference and performance as well as coordinates
                # for the position of the triangles.

                x, y, alt, color, triangle, size = self.get_plot_symbols(
                    run_index, var_index, avg_skill, avg_diff, metric
                )

                ax.scatter(x, y, marker=triangle, color=color, s=size.values, zorder=3)

                # Perform Wilcoxon test
                if len(diff["forecast_step"].values) > 1:
                    stat, p = wilcoxon(diff, alternative=alt)

                    # Draw rectangle border for significance
                    if p < 0.05:
                        lw = 2 if p < 0.01 else 1
                        rect_color = color
                        rect = plt.Rectangle(
                            (x - 0.25, y - 0.25),
                            0.5,
                            0.5,
                            fill=False,
                            edgecolor=rect_color,
                            linewidth=lw,
                            zorder=2,
                        )
                        ax.add_patch(rect)

            skill_models.append(skill_model / n_common_channels)

        # Set axis labels
        ylabels = [
            f"{var}\n({baseline.coords['metric'].item().upper()}={baseline.sel(channel=var).mean().values.squeeze():.3f})"
            for var in common_channels
        ]
        xlabels = [
            f"{model_name}\nSkill: {skill_models[i]:.3f}" for i, model_name in enumerate(runs[1::])
        ]
        ax.set_xticks(np.arange(1, n_runs))
        ax.set_xticklabels(xlabels, fontsize=10)
        ax.set_yticks(np.arange(n_common_channels) + 0.5)
        ax.set_yticklabels(ylabels, fontsize=10)
        for label in ax.get_yticklabels():
            label.set_horizontalalignment("center")
            label.set_x(-0.17)
        ax.set_ylabel("Variable", fontsize=14)
        ax.set_title(
            f"Model Scorecard vs. Baseline '{runs[0]}'",
            fontsize=16,
            pad=20,
        )
        for x in np.arange(0.5, n_runs - 1, 1):
            ax.axvline(x, color="gray", linestyle="--", linewidth=0.5, zorder=0, alpha=0.5)
        ax.set_xlim(0.5, n_runs - 0.5)
        ax.set_ylim(0, n_common_channels)

        legend = [
            Line2D(
                [0],
                [0],
                marker="^",
                color="white",
                label=f"{self.improvement * 100:.0f}% improvement",
                markerfacecolor="blue",
                markersize=np.sqrt(200),
            )
        ]
        plt.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.02, 1.0))

        _logger.info(f"Saving scorecards to: {self.out_plot_dir}")

        parts = ["score_card", tag] + runs
        name = "_".join(filter(None, parts))
        plt.savefig(
            f"{self.out_plot_dir.joinpath(name)}.{self.image_format}",
            bbox_inches="tight",
            dpi=self.dpi_val,
        )
        plt.close(fig)

    def extract_common_channels(self, data, channels, n_runs):
        common_channels = []
        for run_index in range(1, n_runs):
            data0_channels = [str(x) for x in np.atleast_1d(data[0].channel.values)]
            data_idx_channels = [str(x) for x in np.atleast_1d(data[run_index].channel.values)]
            for var in channels:
                if var not in data0_channels or var not in data_idx_channels:
                    continue
                common_channels.append(var)
        common_channels = list(set(common_channels))
        n_vars = len(common_channels)
        return common_channels, n_vars

    def compare_models(
        self,
        data: list[xr.DataArray],
        baseline: xr.DataArray,
        run_index: int,
        var: str,
        metric: str,
        x_dim="forecast_step",
    ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """
        Compare a model with a baseline model and calculate skill scores.

        Parameters
        ----------
        data: list[xr.DataArray]
            List of all scores in xarray format for each model.

        baseline: xarray DataArray
            The baseline scores in xarrays format.

        run_index: int
            The order index over the run_ids.

        var: str
            The specified channel over which we compare.

        xdim: str
            The dimension for which an average will not be calculated.

        Returns
        ----------
        diff: xr.DataArray
            Difference in scores between baseline and model.

        diff.mean(dim="forecast_step"): xr.DataArray
            Average difference in scores over all forecast steps between baseline and model .

        skill.mean(dim="forecast_step"): xr.DataArray
            Average skill scores over all forecast steps between baseline and model .

        """
        baseline_var = baseline.sel({"channel": var})
        data_var = data[run_index].sel({"channel": var})

        baseline_score, model_score = calculate_average_over_dim(x_dim, baseline_var, data_var)
        diff = baseline_score - model_score

        skill = self.get_skill_score(model_score, baseline_score, metric)
        return diff, diff.mean(dim=x_dim), skill.mean(dim=x_dim)

    def get_skill_score(
        self, score_model: xr.DataArray, score_ref: xr.DataArray, metric: str
    ) -> xr.DataArray:
        """
        Calculate skill score comparing a model against a baseline.

        Skill score is defined as: (model_score - baseline_score) / (perfect_score - baseline_score)

        Parameters
        ----------
        score_model : xr.DataArray
            The scores of the model being evaluated
        score_ref : xr.DataArray
            The scores of the reference/baseline model
        metric : str
            The metric name for which to calculate skill score

        Returns
        -------
        xr.DataArray
            Skill scores comparing model to baseline
        """
        perf_score = self.get_perf_score(metric)
        skill_score = (score_model - score_ref) / (perf_score - score_ref)
        return skill_score

    def get_perf_score(self, metric: str) -> float:
        """
        Get the perfect score for a given metric.

        Perfect scores represent ideal performance:
        - Error metrics: 0 (lower is better)
        - Skill/score metrics: 1 (higher is better)
        - PSNR: 100 (higher is better)

        Parameters
        ----------
        metric : str
            Metric name

        Returns
        -------
        float
            Perfect score for the specified metric
        """
        # Metrics where lower values indicate better performance (error metrics)
        if lower_is_better(metric):
            return 0.0

        # Metrics where higher values indicate better performance (with specific perfect score)
        elif metric in ["psnr"]:
            return 100.0

        # Metrics where higher values indicate better performance (default perfect score)
        else:
            return 1.0

    def get_plot_symbols(
        self,
        run_index: int,
        var_index: int,
        avg_skill: xr.DataArray,
        avg_diff: xr.DataArray,
        metric: str,
    ) -> tuple[int, float, str, str, str, xr.DataArray]:
        """
        Determine plot symbol properties based on performance difference.

        Parameters
        ----------
        run_index : int
            Index of the model.
        var_index : int
            Index of the variable/channel.
        avg_skill : xr.DataArray
            Average skill score of the model.
        avg_diff : xr.DataArray
            Average difference between baseline and model.
        metric : str
            Metric used for interpretation.

        Returns
        -------
        Tuple[int, float, str, str, str, xr.DataArray]
            x, y coordinates, alternative hypothesis, color, triangle symbol, size.
        """
        # Conservative choice
        alt = "two-sided"
        modus = "different"
        color = "gray"

        # Determine if diff_mean indicates improvement
        is_improvement = (avg_diff > 0 and lower_is_better(metric)) or (
            avg_diff < 0 and not lower_is_better(metric)
        )

        if is_improvement:
            alt = "greater"
            modus = "better"
            color = "blue"
        elif not is_improvement and avg_diff != 0:
            alt = "less"
            modus = "worse"
            color = "red"
        else:
            alt = "two-sided"
            modus = "different"

        triangle = "^" if modus == "better" else "v"

        # Triangle coordinates
        x = run_index
        # First row is model 1 vs model 0
        y = var_index + 0.5

        size = 200 * (1 - (1 / (1 + abs(avg_skill) / self.improvement)))  # Add base size to all

        return x, y, alt, color, triangle, size
