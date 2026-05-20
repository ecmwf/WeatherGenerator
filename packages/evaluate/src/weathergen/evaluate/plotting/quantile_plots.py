"""Quantile plot classes for the evaluation plotting subpackage."""

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class QuantilePlots:
    def __init__(self, plotter_cfg: dict, output_basedir: str | Path):
        """
        Initialize the QuantilePlots class.

        Parameters
        ----------
        plotter_cfg:
            Configuration dictionary containing basic information for plotting.
            Expected keys are:
                - image_format: Format of the saved images (e.g., 'png', 'pdf', etc.)
                - dpi_val: DPI value for the saved images
                - fig_size: Size of the figure (width, height) in inches
        output_basedir:
            Base directory under which the plots will be saved.
            Expected scheme `<results_base_dir>/<run_id>`.
        """
        self.image_format = plotter_cfg.get("image_format")
        self.dpi_val = plotter_cfg.get("dpi_val")
        self.fig_size = plotter_cfg.get("fig_size")
        self.out_plot_dir = Path(output_basedir) / "quantile_plots"

        if not os.path.exists(self.out_plot_dir):
            _logger.info(f"Creating dir {self.out_plot_dir}")
            os.makedirs(self.out_plot_dir, exist_ok=True)

    def _check_lengths(self, data: xr.DataArray | list, labels: str | list) -> tuple[list, list]:
        """
        Check if the lengths of data and labels match.

        Parameters
        ----------
        data:
            DataArray or list of DataArrays to be plotted
        labels:
            Label or list of labels for each dataset

        Returns
        -------
            data_list, label_list - lists of data and labels
        """
        assert isinstance(data, xr.DataArray | list), (
            "QuantilePlots::_check_lengths - Data should be of type xr.DataArray or list"
        )
        assert isinstance(labels, str | list), (
            "QuantilePlots::_check_lengths - Labels should be of type str or list"
        )

        data_list = [data] if isinstance(data, xr.DataArray) else data
        label_list = [labels] if isinstance(labels, str) else labels

        assert len(data_list) == len(label_list), (
            "QuantilePlots::_check_lengths - Data and Labels do not match"
        )

        return data_list, label_list

    def qq_plot(
        self,
        qq_data: list[xr.Dataset],
        labels: str | list,
        tag: str = "",
        metric: str = "qq_analysis",
        extreme_percentiles: tuple[float, float] | None = None,
    ) -> None:
        """
        Create quantile-quantile (Q-Q) plots for extreme value analysis.

        This method generates comprehensive Q-Q plots comparing forecast quantiles
        against ground truth quantiles, with emphasis on extreme values.

        Parameters
        ----------
        qq_data:
            Dataset or list of Datasets containing Q-Q analysis results.
            Each dataset should contain:
            - 'quantile_levels': Theoretical quantile levels (0 to 1)
            - 'p_quantiles': Quantile values from prediction data
            - 'gt_quantiles': Quantile values from ground truth data
            - 'qq_deviation': Absolute difference between quantiles
            - 'extreme_low_mse': MSE for lower extreme quantiles
            - 'extreme_high_mse': MSE for upper extreme quantiles
        labels:
            Label or list of labels for each dataset
        tag:
            Tag to be added to the plot title and filename
        metric:
            Name of the metric (default: 'qq_analysis')
        extreme_percentiles:
            Lower and upper percentile thresholds for extreme regions.

        Returns
        -------
            None
        """
        data_list, label_list = self._check_lengths(qq_data, labels)

        # Use extreme_percentiles from data if not explicitly provided
        if extreme_percentiles is None:
            extreme_percentiles = tuple(data_list[0].attrs.get("extreme_percentiles", (5.0, 95.0)))

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 6), dpi=self.dpi_val)
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.3)

        ax_qq = fig.add_subplot(gs[0])  # Main Q-Q plot
        ax_dev = fig.add_subplot(gs[1])  # Deviation plot

        colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))

        for _i, (ds, label, color) in enumerate(zip(data_list, label_list, colors, strict=False)):
            # Extract quantile data
            quantile_levels = ds["quantile_levels"].values
            p_quantiles = ds["p_quantiles"].values
            gt_quantiles = ds["gt_quantiles"].values
            qq_deviation = ds["qq_deviation"].values

            # Main Q-Q plot
            ax_qq.scatter(
                gt_quantiles,
                p_quantiles,
                alpha=0.6,
                s=20,
                c=[color],
                label=label,
                edgecolors="none",
            )

            # Deviation plot
            ax_dev.plot(
                quantile_levels,
                qq_deviation,
                label=label,
                color=color,
                linewidth=2,
                alpha=0.8,
            )

        # Format main Q-Q plot
        ax_qq.set_xlabel("Ground Truth Quantiles", fontsize=12)
        ax_qq.set_ylabel("Prediction Quantiles", fontsize=12)
        ax_qq.set_title("Quantile-Quantile Plot for Extreme Value Analysis", fontsize=14)

        # Add perfect agreement line (y=x)
        min_val = min([ds["gt_quantiles"].min().values for ds in data_list])
        max_val = max([ds["gt_quantiles"].max().values for ds in data_list])
        ax_qq.plot(
            [min_val, max_val],
            [min_val, max_val],
            "k--",
            linewidth=2,
            label="Perfect Agreement",
            alpha=0.7,
        )

        # Add shaded regions for extremes
        if len(data_list) > 0:
            ds_ref = data_list[0]
            quantile_levels = ds_ref["quantile_levels"].values

            # Find extreme regions using configurable thresholds
            lower_extreme_idx = quantile_levels < (extreme_percentiles[0] / 100)
            upper_extreme_idx = quantile_levels > (extreme_percentiles[1] / 100)

            if np.any(lower_extreme_idx):
                lower_q = ds_ref["gt_quantiles"].values[lower_extreme_idx]
                ax_qq.axvspan(
                    min_val,
                    lower_q.max() if len(lower_q) > 0 else min_val,
                    alpha=0.1,
                    color="blue",
                    label="Lower Extreme Zone",
                )

            if np.any(upper_extreme_idx):
                upper_q = ds_ref["gt_quantiles"].values[upper_extreme_idx]
                ax_qq.axvspan(
                    upper_q.min() if len(upper_q) > 0 else max_val,
                    max_val,
                    alpha=0.1,
                    color="red",
                    label="Upper Extreme Zone",
                )

        ax_qq.legend(frameon=False, loc="upper left", fontsize=10)
        ax_qq.grid(True, linestyle="--", alpha=0.3)

        # Format deviation plot
        ax_dev.set_xlabel("Quantile Level", fontsize=12)
        ax_dev.set_ylabel("Absolute Deviation", fontsize=12)
        ax_dev.set_title("Quantile Deviation", fontsize=14)
        ax_dev.legend(frameon=False, fontsize=10)
        ax_dev.grid(True, linestyle="--", alpha=0.3)

        # Highlight extreme regions in deviation plot
        lower_threshold = extreme_percentiles[0] / 100
        upper_threshold = extreme_percentiles[1] / 100
        ax_dev.axvspan(0.0, lower_threshold, alpha=0.1, color="blue")
        ax_dev.axvspan(upper_threshold, 1.0, alpha=0.1, color="red")

        plt.tight_layout()

        # Save the plot
        parts = ["qq_analysis", tag]
        name = "_".join(filter(None, parts))
        save_path = self.out_plot_dir.joinpath(f"{name}.{self.image_format}")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        _logger.info(f"Q-Q plot saved to {save_path}")
