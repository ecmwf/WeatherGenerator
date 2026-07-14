# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Line plot classes for the evaluation plotting subpackage."""

import logging
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr

from weathergen.evaluate.plotting.plot_utils import (
    align_labels,
    channel_sort_key,
    clean_label,
    lower_is_better,
)

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class LinePlots:
    def __init__(self, plotter_cfg: dict, output_basedir: str | Path):
        """
        Initialize the LinePlots class.

        Parameters
        ----------
        plotter_cfg:
            Configuration dictionary containing basic information for plotting.
            Expected keys are:
                - image_format: Format of the saved images (e.g., 'png', 'pdf', etc.)
                - dpi_val: DPI value for the saved images
                - fig_size: Size of the figure (width, height) in inches
                -  plot_ensemble:
                    If True, plot ensemble spread if 'ens' dimension is present. Options are:
                        - False: do not plot ensemble spread
                        - "std": plot mean +/- standard deviation
                        - "minmax": plot min-max range
                        - "members": plot individual ensemble members
        output_basedir:
            Base directory under which the plots will be saved.
            Expected scheme `<results_base_dir>/<run_id>`.
        """

        self.image_format = plotter_cfg.get("image_format")
        self.dpi_val = plotter_cfg.get("dpi_val")
        self.fig_size = plotter_cfg.get("fig_size")
        self.log_scale = plotter_cfg.get("log_scale")
        self.add_grid = plotter_cfg.get("add_grid")
        self.plot_ensemble = plotter_cfg.get("plot_ensemble", False)
        self.baseline = plotter_cfg.get("baseline")
        self.out_plot_dir_lines = Path(output_basedir) / "line_plots"
        self.out_plot_dir_ratio = Path(output_basedir) / "ratio_plots"
        self.out_plot_dir_psd = Path(output_basedir) / "psd_plots"
        if not os.path.exists(self.out_plot_dir_lines):
            _logger.info(f"Creating dir {self.out_plot_dir_lines}")
            os.makedirs(self.out_plot_dir_lines, exist_ok=True)
        if not os.path.exists(self.out_plot_dir_ratio):
            _logger.info(f"Creating dir {self.out_plot_dir_ratio}")
            os.makedirs(self.out_plot_dir_ratio, exist_ok=True)
        if not os.path.exists(self.out_plot_dir_psd):
            _logger.info(f"Creating dir {self.out_plot_dir_psd}")
            os.makedirs(self.out_plot_dir_psd, exist_ok=True)

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
            "Compare::plot - Data should be of type xr.DataArray or list"
        )
        assert isinstance(labels, str | list), (
            "Compare::plot - Labels should be of type str or list"
        )

        data_list = [data] if isinstance(data, xr.DataArray) else data
        label_list = [labels] if isinstance(labels, str) else labels

        assert len(data_list) == len(label_list), "Compare::plot - Data and Labels do not match"

        return data_list, label_list

    def _check_n_fsteps(self, data_list):
        """
        Check the number of forecast steps in each dataset and return the index of the dataset with
        the maximum number of forecast steps.

        Parameters
        ----------
        data_list:
            data_list - list of DataArrays to be plotted

        Returns
        -------
            min_idx - index of the dataset with the minimum number of forecast steps
        """

        min_idx = min(
            range(len(data_list)), key=lambda i: data_list[i].sizes.get("forecast_step", 0)
        )
        return min_idx

    def print_all_points_from_graph(self, fig: plt.Figure) -> None:
        """Log all data points from every line in a matplotlib figure.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure whose axes and lines will be iterated.

        Returns
        -------
        None
        """
        for ax in fig.get_axes():
            for line in ax.get_lines():
                ydata = line.get_ydata()
                xdata = line.get_xdata()
                label = line.get_label()
                _logger.info(f"Summary for {label} plot:")
                for xi, yi in zip(xdata, ydata, strict=False):
                    xi = xi if isinstance(xi, str) else f"{float(xi):.3f}"
                    yi = yi if isinstance(yi, str) else f"{float(yi):.3f}"
                    _logger.info(f"  x: {xi}, y: {yi}")
                _logger.info("--------------------------")
        return

    def _plot_ensemble(
        self, data: xr.DataArray, x_dim: str, label: str, color: str | None = None
    ) -> None:
        """
        Plot ensemble spread for a data array.

        Parameters
        ----------
        data: xr.xArray
            DataArray to be plotted
        x_dim: str
            Dimension to be used for the x-axis.
        label: str
            Label for the dataset
        color: str or None
            Color for the line. If None, matplotlib auto-cycles.
        Returns
        -------
            None
        """
        averaged = data.mean(dim=[dim for dim in data.dims if dim != x_dim], skipna=True).sortby(
            x_dim
        )

        plot_kwargs = dict(
            label=label,
            marker="o",
            markersize=4,
            linewidth=1.2,
            linestyle="-",
        )
        if color is not None:
            plot_kwargs["color"] = color

        lines = plt.plot(
            averaged[x_dim],
            averaged.values,
            **plot_kwargs,
        )
        line = lines[0]
        color = line.get_color()

        ens = data.mean(
            dim=[dim for dim in data.dims if dim not in [x_dim, "ens"]], skipna=True
        ).sortby(x_dim)

        if self.plot_ensemble == "std":
            std_dev = ens.std(dim="ens", skipna=True).sortby(x_dim)
            plt.fill_between(
                averaged[x_dim],
                (averaged - std_dev).values,
                (averaged + std_dev).values,
                label=f"{label} - std dev",
                color=color,
                alpha=0.2,
            )

        elif self.plot_ensemble == "minmax":
            ens_min = ens.min(dim="ens", skipna=True).sortby(x_dim)
            ens_max = ens.max(dim="ens", skipna=True).sortby(x_dim)

            plt.fill_between(
                averaged[x_dim],
                ens_min.values,
                ens_max.values,
                label=f"{label} - min max",
                color=color,
                alpha=0.2,
            )

        elif self.plot_ensemble == "members":
            for j in range(ens.ens.size):
                plt.plot(
                    ens[x_dim],
                    ens.isel(ens=j).values,
                    color=color,
                    alpha=0.2,
                )
        else:
            _logger.warning(
                f"LinePlot:: Unknown option for plot_ensemble: {self.plot_ensemble}. "
                "Skipping ensemble plotting."
            )

    def _preprocess_data(
        self, data: xr.DataArray, x_dim: str | list[str], verbose: bool = True
    ) -> xr.DataArray:
        """
        Average all dimensions except x_dim (which may be a string or list)
        and then sort the result.

        Parameters
        ----------
        data : xr.DataArray
            DataArray to be preprocessed.
        x_dim : str or list of str
            Dimension(s) to be preserved for the x-axis.
        verbose : bool
            Log information about averaging.

        Returns
        -------
        xr.DataArray
            Preprocessed DataArray.
        """

        x_dims = [x_dim] if isinstance(x_dim, str) else list(x_dim)

        non_x_dims = [dim for dim in data.dims if dim not in x_dims]

        if any(data.sizes.get(dim, 1) > 1 for dim in non_x_dims) and verbose:
            logging.info(f"Averaging over dimensions: {non_x_dims}")

        out = data.mean(dim=non_x_dims, skipna=True)

        for xd in x_dims:
            out = out.sortby(xd)

        return out

    def plot(
        self,
        data: xr.DataArray | list,
        labels: str | list,
        tag: str = "",
        x_dim: str = "lead_time",
        y_dim: str = "value",
        print_summary: bool = False,
        title: str | None = None,
        colors: list[str | None] | None = None,
        line: float | None = None,
    ) -> None:
        """
        Plot a line graph comparing multiple datasets.

        Parameters
        ----------
        data:
            DataArray or list of DataArrays to be plotted
        labels:
            Label or list of labels for each dataset
        tag:
            Tag to be added to the plot title and filename
        x_dim:
            Dimension to be used for the x-axis. The code will average over all other dimensions.
        y_dim:
            Name of the dimension to be used for the y-axis.
        print_summary:
            If True, print a summary of the values from the graph.
        line:
            If provided, draw a horizontal reference line at the given y-value
            (e.g. the optimal value of a metric).
        Returns
        -------
            None
        """

        data_list, label_list = self._check_lengths(data, labels)

        assert x_dim in data_list[0].dims or x_dim in data_list[0].coords, (
            f"x dimension '{x_dim}' not found in data dimensions "
            f"{data_list[0].dims} or coords {data_list[0].coords}."
        )

        fig = plt.figure(figsize=(12, 6), dpi=self.dpi_val)
        ax = fig.add_subplot(111)

        for i, data in enumerate(data_list):
            non_zero_dims = [dim for dim in data.dims if dim != x_dim and data[dim].shape[0] > 1]
            color = colors[i] if colors and i < len(colors) else None

            if self.plot_ensemble and "ens" in non_zero_dims:
                _logger.info(f"LinePlot:: Plotting ensemble with option {self.plot_ensemble}.")
                self._plot_ensemble(data, x_dim, label_list[i], color=color)
            else:
                averaged = self._preprocess_data(data, x_dim)

                plot_kwargs = dict(
                    label=label_list[i],
                    marker="o",
                    markersize=4,
                    linewidth=1.2,
                    linestyle="-",
                )
                if color is not None:
                    plot_kwargs["color"] = color

                ax.plot(
                    averaged[x_dim],
                    averaged.values,
                    **plot_kwargs,
                )

        parts = ["compare", tag]
        name = "_".join(filter(None, parts))

        # TODO: generalise this for other x_dims by introducing a "units"
        # entry in the function if needed
        xunits = "hr" if x_dim == "lead_time" else None
        x_dim_opts = (x_dim, None, xunits) if xunits else x_dim
        self._plot_base(
            fig,
            name,
            x_dim_opts,
            y_dim,
            print_summary,
            line=line,
            title=title,
            out_plot_dir=self.out_plot_dir_lines,
        )

    def _plot_base(
        self,
        fig: plt.Figure,
        name: str,
        x_dim: tuple[str, ...],
        y_dim: tuple[str, ...],
        print_summary: bool = False,
        line: float | None = None,
        vlines: bool = False,
        title: str | None = None,
        out_plot_dir: Path = None,
        range: tuple[float, float] | None = None,
    ) -> None:
        """
        Apply labels, title, legend, save and optionally print summary.
        Parameters
        ----------
        fig:
            Matplotlib figure to be finalized
        name:
            Name of the plot file
        x_dim:
            Label for the x-axis, can be a tuple if the x-axis has units and/or
            a description is needed (e.g. ("improvement"))
            syntax: (label, units, description)
        y_dim:
            Label for the y-axis, can be a tuple if the y-axis has units and/or
            a description is needed (e.g. ("improvement"))
            syntax: (label, units, description)
        print_summary:
            If True, print a summary of the values from the graph.
        line:
            If provided, draw a horizontal line at the given y-value.
        vlines:
            If True, draw vertical lines to separate each group of variables.
        title:
            Title for the plot.
        out_plot_dir:
            Directory where the plot will be saved.
        range:
            Tuple specifying the y-axis range (min, max).
        Returns
        -------
            None
        """
        x_dim, x_dim_descr, xunits = x_dim if isinstance(x_dim, tuple) else (x_dim, None, None)
        y_dim, y_dim_descr, yunits = y_dim if isinstance(y_dim, tuple) else (y_dim, None, None)

        xlabel = (
            clean_label(x_dim)
            + (f" [{xunits}]" if xunits else "")
            + (f" ({x_dim_descr})" if x_dim_descr else "")
        )
        ylabel = (
            clean_label(y_dim).upper()
            + (f" [{yunits}]" if yunits else "")
            + (f" ({y_dim_descr})" if y_dim_descr else "")
        )

        ax = fig.gca()

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)

        clean_title = title if title is not None else clean_label(name)
        ax.set_title(
            clean_title,
            fontsize=11,
            fontweight="medium",
        )
        ax.legend(frameon=False, fancybox=False, edgecolor="0.6", fontsize=8)
        ax.tick_params(axis="both", labelsize=9, direction="in", top=True, right=True)

        # Thin spines
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

        if self.add_grid:
            ax.grid(True, linestyle="--", color="gray", alpha=0.3, linewidth=0.5)

        if self.log_scale:
            ax.set_yscale("log")

        if print_summary:
            _logger.info(f"Summary values for {name}")
            self.print_all_points_from_graph(fig)

        if line:
            ax.axhline(y=line, color="black", linestyle="--", linewidth=0.8, zorder=1)

        if vlines:
            vlines = []
            last_prefix = None

            channels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]

            for idx, ch in enumerate(channels):
                m = re.match(r"([a-zA-Z]+)_\d+", ch)
                prefix = m.group(1) if m else ch
                if last_prefix is not None and prefix != last_prefix:
                    vlines.append(idx - 0.5)
                last_prefix = prefix
            for vl in vlines:
                ax.axvline(x=vl, color="#001f3f", linestyle="-", linewidth=0.5, zorder=1)

        plt.tight_layout()
        if out_plot_dir is None:
            raise ValueError("Output plot directory not provided.")
        plt.savefig(f"{out_plot_dir.joinpath(name)}.{self.image_format}")
        plt.close()

    def ratio_plot(
        self,
        data: xr.DataArray | list,
        run_ids: list[str],
        labels: str | list,
        y_dim: str = "value",
        tag: str = "",
        print_summary: bool = False,
        colors: list[str | None] | None = None,
    ) -> None:
        """Plot a ratio plot comparing multiple datasets to a baseline.

        Each non-baseline dataset is divided element-wise by the baseline,
        and the resulting ratio is plotted per channel.

        Parameters
        ----------
        data : xr.DataArray or list
            DataArray or list of DataArrays to be compared.
        run_ids : list of str
            Run identifiers corresponding to each element in *data*.
        labels : str or list
            Label or list of labels for the legend.
        tag : str
            Tag appended to the plot title and filename.
        y_dim : str
            Dimension used for the y-axis label (default ``'value'``).
        print_summary : bool
            If ``True``, print data-point values to the log.
        colors : list of str or None, optional
            Per-run colour overrides. Entries that are ``None`` fall back to
            matplotlib's default colour cycle.

        Returns
        -------
        None
        """

        data_list, label_list = self._check_lengths(data, labels)
        min_index = self._check_n_fsteps(data_list)

        if len(data_list) < 2:
            baseline = xr.full_like(data_list[min_index], 1.0)
            baseline_name = "ones"
            descr = "scores"
        else:
            descr = "ratio_plot"
            baseline_name = self.baseline
            baseline_idx = run_ids.index(self.baseline) if self.baseline in run_ids else None
            if baseline_idx is not None:
                _logger.info(f"Using baseline run ID '{self.baseline}' for ratio plot.")
                baseline = data_list[baseline_idx]
            else:
                baseline_name = run_ids[0]
                baseline = data_list[0]

        ref_raw = self._preprocess_data(baseline, ["forecast_step", "channel"], verbose=False)

        channel_names = set(ref_raw.channel.values)
        for data in data_list[1:]:
            channel_names.update(data.channel.values)

        ref_channel_names = sorted(channel_names, key=channel_sort_key)

        ref = align_labels(ref_raw, ref_channel_names, "channel").reindex(channel=ref_channel_names)

        # Build a run_id to color map, skipping the baseline
        color_map = {}
        if colors:
            for rid, c in zip(run_ids, colors, strict=False):
                if c is not None:
                    color_map[rid] = c

        # Pre-compute all ratios and global y-axis range
        all_fsteps = sorted(data_list[min_index]["forecast_step"].values)
        # ratios_by_fstep[f_step] = [(run_id, label, ratio_values), ...]
        ratios_by_fstep: dict[int, list[tuple[str, str, object]]] = {f: [] for f in all_fsteps}
        global_ymin = float("inf")
        global_ymax = float("-inf")

        for data, run_id, lbl in zip(data_list, run_ids, label_list, strict=False):
            if run_id == baseline_name:
                continue
            num_raw = self._preprocess_data(data, ["forecast_step", "channel"], verbose=False)
            num = align_labels(num_raw, ref_channel_names, "channel").reindex(
                channel=ref_channel_names
            )
            for f_step in all_fsteps:
                ratio = num.sel(channel=ref_channel_names, forecast_step=f_step) / ref.sel(
                    channel=ref_channel_names, forecast_step=f_step
                )
                ratios_by_fstep[f_step].append((run_id, lbl, ratio.values))
                finite = ratio.values[~xr.DataArray(ratio).isnull().values]
                if len(finite) > 0:
                    global_ymin = min(global_ymin, float(finite.min()))
                    global_ymax = max(global_ymax, float(finite.max()))

        # Add a small margin (5%) around the range
        if global_ymin != float("inf"):
            margin = 0.05 * (global_ymax - global_ymin) if global_ymax > global_ymin else 0.1
            global_ymin -= margin
            global_ymax += margin

        # Plot each forecast step
        for f_step in all_fsteps:
            fig = plt.figure(figsize=(max(12, len(ref_channel_names) * 0.25), 6))
            for run_id, lbl, ratio_vals in ratios_by_fstep[f_step]:
                plot_kwargs = dict(label=lbl, marker="o", linestyle="-")
                if run_id in color_map:
                    plot_kwargs["color"] = color_map[run_id]
                plt.plot(ref_channel_names, ratio_vals, **plot_kwargs)

            if global_ymin != float("inf"):
                plt.ylim(global_ymin, global_ymax)

            parts = [descr, f"fstep_0{f_step}", tag]
            name = "_".join(filter(None, parts))
            plt.xticks(rotation=90, ha="right")
            plt.grid(True, linestyle="--", color="gray", alpha=0.2)
            title = (
                f"{descr.replace('_', ' ')} {tag.split('_')[0]} | fstep {f_step} | "
                f" {tag.split('_')[-1]} (baseline: {baseline_name})"
            )

            y_dim_opts = (y_dim, "improvement", None)
            self._plot_base(
                fig,
                name,
                "channel",
                y_dim_opts,
                print_summary,
                line=1.0,
                vlines=True,
                title=title,
                out_plot_dir=self.out_plot_dir_ratio,
            )

    def heat_map(
        self,
        data: xr.DataArray | list,
        labels: str | list,
        metric: str,
        x_dim,
        tag: str = "",
    ) -> None:
        """Plot a heat map comparing multiple datasets across forecast steps.

        For each run a heat map is drawn showing the ratio of each
        forecast-step score to the first forecast-step score, per channel.

        Parameters
        ----------
        data : xr.DataArray or list
            DataArray or list of DataArrays to be plotted.
        labels : str or list
            Label or list of labels for each dataset.
        metric : str
            Metric name used to select the colourmap direction.
        x_dim : str
            Dimension used for the x-axis (e.g. ``'forecast_step'``).
        tag : str
            Tag appended to the filename.

        Returns
        -------
        None
        """

        data_list, label_list = self._check_lengths(data, labels)

        n_runs = len(data_list)

        x_ticks_names = set()

        for data in data_list:
            da = data.isel({x_dim: 0})
            x_ticks_names.update(map(str, da.channel.values))

        ref_ticks_names = sorted(x_ticks_names, key=channel_sort_key)

        fig, axes = plt.subplots(
            1, n_runs, figsize=(8 * n_runs, max(12, len(ref_ticks_names) * 0.25)), squeeze=False
        )

        global_min = float("inf")
        global_max = float("-inf")

        for ax, data, label in zip(axes[0], data_list, labels, strict=False):
            time_steps = sorted(data[x_dim].values)

            ref = data.reindex(channel=ref_ticks_names).sel({x_dim: time_steps[0]})
            ref = self._preprocess_data(ref, "channel", verbose=False)

            if ref.isnull().all():
                _logger.warning(
                    f"Heatmap:: Reference data for metric {metric} and label {label} contains "
                    "only NaNs. Skipping heatmap."
                )
                continue

            num = self._preprocess_data(data, [x_dim, "channel"], verbose=False)
            num = num.reindex(channel=ref_ticks_names).sel({x_dim: time_steps})

            heatmap_data = num / ref

            cmap = plt.get_cmap("magma_r") if lower_is_better(metric) else plt.get_cmap("magma")
            global_min = min(global_min, float(heatmap_data.min()))
            global_max = max(global_max, float(heatmap_data.max()))

            last_hm = sns.heatmap(
                heatmap_data.values.T,
                ax=ax,
                cmap=cmap,
                vmin=global_min,
                vmax=global_max,
                xticklabels=time_steps,
                yticklabels=ref_ticks_names,
                annot=False,
                fmt=".2f",
                cbar=False,
            )
            ax.set_title(f"Heatmap {metric} – {label}")
            ax.set_xlabel(f"{x_dim.replace('_', ' ').title()} (h)")
            ax.set_ylabel("Variable")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        cbar = fig.colorbar(
            last_hm.collections[0],
            ax=axes.ravel().tolist(),
            shrink=0.6,
            location="right",
            pad=0.02,
        )
        cbar.set_label(rf"{metric} - $t_{{\mathrm{{step}}}}[0] / t_{{\mathrm{{step}}}}[x]$")
        parts = ["heat_map", metric, tag]
        name = "_".join(filter(None, parts))
        plt.savefig(f"{self.out_plot_dir.joinpath(name)}.{self.image_format}")

    # ------------------------------------------------------------------
    # PSD summary plot
    # ------------------------------------------------------------------

    def psd_plot(
        self,
        psd_datasets: list[dict],
        labels: list[str],
        tag: str = "",
        variable: str = "",
        forecast_step: str = "",
    ) -> None:
        """Create a PSD summary plot overlaying multiple runs.

        Each entry in *psd_datasets* is a dict with keys
        ``frequencies``, ``psd_target``, ``psd_prediction``, ``psd_method``.

        Parameters
        ----------
        psd_datasets : list[dict]
            One dict per run, each containing the PSD arrays stored by
            ``Scores.calc_psd`` in ``.attrs``.
        labels : list[str]
            Human-readable label for each run.
        tag : str
            Filename tag.
        """
        out_dir = Path(self.out_plot_dir_psd)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Use the target from the first run as reference
        freq = np.asarray(psd_datasets[0]["frequencies"])
        tar_psd = np.asarray(psd_datasets[0]["psd_target"])

        fig, (ax_spec, ax_ratio) = plt.subplots(
            2,
            1,
            figsize=self.fig_size or (10, 8),
            gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08},
        )

        # Upper panel: log-log spectra
        ax_spec.loglog(freq, tar_psd, color="black", lw=1.5, label="Target")
        colors = plt.cm.tab10.colors
        for i, (ds, label) in enumerate(zip(psd_datasets, labels, strict=False)):
            c = colors[i % len(colors)]
            ax_spec.loglog(
                np.asarray(ds["frequencies"]),
                np.asarray(ds["psd_prediction"]),
                color=c,
                lw=1.5,
                label=label,
            )
        ax_spec.set_ylabel("Power")
        psd_method = psd_datasets[0].get("psd_method", "sht")
        title_parts = [f"PSD ({psd_method})"]
        if variable:
            title_parts.append(variable)
        if forecast_step:
            title_parts.append(f"step {forecast_step}")
        ax_spec.set_title(" – ".join(title_parts))
        ax_spec.legend(frameon=False, fontsize=7)
        ax_spec.grid(True, which="both", ls="--", alpha=0.4)

        # Lower panel: ratio (pred / target)
        for i, (ds, label) in enumerate(zip(psd_datasets, labels, strict=False)):
            c = colors[i % len(colors)]
            pred = np.asarray(ds["psd_prediction"])
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(tar_psd > 0, pred / tar_psd, np.nan)
            ax_ratio.semilogx(freq, ratio, color=c, lw=1.2, label=label)
        ax_ratio.axhline(1.0, ls="--", color="gray", lw=0.8)
        ax_ratio.set_ylabel("Pred / Target")
        ax_ratio.set_xlabel("Frequency (1/deg)")
        ax_ratio.set_ylim(0, 2)
        ax_ratio.grid(True, which="both", ls="--", alpha=0.4)

        name = tag or "psd"
        fname = out_dir / f"{name}.{self.image_format}"
        _logger.debug(f"Saving PSD summary plot to {fname}")
        fig.savefig(str(fname), bbox_inches="tight", dpi=self.dpi_val)
        plt.close(fig)
