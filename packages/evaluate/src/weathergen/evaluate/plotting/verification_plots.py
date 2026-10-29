# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Plots of the scores comparing all runs: lead time, ratio, heat map, score card, bar, Q-Q, PSD,
by init hour.

They are drawn from the scores (``MetricResult``) of all runs, once all runs are evaluated.
The drawing itself is done by ``LinePlots``, ``BarPlots``, ``ScoreCards`` and
``QuantilePlots``.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from weathergen.evaluate.plotting.bar_plots import BarPlots
from weathergen.evaluate.plotting.line_plots import LinePlots
from weathergen.evaluate.plotting.pdf_merge import merge_pdf_subdirectories
from weathergen.evaluate.plotting.plot_utils import (
    PlotSubdir,
    _assign_time_coord,
    apply_font_settings,
    collect_channels,
    collect_streams,
    create_filename,
)
from weathergen.evaluate.plotting.quantile_plots import QuantilePlots
from weathergen.evaluate.plotting.score_cards import ScoreCards

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score plots across runs
# ---------------------------------------------------------------------------


def plot_scores(cfg: dict, scores_dict: dict, init_hour_scores: dict, summary_dir: Path) -> None:
    """Plot the scores of all runs: one figure per ``score_plots`` entry, metric and region.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.
    scores_dict : dict
        ``scores_dict[metric][region][stream][run_id] -> DataArray`` (``MetricResult.score``).
    init_hour_scores : dict
        Same nesting with the ``MetricResult.by_init_hour`` arrays; empty unless
        ``init_hour`` is requested.
    summary_dir : Path
        Directory to write plots to.
    """
    runs = cfg.run_ids
    metrics = cfg.evaluation.metrics
    print_summary = cfg.evaluation.get("print_summary", False)
    # image_format / dpi_val etc. live at the top level of the config,
    # not under a "global_plotting_options" sub-key.
    plt_opt = cfg.get("global_plotting_options", cfg)
    eval_opt = cfg.get("evaluation", {})

    plot_cfg = {
        "image_format": plt_opt.get("image_format", "png"),
        "dpi_val": plt_opt.get("dpi_val", 300),
        "fig_size": plt_opt.get("fig_size", (8, 10)),
        "log_scale": eval_opt.get("log_scale", False),
        "add_grid": eval_opt.get("add_grid", False),
        "plot_ensemble": eval_opt.get("plot_ensemble", False),
        "baseline": eval_opt.get("baseline", None),
        "font_size": plt_opt.get("font_size"),
        "font_type": plt_opt.get("font_type"),
    }

    apply_font_settings(plot_cfg)
    # Prefix the output directory with a run_ids identifier so that
    # different evaluation configs can coexist in the same base directory.
    run_ids_str = "_".join(sorted(runs.keys()))
    output_basedir = summary_dir / run_ids_str

    plotter = LinePlots(plot_cfg, output_basedir)
    sc_plotter = ScoreCards(plot_cfg, output_basedir)
    br_plotter = BarPlots(plot_cfg, output_basedir)
    quantile_plotter = QuantilePlots(plot_cfg, output_basedir)

    # Resolve which summary plots to produce: prefer new-style score_plots list,
    # fall back to old-style individual booleans.
    score_plots_list = eval_opt.get("score_plots", [])
    _sp = set(score_plots_list)
    do_metric_plots = "metric_plots" in _sp
    do_ratio = "ratio" in _sp or eval_opt.get("ratio_plots", False)
    do_heatmap = "heatmap" in _sp or eval_opt.get("heat_maps", False)
    do_scorecard = "scorecard" in _sp or eval_opt.get("score_cards", False)
    do_bar = "bar" in _sp or eval_opt.get("bar_plots", False)
    # by-init-hour scores are only computed when init_hour is requested (_process_stream)
    do_init_hour = bool(init_hour_scores)
    # Drawn with raw matplotlib: own directory and dpi default until it goes through LinePlots.
    init_hour_dir = summary_dir / "_".join(runs) / "score_by_init_hour"
    init_hour_dpi = plt_opt.get("dpi_val", 150)

    # Map each resolved plot option to the subdir(s) it produces, so PDF merging
    # can reuse the same flags without re-deriving them from eval_opt.
    plot_option_subdirs = {
        "metric_plots": [PlotSubdir.line_plots, PlotSubdir.psd_plots, PlotSubdir.qq_plots],
        "ratio": [PlotSubdir.ratio_plots],
        "scorecard": [PlotSubdir.score_cards],
        "bar": [PlotSubdir.bar_plots],
    }
    enabled_opts = {
        "metric_plots": do_metric_plots,
        "ratio": do_ratio,
        "scorecard": do_scorecard,
        "bar": do_bar,
    }

    for metric in metrics:
        for region in scores_dict[metric].keys():
            # Set metric/region subdirectory for all plotters
            plotter.set_subdir(metric, region)
            sc_plotter.set_subdir(metric, region)
            br_plotter.set_subdir(metric, region)
            quantile_plotter.set_subdir(metric, region)

            # metric_plots draws each metric in its standard form: a PSD plot for psd,
            # a Q-Q plot for qq_analysis, and score vs lead time for all others.
            # psd is a spectrum, so none of the other score plots apply to it.
            if metric == "psd":
                if do_metric_plots:
                    psd_plot_metric_region(metric, region, runs, scores_dict, plotter)
                continue

            if do_metric_plots:
                if metric == "qq_analysis":
                    quantile_plot_metric_region(metric, region, runs, scores_dict, quantile_plotter)
                else:
                    lead_time_plot_metric_region(
                        metric, region, runs, scores_dict, plotter, print_summary
                    )
            if do_ratio:
                ratio_plot_metric_region(metric, region, runs, scores_dict, plotter, print_summary)
            if do_heatmap:
                heat_maps_metric_region(metric, region, runs, scores_dict, plotter)
            if do_scorecard:
                score_card_metric_region(metric, region, runs, scores_dict, sc_plotter)
            if do_bar:
                bar_plot_metric_region(metric, region, runs, scores_dict, br_plotter)
            if do_init_hour and region in init_hour_scores.get(metric, {}):
                init_hour_metric_region(
                    metric,
                    region,
                    runs,
                    init_hour_scores,
                    plot_cfg["image_format"],
                    init_hour_dpi,
                    init_hour_dir,
                )

    if do_init_hour:
        _logger.info(f"Scores by initialisation hour plots saved to {init_hour_dir}.")

    # Merge individual PDFs into combined documents for easier browsing
    if plot_cfg["image_format"] == "pdf":
        enabled_subdirs = [
            subdir
            for opt, subdirs in plot_option_subdirs.items()
            if enabled_opts[opt]
            for subdir in subdirs
        ]
        merge_pdf_subdirectories(output_basedir, run_ids=list(runs.keys()), subdirs=enabled_subdirs)


# ---------------------------------------------------------------------------
# Drivers: one (metric, region) with all runs on the figure
# ---------------------------------------------------------------------------


def lead_time_plot_metric_region(
    metric: str,
    region: str,
    runs: dict,
    scores_dict: dict,
    plotter: object,
    print_summary: bool,
) -> None:
    """Plot data for all streams and channels for a given metric and region.

    Parameters
    ----------
    metric: str
        String specifying the metric to plot
    region: str
        String specifying the region to plot
    runs: dict
        Dictionary containing the config for all runs
    scores_dict : dict
        The dictionary containing all computed metrics.
    plotter:
        Plotter object to handle the plotting part
    print_summary: bool
        Option to print plot values to screen

    """
    streams_set = collect_streams(runs)
    channels_set = collect_channels(scores_dict, metric, region, runs)

    for stream in streams_set:
        for ch in channels_set:
            selected_data, labels, run_ids, colors = [], [], [], []

            for run_id, data in scores_dict[metric][region].get(stream, {}).items():
                # skip if channel is missing or contains NaN
                if ch not in np.atleast_1d(data.channel.values) or data.isnull().all():
                    continue

                selected_data.append(data.sel(channel=ch))
                label = runs[run_id].get("label", run_id)
                if label != run_id:
                    label = f"{run_id} - {label}"
                labels.append(label)
                run_ids.append(run_id)
                colors.append(runs[run_id].get("color", None))

            if selected_data:
                _logger.info(f"Creating plot for {metric} - {region} - {stream} - {ch}.")

                name = create_filename(
                    prefix=[metric, region], middle=sorted(set(run_ids)), suffix=[stream, ch]
                )

                selected_data, time_dim = _assign_time_coord(selected_data)

                title = f"{metric.upper()} | {stream} | {ch}"

                ref_line_dict = {"ssr_adj": 1.0}
                line = ref_line_dict.get(metric)

                plotter.plot(
                    selected_data,
                    labels,
                    tag=name,
                    x_dim=time_dim,
                    y_dim=metric,
                    print_summary=print_summary,
                    title=title,
                    colors=colors,
                    line=line,
                )


def ratio_plot_metric_region(
    metric: str,
    region: str,
    runs: dict,
    scores_dict: dict,
    plotter: object,
    print_summary: bool,
) -> None:
    """Plot ratio data for all streams and channels for a given metric and region.

    Parameters
    ----------
    metric: str
        String specifying the metric to plot
    region: str
        String specifying the region to plot
    runs: dict
        Dictionary containing the config for all runs
    scores_dict : dict
        The dictionary containing all computed metrics.
    plotter:
        Plotter object to handle the plotting part
    print_summary: bool
        Option to print plot values to screen

    """
    streams_set = collect_streams(runs)

    for stream in streams_set:
        selected_data = []
        labels = []
        run_ids = []
        colors = []
        for run_id, run_data in runs.items():
            data = scores_dict.get(metric, {}).get(region, {}).get(stream, {}).get(run_id)
            if data is None or data.isnull().all():
                continue
            selected_data.append(data)
            label = run_data.get("label", run_id)
            if label != run_id:
                label = f"{run_id} - {label}"
            labels.append(label)
            run_ids.append(run_id)
            colors.append(run_data.get("color", None))

        if len(selected_data) > 0:
            _logger.info(f"Creating ratio plot for {metric} - {stream}")

            name = create_filename(
                prefix=[metric, region], middle=sorted(set(run_ids)), suffix=[stream]
            )
            plotter.ratio_plot(
                data=selected_data,
                run_ids=run_ids,
                labels=labels,
                y_dim=metric,
                tag=name,
                print_summary=print_summary,
                colors=colors,
            )


def heat_maps_metric_region(
    metric: str,
    region: str,
    runs: dict,
    scores_dict: dict,
    plotter: object,
) -> None:
    """Plot heat map data for all streams and channels for a given metric and region.

    Parameters
    ----------
    metric: str
        String specifying the metric to plot
    region: str
        String specifying the region to plot
    runs: dict
        Dictionary containing the config for all runs
    scores_dict : dict
        The dictionary containing all computed metrics.
    plotter:
        Plotter object to handle the plotting part
    print_summary: bool
        Option to print plot values to screen

    """
    streams_set = collect_streams(runs)

    for stream in streams_set:
        selected_data = []
        labels = []
        run_ids = []
        for run_id in runs:
            data = scores_dict.get(metric, {}).get(region, {}).get(stream, {}).get(run_id)
            if data is None or data.isnull().all():
                continue

            selected_data.append(data)
            label = runs[run_id].get("label", run_id)
            if label != run_id:
                label = f"{run_id} - {label}"
            labels.append(label)
            run_ids.append(run_id)

        if len(selected_data) > 0:
            _logger.info(f"Creating heat maps for {metric} - {stream}")
            name = create_filename(
                prefix=[metric, region], middle=sorted(set(run_ids)), suffix=[stream]
            )
            selected_data, time_dim = _assign_time_coord(selected_data)

            plotter.heat_map(
                selected_data,
                labels,
                metric=metric,
                tag=name,
                x_dim=time_dim,
            )


def score_card_metric_region(
    metric: str,
    region: str,
    runs: dict,
    scores_dict: dict,
    sc_plotter: object,
) -> None:
    """
    Create score cards for all streams and channels for a given metric and region.

    Parameters
    ----------
    metric: str
        String specifying the metric to plot
    region: str
        String specifying the region to plot
    runs: dict
        Dictionary containing the config for all runs
    scores_dict : dict
        The dictionary containing all computed metrics.
    sc_plotter:
        Plotter object to handle the plotting part
    """
    streams_set = collect_streams(runs)
    channels_set = collect_channels(scores_dict, metric, region, runs)

    for stream in streams_set:
        selected_data, run_ids = [], []
        for run_id, data in scores_dict[metric][region].get(stream, {}).items():
            if data.isnull().all():
                continue
            selected_data.append(data)
            run_ids.append(run_id)

        if len(selected_data) >= 2:
            _logger.info(f"Creating score cards for {metric} - {region} - {stream}.")
            name = "_".join([metric, region, stream])
            sc_plotter.plot(selected_data, run_ids, metric, channels_set, name)
        elif len(selected_data) == 1:
            _logger.info(
                f"Skipping score card for {metric} - {region} - {stream}: "
                f"only one run available (need at least 2 to compare)."
            )


def bar_plot_metric_region(
    metric: str,
    region: str,
    runs: dict,
    scores_dict: dict,
    br_plotter: object,
) -> None:
    """
    Create bar plots for all streams and run_ids for a given metric and region.

    Parameters
    ----------
    metric: str
        String specifying the metric to plot
    region: str
        String specifying the region to plot
    runs: dict
        Dictionary containing the config for all runs
    scores_dict : dict
        The dictionary containing all computed metrics.
    plotter:
        Plotter object to handle the plotting part
    """
    streams_set = collect_streams(runs)
    channels_set = collect_channels(scores_dict, metric, region, runs)

    for stream in streams_set:
        selected_data, run_ids = [], []

        for run_id, data in scores_dict[metric][region].get(stream, {}).items():
            if data.isnull().all():
                continue
            selected_data.append(data)
            run_ids.append(run_id)

        if selected_data:
            _logger.info(f"Creating bar plots for {metric} - {region} - {stream}.")
            name = "_".join([metric, region, stream])
            br_plotter.plot(selected_data, run_ids, metric, channels_set, name)


def quantile_plot_metric_region(
    metric: str,
    region: str,
    runs: dict,
    scores_dict: dict,
    plotter: object,
) -> None:
    """
    Create quantile-quantile (Q-Q) plots for extreme value analysis for all streams
    and channels for a given metric and region.

    Parameters
    ----------
    metric: str
        String specifying the metric to plot (should be 'qq_analysis')
    region: str
        String specifying the region to plot
    runs: dict
        Dictionary containing the config for all runs
    scores_dict : dict
        The dictionary containing all computed metrics.
    plotter:
        Plotter object to handle the plotting part. Must have a qq_plot method.
    """
    streams_set = collect_streams(runs)
    channels_set = collect_channels(scores_dict, metric, region, runs)

    for stream in streams_set:
        for ch in channels_set:
            selected_data, labels, run_ids = [], [], []
            qq_full_data = []  # Store full Q-Q datasets for detailed plotting

            for run_id, data in scores_dict[metric][region].get(stream, {}).items():
                # skip if channel is missing
                if ch not in np.atleast_1d(data.channel.values):
                    continue

                # Select channel
                data_for_channel = data.sel(channel=ch) if "channel" in data.dims else data

                # Check for NaN
                if data_for_channel.isnull().all():
                    continue

                # For qq_analysis, extract Q-Q data from attributes
                if metric == "qq_analysis" and "p_quantiles" in data_for_channel.attrs:
                    attrs = data_for_channel.attrs
                    # Convert to numpy arrays once
                    p_quantiles_arr = np.array(attrs["p_quantiles"])
                    gt_quantiles_arr = np.array(attrs["gt_quantiles"])
                    qq_deviation_arr = np.array(attrs["qq_deviation"])
                    qq_deviation_norm_arr = np.array(attrs["qq_deviation_normalized"])
                    quantile_levels = np.array(attrs["quantile_levels"])
                    extreme_low_mse = float(np.mean(np.array(attrs["extreme_low_mse"])))
                    extreme_high_mse = float(np.mean(np.array(attrs["extreme_high_mse"])))

                    qq_dataset = xr.Dataset(
                        {
                            "quantile_levels": (["quantile"], quantile_levels),
                            "p_quantiles": (["quantile"], _flatten_or_average(p_quantiles_arr)),
                            "gt_quantiles": (["quantile"], _flatten_or_average(gt_quantiles_arr)),
                            "qq_deviation": (["quantile"], _flatten_or_average(qq_deviation_arr)),
                            "qq_deviation_normalized": (
                                ["quantile"],
                                _flatten_or_average(qq_deviation_norm_arr),
                            ),
                            "extreme_low_mse": ([], extreme_low_mse),
                            "extreme_high_mse": ([], extreme_high_mse),
                        }
                    )
                    # Store extreme percentiles for plotting
                    qq_dataset.attrs["extreme_percentiles"] = tuple(attrs["extreme_percentiles"])
                    qq_full_data.append(qq_dataset)

                selected_data.append(data_for_channel)
                label = runs[run_id].get("label", run_id)
                if label != run_id:
                    label = f"{run_id} - {label}"
                labels.append(label)
                run_ids.append(run_id)

            if selected_data:
                _logger.info(f"Creating Q-Q plot for {metric} - {region} - {stream} - {ch}.")

                name = create_filename(
                    prefix=[metric, region], middle=sorted(set(run_ids)), suffix=[stream, ch]
                )

                # Check if plotter has qq_plot method and Q-Q data is available
                if hasattr(plotter, "qq_plot") and qq_full_data:
                    _logger.info(f"Creating Q-Q plot with {len(qq_full_data)} dataset(s).")
                    # Extract extreme_percentiles from dataset
                    extreme_pct = qq_full_data[0].attrs["extreme_percentiles"]
                    plotter.qq_plot(
                        qq_full_data,
                        labels,
                        tag=name,
                        metric=metric,
                        extreme_percentiles=extreme_pct,
                    )
                else:
                    # Skip plotting if no Q-Q data available
                    _logger.warning(
                        f"Q-Q data not available for {metric} - {region} - {stream} - {ch}. "
                        f"Skipping plot generation."
                    )


def _flatten_or_average(arr: NDArray) -> NDArray:
    """Flatten array or average across non-quantile dimensions.

    Parameters
    ----------
    arr : np.ndarray
        Input array, possibly multi-dimensional.

    Returns
    -------
    np.ndarray
        Flattened 1D array, averaged across extra dimensions if needed.
    """
    if arr.ndim > 1:
        return np.mean(arr, axis=tuple(range(1, arr.ndim))).flatten()
    return arr


def _extract_psd_attrs(data_ch: xr.DataArray, fstep: int, ch: str) -> list[dict] | None:
    """Extract PSD curve data from DataArray attrs for a given fstep/channel.

    Returns a single-element list of dicts ready for the plotter, or None if keys are missing.
    """
    attrs = data_ch.attrs
    fp = f"fstep_{fstep}/"

    for prefix in (f"{fp}{ch}/", fp):
        if f"{prefix}frequencies" in attrs and f"{prefix}psd_target" in attrs:
            return [
                {
                    "frequencies": np.array(attrs[f"{prefix}frequencies"]),
                    "psd_target": np.array(attrs[f"{prefix}psd_target"]),
                    "psd_prediction": np.array(attrs[f"{prefix}psd_prediction"]),
                    "psd_method": attrs.get(f"{fp}psd_method", attrs.get("psd_method", "sht")),
                }
            ]
    return None


def psd_plot_metric_region(
    metric: str,
    region: str,
    runs: dict,
    scores_dict: dict,
    plotter: object,
) -> None:
    """Create PSD plots for all streams and channels for a given metric and region.

    PSD curves (frequencies, target PSD, prediction PSD) are stored in
    ``score.attrs`` by ``Scores.calc_psd`` and read back here.
    """
    streams_set = collect_streams(runs)
    channels_set = collect_channels(scores_dict, metric, region, runs)

    for stream in streams_set:
        for ch in channels_set:
            for run_id, data in scores_dict[metric][region].get(stream, {}).items():
                if ch not in np.atleast_1d(data.channel.values):
                    continue

                data_ch = data.sel(channel=ch) if "channel" in data.dims else data
                if data_ch.isnull().all():
                    continue

                attr_fsteps = data_ch.attrs.get("attr_fsteps", [])
                if not attr_fsteps:
                    _logger.warning(f"PSD attrs missing for {run_id}/{stream}/{ch}. Skipping.")
                    continue

                label = runs[run_id].get("label", run_id)

                for fstep in attr_fsteps:
                    psd_datasets = _extract_psd_attrs(data_ch, fstep, ch)
                    if psd_datasets is None:
                        continue

                    method_tag = psd_datasets[0].get("psd_method", "sht")
                    name = create_filename(
                        prefix=[metric, method_tag, region],
                        middle=[run_id],
                        suffix=[stream, ch, f"fstep{fstep}"],
                    )
                    plotter.psd_plot(
                        psd_datasets,
                        [label],
                        tag=name,
                        variable=ch,
                        forecast_step=str(fstep),
                    )
    _logger.info(f"PSD plots saved successfully into: {plotter.out_plot_dir_psd}")


def init_hour_metric_region(
    metric: str,
    region: str,
    runs: dict,
    init_hour_scores: dict,
    image_format: str,
    dpi_val: int,
    out_dir: Path,
) -> None:
    """Plot the score against the initialisation hour of the day, all runs on one figure.

    One figure per stream, channel and forecast step.

    Parameters
    ----------
    metric: str
        String specifying the metric to plot
    region: str
        String specifying the region to plot
    runs: dict
        Dictionary containing the config for all runs
    init_hour_scores : dict
        ``init_hour_scores[metric][region][stream][run_id] -> DataArray`` with dims
        ``(init_hour, forecast_step, channel[, ens])`` (``MetricResult.by_init_hour``).
    image_format : str
        Image format of the figures.
    dpi_val : int
        Resolution of the figures.
    out_dir : Path
        Directory to write the figures to.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for stream, run_dict in init_hour_scores[metric][region].items():
        if not run_dict:
            continue

        # Determine fsteps and channels from first run
        first_run_scores = next(iter(run_dict.values()))
        fsteps = sorted(int(f) for f in first_run_scores.forecast_step.values)
        sample_score = first_run_scores.isel(forecast_step=0)
        channels = (
            list(sample_score.coords["channel"].values)
            if "channel" in sample_score.dims
            else [None]
        )

        for channel in channels:
            for fstep in fsteps:
                plt.figure(figsize=(10, 6), dpi=dpi_val)

                for run_id, fstep_scores in run_dict.items():
                    if fstep not in fstep_scores.forecast_step.values:
                        continue
                    score = fstep_scores.sel(forecast_step=fstep)
                    score_vals = score.sel(channel=channel) if channel is not None else score
                    hours = score_vals.coords["init_hour"].values
                    values = score_vals.values.flatten()
                    run_label = runs[run_id].get("label", run_id)
                    label = f"{run_label} ({run_id})"
                    color = runs[run_id].get("color", None)
                    plt.plot(
                        hours,
                        values,
                        marker="o",
                        linewidth=2,
                        label=label,
                        color=color,
                    )

                ch_label = channel if channel is not None else "all"
                title = (
                    f"{metric.upper()} vs initialisation hour | "
                    f"{ch_label} | fstep {fstep} | {region} | {stream}"
                )
                plt.title(title)
                plt.xlabel("Initialisation hour [UTC]")
                plt.ylabel(metric.upper())
                if hours is not None and len(hours) > 0:
                    plt.xlim(min(hours) - 0.5, max(hours) + 0.5)
                    plt.xticks(sorted(hours))
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()

                run_ids_str = "_".join(sorted(run_dict.keys()))
                plot_path = (
                    out_dir / f"{metric}_{ch_label}_{region}_{stream}"
                    f"_{run_ids_str}"
                    f"_fstep_{fstep}_by_init_hour.{image_format}"
                )
                plt.savefig(plot_path, bbox_inches="tight")
                plt.close()
