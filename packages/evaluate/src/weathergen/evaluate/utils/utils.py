# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# Standard library
import json
import logging
from collections import defaultdict
from pathlib import Path

# Third-party
import numpy as np
import omegaconf as oc
import xarray as xr
from tqdm import tqdm

# Local application / package
from weathergen.evaluate.io.io_reader import Reader
from weathergen.evaluate.plotting.plot_utils import (
    bar_plot_metric_region,
    heat_maps_metric_region,
    plot_metric_region,
    ratio_plot_metric_region,
    score_card_metric_region,
)
from weathergen.evaluate.plotting.plotter import BarPlots, LinePlots, Plotter, ScoreCards
from weathergen.evaluate.scores.score import VerifiedData, get_score
from weathergen.evaluate.utils.clim_utils import get_climatology
from weathergen.evaluate.utils.regions import RegionBoundingBox

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


def get_next_data(fstep, da_preds, da_tars, fsteps):
    """
    Get the next forecast step data for the given forecast step.
    """
    fstep_idx = fsteps.index(fstep)
    # Get the next forecast step
    next_fstep = fsteps[fstep_idx + 1] if fstep_idx + 1 < len(fsteps) else None
    if next_fstep is not None:
        preds_next = da_preds.get(next_fstep, None)
        tars_next = da_tars.get(next_fstep, None)
    else:
        preds_next = None
        tars_next = None

    return preds_next, tars_next


def calc_scores_per_stream(reader, stream, regions, metrics, plot_score_maps=False):
    """
    Calculate scores for a given run and stream using the specified metrics.

    Parameters
    ----------
    reader : Reader
        Reader object containing all info about a particular run.
    stream :
        Stream name to calculate scores for.
    scores_dict:
        Dictionary for scores with structure scores_dict[metric][region][stream][run_id]
    regions :
        List of regions to calculate scores on.
    metrics :
        List of metric names to calculate.
    plot_score_maps :
        When it is True and the stream is on a regular grid the scores are
        recomputed as a function of the "ipoint" and plotted on a 2D scatter map.
        NOTE: the scores are averaged over the "sample" dimension and for most
        of the metrics this does not give the same results as averaging over
        the "ipoint" dimension.
    Returns
    -------
    Dictionary containing scores for each metric and stream.
    """
    local_scores = {}  # top-level dict: metric -> region -> stream -> run_id

    _logger.info(f"RUN {reader.run_id} - {stream}: Calculating scores for metrics {metrics}...")
    if plot_score_maps:
        _logger.info(f"RUN {reader.run_id} - {stream}: Plotting scores is enabled.")

        map_dir = reader.runplot_dir / "plots" / stream / "score_maps"
        map_dir.mkdir(parents=True, exist_ok=True)

        _logger.info(f"RUN {reader.run_id} - {stream}: Saving plotted scores to {map_dir}")

    available_data = reader.check_availability(stream, mode="evaluation")
    fsteps = available_data.fsteps
    samples = available_data.samples
    channels = available_data.channels
    ensemble = available_data.ensemble
    is_regular = reader.is_regular(stream)
    group_by_coord = None if is_regular else "sample"

    output_data = reader.get_data(
        stream,
        fsteps=fsteps,
        samples=samples,
        channels=channels,
        ensemble=ensemble,
        return_counts=True,
    )
    da_preds = output_data.prediction
    da_tars = output_data.target

    aligned_clim_data = get_climatology(reader, da_tars, stream)

    for region in regions:
        bbox = RegionBoundingBox.from_region_name(region)

        metric_stream = xr.DataArray(
            np.full(
                (len(samples), len(fsteps), len(channels), len(metrics), len(ensemble)),
                np.nan,
            ),
            coords={
                "sample": samples,
                "forecast_step": fsteps,
                "channel": channels,
                "metric": metrics,
                "ens": ensemble,
            },
        )

        for (fstep, tars), (_, preds) in zip(da_tars.items(), da_preds.items(), strict=False):
            if preds.ipoint.size == 0:
                _logger.warning(
                    f"No data for stream {stream} at fstep {fstep} in region {region}. Skipping."
                )
                continue

            _logger.debug(f"Verifying data for stream {stream}...")

            preds_next, tars_next = get_next_data(fstep, da_preds, da_tars, fsteps)

            if region != "global":
                _logger.debug(
                    f"Applying bounding box mask for region '{region}' to targets and predictions."
                )

            tars, preds, tars_next, preds_next = [
                bbox.apply_mask(x) if x is not None else None
                for x in (tars, preds, tars_next, preds_next)
            ]
            climatology = aligned_clim_data[fstep] if aligned_clim_data else None
            score_data = VerifiedData(preds, tars, preds_next, tars_next, climatology)
            # Build up computation graphs for all metrics
            _logger.debug(f"Build computation graphs for metrics for stream {stream}...")

            # Add it only if it is not None
            valid_scores = []
            for metric in metrics:
                score = get_score(
                    score_data, metric, agg_dims="ipoint", group_by_coord=group_by_coord
                )
                if score is not None:
                    valid_scores.append(score)

            valid_metric_names = [
                metric
                for metric, score in zip(metrics, valid_scores, strict=False)
                if score is not None
            ]
            if not valid_scores:
                continue

            combined_metrics = xr.concat(valid_scores, dim="metric")
            combined_metrics = combined_metrics.assign_coords(metric=valid_metric_names)
            combined_metrics = combined_metrics.compute()

            for coord in ["channel", "sample", "ens"]:
                combined_metrics = scalar_coord_to_dim(combined_metrics, coord)

            criteria = {
                "forecast_step": int(fstep),
                "sample": combined_metrics.sample,
                "channel": combined_metrics.channel,
                "metric": combined_metrics.metric,
            }
            if "ens" in combined_metrics.dims:
                criteria["ens"] = combined_metrics.ens

            metric_stream.loc[criteria] = combined_metrics

            if is_regular and plot_score_maps:
                _logger.info(f"Plotting scores on a map {stream} - forecast step: {fstep}...")
                _plot_score_maps_per_stream(
                    reader, map_dir, stream, region, score_data, metrics, fstep
                )

        _logger.info(f"Scores for run {reader.run_id} - {stream} calculated successfully.")

        # Build local dictionary for this region
        for metric in metrics:
            local_scores.setdefault(metric, {}).setdefault(region, {}).setdefault(stream, {})[
                reader.run_id
            ] = metric_stream.sel({"metric": metric})

    return local_scores


def _plot_score_maps_per_stream(
    reader: Reader,
    map_dir: str,
    stream: str,
    region: str,
    score_data: VerifiedData,
    metrics: list[str],
    fstep: int,
) -> None:
    """Plot 2D score maps for all metrics and channels.
    Parameters
    ----------
    reader: Reader
        Reader object containing all infos about the run
    map_dir: str
        Directory where the plots are saved.
    stream: str
        Stream name to plot score maps for.
     region :
        Region name to plot score maps for.
    score_data: VerifiedData
        prediction and target stored in the data class.
    metrics: str
        List of all metrics to plot.
    fstep:
        forecast step to plot.

    Return
    ------
    None
    """

    cfg = reader.global_plotting_options

    # TODO: add support for climatology-dependent metrics as well

    plotter = Plotter(
        {
            "image_format": cfg.get("image_format", "png"),
            "dpi_val": cfg.get("dpi_val", 300),
            "fig_size": cfg.get("fig_size", (8, 10)),
        },
        reader.runplot_dir,
        stream,
    )

    preds = score_data.prediction

    plot_metrics = xr.concat(
        [get_score(score_data, m, agg_dims="sample") for m in metrics], dim="metric"
    )

    plot_metrics = plot_metrics.assign_coords(
        lat=preds.lat.reset_coords(drop=True),
        lon=preds.lon.reset_coords(drop=True),
        metric=metrics,
    ).compute()

    if "ens" in preds.dims:
        plot_metrics["ens"] = preds.ens

    has_ens = "ens" in plot_metrics.coords
    ens_values = plot_metrics.coords["ens"].values if has_ens else [None]

    for metric in plot_metrics.coords["metric"].values:
        for ens_val in tqdm(ens_values, f"Plotting metric - {metric}"):
            tag = f"score_maps_{metric}_fstep_{fstep}" + (
                f"_ens_{ens_val}" if ens_val is not None else ""
            )
            for channel in plot_metrics.coords["channel"].values:
                sel = {"metric": metric, "channel": channel}
                if ens_val is not None:
                    sel["ens"] = ens_val

                data = plot_metrics.sel(**sel).squeeze()
                title = f"{metric} - {channel}: fstep {fstep}" + (
                    f", ens {ens_val}" if ens_val is not None else ""
                )
                plotter.scatter_plot(data, map_dir, channel, region, tag=tag, title=title)


def plot_data(reader: Reader, stream: str, global_plotting_opts: dict) -> None:
    """
    Plot the data for a given run and stream.

    Parameters
    ----------
    reader: Reader
        Reader object containing all infos about the run
    stream: str
        Stream name to plot data for.
    global_plotting_opts: dict
        Dictionary containing all plotting options that apply globally to all run_ids
    """
    run_id = reader.run_id

    # get stream dict from evaluation config (assumed to be part of cfg at this point)
    stream_cfg = reader.get_stream(stream)

    # handle plotting settings
    plot_settings = stream_cfg.get("plotting", {})

    # return early if no plotting is requested
    if not (
        plot_settings
        and (
            plot_settings.get("plot_maps", False)
            or plot_settings.get("plot_histograms", False)
            or plot_settings.get("plot_animations", False)
        )
    ):
        return

    plotter_cfg = {
        "image_format": global_plotting_opts.get("image_format", "png"),
        "dpi_val": global_plotting_opts.get("dpi_val", 300),
        "fig_size": global_plotting_opts.get("fig_size", (8, 10)),
        "fps": global_plotting_opts.get("fps", 2),
        "regions": global_plotting_opts.get("regions", ["global"]),
        "plot_subtimesteps": reader.get_inference_stream_attr(stream, "tokenize_spacetime", False),
    }
    plotter = Plotter(plotter_cfg, reader.runplot_dir)

    available_data = reader.check_availability(stream, mode="plotting")

    # Check if maps should be plotted and handle configuration if provided
    plot_maps = plot_settings.get("plot_maps", False)
    if not isinstance(plot_maps, bool):
        raise TypeError("plot_maps must be a boolean.")

    plot_target = plot_settings.get("plot_target", True)
    if not isinstance(plot_target, bool):
        raise TypeError("plot_target must be a boolean.")

    # Check if histograms should be plotted
    plot_histograms = plot_settings.get("plot_histograms", False)
    if not isinstance(plot_histograms, bool):
        raise TypeError("plot_histograms must be a boolean.")

    plot_animations = plot_settings.get("plot_animations", False)
    if not isinstance(plot_animations, bool):
        raise TypeError("plot_animations must be a boolean.")

    model_output = reader.get_data(
        stream,
        samples=available_data.samples,
        fsteps=available_data.fsteps,
        channels=available_data.channels,
        ensemble=available_data.ensemble,
    )

    da_tars = model_output.target
    da_preds = model_output.prediction

    if not da_tars:
        _logger.info(f"Skipping Plot Data for {stream}. Targets are empty.")
        return

    # get common ranges across all run_ids
    if not isinstance(global_plotting_opts.get(stream), oc.DictConfig):
        global_plotting_opts[stream] = oc.DictConfig({})
    maps_config = common_ranges(
        da_tars, da_preds, available_data.channels, global_plotting_opts[stream]
    )

    for (fstep, tars), (_, preds) in zip(da_tars.items(), da_preds.items(), strict=False):
        plot_chs = list(np.atleast_1d(tars.channel.values))
        plot_samples = list(np.unique(tars.sample.values))

        for sample in tqdm(plot_samples, desc=f"Plotting {run_id} - {stream} - fstep {fstep}"):
            data_selection = {
                "sample": sample,
                "stream": stream,
                "forecast_step": fstep,
            }

            if plot_maps:
                if plot_target:
                    plotter.create_maps_per_sample(
                        tars, plot_chs, data_selection, "targets", maps_config
                    )
                for ens in available_data.ensemble:
                    preds_ens = (
                        preds.sel(ens=ens) if "ens" in preds.dims and ens != "mean" else preds
                    )
                    preds_tag = "" if "ens" not in preds.dims else f"ens_{ens}"
                    preds_name = "_".join(
                        filter(None, ["preds", preds_tag])
                    )  # avoid trailing underscore

                    plotter.create_maps_per_sample(
                        preds_ens, plot_chs, data_selection, preds_name, maps_config
                    )

                    if plot_histograms:
                        plotter.create_histograms_per_sample(
                            tars, preds_ens, plot_chs, data_selection, preds_tag
                        )

            plotter = plotter.clean_data_selection()

    if plot_animations:
        plot_fsteps = da_tars.keys()
        for ens in available_data.ensemble:
            preds_name = "preds" if "ens" not in preds.dims else f"preds_ens_{ens}"
            plotter.animation(plot_samples, plot_fsteps, plot_chs, data_selection, preds_name)
        if plot_target:
            plotter.animation(plot_samples, plot_fsteps, plot_chs, data_selection, "targets")

    return


def metric_list_to_json(
    reader: Reader,
    metrics_list: list[xr.DataArray],
    npoints_sample_list: list[xr.DataArray],
    streams: list[str],
    region: str,
):
    """
    Write the evaluation results collected in a list of xarray DataArrays for the metrics
    to stream- and metric-specific JSON files.

    Parameters
    ----------
    reader:
        Reader object containing all info about the run_id.
    metrics_list :
        Metrics per stream.
    npoints_sample_list :
        Number of points per sample per stream.
    streams :
        Stream names.
    region :
        Region name.
    metric_dir :
        Output directory.
    run_id :
        Identifier of the inference run.
    mini_epoch :
        Mini_epoch number.
    """
    assert len(metrics_list) == len(npoints_sample_list) == len(streams), (
        "The lengths of metrics_list, npoints_sample_list, and streams must be the same."
    )

    reader.metrics_dir.mkdir(parents=True, exist_ok=True)

    for s_idx, stream in enumerate(streams):
        metrics_stream, npoints_sample_stream = (
            metrics_list[s_idx],
            npoints_sample_list[s_idx],
        )

        for metric in metrics_stream.coords["metric"].values:
            metric_now = metrics_stream.sel(metric=metric)

            # Save as individual DataArray, not Dataset
            metric_now.attrs["npoints_per_sample"] = npoints_sample_stream.values.tolist()
            metric_dict = metric_now.to_dict()

            # Match the expected filename pattern
            save_path = (
                reader.metrics_dir
                / f"{reader.run_id}_{stream}_{region}_{metric}_chkpt{reader.mini_epoch:05d}.json"
            )

            _logger.info(f"Saving results to {save_path}")
            with open(save_path, "w") as f:
                json.dump(metric_dict, f, indent=4)

    _logger.info(
        f"Saved all results of inference run {reader.run_id} - mini_epoch {reader.mini_epoch:d} "
        f"successfully to {reader.metrics_dir}."
    )


def plot_summary(cfg: dict, scores_dict: dict, summary_dir: Path):
    """
    Plot summary of the evaluation results.
    This function is a placeholder for future implementation.

    Parameters
    ----------
    cfg :
        Configuration dictionary containing all information for the evaluation.
    scores_dict :
        Dictionary containing scores for each metric and stream.
    """
    _logger.info("Plotting summary of evaluation results...")

    runs = cfg.run_ids
    metrics = cfg.evaluation.metrics
    print_summary = cfg.evaluation.get("print_summary", False)
    regions = cfg.evaluation.get("regions", ["global"])
    plt_opt = cfg.get("global_plotting_options", {})
    eval_opt = cfg.get("evaluation", {})

    plot_cfg = {
        "image_format": plt_opt.get("image_format", "png"),
        "dpi_val": plt_opt.get("dpi_val", 300),
        "fig_size": plt_opt.get("fig_size", (8, 10)),
        "log_scale": eval_opt.get("log_scale", False),
        "add_grid": eval_opt.get("add_grid", False),
        "plot_ensemble": eval_opt.get("plot_ensemble", False),
        "baseline": eval_opt.get("baseline", None),
    }

    plotter = LinePlots(plot_cfg, summary_dir)
    sc_plotter = ScoreCards(plot_cfg, summary_dir)
    br_plotter = BarPlots(plot_cfg, summary_dir)
    for region in regions:
        for metric in metrics:
            if eval_opt.get("summary_plots", True):
                plot_metric_region(metric, region, runs, scores_dict, plotter, print_summary)
            if eval_opt.get("ratio_plots", False):
                ratio_plot_metric_region(metric, region, runs, scores_dict, plotter, print_summary)
            if eval_opt.get("heat_maps", False):
                heat_maps_metric_region(metric, region, runs, scores_dict, plotter)
            if eval_opt.get("score_cards", False):
                score_card_metric_region(metric, region, runs, scores_dict, sc_plotter)
            if eval_opt.get("bar_plots", False):
                bar_plot_metric_region(metric, region, runs, scores_dict, br_plotter)


############# Utility functions ############


def common_ranges(
    data_tars: list[dict],
    data_preds: list[dict],
    plot_chs: list[str],
    maps_config: oc.dictconfig.DictConfig,
) -> oc.dictconfig.DictConfig:
    """
    Calculate common ranges per stream and variables.

    Parameters
    ----------
    data_tars :
        the (target) list of dictionaries with the forecasteps and respective xarray
    data_preds :
        the (prediction) list of dictionaries with the forecasteps and respective xarray
    plot_chs:
        the variables to be plotted as given by the configuration file
    maps_config:
        the global plotting configuration
    Returns
    -------
    maps_config :
        the global plotting configuration with the ranges added and included for each variable (and
        for each stream).
    """
    for var in plot_chs:
        if var in maps_config:
            if not isinstance(maps_config[var].get("vmax"), (int | float)):
                list_max = calc_bounds(data_tars, data_preds, var, "max")
                list_max = np.concatenate([arr.flatten() for arr in list_max]).tolist()
                maps_config[var].update({"vmax": float(max(list_max))})

            if not isinstance(maps_config[var].get("vmin"), (int | float)):
                list_min = calc_bounds(data_tars, data_preds, var, "min")
                list_min = np.concatenate([arr.flatten() for arr in list_min]).tolist()
                maps_config[var].update({"vmin": float(min(list_min))})

        else:
            list_max = calc_bounds(data_tars, data_preds, var, "max")
            list_max = np.concatenate([arr.flatten() for arr in list_max]).tolist()
            list_min = calc_bounds(data_tars, data_preds, var, "min")
            list_min = np.concatenate([arr.flatten() for arr in list_min]).tolist()

            maps_config.update({var: {"vmax": float(max(list_max)), "vmin": float(min(list_min))}})

    return maps_config


def calc_val(x: xr.DataArray, bound: str) -> list[float]:
    """
    Calculate the maximum or minimum value per variable for all forecasteps.
    Parameters
    ----------
    x :
        the xarray DataArray with the forecasteps and respective values
    bound :
        the bound to be calculated, either "max" or "min"
    Returns
    -------
        a list with the maximum or minimum values for a specific variable.
    """
    if bound == "max":
        return x.max(dim=("ipoint")).values
    elif bound == "min":
        return x.min(dim=("ipoint")).values
    else:
        raise ValueError("bound must be either 'max' or 'min'")


def calc_bounds(
    data_tars,
    data_preds,
    var,
    bound,
):
    """
    Calculate the minimum and maximum values per variable for all forecasteps for both targets and
    predictions

    Parameters
    ----------
    data_tars :
        the (target) list of dictionaries with the forecasteps and respective xarray
    data_preds :
        the (prediction) list of dictionaries with the forecasteps and respective xarray
    Returns
    -------
    list_bound :
        a list with the maximum or minimum values for a specific variable.
    """
    list_bound = []
    for da_tars, da_preds in zip(data_tars.values(), data_preds.values(), strict=False):
        list_bound.extend(
            (
                calc_val(da_tars.where(da_tars.channel == var, drop=True), bound),
                calc_val(da_preds.where(da_preds.channel == var, drop=True), bound),
            )
        )

    return list_bound


def scalar_coord_to_dim(da: xr.DataArray, name: str, axis: int = -1) -> xr.DataArray:
    """
    Convert a scalar coordinate to a dimension in an xarray DataArray.
    If the coordinate is already a dimension, it is returned unchanged.

    Parameters
    ----------
    da : xarray.DataArray
        The DataArray to modify.
    name : str
        The name of the coordinate to convert.
    axis : int, optional
        The axis along which to expand the dimension. Default is -1 (last axis).
    Returns
    -------
    xarray.DataArray
        The modified DataArray with the scalar coordinate converted to a dimension.
    """
    if name in da.dims:
        return da  # already a dimension
    if name in da.coords and da.coords[name].ndim == 0:
        val = da.coords[name].item()
        da = da.drop_vars(name)
        da = da.expand_dims({name: [val]}, axis=axis)
    return da


def nested_dict():
    """Two-level nested dict factory: dict[key1][key2] = value"""
    return defaultdict(dict)


def triple_nested_dict():
    """Three-level nested dict factory: dict[key1][key2][key3] = value"""
    return defaultdict(nested_dict)
