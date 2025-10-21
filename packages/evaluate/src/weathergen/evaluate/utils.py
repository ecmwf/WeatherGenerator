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
from pathlib import Path

import numpy as np
import omegaconf as oc
import xarray as xr
from tqdm import tqdm

from weathergen.evaluate.io_reader import Reader
from weathergen.evaluate.plot_utils import plot_metric_region
from weathergen.evaluate.plotter import LinePlots, Plotter
from weathergen.evaluate.score import VerifiedData, get_score

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


def calc_scores_per_stream(
    reader: Reader, stream: str, region: str, metrics: list[str]
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate scores for a given run and stream using the specified metrics.

    Parameters
    ----------
    reader : Reader
        Reader object containing all info about a particular run.
    stream :
        Stream name to calculate scores for.
    region :
        Region name to calculate scores for.
    metrics :
        List of metric names to calculate.

    Returns
    -------
    Tuple of xarray DataArray containing the scores and the number of points per sample.
    """

    _logger.info(
        f"RUN {reader.run_id} - {stream}: Calculating scores for metrics {metrics}..."
    )

    available_data = reader.check_availability(stream, mode="evaluation")

    output_data = reader.get_data(
        stream,
        region=region,
        fsteps=available_data.fsteps,
        samples=available_data.samples,
        channels=available_data.channels,
        return_counts=True,
    )

    da_preds = output_data.prediction
    da_tars = output_data.target
    points_per_sample = output_data.points_per_sample

    # Print ensemble spread before selection
    print_ensemble_spread(da_preds, stream)

    # Select first ensemble member if ens dimension exists
    for fstep in da_preds.keys():
        if "ens" in da_preds[fstep].dims:
            #da_preds[fstep] = da_preds[fstep].isel(ens=0, drop=True)
            #_logger.info(f"Selected first ensemble member for forecast step {fstep}")
            # or compute the mean: .mean(dim='ens'):
            da_preds[fstep] = da_preds[fstep].mean(dim='ens')
            _logger.info(f"Averaged ensemble members for forecast step {fstep}")

    # get coordinate information from retrieved data
    fsteps = [int(k) for k in da_tars.keys()]

    first_da = list(da_preds.values())[0]

    # TODO: improve the way we handle samples.
    samples = list(np.atleast_1d(np.unique(first_da.sample.values)))
    channels = list(np.atleast_1d(first_da.channel.values))

    metric_list = []

    metric_stream = xr.DataArray(
        np.full(
            (len(samples), len(fsteps), len(channels), len(metrics)),
            np.nan,
        ),
        coords={
            "sample": samples,
            "forecast_step": fsteps,
            "channel": channels,
            "metric": metrics,
        },
    )

    for (fstep, tars), (_, preds) in zip(
        da_tars.items(), da_preds.items(), strict=False
    ):
        _logger.debug(f"Verifying data for stream {stream}...")

        preds_next, tars_next = get_next_data(fstep, da_preds, da_tars, fsteps)

        if preds.ipoint.size > 0:
            score_data = VerifiedData(preds, tars, preds_next, tars_next)
            # Build up computation graphs for all metrics
            _logger.debug(
                f"Build computation graphs for metrics for stream {stream}..."
            )

            combined_metrics = [
                get_score(
                    score_data,
                    metric,
                    agg_dims="ipoint",
                    group_by_coord="sample",
                )
                for metric in metrics
            ]

            combined_metrics = xr.concat(combined_metrics, dim="metric")
            combined_metrics["metric"] = metrics

            _logger.debug(f"Running computation of metrics for stream {stream}...")
            combined_metrics = combined_metrics.compute()
            combined_metrics = scalar_coord_to_dim(combined_metrics, "channel")
            combined_metrics = scalar_coord_to_dim(combined_metrics, "sample")
        else:
            # depending on the datset, there might be no data (e.g. no CERRA in southern hemisphere region)
            _logger.warning(
                f"No data available for stream {stream} at forecast step {fstep} in region {region}. Skipping metrics calculation."
            )
            continue

        metric_list.append(combined_metrics)

        metric_stream.loc[{"forecast_step": int(fstep)}] = combined_metrics

    _logger.info(f"Scores for run {reader.run_id} - {stream} calculated successfully.")

    metric_stream = xr.concat(metric_list, dim="forecast_step")
    metric_stream = metric_stream.assign_coords({"forecast_step": fsteps})

    return metric_stream, points_per_sample


def plot_data(reader: Reader, stream: str, global_plotting_opts: dict) -> list[str]:
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

    Returns
    -------
    List of plot names generated during the plotting process.
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
        "plot_subtimesteps": reader.get_inference_stream_attr(
            stream, "tokenize_spacetime", False
        ),
    }

    plotter = Plotter(plotter_cfg, reader.runplot_dir)

    available_data = reader.check_availability(stream, mode="plotting")

    # Check if maps should be plotted and handle configuration if provided
    plot_maps = plot_settings.get("plot_maps", False)
    if not isinstance(plot_maps, bool):
        raise TypeError("plot_maps must be a boolean.")

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
    )

    da_tars = model_output.target
    da_preds = model_output.prediction

    # ensemble spread before selection
    print_ensemble_spread(da_preds, stream)

    # Here we select the first ensemble member if the dimension exists
    for fstep in da_preds.keys():
        if "ens" in da_preds[fstep].dims:
            #da_preds[fstep] = da_preds[fstep].isel(ens=0, drop=True)
            #_logger.info(f"Selected first ensemble member for forecast step {fstep}")
            # or compute the mean: .mean(dim='ens'):
            da_preds[fstep] = da_preds[fstep].mean(dim='ens')
            _logger.info(f"Averaged ensemble members for forecast step {fstep}")
 

    if not da_tars:
        _logger.info(f"Skipping Plot Data for {stream}. Targets are empty.")
        return

    # get common ranges across all run_ids
    if not isinstance(global_plotting_opts.get(stream), oc.DictConfig):
        global_plotting_opts[stream] = oc.DictConfig({})
    maps_config = common_ranges(
        da_tars, da_preds, available_data.channels, global_plotting_opts[stream]
    )

    plot_names = []

    for (fstep, tars), (_, preds) in zip(
        da_tars.items(), da_preds.items(), strict=False
    ):
        plot_chs = list(np.atleast_1d(tars.channel.values))
        plot_samples = list(np.unique(tars.sample.values))

        for sample in tqdm(
            plot_samples, desc=f"Plotting {run_id} - {stream} - fstep {fstep}"
        ):
            plots = []

            data_selection = {
                "sample": sample,
                "stream": stream,
                "forecast_step": fstep,
            }

            if plot_maps:
                map_tar = plotter.create_maps_per_sample(
                    tars, plot_chs, data_selection, "targets", maps_config
                )

                map_pred = plotter.create_maps_per_sample(
                    preds, plot_chs, data_selection, "preds", maps_config
                )
                plots.extend([map_tar, map_pred])

            if plot_histograms:
                h = plotter.create_histograms_per_sample(
                    tars, preds, plot_chs, data_selection
                )
                plots.append(h)

            plotter = plotter.clean_data_selection()

            plot_names.append(plots)

    if plot_animations:
        plot_fsteps = da_tars.keys()
        h = plotter.animation(
            plot_samples, plot_fsteps, plot_chs, data_selection, "preds"
        )
        h = plotter.animation(
            plot_samples, plot_fsteps, plot_chs, data_selection, "targets"
        )

    plot_spread_skill = plot_settings.get("plot_spread_skill", False)
    if plot_spread_skill:
        _logger.info(f"Creating ensemble spread-skill diagnostic maps for {stream}...")
        
        # Need to reload data WITHOUT selecting first ensemble member
        model_output_ens = reader.get_data(
            stream,
            samples=available_data.samples,
            fsteps=available_data.fsteps,
            channels=available_data.channels,
        )
        
        # Check if ensemble dimension exists
        first_pred = list(model_output_ens.prediction.values())[0]
        if "ens" in first_pred.dims:
            spread_skill_plots = plot_spread_skill_maps(reader, stream, global_plotting_opts)
            plot_names.extend(spread_skill_plots)
        else:
            _logger.warning(f"Cannot plot spread-skill maps: no ensemble dimension found in {stream}")


    return plot_names


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
    epoch :
        Epoch number.
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
            metric_now.attrs["npoints_per_sample"] = (
                npoints_sample_stream.values.tolist()
            )
            metric_dict = metric_now.to_dict()

            # Match the expected filename pattern
            save_path = (
                reader.metrics_dir
                / f"{reader.run_id}_{stream}_{region}_{metric}_epoch{reader.epoch:05d}.json"
            )

            _logger.info(f"Saving results to {save_path}")
            with open(save_path, "w") as f:
                json.dump(metric_dict, f, indent=4)

    _logger.info(
        f"Saved all results of inference run {reader.run_id} - epoch {reader.epoch:d} successfully to {reader.metrics_dir}."
    )


def retrieve_metric_from_json(reader: Reader, stream: str, region: str, metric: str):
    """
    Retrieve the score for a given run, stream, metric, epoch, and rank from a JSON file.

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
    xr.DataArray
        The metric DataArray.
    """
    score_path = (
        Path(reader.metrics_dir)
        / f"{reader.run_id}_{stream}_{region}_{metric}_epoch{reader.epoch:05d}.json"
    )
    _logger.debug(f"Looking for: {score_path}")

    if score_path.exists():
        with open(score_path) as f:
            data_dict = json.load(f)
            return xr.DataArray.from_dict(data_dict)
    else:
        raise FileNotFoundError(f"File {score_path} not found in the archive.")


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
    }

    plotter = LinePlots(plot_cfg, summary_dir)

    for region in regions:
        for metric in metrics:
            plot_metric_region(
                metric, region, runs, scores_dict, plotter, print_summary
            )


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
        the global plotting configuration with the ranges added and included for each variable (and for each stream).
    """

    for var in plot_chs:
        if var in maps_config:
            if not isinstance(maps_config[var].get("vmax"), (int | float)):
                list_max = calc_bounds(data_tars, data_preds, var, "max")

                maps_config[var].update({"vmax": float(max(list_max))})

            if not isinstance(maps_config[var].get("vmin"), (int | float)):
                list_min = calc_bounds(data_tars, data_preds, var, "min")

                maps_config[var].update({"vmin": float(min(list_min))})

        else:
            list_max = calc_bounds(data_tars, data_preds, var, "max")

            list_min = calc_bounds(data_tars, data_preds, var, "min")

            maps_config.update(
                {var: {"vmax": float(max(list_max[0])), "vmin": float(min(list_min[0]))}}
            )

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
    Calculate the minimum and maximum values per variable for all forecasteps for both targets and predictions

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


def print_ensemble_spread(da_preds: dict[str, xr.DataArray], stream: str):
    """
    Simple utility to print mean ensemble spread per channel.
    
    Spread is computed as sqrt(mean variance) with Bessel's correction (ddof=1).
    
    Parameters
    ----------
    da_preds : dict
        Dictionary of predictions with forecast steps as keys
    stream : str
        Stream name for logging
    """
    print(f"\n{'='*60}")
    print(f"ENSEMBLE SPREAD STATISTICS - {stream}")
    print(f"{'='*60}")
    
    for fstep, preds in da_preds.items():
        if "ens" in preds.dims:
            # Compute variance across ensemble members with Bessel's correction
            # ddof=1 means divide by (N-1) instead of N
            variance = preds.var(dim="ens", ddof=1)  # [ipoint, channel]
            
            # Mean variance per channel (averaging over all points)
            mean_variance_per_channel = variance.mean(dim="ipoint")  # [channel]
            
            # Spread is sqrt of mean variance
            mean_spread_per_channel = np.sqrt(mean_variance_per_channel)
            
            print(f"\nForecast step: {fstep}")
            print(f"  Number of ensemble members: {preds.sizes['ens']}")
            print(f"  Number of points (all samples): {preds.sizes['ipoint']}")
            print(f"  Using Bessel's correction: variance = sum((x - mean)²) / (N - 1)")
            
            # Print per-channel spreads
            channels = preds.channel.values
            for ch_idx, channel in enumerate(channels):
                ch_spread = mean_spread_per_channel.isel(channel=ch_idx).values.item()
                print(f"  {channel:>15s}: mean spread = {ch_spread:.6f}")
            
            # Overall mean spread (sqrt of mean variance across all channels and points)
            overall_spread = np.sqrt(variance.mean()).values.item()
            print(f"  {'OVERALL':>15s}: mean spread = {overall_spread:.6f}")
        else:
            print(f"\nForecast step: {fstep} - No ensemble dimension found")
    
    print(f"{'='*60}\n")


def plot_spread_skill_maps(
    reader: Reader,
    stream: str,
    global_plotting_opts: dict,
) -> list[str]:
    """
    Plot spread, skill (RMSE), and spread-skill ratio maps for ensemble predictions.
    
    Creates three types of maps:
    1. Ensemble spread (sqrt of variance across members, with Bessel's correction)
    2. Ensemble skill (RMSE of ensemble mean)
    3. Spread-skill ratio (calibration diagnostic)
    
    Spread is computed as sqrt(variance) with ddof=1 (Bessel's correction).
    
    Parameters
    ----------
    reader : Reader
        Reader object containing all info about the run
    stream : str
        Stream name to plot data for
    global_plotting_opts : dict
        Dictionary containing all plotting options
        
    Returns
    -------
    list[str]
        List of plot filenames generated
    """
    _logger.info(f"Creating spread-skill maps for {stream}...")
    
    plotter_cfg = {
        "image_format": global_plotting_opts.get("image_format", "png"),
        "dpi_val": global_plotting_opts.get("dpi_val", 300),
        "fig_size": global_plotting_opts.get("fig_size", (8, 10)),
        "plot_subtimesteps": reader.get_inference_stream_attr(
            stream, "tokenize_spacetime", False
        ),
    }
    
    plotter = Plotter(plotter_cfg, reader.runplot_dir)
    
    available_data = reader.check_availability(stream, mode="plotting")
    
    model_output = reader.get_data(
        stream,
        samples=available_data.samples,
        fsteps=available_data.fsteps,
        channels=available_data.channels,
    )
    
    da_tars = model_output.target
    da_preds = model_output.prediction
    
    plot_names = []
    
    for (fstep, tars), (_, preds) in zip(
        da_tars.items(), da_preds.items(), strict=False
    ):
        if "ens" not in preds.dims:
            _logger.warning(f"No ensemble dimension found for {stream} fstep {fstep}")
            continue
        
        plot_chs = list(np.atleast_1d(tars.channel.values))
        
        # Compute ensemble statistics
        # 1. Spread: sqrt(variance) across ensemble members with Bessel's correction
        variance = preds.var(dim="ens", ddof=1)  # [ipoint, channel], divide by (N-1)
        spread = np.sqrt(variance)  # [ipoint, channel]
        
        # 2. Skill: RMSE of ensemble mean
        ens_mean = preds.mean(dim="ens")  # [ipoint, channel]
        skill = np.sqrt(((ens_mean - tars) ** 2))  # [ipoint, channel]
        
        # 3. Spread-skill ratio (clip to avoid extreme values from near-zero skill)
        spread_skill_ratio = spread / skill  # [ipoint, channel]
        spread_skill_ratio = spread_skill_ratio.clip(max=1.5)
        
        # Set the data selection so plotter knows the stream
        plot_samples = list(np.unique(tars.sample.values)) if "sample" in tars.dims else []
        data_selection = {
            "stream": stream,
            "forecast_step": fstep,
        }
        if plot_samples:
            data_selection["sample"] = plot_samples[0]
        
        plotter.update_data_selection(data_selection)
        
        # Initialize global plotting config if needed
        if not isinstance(global_plotting_opts.get(stream), oc.DictConfig):
            global_plotting_opts[stream] = oc.DictConfig({})
        
        # Create maps for each channel
        for channel in plot_chs:
            # Calculate common vmin/vmax for spread and skill to ensure consistent scales
            spread_ch = spread.sel(channel=channel)  # [ipoint]
            skill_ch = skill.sel(channel=channel)  # [ipoint]
            
            # Compute common range (use max of both for vmax, 0 for vmin)
            vmin_common = 0.0
            vmax_common = max(
                float(spread_ch.max().values),
                float(skill_ch.max().values)
            )
            
            # Add some padding to vmax (5%)
            vmax_common *= 1.05
            
            _logger.info(f"Common colorbar range for {channel}: [{vmin_common:.4f}, {vmax_common:.4f}]")
            
            # ← REMOVED: Don't modify global_plotting_opts to avoid conflicts
            # Just pass settings directly via map_kwargs
            
            # 1. Plot SPREAD map
            map_output_dir = plotter.get_map_output_dir("ensemble_spread")
            if not map_output_dir.exists():
                map_output_dir.mkdir(parents=True, exist_ok=True)
            
            spread_plot = plotter.scatter_plot(
                spread_ch,
                map_output_dir,
                channel,
                tag=f"spread_fstep{fstep}",
                map_kwargs={
                    "vmin": vmin_common,
                    "vmax": vmax_common,
                },
            )
            plot_names.append(spread_plot)
            _logger.info(f"Created spread map for {stream} - {channel} - fstep {fstep}")
            
            # 2. Plot SKILL (RMSE) map with same range
            map_output_dir = plotter.get_map_output_dir("ensemble_skill")
            if not map_output_dir.exists():
                map_output_dir.mkdir(parents=True, exist_ok=True)
            
            skill_plot = plotter.scatter_plot(
                skill_ch,
                map_output_dir,
                channel,
                tag=f"skill_fstep{fstep}",
                map_kwargs={
                    "vmin": vmin_common,
                    "vmax": vmax_common,
                },
            )
            plot_names.append(skill_plot)
            _logger.info(f"Created skill map for {stream} - {channel} - fstep {fstep}")
            
            # 3. Plot SPREAD-SKILL RATIO map
            map_output_dir = plotter.get_map_output_dir("spread_skill_ratio")
            if not map_output_dir.exists():
                map_output_dir.mkdir(parents=True, exist_ok=True)
            
            ratio_plot = plotter.scatter_plot(
                spread_skill_ratio.sel(channel=channel),
                map_output_dir,
                channel,
                tag=f"spread_skill_ratio_fstep{fstep}",
                map_kwargs={
                    "vmin": 0.0,
                    "vmax": 1.2,
                },
            )
            plot_names.append(ratio_plot)
            _logger.info(f"Created spread-skill ratio map for {stream} - {channel} - fstep {fstep}")
        
        # Clean up data selection after each forecast step
        plotter.clean_data_selection()
    
    return plot_names