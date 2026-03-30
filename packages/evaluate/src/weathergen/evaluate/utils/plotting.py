# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Plotting helpers: per-sample maps, score maps, and summary plots."""

import contextlib
import logging
import os
import resource
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import numpy as np
import omegaconf as oc
import xarray as xr
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from tqdm import tqdm

from weathergen.evaluate.io.io_reader import Reader, ReaderOutput
from weathergen.evaluate.plotting.plot_utils import (
    bar_plot_metric_region,
    heat_maps_metric_region,
    plot_metric_region,
    quantile_plot_metric_region,
    ratio_plot_metric_region,
    score_card_metric_region,
)
from weathergen.evaluate.plotting.plotter import (
    BarPlots,
    LinePlots,
    Plotter,
    QuantilePlots,
    ScoreCards,
)
from weathergen.evaluate.scores.score import VerifiedData, get_score
from weathergen.evaluate.utils.array_utils import bias_ranges, common_ranges
from weathergen.evaluate.utils.clim_utils import get_climatology
from weathergen.evaluate.utils.regions import RegionBoundingBox
from weathergen.evaluate.utils.scoring import get_next_data

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Process headroom helper
# ---------------------------------------------------------------------------


def _resolve_num_plot_workers(requested: int = 0) -> int:
    """Return a safe number of parallel plot workers.

    Parameters
    ----------
    requested : int
        Value from config (``num_plot_workers``). ``0`` means auto-detect.
    """
    if requested > 0:
        return min(requested, os.cpu_count() or 12)

    try:
        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NPROC)
        if soft_limit == resource.RLIM_INFINITY:
            soft_limit = 65536

        result = subprocess.run(
            ["ps", "-u", str(os.getuid()), "--no-headers", "-o", "pid"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        user_procs = len(result.stdout.strip().splitlines()) if result.returncode == 0 else 0

        available = soft_limit - user_procs
        if available < 64:
            _logger.info(
                f"Parallel plotting: low process headroom "
                f"({available}/{soft_limit} slots free). Using sequential plotting."
            )
            return 1

        n = min(available // 8, os.cpu_count() or 12, 12)
        n = max(n, 1)
        _logger.info(
            f"Parallel plotting: process headroom {available}/{soft_limit} free. "
            f"Using num_plot_workers={n}."
        )
        return n

    except Exception as e:
        _logger.debug(
            f"Could not auto-detect process limits for plotting ({e}). Defaulting to sequential."
        )
        return 1


# ---------------------------------------------------------------------------
# Score maps
# ---------------------------------------------------------------------------


def plot_score_maps_per_stream(
    reader: Reader,
    stream: str,
    regions: list[str],
    metrics_dict: dict,
    output_data: "ReaderOutput | None" = None,
) -> None:
    """Plot spatial score maps for all regions and forecast steps.

    Parameters
    ----------
    reader : Reader
        Reader object containing all info about a particular run.
    stream : str
        Stream name to plot score maps for.
    regions : list[str]
        List of regions to plot.
    metrics_dict : dict
        Dictionary mapping region names to metric dicts.
    output_data : ReaderOutput | None
        Pre-loaded data; when provided ``reader.get_data()`` is skipped.
    """
    if not reader.is_gridded_data(stream):
        _logger.debug(f"RUN {reader.run_id} - {stream}: Skipping score maps (non-gridded data).")
        return

    map_dir = reader.runplot_dir / "plots" / stream / "score_maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    _logger.info(f"RUN {reader.run_id} - {stream}: Plotting score maps → {map_dir}")

    available_data = reader.check_availability(stream, mode="evaluation")
    fsteps = available_data.fsteps
    samples = available_data.samples
    channels = available_data.channels
    ensemble = available_data.ensemble

    if output_data is None:
        output_data = reader.get_data(
            stream, fsteps=fsteps, samples=samples, channels=channels, ensemble=ensemble
        )

    da_preds = output_data.prediction
    da_tars = output_data.target
    fsteps = sorted(da_preds.keys())
    aligned_clim_data = get_climatology(reader, da_tars, stream)

    n_plot_workers = _resolve_num_plot_workers(int(reader.eval_cfg.get("num_plot_workers", 0)))

    cfg = reader.global_plotting_options
    plotter_cfg = {
        "image_format": cfg.get("image_format", "png"),
        "dpi_val": cfg.get("dpi_val", 300),
        "fig_size": cfg.get("fig_size", (8, 10)),
    }
    output_basedir = str(reader.runplot_dir)
    run_id = reader.run_id

    fstep_tasks: list[dict] = []
    for region in regions:
        bbox = RegionBoundingBox.from_region_name(region)
        metrics = metrics_dict[region]
        for fstep in fsteps:
            tars_fs = da_tars[fstep]
            preds_fs = da_preds[fstep]
            preds_next, tars_next = get_next_data(fstep, da_preds, da_tars, fsteps)
            climatology = aligned_clim_data[fstep] if aligned_clim_data else None
            tars_r, preds_r, tars_next_r, preds_next_r = [
                bbox.apply_mask(x) if x is not None else None
                for x in (tars_fs, preds_fs, tars_next, preds_next)
            ]
            score_data = VerifiedData(preds_r, tars_r, preds_next_r, tars_next_r, climatology)
            fstep_tasks.append(
                {
                    "plotter_cfg": plotter_cfg,
                    "output_basedir": output_basedir,
                    "map_dir": str(map_dir),
                    "stream": stream,
                    "region": region,
                    "score_data": score_data,
                    "metrics": dict(metrics),
                    "fstep": fstep,
                    "run_id": run_id,
                }
            )

    n_tasks = len(fstep_tasks)
    effective = min(n_plot_workers, n_tasks)
    _logger.info(
        f"RUN {run_id} - {stream}: Plotting {n_tasks} score-map tasks "
        f"({len(regions)} region(s) × {len(fsteps)} fstep(s)) "
        f"with {effective} worker(s)."
    )

    if effective > 1 and n_tasks > 1:
        try:
            Parallel(n_jobs=effective, backend="loky", verbose=2)(
                delayed(_score_map_fstep_worker)(**t) for t in fstep_tasks
            )
            with contextlib.suppress(Exception):
                get_reusable_executor().shutdown(wait=True)
        except Exception as exc:
            _logger.warning(
                f"Parallel score-map fstep dispatch failed "
                f"({type(exc).__name__}: {exc}). Falling back to sequential."
            )
            with contextlib.suppress(Exception):
                get_reusable_executor().shutdown(wait=True)
            for t in tqdm(fstep_tasks, desc=f"Score maps {stream} (sequential)"):
                _score_map_fstep_worker(**t)
    else:
        for t in tqdm(fstep_tasks, desc=f"Score maps {stream}"):
            _score_map_fstep_worker(**t)


def _score_map_fstep_worker(
    plotter_cfg: dict,
    output_basedir: str,
    map_dir: str,
    stream: str,
    region: str,
    score_data: "VerifiedData",
    metrics: dict,
    fstep: int,
    run_id: str,
) -> None:
    """Module-level loky worker: compute scores + plot maps for one (region, fstep)."""
    _plot_score_maps_per_stream(
        plotter_cfg=plotter_cfg,
        output_basedir=output_basedir,
        map_dir=map_dir,
        stream=stream,
        region=region,
        score_data=score_data,
        metrics=metrics,
        fstep=fstep,
        run_id=run_id,
    )


def _plot_score_maps_per_stream(
    plotter_cfg: dict,
    output_basedir: str,
    map_dir: str,
    stream: str,
    region: str,
    score_data: "VerifiedData",
    metrics: dict[str, object],
    fstep: int,
    run_id: str = "",
) -> None:
    """Plot 2D score maps for all metrics/channels for one (region, fstep)."""
    preds = score_data.prediction

    metric_names = list(metrics.keys())
    metric_params = list(metrics.values())
    score_results: list[xr.DataArray | None] = [None] * len(metric_names)
    with ThreadPoolExecutor(max_workers=min(12, len(metric_names))) as executor:
        future_to_idx = {
            executor.submit(get_score, score_data, m, agg_dims="sample", parameters=p): i
            for i, (m, p) in enumerate(zip(metric_names, metric_params, strict=False))
        }
        for future in as_completed(future_to_idx):
            score_results[future_to_idx[future]] = future.result()

    valid = [(m, r) for m, r in zip(metric_names, score_results, strict=False) if r is not None]
    if not valid:
        return

    plot_metrics = xr.concat(
        [r for _, r in valid],
        dim="metric",
        coords="minimal",
        combine_attrs="drop_conflicts",
    )
    plot_metrics = plot_metrics.assign_coords(
        lat=preds.lat.reset_coords(drop=True),
        lon=preds.lon.reset_coords(drop=True),
        metric=[m for m, _ in valid],
    ).compute()

    if "ens" in preds.dims:
        plot_metrics["ens"] = preds.ens

    has_ens = "ens" in plot_metrics.coords
    ens_values = plot_metrics.coords["ens"].values if has_ens else [None]

    plot_tasks: list[dict] = []
    for metric in plot_metrics.coords["metric"].values:
        for ens_val in ens_values:
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
                plot_tasks.append(
                    {
                        "plotter_cfg": plotter_cfg,
                        "output_basedir": output_basedir,
                        "stream": stream,
                        "data": data,
                        "map_dir": str(map_dir),
                        "channel": str(channel),
                        "region": region,
                        "tag": tag,
                        "title": title,
                    }
                )

    for t in plot_tasks:
        _scatter_plot_single(**t)


def _scatter_plot_single(
    plotter_cfg: dict,
    output_basedir: str,
    stream: str,
    data: xr.DataArray,
    map_dir: str,
    channel: str,
    region: str,
    tag: str,
    title: str,
) -> None:
    """Plot a single score-map scatter plot (picklable for loky workers)."""
    matplotlib.use("Agg")
    plotter = Plotter(plotter_cfg, Path(output_basedir), stream)
    plotter.scatter_plot(data, Path(map_dir), channel, region, tag=tag, title=title)


# ---------------------------------------------------------------------------
# Per-sample map / histogram plots
# ---------------------------------------------------------------------------


def _plot_single_sample(
    plotter_cfg: dict,
    output_basedir: str,
    tars: xr.DataArray,
    preds: xr.DataArray,
    bias_data: xr.DataArray | None,
    sample: int | str,
    fstep: int | str,
    stream: str,
    plot_chs: list[str],
    ensemble: list,
    plot_maps: bool,
    plot_bias: bool,
    plot_target: bool,
    plot_histograms: bool,
    maps_config: dict,
    bias_config: dict,
) -> None:
    """Plot all maps/histograms for a single (fstep, sample) pair (loky worker)."""
    matplotlib.use("Agg")

    maps_cfg = oc.OmegaConf.create(maps_config)
    bias_cfg = oc.OmegaConf.create(bias_config)
    plotter = Plotter(plotter_cfg, Path(output_basedir))

    data_selection = {"sample": sample, "stream": stream, "forecast_step": fstep}

    if plot_maps:
        if plot_target:
            plotter.create_maps_per_sample(tars, plot_chs, data_selection, "targets", maps_cfg)

        if plot_bias and bias_data is not None:
            plotter.create_maps_per_sample(bias_data, plot_chs, data_selection, "bias", bias_cfg)

        for ens in ensemble:
            preds_ens = preds.sel(ens=ens) if "ens" in preds.dims and ens != "mean" else preds
            preds_tag = "" if "ens" not in preds.dims else f"ens_{ens}"
            preds_name = "_".join(filter(None, ["preds", preds_tag]))
            plotter.create_maps_per_sample(
                preds_ens, plot_chs, data_selection, preds_name, maps_cfg
            )

            if plot_histograms:
                plotter.create_histograms_per_sample(
                    tars, preds_ens, plot_chs, data_selection, preds_tag
                )

    plotter.clean_data_selection()


def plot_data(
    reader: Reader,
    stream: str,
    global_plotting_opts: dict,
    output_data: ReaderOutput | None = None,
) -> None:
    """Plot prediction/target maps and histograms for a given run and stream.

    Parameters
    ----------
    reader : Reader
        Reader object containing all info about the run.
    stream : str
        Stream name to plot data for.
    global_plotting_opts : dict
        Global plotting options (applies to all run_ids).
    output_data : ReaderOutput | None
        Pre-loaded data; when provided ``reader.get_data()`` is skipped.
    """
    run_id = reader.run_id
    stream_cfg = reader.get_stream(stream)
    plot_settings = stream_cfg.get("plotting", {})

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
        "plot_subtimesteps": reader.get_inference_stream_attr(stream, "tokenize_spacetime", False)
        | plot_settings.get("plot_subtimesteps", False),
    }
    plotter = Plotter(plotter_cfg, reader.runplot_dir)

    available_data = reader.check_availability(stream, mode="plotting")

    plot_maps = plot_settings.get("plot_maps", False)
    if not isinstance(plot_maps, bool):
        raise TypeError("plot_maps must be a boolean.")
    plot_bias = plot_settings.get("plot_bias", True)
    if not isinstance(plot_bias, bool):
        raise TypeError("plot_bias must be a boolean.")
    plot_target = plot_settings.get("plot_target", True)
    if not isinstance(plot_target, bool):
        raise TypeError("plot_target must be a boolean.")
    plot_histograms = plot_settings.get("plot_histograms", False)
    if not isinstance(plot_histograms, bool):
        raise TypeError("plot_histograms must be a boolean.")
    plot_animations = plot_settings.get("plot_animations", False)
    if not isinstance(plot_animations, bool):
        raise TypeError("plot_animations must be a boolean.")

    if output_data is None:
        model_output = reader.get_data(
            stream,
            samples=available_data.samples,
            fsteps=available_data.fsteps,
            channels=available_data.channels,
            ensemble=available_data.ensemble,
        )
    else:
        model_output = output_data

    da_tars = model_output.target
    da_preds = model_output.prediction

    if not da_tars:
        _logger.info(f"Skipping Plot Data for {stream}. Targets are empty.")
        return

    plot_fstep_set = set(available_data.fsteps) if available_data.fsteps is not None else None
    plot_sample_set = set(available_data.samples) if available_data.samples is not None else None
    plot_channel_set = (
        set(available_data.channels) if available_data.channels is not None else None
    )

    output_fstep_keys = set(da_tars.keys())
    if plot_fstep_set is not None and output_fstep_keys - plot_fstep_set:
        zarr_fsteps = set(int(f) for f in reader.get_forecast_steps())
        if plot_fstep_set == zarr_fsteps:
            _logger.debug(
                f"Sub-step expansion detected: output has {len(output_fstep_keys)} "
                f"entries vs {len(zarr_fsteps)} zarr fsteps. "
                f"Expanding plotting filter to all output fsteps."
            )
            plot_fstep_set = output_fstep_keys

    if plot_fstep_set is not None:
        da_tars = {fs: da for fs, da in da_tars.items() if fs in plot_fstep_set}
        da_preds = {fs: da for fs, da in da_preds.items() if fs in plot_fstep_set}

    if not da_tars:
        _logger.info(f"Skipping Plot Data for {stream}. No matching fsteps after filtering.")
        return

    if not isinstance(global_plotting_opts.get(stream), oc.DictConfig):
        global_plotting_opts[stream] = oc.DictConfig({})
    maps_config = common_ranges(
        da_tars, da_preds, available_data.channels, global_plotting_opts[stream]
    )
    bias_config = bias_ranges(
        da_tars, da_preds, available_data.channels, global_plotting_opts[stream]
    )

    maps_config_dict = oc.OmegaConf.to_container(maps_config, resolve=True)
    bias_config_dict = oc.OmegaConf.to_container(bias_config, resolve=True)
    output_basedir = str(reader.runplot_dir)

    num_plot_workers = _resolve_num_plot_workers(int(reader.eval_cfg.get("num_plot_workers", 0)))

    tasks: list[dict] = []
    for (fstep, tars), (_, preds) in zip(da_tars.items(), da_preds.items(), strict=False):
        all_chs = list(np.atleast_1d(tars.channel.values))
        plot_chs = (
            [ch for ch in all_chs if ch in plot_channel_set]
            if plot_channel_set is not None
            else all_chs
        )
        if not plot_chs:
            continue

        all_samples = list(np.unique(tars.sample.values))
        plot_samples = (
            [s for s in all_samples if s in plot_sample_set]
            if plot_sample_set is not None
            else all_samples
        )
        if not plot_samples:
            continue

        bias_data = (preds - tars) if plot_bias else None

        for sample in plot_samples:
            tasks.append(
                {
                    "plotter_cfg": plotter_cfg,
                    "output_basedir": output_basedir,
                    "tars": tars,
                    "preds": preds,
                    "bias_data": bias_data,
                    "sample": sample,
                    "fstep": fstep,
                    "stream": stream,
                    "plot_chs": plot_chs,
                    "ensemble": list(available_data.ensemble),
                    "plot_maps": plot_maps,
                    "plot_bias": plot_bias,
                    "plot_target": plot_target,
                    "plot_histograms": plot_histograms,
                    "maps_config": maps_config_dict,
                    "bias_config": bias_config_dict,
                }
            )

    effective_workers = min(num_plot_workers, len(tasks))

    if effective_workers > 1 and len(tasks) > 1:
        _logger.info(
            f"Parallel plotting: dispatching {len(tasks)} (fstep, sample) tasks "
            f"across {effective_workers} loky workers."
        )
        try:
            Parallel(n_jobs=effective_workers, backend="loky", verbose=2)(
                delayed(_plot_single_sample)(**task) for task in tasks
            )
            with contextlib.suppress(Exception):
                get_reusable_executor().shutdown(wait=True)
        except Exception as exc:
            _logger.warning(
                f"Parallel plotting failed ({type(exc).__name__}: {exc}). "
                f"Falling back to sequential plotting."
            )
            with contextlib.suppress(Exception):
                get_reusable_executor().shutdown(wait=True)
            for task in tqdm(tasks, desc=f"Plotting {run_id} - {stream} (sequential fallback)"):
                _plot_single_sample(**task)
    else:
        for task in tqdm(tasks, desc=f"Plotting {run_id} - {stream}"):
            _plot_single_sample(**task)

    if plot_animations:
        plotter = Plotter(plotter_cfg, reader.runplot_dir)
        last_fstep = list(da_tars.keys())[-1]
        last_tars = da_tars[last_fstep]
        last_preds = da_preds[last_fstep]
        all_chs = list(np.atleast_1d(last_tars.channel.values))
        plot_chs = (
            [ch for ch in all_chs if ch in plot_channel_set]
            if plot_channel_set is not None
            else all_chs
        )
        all_samples = list(np.unique(last_tars.sample.values))
        plot_samples = (
            [s for s in all_samples if s in plot_sample_set]
            if plot_sample_set is not None
            else all_samples
        )
        plot_fsteps = da_tars.keys()
        data_selection = {
            "sample": plot_samples[-1],
            "stream": stream,
            "forecast_step": last_fstep,
        }
        for ens in available_data.ensemble:
            preds_name = "preds" if "ens" not in last_preds.dims else f"preds_ens_{ens}"
            plotter.animation(plot_samples, plot_fsteps, plot_chs, data_selection, preds_name)
        if plot_target:
            plotter.animation(plot_samples, plot_fsteps, plot_chs, data_selection, "targets")
        if plot_bias:
            plotter.animation(plot_samples, plot_fsteps, plot_chs, data_selection, "bias")


# ---------------------------------------------------------------------------
# Summary plots
# ---------------------------------------------------------------------------


def plot_summary(cfg: dict, scores_dict: dict, summary_dir: Path):
    """Plot summary of the evaluation results.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.
    scores_dict : dict
        Scores for each metric and stream.
    summary_dir : Path
        Directory to write plots to.
    """
    runs = cfg.run_ids
    metrics = cfg.evaluation.metrics
    print_summary = cfg.evaluation.get("print_summary", False)
    regions = cfg.evaluation.get("regions", ["global"])
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
    }

    plotter = LinePlots(plot_cfg, summary_dir)
    sc_plotter = ScoreCards(plot_cfg, summary_dir)
    br_plotter = BarPlots(plot_cfg, summary_dir)
    quantile_plotter = QuantilePlots(plot_cfg, summary_dir)
    plotting_log_emitted = False
    for region in regions:
        for metric in metrics:
            if eval_opt.get("summary_plots", True):
                plot_metric_region(metric, region, runs, scores_dict, plotter, print_summary)
            if eval_opt.get("ratio_plots", False):
                ratio_plot_metric_region(metric, region, runs, scores_dict, plotter, print_summary)
            if eval_opt.get("heat_maps", False):
                heat_maps_metric_region(metric, region, runs, scores_dict, plotter)
            if eval_opt.get("score_cards", False):
                if not plotting_log_emitted:
                    _logger.info(f"Saving score cards to: {summary_dir}")
                score_card_metric_region(metric, region, runs, scores_dict, sc_plotter)
            if eval_opt.get("bar_plots", False):
                if not plotting_log_emitted:
                    _logger.info(f"Saving bar plots to: {summary_dir}")
                bar_plot_metric_region(metric, region, runs, scores_dict, br_plotter)
            if metric == "qq_analysis":
                if not plotting_log_emitted:
                    _logger.info(f"Saving quantile plots to: {summary_dir}")
                quantile_plot_metric_region(metric, region, runs, scores_dict, quantile_plotter)
            plotting_log_emitted = True
