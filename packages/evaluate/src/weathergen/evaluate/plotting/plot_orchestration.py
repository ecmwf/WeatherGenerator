# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Plotting orchestration: parallel dispatch of per-sample maps, score maps, and summary plots."""

import logging
from pathlib import Path

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import omegaconf as oc
import xarray as xr
from joblib import delayed
from PIL import Image
from tqdm import tqdm

from weathergen.evaluate.io.data.io_orchestration import dispatch_parallel, get_num_workers
from weathergen.evaluate.io.io_reader import Reader, ReaderOutput
from weathergen.evaluate.plotting.bar_plots import BarPlots
from weathergen.evaluate.plotting.line_plots import LinePlots
from weathergen.evaluate.plotting.plot_orchestration_utils import (
    _compute_ranges,
    _compute_scores,
    group_by_init_hour,
)
from weathergen.evaluate.plotting.plot_utils import (
    bar_plot_metric_region,
    heat_maps_metric_region,
    plot_metric_region,
    psd_plot_metric_region,
    quantile_plot_metric_region,
    ratio_plot_metric_region,
    score_card_metric_region,
)
from weathergen.evaluate.plotting.plotter import Plotter
from weathergen.evaluate.plotting.quantile_plots import QuantilePlots
from weathergen.evaluate.plotting.score_cards import ScoreCards
from weathergen.evaluate.scores.score import VerifiedData, get_score
from weathergen.evaluate.utils.array_utils import bias_ranges, common_ranges
from weathergen.evaluate.utils.clim_utils import get_climatology, needs_climatology

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# timeseries
# ---------------------------------------------------------------------------
def run_score_timeseries_pipeline(
    reader: Reader,
    stream: str,
    regions: list[str],
    metrics_dict: dict,
    output_data: "ReaderOutput | None" = None,
    global_plotting_options: dict | None = None,
) -> dict[str, dict[str, dict[int, xr.DataArray]]]:
    """Plot timeseries of score values across forecast steps for all regions.

    Parameters
    ----------
    reader : Reader
        Reader object containing all info about a particular run.
    stream : str
        Stream name to plot score timeseries for.
    regions : list[str]
        List of regions to plot.
    metrics_dict : dict
        Dictionary mapping region names to metric dicts.
    output_data : ReaderOutput | None
        Pre-loaded data; when provided ``reader.get_data()`` is skipped.
    global_plotting_options : dict | None
        Global plotting options. These can be passed to the plotter and can be used to set options.

    Returns
    -------
    dict[str, dict[str, dict[int, xr.DataArray]]]
        Nested dict: ``scores_by_hour[metric][region][fstep] = xr.DataArray`` with
        dims ``(source_end_hour, channel)``.  Returns empty dict on failure.
    """

    available_data = reader.check_availability(stream, mode="evaluation")
    if not available_data.score_availability:
        _logger.warning(f"RUN {reader.run_id} - {stream}: No data available for score timeseries.")
        return {}

    da_tars = output_data.target
    da_preds = output_data.prediction
    if not da_tars:
        return {}

    # Group samples by hour of day of source_interval_end
    hour_to_samples = group_by_init_hour(output_data)
    if not hour_to_samples:
        _logger.warning(f"RUN {reader.run_id} - {stream}: Could not group by init hour. Skipping.")
        return {}

    unique_hours = sorted(hour_to_samples.keys())
    fsteps = sorted(da_preds.keys())

    n_workers = get_num_workers(
        check_process_headroom=True,
        max_workers=reader.eval_cfg.get("max_workers", None),
    )

    # --- Parallel score computation across (region, fstep) pairs ---
    score_tasks: list[dict] = []
    for fstep in fsteps:
        preds_fs = da_preds[fstep]
        tars_fs = da_tars[fstep]

        # Assign source_end_hour coordinate from the grouping
        hour_values = np.full(len(preds_fs.sample), -1, dtype=int)
        sample_vals = preds_fs.sample.values
        for hour, sample_indices in hour_to_samples.items():
            mask = np.isin(sample_vals, sample_indices)
            hour_values[mask] = hour

        source_end_hour = xr.DataArray(
            hour_values, dims=("sample",), coords={"sample": sample_vals}
        )
        preds_with_hour = preds_fs.assign_coords(source_end_hour=source_end_hour)
        tars_with_hour = tars_fs.assign_coords(source_end_hour=source_end_hour)

        for region in regions:
            region_metrics = metrics_dict.get(region)
            metric_names = list(region_metrics.keys())
            metric_params = list(region_metrics.values())
            score_tasks.append(
                dict(
                    fstep=fstep,
                    region=region,
                    metric_names=metric_names,
                    metric_params=metric_params,
                    preds_with_hour=preds_with_hour,
                    tars_with_hour=tars_with_hour,
                    unique_hours=unique_hours,
                )
            )

    _logger.info(
        f"RUN {reader.run_id} - {stream}: Computing score timeseries for "
        f"{len(score_tasks)} (region, fstep) tasks with up to {n_workers} worker(s)."
    )

    calls = [delayed(_compute_timeseries_scores_for_fstep)(**t) for t in score_tasks]
    raw_results = dispatch_parallel(
        calls, n_workers=n_workers, backend="loky", desc=f"Score timeseries {stream}"
    )

    # Accumulate results into scores_by_hour[metric][region][fstep]
    scores_by_hour: dict[str, dict[str, dict[int, xr.DataArray]]] = {}
    for fstep, region, metric_scores in raw_results:
        for metric_name, score in metric_scores.items():
            scores_by_hour.setdefault(metric_name, {}).setdefault(region, {})[fstep] = score

    _logger.info(
        f"RUN {reader.run_id} - {stream}: Score timeseries computed for "
        f"{len(fsteps)} fsteps × {len(unique_hours)} init hours."
    )

    return scores_by_hour


def _compute_timeseries_scores_for_fstep(
    fstep: int,
    region: str,
    metric_names: list[str],
    metric_params: list,
    preds_with_hour: xr.DataArray,
    tars_with_hour: xr.DataArray,
    unique_hours: list[int],
) -> tuple[int, str, dict[str, xr.DataArray]]:
    """Compute grouped scores for one (region, fstep) pair (parallelisable worker).

    Returns ``(fstep, region, {metric_name: score_da})``.
    """
    group_by_coord = "source_end_hour" if len(unique_hours) > 1 else None
    agg_dims = ["sample", "ipoint"]

    metric_scores: dict[str, xr.DataArray] = {}
    for metric_name, parameters in zip(metric_names, metric_params, strict=False):
        score = get_score(
            VerifiedData(preds_with_hour, tars_with_hour, None, None, None),
            metric_name,
            agg_dims=agg_dims,
            group_by_coord=group_by_coord,
            compute=True,
            parameters=parameters,
        )
        if group_by_coord is None:
            score = score.expand_dims({"source_end_hour": [int(unique_hours[0])]})
        metric_scores[metric_name] = score

    return fstep, region, metric_scores


# ---------------------------------------------------------------------------
# Score maps
# ---------------------------------------------------------------------------


def run_score_map_pipeline(
    reader: Reader,
    stream: str,
    regions: list[str],
    metrics_dict: dict,
    output_data: "ReaderOutput | None" = None,
    global_plotting_options: dict | None = None,
    plot_score_options: dict | None = None,
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
    global_plotting_options : dict | None
        Global plotting options. These can be passed to the plotter and can be used to set options.
    plot_score_options : dict | None
        Dictionary containing all common score calculation options.
    """

    if not reader.is_gridded_data(stream):
        _logger.debug(f"RUN {reader.run_id} - {stream}: Skipping score maps (non-gridded data).")
        return

    map_dir = reader.runplot_dir / "plots" / stream / "score_maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    _logger.info(f"RUN {reader.run_id} - {stream}: Saving score maps to {map_dir}")

    available_data = reader.check_availability(stream, mode="evaluation")
    if not available_data.score_availability:
        _logger.warning(
            f"RUN {reader.run_id} - {stream}: No evaluation config. Skipping score maps."
        )
        return
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
    needs_clim = needs_climatology(metrics_dict)
    aligned_clim_data = get_climatology(reader, da_tars, stream) if needs_clim else None

    n_plot_workers = get_num_workers(
        check_process_headroom=True,
        max_workers=reader.eval_cfg.get("max_workers", None),
    )

    cfg = global_plotting_options
    plotter_cfg = {
        "image_format": cfg.get("image_format", "png"),
        "dpi_val": cfg.get("dpi_val", 300),
        "fig_size": cfg.get("fig_size", None),
        "animation_format": cfg.get("animation_format", "gif"),
        "fps": cfg.get("fps", 2),
        "log_colorbar": cfg.get("log_colorbar", False),
    }
    output_basedir = str(reader.runplot_dir)
    run_id = reader.run_id

    _computed, raw_results = _compute_scores(
        regions,
        metrics_dict,
        fsteps,
        da_preds,
        da_tars,
        aligned_clim_data,
        n_workers=n_plot_workers,
    )

    score_ranges_dict = _compute_ranges(raw_results)

    fstep_tasks: list[dict] = []
    for region in regions:
        for fstep in fsteps:
            fstep_tasks.append(
                {
                    "plotter_cfg": plotter_cfg,
                    "score_ranges_dict": score_ranges_dict,
                    "output_basedir": output_basedir,
                    "map_dir": str(map_dir),
                    "stream": stream,
                    "region": region,
                    "computed": _computed[(region, fstep)],
                    "fstep": fstep,
                    "run_id": run_id,
                }
            )

    _logger.info(
        f"RUN {run_id} - {stream}: Plotting {len(fstep_tasks)} score-map tasks "
        f"({len(regions)} region(s) × {len(fsteps)} fstep(s)) "
        f"with up to {n_plot_workers} worker(s)."
    )

    calls = [delayed(_plot_score_maps_per_stream)(**t) for t in fstep_tasks]
    dispatch_parallel(calls, n_workers=n_plot_workers, backend="loky", desc=f"Score maps {stream}")

    plot_score_animations = plot_score_options.get("score_animation", False)
    if plot_score_animations:
        _dispatch_score_map_animations(
            map_dir=map_dir,
            plotter_cfg=plotter_cfg,
            run_id=run_id,
            stream=stream,
            metrics=list(dict.fromkeys(m for metrics in metrics_dict.values() for m in metrics)),
            regions=regions,
            variables=channels,
            ens_values=list(ensemble) if ensemble else [None],
            fsteps=fsteps,
            n_workers=n_plot_workers,
        )


def _plot_score_maps_per_stream(
    plotter_cfg: dict,
    score_ranges_dict: dict,
    output_basedir: str,
    map_dir: str,
    stream: str,
    region: str,
    computed: tuple[list, xr.DataArray, list[str]],
    fstep: int,
    run_id: str = "",
) -> None:
    """Plot 2D score maps for all metrics/channels for one (region, fstep)."""

    score_results, preds, metric_names = computed
    valid = [
        (m, r)
        for m, r in zip(metric_names, score_results, strict=False)
        if r is not None and "ipoint" in r.dims
    ]
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
            tag = "score_maps" + (f"_ens_{ens_val}" if ens_val is not None else "") + f"_{metric}"
            for channel in plot_metrics.coords["channel"].values:
                sel = {"metric": metric, "channel": channel}
                if ens_val is not None:
                    sel["ens"] = ens_val
                data = plot_metrics.sel(**sel).squeeze()
                title = f"{metric} - {channel}: fstep {fstep}" + (
                    f", ens {ens_val}" if ens_val is not None else ""
                )
                scores_cfg = score_ranges_dict.get(metric, {}).get(region, {}).get(channel, {})
                plot_tasks.append(
                    {
                        "plotter_cfg": plotter_cfg,
                        "scores_cfg": scores_cfg,
                        "output_basedir": output_basedir,
                        "stream": stream,
                        "data": data,
                        "map_dir": str(map_dir),
                        "channel": str(channel),
                        "region": region,
                        "fstep": fstep,
                        "tag": tag,
                        "title": title,
                    }
                )

    for t in plot_tasks:
        _scatter_plot_single(**t)


def _scatter_plot_single(
    plotter_cfg: dict,
    scores_cfg: dict,
    output_basedir: str,
    stream: str,
    data: xr.DataArray,
    map_dir: str,
    channel: str,
    region: str,
    fstep: int,
    tag: str,
    title: str,
) -> None:
    """Plot a single score-map scatter plot (picklable for loky workers)."""
    matplotlib.use("Agg")
    plotter = Plotter(plotter_cfg, Path(output_basedir), stream)
    plotter.update_data_selection({"sample": None, "stream": stream, "forecast_step": fstep})
    plotter.scatter_plot(
        data, Path(map_dir), channel, region, tag=tag, map_kwargs=scores_cfg, title=title
    )


# ---------------------------------------------------------------------------
# Animations
# ---------------------------------------------------------------------------


def _build_single_animation(
    output_dir: Path,
    run_id: str,
    tag: str,
    stream: str,
    region: str | None,
    var: str,
    sample: object,
    fsteps: list,
    image_format: str,
    animation_format: str,
    duration_ms: int,
    prefix: str = "map",
) -> list[str]:
    """Build one animation for a single (region, sample/ens, variable) combination.

    All work is I/O + Pillow — no matplotlib state involved.

    The function scans ``output_dir`` for per-sample map/histogram frames whose filenames follow:

        {prefix}_{run_id}_{tag}_{sample}_{valid_time}_{stream}_{region}_{var}_{fstep:03d}

    When ``score_animation=True`` filenames are constructed deterministically because
    the fstep is embedded in the tag (``score_maps_{metric}_fstep_{N}``) rather
    than being a zero-padded suffix.  Pass ``tag="score_maps_{metric}"`` and
    ``sample`` as the ensemble value (or ``None`` for no ensemble).

    Returns the list of source frame paths assembled into the animation, or an
    empty list when no (or fewer than two for score maps) frames were found.
    """
    if not output_dir.is_dir():
        return []

    region_part = region if region else ""
    if sample is not None:
        head = "_".join(filter(None, [prefix, run_id, tag, str(sample)]))
    else:
        head = "_".join(filter(None, [prefix, run_id, tag]))
    tail = "_".join(filter(None, [stream, region_part, var]))
    suffix = f".{image_format}"
    fstep_strs = {str(f).zfill(3) for f in fsteps}
    image_paths = sorted(
        str(f)
        for f in output_dir.iterdir()
        if f.name.startswith(head + "_")
        and f.name.endswith(suffix)
        and f"_{tail}_" in f.name
        and f.stem.rsplit("_", 1)[-1] in fstep_strs
    )
    if not image_paths:
        return []
    if sample is not None:
        anim_parts = ["animation", run_id, tag, str(sample), stream]
    else:
        anim_parts = ["animation", run_id, tag, stream]
    if region:
        anim_parts.append(region)
    anim_parts.append(var)
    out_path = f"{output_dir / '_'.join(filter(None, anim_parts))}.{animation_format}"

    if animation_format.lower() == "mp4":
        frames = [imageio.imread(p) for p in image_paths]
        fps = 1000 / duration_ms if duration_ms > 0 else 2
        imageio.mimsave(out_path, frames, fps=fps, ffmpeg_params=["-crf", "18"])
    else:
        images = [Image.open(p) for p in image_paths]
        images[0].save(
            out_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
        )
        for img in images:
            img.close()
    _logger.debug(f"Saved animation to {out_path}")
    return image_paths


def _dispatch_animations(
    plotter: "Plotter",
    samples: list,
    fsteps,
    variables: list[str],
    select: dict,
    tag: str,
    max_workers: int | None = None,
) -> list[str]:
    """Build GIF animations in parallel for all (region, sample, variable) combinations.

    Animations are built for both maps and histograms — whichever image files
    exist on disk will be picked up automatically.

    Parameters
    ----------
    plotter : Plotter
        Plotter instance (used only for config: regions, fps, image_format, run_id, stream).
    samples, fsteps, variables, select, tag
        Same arguments that ``Plotter.animation`` used to accept.

    Returns
    -------
    list[str]
        Paths of all source frames that were assembled into GIFs.
    """
    plotter.update_data_selection(select)

    duration_ms = int(1000 / plotter.fps) if plotter.fps > 0 else 400

    prefixes = [
        ("map", plotter.get_map_output_dir(tag)),
        ("histogram", plotter.get_hist_output_dir()),
    ]

    tasks = [
        {
            "output_dir": output_dir,
            "run_id": plotter.run_id,
            "tag": tag,
            "stream": plotter.stream,
            "region": region,
            "var": var,
            "sample": sample,
            "fsteps": list(fsteps),
            "image_format": plotter.image_format,
            "animation_format": plotter.animation_format,
            "duration_ms": duration_ms,
            "prefix": prefix,
        }
        for prefix, output_dir in prefixes
        for region in plotter.regions
        for sample in samples
        for var in variables
    ]

    calls = [
        delayed(_build_single_animation)(**t)
        for t in tqdm(tasks, desc=f"Creating animations {plotter.stream} {tag}")
    ]
    results = dispatch_parallel(
        calls,
        n_workers=get_num_workers(max_workers=max_workers),
        backend="loky",
        desc="Animations",
    )
    return [p for r in results if r for p in r]


def _dispatch_score_map_animations(
    map_dir: Path,
    plotter_cfg: dict,
    run_id: str,
    stream: str,
    metrics: list[str],
    regions: list[str],
    variables: list[str],
    ens_values: list,
    fsteps: list,
    n_workers: int | None = None,
) -> list[str]:
    """Build score-map animations in parallel for all (metric, region, variable[, ens]) combos.

    Returns the paths of all source frames assembled into animations.
    """
    duration_ms = int(1000 / plotter_cfg["fps"]) if plotter_cfg["fps"] > 0 else 400

    tasks = [
        dict(
            output_dir=map_dir,
            run_id=run_id,
            tag="score_maps" + (f"_ens_{ens_val}" if ens_val is not None else "") + f"_{metric}",
            stream=stream,
            region=region,
            var=var,
            sample=None,
            fsteps=list(fsteps),
            image_format=plotter_cfg["image_format"],
            animation_format=plotter_cfg["animation_format"],
            duration_ms=duration_ms,
            score_animation=True,
        )
        for metric in metrics
        for region in regions
        for var in variables
        for ens_val in ens_values
    ]

    calls = [delayed(_build_single_animation)(**t) for t in tasks]
    results = dispatch_parallel(
        calls,
        n_workers=n_workers,
        backend="loky",
        desc=f"Score map animations {stream}",
    )
    return [p for r in results if r for p in r]


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
    plot_histograms: bool | str,
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

        # Plot bias once if it doesn't carry an ensemble dimension,
        # otherwise it will be sliced per member inside the loop below.
        bias_has_ens = bias_data is not None and "ens" in bias_data.dims
        if plot_bias and bias_data is not None and not bias_has_ens:
            plotter.create_maps_per_sample(bias_data, plot_chs, data_selection, "bias", bias_cfg)

    for ens in ensemble:
        has_ens = "ens" in preds.dims and ens != "mean"
        preds_ens = preds.sel(ens=ens) if has_ens else preds
        preds_tag = "" if "ens" not in preds.dims else f"ens_{ens}"
        preds_name = "_".join(filter(None, ["preds", preds_tag]))

        if plot_maps:
            plotter.create_maps_per_sample(
                preds_ens, plot_chs, data_selection, preds_name, maps_cfg
            )

            if plot_bias and bias_has_ens:
                bias_ens = bias_data.sel(ens=ens) if ens != "mean" else bias_data
                bias_tag = "_".join(filter(None, ["bias", preds_tag]))
                plotter.create_maps_per_sample(
                    bias_ens, plot_chs, data_selection, bias_tag, bias_cfg
                )

        if plot_histograms is True or plot_histograms == "per-sample":
            plotter.create_histograms(
                tars,
                preds_ens,
                plot_chs,
                data_selection,
                preds_name,
                ranges=maps_config,
            )

    plotter.clean_data_selection()


def _plot_all_samples(
    plotter_cfg: dict,
    output_basedir: str,
    tars: xr.DataArray,
    preds: xr.DataArray,
    bias_data: xr.DataArray | None,
    fstep: int | str,
    stream: str,
    plot_chs: list[str],
    ensemble: list,
    plot_histograms: bool | str,
    maps_config: dict,
    bias_config: dict,
) -> None:
    """Plot histograms across all samples for a single fstep.

    Unlike per-sample histograms, these aggregate all samples together.
    The output filename uses 'global' instead of a sample id and omits the timestep.
    """
    if not (plot_histograms is True or plot_histograms == "across-samples"):
        return

    matplotlib.use("Agg")
    plotter = Plotter(plotter_cfg, Path(output_basedir))

    data_selection = {"sample": "all_samples", "stream": stream, "forecast_step": fstep}

    for ens in ensemble:
        has_ens = "ens" in preds.dims and ens != "mean"
        preds_ens = preds.sel(ens=ens) if has_ens else preds
        preds_tag = "" if "ens" not in preds.dims else f"ens_{ens}"
        preds_name = "_".join(filter(None, ["preds", preds_tag]))

        plotter.create_histograms(
            tars,
            preds_ens,
            plot_chs,
            data_selection,
            preds_name,
            ranges=maps_config,
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

    plot_keys = ("plot_maps", "plot_histograms", "plot_animations")
    if not plot_settings or not any(plot_settings.get(k, False) for k in plot_keys):
        return

    plotter_cfg = {
        "image_format": global_plotting_opts.get("image_format", "png"),
        "animation_format": global_plotting_opts.get("animation_format", "gif"),
        "dpi_val": global_plotting_opts.get("dpi_val", 300),
        "fig_size": global_plotting_opts.get("fig_size"),
        "fps": global_plotting_opts.get("fps", 2),
        "regions": global_plotting_opts.get("regions", stream_cfg.get("regions", ["global"])),
        "log_x": global_plotting_opts.get("log_x", False),
        "log_y": global_plotting_opts.get("log_y", False),
        "n_bins": global_plotting_opts.get("n_bins", 50),
        "plot_subtimesteps": reader.get_inference_stream_attr(stream, "tokenize_spacetime", False)
        | plot_settings.get("plot_subtimesteps", False),
    }

    plotter = Plotter(plotter_cfg, reader.runplot_dir)

    available_data = reader.check_availability(stream, mode="plotting")
    if not available_data.score_availability:
        _logger.warning(f"RUN {reader.run_id} - {stream}: No plotting config. Skipping plots.")
        return

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
    if not isinstance(plot_histograms, bool) and plot_histograms not in {
        "across-samples",
        "per-sample",
    }:
        raise TypeError("plot_histograms must be true, false, 'across-samples', or 'per-sample'. ")
    plot_animations = plot_settings.get("plot_animations", False)
    if not isinstance(plot_animations, bool):
        raise TypeError("plot_animations must be a boolean.")

    model_output = output_data
    if output_data is None:
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

    plot_fstep_set = set(available_data.fsteps) if available_data.fsteps is not None else None
    plot_sample_set = set(available_data.samples) if available_data.samples is not None else None
    plot_channel_set = set(available_data.channels) if available_data.channels is not None else None

    output_dir = str(reader.runplot_dir)
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
    _range_args = (da_tars, da_preds, available_data.channels, global_plotting_opts[stream])
    maps_config_dict = oc.OmegaConf.to_container(common_ranges(*_range_args), resolve=True)
    bias_config_dict = oc.OmegaConf.to_container(bias_ranges(*_range_args), resolve=True)

    num_plot_workers = get_num_workers(
        check_process_headroom=True,
        max_workers=reader.eval_cfg.get("max_workers", None),
    )

    tasks: list[dict] = []
    all_samples_tasks: list[dict] = []
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

        all_samples_tasks.append(
            {
                "plotter_cfg": plotter_cfg,
                "output_basedir": output_dir,
                "tars": tars,
                "preds": preds,
                "bias_data": bias_data,
                "fstep": fstep,
                "stream": stream,
                "plot_chs": plot_chs,
                "ensemble": list(available_data.ensemble),
                "plot_histograms": plot_histograms,
                "maps_config": maps_config_dict,
                "bias_config": bias_config_dict,
            }
        )

        for sample in plot_samples:
            tasks.append(
                {
                    "plotter_cfg": plotter_cfg,
                    "output_basedir": output_dir,
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

    _logger.info(
        f"Parallel plotting: dispatching {len(tasks)} (fstep, sample) tasks "
        f"across up to {num_plot_workers} loky workers."
    )
    calls = [delayed(_plot_single_sample)(**task) for task in tasks]
    dispatch_parallel(
        calls, n_workers=num_plot_workers, backend="loky", desc=f"Plotting {run_id} - {stream}"
    )

    if all_samples_tasks:
        _logger.info(
            f"Parallel plotting: dispatching {len(all_samples_tasks)} across-samples "
            f"tasks using up to {num_plot_workers} loky workers."
        )
        as_calls = [delayed(_plot_all_samples)(**t) for t in all_samples_tasks]
        dispatch_parallel(
            as_calls,
            n_workers=num_plot_workers,
            backend="loky",
            desc=f"Across-samples plots {run_id} - {stream}",
        )

    if plot_animations:
        last_fstep = list(da_tars.keys())[-1]
        last_preds = da_preds[last_fstep]
        last_tars = da_tars[last_fstep]
        has_ens = "ens" in last_preds.dims

        _sel = lambda items, allowed: [x for x in items if x in allowed] if allowed else items
        plot_chs = _sel(list(np.atleast_1d(last_tars.channel.values)), plot_channel_set)
        plot_samples = _sel(list(np.unique(last_tars.sample.values)), plot_sample_set)

        max_wk = reader.eval_cfg.get("max_workers", None)
        anim_samples = plot_samples + (["all_samples"] if plot_histograms else [])
        anim_kw = dict(
            plotter=plotter,
            samples=anim_samples,
            fsteps=da_tars.keys(),
            variables=plot_chs,
            max_workers=max_wk,
            select={"sample": plot_samples[-1], "stream": stream, "forecast_step": last_fstep},
        )

        tags: list[str] = []
        for ens in available_data.ensemble:
            tags.append("preds" if not has_ens else f"preds_ens_{ens}")
        if plot_target:
            tags.append("targets")
        if plot_bias:
            for ens in available_data.ensemble:
                tags.append("bias" if not has_ens else f"bias_ens_{ens}")

        for tag in tags:
            _dispatch_animations(**anim_kw, tag=tag)


# ---------------------------------------------------------------------------
# Summary plots
# ---------------------------------------------------------------------------


def plot_timeseries_summary(
    cfg: dict,
    timeseries_scores: dict,
    summary_dir: Path,
) -> None:
    """Plot timeseries summary comparing multiple run_ids on the same figure.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary (used for runs, labels, colors, plotting options).
    timeseries_scores : dict
        Nested dict: ``timeseries_scores[metric][region][stream][run_id][fstep] = xr.DataArray``
        where each DataArray has dims ``(source_end_hour,)`` or ``(source_end_hour, channel)``.
    summary_dir : Path
        Directory to write summary plots to.
    """
    if not timeseries_scores:
        return

    runs = cfg.run_ids
    plt_opt = cfg.get("global_plotting_options", cfg)
    image_format = plt_opt.get("image_format", "png")
    dpi_val = plt_opt.get("dpi_val", 150)

    ts_dir = summary_dir / "score_init_time_series"
    ts_dir.mkdir(parents=True, exist_ok=True)

    for metric_name, region_dict in timeseries_scores.items():
        for region, stream_dict in region_dict.items():
            for stream, run_dict in stream_dict.items():
                if not run_dict:
                    continue

                # Determine fsteps and channels from first run
                first_run_scores = next(iter(run_dict.values()))
                fsteps = sorted(first_run_scores.keys())
                sample_score = first_run_scores[fsteps[0]]
                channels = (
                    list(sample_score.coords["channel"].values)
                    if "channel" in sample_score.dims
                    else [None]
                )

                for channel in channels:
                    for fstep in fsteps:
                        plt.figure(figsize=(10, 6), dpi=dpi_val)

                        for run_id, fstep_scores in run_dict.items():
                            if fstep not in fstep_scores:
                                continue
                            score = fstep_scores[fstep]
                            score_vals = (
                                score.sel(channel=channel) if channel is not None else score
                            )
                            hours = score_vals.coords["source_end_hour"].values
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
                            f"{metric_name.upper()} vs source end hour | "
                            f"{ch_label} | fstep {fstep} | {region} | {stream}"
                        )
                        plt.title(title)
                        plt.xlabel("Source window end hour [UTC]")
                        plt.ylabel(metric_name.upper())
                        if hours is not None and len(hours) > 0:
                            plt.xlim(min(hours) - 0.5, max(hours) + 0.5)
                            plt.xticks(sorted(hours))
                        plt.grid(True, alpha=0.3)
                        plt.legend()
                        plt.tight_layout()

                        run_ids_str = "_".join(sorted(run_dict.keys()))
                        plot_path = (
                            ts_dir / f"{metric_name}_{ch_label}_{region}_{stream}"
                            f"_{run_ids_str}"
                            f"_fstep_{fstep}_by_source_end_hour.{image_format}"
                        )
                        plt.savefig(plot_path, bbox_inches="tight")
                        plt.close()

    _logger.info(f"Timeseries summary plots saved to {ts_dir}.")


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
    for metric in metrics:
        for region in scores_dict[metric].keys():
            if eval_opt.get("summary_plots", False):
                if metric == "psd":
                    psd_plot_metric_region(metric, region, runs, scores_dict, plotter)
                elif metric == "qq_analysis":
                    quantile_plot_metric_region(metric, region, runs, scores_dict, quantile_plotter)
                else:
                    plot_metric_region(metric, region, runs, scores_dict, plotter, print_summary)
            if eval_opt.get("ratio_plots", False):
                ratio_plot_metric_region(metric, region, runs, scores_dict, plotter, print_summary)
            if eval_opt.get("heat_maps", False):
                heat_maps_metric_region(metric, region, runs, scores_dict, plotter)
            if eval_opt.get("score_cards", False):
                score_card_metric_region(metric, region, runs, scores_dict, sc_plotter)
            if eval_opt.get("bar_plots", False):
                bar_plot_metric_region(metric, region, runs, scores_dict, br_plotter)
