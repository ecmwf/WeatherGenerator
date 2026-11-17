# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Plots of the prediction and target fields: maps, histograms and timeseries.

The maps and histograms are drawn from the loaded data of one (run, stream)
(``ReaderOutput``), so they can be drawn per chunk of samples and forecast steps; the
drawing itself is done by ``Plotter``, the videos by ``animations`` once all the frames are
written.  The timeseries are the region means of the data (``Timeseries.region_means``),
which ``plot_data`` computes and returns per chunk; ``plot_timeseries`` draws them once all
runs are done, every run on the same figure.
"""

import logging
from pathlib import Path

import matplotlib
import numpy as np
import omegaconf as oc
import xarray as xr
from joblib import delayed
from scipy.stats import wasserstein_distance as wd

from weathergen.evaluate.io.data.io_orchestration import dispatch_parallel, get_num_workers
from weathergen.evaluate.io.io_reader import Reader, ReaderOutput
from weathergen.evaluate.plotting.colour_ranges import plot_colour_ranges
from weathergen.evaluate.plotting.plot_utils import (
    apply_font_settings,
    build_plotter_cfg,
    filter_set,
)
from weathergen.evaluate.plotting.plotter import DistStats, Plotter, histogram_bins
from weathergen.evaluate.plotting.timeseries import Timeseries, plot_single_timeseries
from weathergen.evaluate.utils.array_utils import select_ensemble_data
from weathergen.evaluate.utils.config_compat import parse_data_plots
from weathergen.evaluate.utils.regions import RegionBoundingBox

_logger = logging.getLogger(__name__)


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
    std_config: dict,
) -> None:
    """Plot all maps/histograms for a single (fstep, sample) pair (loky worker)."""
    matplotlib.use("Agg")

    maps_cfg = oc.OmegaConf.create(maps_config)
    bias_cfg = oc.OmegaConf.create(bias_config)
    std_cfg = oc.OmegaConf.create(std_config)
    plotter = Plotter(plotter_cfg, Path(output_basedir))

    data_selection = {"sample": sample, "stream": stream, "forecast_step": fstep}

    # Prediction, target and bias maps are enabled independently.
    if plot_target:
        plotter.create_maps_per_sample(tars, plot_chs, data_selection, "targets", maps_cfg)

    # Plot bias once if it doesn't carry an ensemble dimension,
    # otherwise it will be sliced per member inside the loop below.
    bias_has_ens = bias_data is not None and "ens" in bias_data.dims
    if plot_bias and bias_data is not None and not bias_has_ens:
        plotter.create_maps_per_sample(bias_data, plot_chs, data_selection, "bias", bias_cfg)

    for ens in ensemble:
        preds_ens = select_ensemble_data(preds, ens)
        if ens in ("mean", "std"):
            preds_tag = f"ens_{ens}"
        else:
            preds_tag = "" if "ens" not in preds.dims else f"ens_{ens}"
        preds_name = "_".join(filter(None, ["preds", preds_tag]))

        if plot_maps:
            cfg_to_use = std_cfg if ens == "std" else maps_cfg
            plotter.create_maps_per_sample(
                preds_ens, plot_chs, data_selection, preds_name, cfg_to_use
            )

        if plot_bias and bias_has_ens:
            bias_ens = select_ensemble_data(bias_data, ens)
            bias_tag = "_".join(filter(None, ["bias", preds_tag]))
            plotter.create_maps_per_sample(bias_ens, plot_chs, data_selection, bias_tag, bias_cfg)

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


def histogram_statistics(
    da_tars: dict,
    da_preds: dict,
    samples: list,
    channels: list[str],
    regions: list[str],
    ensemble: list,
    ranges: dict,
    n_bins: int,
) -> xr.Dataset:
    """The across-samples histograms as per-sample counts: the ``histogram`` data statistic.

    Per (sample, forecast step, channel, region) the counts of the target and, per ensemble
    tag (a member, the mean or the std, as ``select_ensemble_data`` names them in the file
    names), of the prediction on the channel's bins (``Plotter.histogram_bins`` of the
    channel's ``vmin`` / ``vmax`` in *ranges*: the same edges in every chunk), with the
    count, min, max and sums of the values, squares and cubes for the statistics panel.
    Summed over the samples, these draw the across-samples histogram of the whole stream
    (``plot_histograms_across_samples``), chunked or not.
    """
    bins = {}
    for channel in channels:
        var_range = ranges.get(channel, {})
        values = np.concatenate(
            [np.asarray(da.sel(channel=channel)).ravel() for da in da_tars.values()]
            + [np.asarray(da.sel(channel=channel)).ravel() for da in da_preds.values()]
        )
        bins[channel] = histogram_bins(
            (var_range.get("vmin"), var_range.get("vmax")), n_bins, values
        )
    first = next(iter(da_preds.values()))
    tags = []
    for ens in ensemble:
        if ens in ("mean", "std"):
            tags.append(f"preds_ens_{ens}")
        else:
            tags.append("preds" if "ens" not in first.dims else f"preds_ens_{ens}")
    fsteps = list(da_tars)
    stats = ["count", "min", "max", "sum", "sum2", "sum3"]
    shape = (len(samples), len(fsteps), len(channels), len(regions))
    target_counts = np.zeros(shape + (n_bins,))
    target_moments = np.zeros(shape + (len(stats),))
    pred_counts = np.zeros(shape + (len(tags), n_bins))
    pred_moments = np.zeros(shape + (len(tags), len(stats)))
    for f, fstep in enumerate(fsteps):
        for r, region in enumerate(regions):
            bbox = None if region == "global" else RegionBoundingBox.from_region_name(region)
            tars = da_tars[fstep] if bbox is None else bbox.apply_mask(da_tars[fstep])
            preds = da_preds[fstep] if bbox is None else bbox.apply_mask(da_preds[fstep])
            for c, channel in enumerate(channels):
                for i, sample in enumerate(samples):
                    v = np.asarray(tars.sel(channel=channel, sample=sample)).ravel()
                    v = v[~np.isnan(v)]
                    target_counts[i, f, c, r] = np.histogram(v, bins=bins[channel])[0]
                    target_moments[i, f, c, r] = _moments(v)
                for t, ens in enumerate(ensemble):
                    preds_ens = select_ensemble_data(preds, ens)
                    for i, sample in enumerate(samples):
                        v = np.asarray(preds_ens.sel(channel=channel, sample=sample)).ravel()
                        v = v[~np.isnan(v)]
                        pred_counts[i, f, c, r, t] = np.histogram(v, bins=bins[channel])[0]
                        pred_moments[i, f, c, r, t] = _moments(v)
    dims = ("sample", "forecast_step", "channel", "region")
    return xr.Dataset(
        {
            "target_counts": (dims + ("bin",), target_counts),
            "target_moments": (dims + ("stat",), target_moments),
            "prediction_counts": (dims + ("tag", "bin"), pred_counts),
            "prediction_moments": (dims + ("tag", "stat"), pred_moments),
        },
        coords={
            "sample": list(samples),
            "forecast_step": fsteps,
            "channel": list(channels),
            "region": list(regions),
            "tag": tags,
            "bin": np.arange(n_bins),
            "stat": stats,
            "edges": (("channel", "edge"), np.stack([bins[c] for c in channels])),
        },
    )


def _moments(v: np.typing.NDArray) -> list[float]:
    """count, min, max and the sums of the values, squares and cubes of *v*."""
    if v.size == 0:
        return [0.0, np.nan, np.nan, 0.0, 0.0, 0.0]
    return [v.size, v.min(), v.max(), v.sum(), (v**2).sum(), (v**3).sum()]


def plot_histograms_across_samples(
    plotter_cfg: dict, output_basedir: str | Path, stream: str, histogram: xr.Dataset
) -> None:
    """The across-samples histograms of one (run, stream) from the ``histogram`` statistic
    (``histogram_statistics``), merged over the chunks: one figure per (forecast step,
    channel, region, ensemble tag), the counts and the statistics summed over the samples.
    The median and the Wasserstein distance are read off the binned counts."""
    matplotlib.use("Agg")
    plotter = Plotter(plotter_cfg, Path(output_basedir))
    for fstep in histogram.forecast_step.values:
        plotter.update_data_selection(
            {"sample": "all_samples", "stream": stream, "forecast_step": int(fstep)}
        )
        hist_output_dir = plotter.get_hist_output_dir()
        hist_output_dir.mkdir(parents=True, exist_ok=True)
        at_step = histogram.sel(forecast_step=fstep).sum("sample", keep_attrs=True)
        extrema = histogram.sel(forecast_step=fstep)
        for region in histogram.region.values:
            for channel in histogram.channel.values:
                bins = histogram.edges.sel(channel=channel).values
                centers = (bins[:-1] + bins[1:]) / 2
                target = at_step.sel(region=region, channel=channel)
                t_counts = target.target_counts.values
                t_s = _stats_of(
                    target.target_moments,
                    extrema.sel(region=region, channel=channel).target_moments,
                    centers,
                    t_counts,
                )
                for tag in histogram.tag.values:
                    pred = target.sel(tag=tag)
                    p_counts = pred.prediction_counts.values
                    p_s = _stats_of(
                        pred.prediction_moments,
                        extrema.sel(region=region, channel=channel, tag=tag).prediction_moments,
                        centers,
                        p_counts,
                    )
                    w_dist = wd(centers, centers, u_weights=t_counts, v_weights=p_counts)
                    plotter.draw_histogram(
                        bins,
                        t_counts,
                        p_counts,
                        t_s,
                        p_s,
                        w_dist,
                        hist_output_dir,
                        str(channel),
                        tag=str(tag),
                        region=str(region),
                    )
    plotter.clean_data_selection()


def _stats_of(
    summed, per_sample, centers: np.typing.NDArray, counts: np.typing.NDArray
) -> DistStats:
    """``DistStats`` of the samples together: the sums from *summed* (``stat`` dim), the min
    and max over the samples from *per_sample*, the median from the *counts* on *centers*."""
    cumulative = np.cumsum(counts)
    median = (
        float(centers[np.searchsorted(cumulative, cumulative[-1] / 2)])
        if cumulative[-1]
        else np.nan
    )
    return DistStats.from_moments(
        float(summed.sel(stat="count")),
        float(per_sample.sel(stat="min").min()),
        float(per_sample.sel(stat="max").max()),
        float(summed.sel(stat="sum")),
        float(summed.sel(stat="sum2")),
        float(summed.sel(stat="sum3")),
        median,
    )


def plot_data(
    reader: Reader,
    stream: str,
    global_plotting_opts: dict,
    output_data: ReaderOutput | None = None,
) -> dict[str, xr.Dataset]:
    """Plot prediction/target maps and histograms for a given run and stream; the plots
    across samples are returned as data statistics instead of drawn: ``{name: Dataset}`` of
    per-sample quantities of the plotting samples, ``timeseries`` being the region means
    (``Timeseries.region_means``).  They are merged over the chunks (``merge_statistics``)
    and drawn once all runs are done (``plot_evaluation``).

    The frames of the videos are drawn here; the videos themselves are built from them by
    ``animations.build_animations`` once all the frames of the run exist.

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
    spec = parse_data_plots(plot_settings)
    if not spec:
        return {}

    plotter_cfg = build_plotter_cfg(
        global_plotting_opts,
        stream_cfg,
        tokenize_spacetime=reader.get_inference_stream_attr(stream, "tokenize_spacetime", False),
    )

    apply_font_settings(plotter_cfg)

    available_data = reader.check_availability(stream, mode="plotting")
    if not available_data.score_availability:
        _logger.warning(f"RUN {reader.run_id} - {stream}: No plotting config. Skipping plots.")
        return {}

    # Per-step images are produced for every requested plot (videos are built from them).
    plot_maps = "predictions" in spec.maps
    plot_target = "target" in spec.maps
    plot_bias = "bias" in spec.maps
    plot_timeseries = spec.timeseries
    hist_per_sample = "per_sample" in spec.histograms
    hist_across = "across_samples" in spec.histograms
    if hist_per_sample and hist_across:
        plot_histograms: bool | str = True
    elif hist_per_sample:
        plot_histograms = "per-sample"
    elif hist_across:
        plot_histograms = "across-samples"
    else:
        plot_histograms = False

    model_output = output_data
    if output_data is None:
        empty = [
            name
            for name, val in (
                ("forecast_step", available_data.fsteps),
                ("sample", available_data.samples),
                ("channel", available_data.channels),
            )
            if val is not None and len(val) == 0
        ]
        if empty:
            _logger.warning(
                f"RUN {run_id} - {stream}: none of the requested {', '.join(empty)}(s) "
                f"for plotting exist in the output (available forecast steps: "
                f"{sorted(int(f) for f in reader.get_forecast_steps())[:5]}...). "
                f"Skipping plots."
            )
            return {}
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
        return {}

    plot_fstep_set = set(available_data.fsteps) if available_data.fsteps is not None else None
    plot_sample_set = set(available_data.samples) if available_data.samples is not None else None
    plot_channel_set = set(available_data.channels) if available_data.channels is not None else None

    output_dir = str(reader.runplot_dir)

    if plot_fstep_set is not None:
        # the requested steps are zarr steps; the data is keyed by sub-step label
        plot_fstep_set = set(reader.substep_labels(stream, sorted(plot_fstep_set)))

    if plot_fstep_set is not None:
        da_tars = {fs: da for fs, da in da_tars.items() if fs in plot_fstep_set}
        da_preds = {fs: da for fs, da in da_preds.items() if fs in plot_fstep_set}

    if not da_tars:
        _logger.info(f"Skipping Plot Data for {stream}. No matching fsteps after filtering.")
        return {}

    if not isinstance(global_plotting_opts.get(stream), oc.DictConfig):
        global_plotting_opts[stream] = oc.DictConfig({})
    maps_config_dict, bias_config_dict, std_config_dict = plot_colour_ranges(
        da_tars, da_preds, available_data.channels, global_plotting_opts[stream]
    )

    num_plot_workers = get_num_workers(
        check_process_headroom=True,
        max_workers=reader.eval_cfg.get("max_workers", None),
    )

    tasks: list[dict] = []
    for (fstep, tars), (_, preds) in zip(da_tars.items(), da_preds.items(), strict=False):
        all_chs = list(np.atleast_1d(tars.channel.values))
        plot_chs = filter_set(all_chs, plot_channel_set)
        if not plot_chs:
            continue

        all_samples = list(np.unique(tars.sample.values))
        plot_samples = filter_set(all_samples, plot_sample_set)
        if not plot_samples:
            continue

        bias_data = (preds - tars) if plot_bias else None

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
                    "std_config": std_config_dict,
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

    # the plots across samples: per-sample statistics of the plotting samples, drawn once
    # all the samples are there (plot_stream_statistics / plot_evaluation)
    statistics = {}
    first = next(iter(da_tars.values()))
    samples = filter_set(list(np.unique(first.sample.values)), plot_sample_set)
    channels = filter_set(list(np.atleast_1d(first.channel.values)), plot_channel_set)
    if "across_samples" in spec.histograms:
        statistics["histogram"] = histogram_statistics(
            da_tars,
            da_preds,
            samples,
            channels,
            plotter_cfg["regions"],
            list(available_data.ensemble),
            maps_config_dict,
            plotter_cfg.get("n_bins", 50),
        )
    if plot_timeseries:
        statistics["timeseries"] = Timeseries(
            {fs: da.sel(sample=samples) for fs, da in da_preds.items()},
            {fs: da.sel(sample=samples) for fs, da in da_tars.items()},
        ).region_means(plotter_cfg["regions"], channels)
    return statistics


def plot_timeseries(
    cfg: dict, timeseries: dict[str, dict[str, xr.Dataset]], summary_dir: Path
) -> None:
    """The timeseries figures of all runs: one per (stream, region, channel, initialisation
    time[, ensemble member]), with the region mean of every run's prediction and of the
    target along the valid time, into ``summary_dir/<run_ids>/timeseries/``.

    Parameters
    ----------
    cfg:
        The evaluation config (``run_ids``, ``max_workers``).
    timeseries:
        ``timeseries[stream][run_id]``: the region means of the run's plotting samples
        (``Timeseries.region_means``), the ``timeseries`` statistic as
        ``accumulate_stream_results`` nests it.
    summary_dir:
        The summary directory of the run.
    """
    output_dir = Path(summary_dir) / "_".join(sorted(cfg.run_ids.keys())) / "timeseries"
    tasks = []
    for stream, by_run in timeseries.items():
        init_times = sorted({t for ds in by_run.values() for t in ds.init_times.values})
        for init_time in init_times:
            # the sample of this init time in each run that has it
            at = {
                run_id: ds.isel(sample=int(np.flatnonzero(ds.init_times.values == init_time)[0]))
                for run_id, ds in by_run.items()
                if (ds.init_times.values == init_time).any()
            }
            target_run = next(iter(at.values()))
            for region in [str(r) for r in target_run.region.values]:
                for channel in [str(c) for c in target_run.channel.values]:
                    predictions = {
                        run_id: ds.prediction.sel(region=region, channel=channel)
                        for run_id, ds in at.items()
                        if region in ds.region.values and channel in ds.channel.values
                    }
                    # the members as the data names them (ints or strings), one figure each
                    members = sorted(
                        {
                            e
                            for p in predictions.values()
                            if "ens" in p.dims
                            for e in p.ens.values.tolist()
                        }
                    ) or [None]
                    for ens in members:
                        tasks.append(
                            {
                                "output_dir": output_dir,
                                "channel": channel,
                                "region": region,
                                "stream": stream,
                                "init_time": init_time,
                                "predictions": predictions,
                                "target": target_run.target.sel(region=region, channel=channel),
                                "ens": ens,
                            }
                        )
    if not tasks:
        return
    _logger.info(f"Timeseries: {len(tasks)} figure(s) of {len(timeseries)} stream(s).")
    dispatch_parallel(
        [delayed(plot_single_timeseries)(**t) for t in tasks],
        n_workers=get_num_workers(check_process_headroom=True, max_workers=cfg.get("max_workers")),
        backend="loky",
        desc="Timeseries",
    )
