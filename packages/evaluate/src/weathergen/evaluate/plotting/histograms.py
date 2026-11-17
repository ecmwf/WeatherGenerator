# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The across-samples histograms: per-sample counts and mergeable statistics computed per
chunk (``histogram_counts``, returned by ``plot_data``), merged over the chunks
(``merge_statistics``) and drawn once all runs are done (``plot_histograms_across_samples``,
called by ``plot_evaluation``).  The per-sample histograms are ``Plotter``'s.
"""

from pathlib import Path

import matplotlib
import numpy as np
import xarray as xr
from scipy.stats import wasserstein_distance as wd

from weathergen.evaluate.plotting.animations import _build_single_animation, duration_of, videos_of
from weathergen.evaluate.plotting.plot_utils import apply_font_settings, build_plotter_cfg
from weathergen.evaluate.plotting.plotter import DistStats, Plotter, histogram_bins
from weathergen.evaluate.utils.array_utils import select_ensemble_data
from weathergen.evaluate.utils.config_compat import parse_data_plots
from weathergen.evaluate.utils.regions import RegionBoundingBox


def histogram_counts(
    da_tars: dict,
    da_preds: dict,
    samples: list,
    channels: list[str],
    regions: list[str],
    ensemble: list,
    ranges: dict,
    n_bins: int,
) -> xr.Dataset:
    """The across-samples histograms as per-sample counts: the ``histogram_counts`` statistic.

    Per (sample, forecast step, channel, region) the counts of the target and, per ensemble
    tag (a member, the mean or the std, as ``select_ensemble_data`` names them in the file
    names), of the prediction on the channel's bins (``Plotter.histogram_bins`` of the
    channel's ``vmin`` / ``vmax`` in *ranges*: the same edges in every chunk), with the
    count, min, max and sums of the values, squares and cubes for the statistics panel.
    Summed over the samples, these draw the across-samples histogram of the whole stream
    (``plot_histograms_across_samples``), chunked or not.  ``plot_data`` records the run's plot
    directory in the attrs (``runplot_dir``), where the figures go.
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
                    target_moments[i, f, c, r] = _mergeable_statistics(v)
                for t, ens in enumerate(ensemble):
                    preds_ens = select_ensemble_data(preds, ens)
                    for i, sample in enumerate(samples):
                        v = np.asarray(preds_ens.sel(channel=channel, sample=sample)).ravel()
                        v = v[~np.isnan(v)]
                        pred_counts[i, f, c, r, t] = np.histogram(v, bins=bins[channel])[0]
                        pred_moments[i, f, c, r, t] = _mergeable_statistics(v)
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


def _mergeable_statistics(v: np.typing.NDArray) -> list[float]:
    """The statistics of one sample's values that merge into the whole's by summing: count,
    min, max and the sums of the values, squares and cubes."""
    if v.size == 0:
        return [0.0, np.nan, np.nan, 0.0, 0.0, 0.0]
    return [v.size, v.min(), v.max(), v.sum(), (v**2).sum(), (v**3).sum()]


def plot_histograms_across_samples(
    cfg: dict, histograms: dict[str, dict[str, xr.Dataset]], summary_dir: Path
) -> None:
    """The across-samples histograms of all runs from their ``histogram_counts``, merged
    over the chunks: per (run, stream) one figure per (forecast step, channel, region,
    ensemble tag) under the run's plot directory, the counts and the statistics summed over
    the samples (the median and the Wasserstein distance read off the binned counts), then
    their video when the stream's ``data_plots`` ask for one.

    Parameters
    ----------
    cfg:
        The evaluation config (``run_ids``, ``global_plotting_options``).
    histograms:
        ``histograms[stream][run_id]``: the ``histogram_counts`` of the run's plotting
        samples, as ``accumulate_stream_results`` nests them.
    summary_dir:
        The summary directory of the run; unused, the figures go with the per-sample ones.
    """
    matplotlib.use("Agg")
    global_plotting_opts = cfg.get("global_plotting_options", {})
    for stream, by_run in histograms.items():
        for run_id, histogram in by_run.items():
            stream_cfg = cfg.run_ids[run_id].get("streams", {}).get(stream, {})
            plotter_cfg = build_plotter_cfg(global_plotting_opts, stream_cfg)
            apply_font_settings(plotter_cfg)
            plotter = Plotter(plotter_cfg, Path(histogram.attrs["runplot_dir"]))
            frames = []
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
                        t_s = _dist_stats_across_samples(
                            target.target_moments,
                            extrema.sel(region=region, channel=channel).target_moments,
                            centers,
                            t_counts,
                        )
                        for tag in histogram.tag.values:
                            pred = target.sel(tag=tag)
                            p_counts = pred.prediction_counts.values
                            p_s = _dist_stats_across_samples(
                                pred.prediction_moments,
                                extrema.sel(
                                    region=region, channel=channel, tag=tag
                                ).prediction_moments,
                                centers,
                                p_counts,
                            )
                            w_dist = wd(centers, centers, u_weights=t_counts, v_weights=p_counts)
                            name = plotter.draw_histogram(
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
                            frames.append(hist_output_dir / f"{name}.{plotter.image_format}")
            plotter.clean_data_selection()
            spec = parse_data_plots(stream_cfg.get("plotting"))
            if "across_samples" in spec.histogram_videos():
                for video, video_frames in videos_of(sorted(frames), global_plotting_opts).items():
                    _build_single_animation(video_frames, video, duration_of(global_plotting_opts))


def _dist_stats_across_samples(
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
