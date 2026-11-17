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
called by ``plot_evaluation``).  The drawing of a histogram figure from its counts
(``draw_histogram``), the bins, the file names and the statistics panel (``DistStats``)
live here too; ``Plotter.plot_histogram`` uses them for the per-sample histograms.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import skew
from scipy.stats import wasserstein_distance as wd

from weathergen.evaluate.plotting.animations import _build_single_animation, duration_of, videos_of
from weathergen.evaluate.plotting.plot_utils import apply_font_settings, build_plotter_cfg
from weathergen.evaluate.utils.array_utils import select_ensemble_data, select_samples
from weathergen.evaluate.utils.config_compat import parse_data_plots
from weathergen.evaluate.utils.regions import RegionBoundingBox


def histogram_bins(xlim: tuple | None, n_bins: int, values=None) -> np.typing.NDArray:
    """The bin edges of a histogram: *xlim* (``vmin``, ``vmax``), a missing limit taken from
    *values*, with a 5% margin on each side so tails are clearly visible."""
    xmin, xmax = xlim if xlim else (None, None)
    if xmin is None:
        xmin = float(np.nanmin(values))
    if xmax is None:
        xmax = float(np.nanmax(values))
    margin = (xmax - xmin) * 0.05
    return np.linspace(xmin - margin, xmax + margin, n_bins + 1)


def histogram_name(run_id, tag, sample, valid_time, stream, region, varname, fstep) -> str:
    """The file name (without extension) of a histogram figure:
    ``histogram_{run_id}[_{tag}]_{sample}[_{valid_time}]_{stream}[_{region}]_{var}_{fstep}``."""
    parts = [
        "histogram",
        str(run_id),
        str(tag) if tag else "",
        str(sample),
        valid_time,
        str(stream),
        region if region else "",
        varname,
        f"{fstep:03d}",
    ]
    return "_".join(filter(None, parts))


@dataclass
class DistStats:
    """Summary statistics for a 1-D distribution."""

    count: int
    min: float
    max: float
    mean: float
    median: float
    std: float
    skewness: float

    @classmethod
    def from_array(cls, v: np.typing.NDArray) -> "DistStats":
        v = np.asarray(v).ravel()
        return cls(
            count=len(v),
            min=float(np.min(v)),
            max=float(np.max(v)),
            mean=float(np.mean(v)),
            median=float(np.median(v)),
            std=float(np.std(v)),
            skewness=float(skew(v, nan_policy="omit")),
        )

    @classmethod
    def from_moments(
        cls, count: int, vmin: float, vmax: float, s1: float, s2: float, s3: float, median: float
    ) -> "DistStats":
        """From the count, min, max and the sums of the values, squares and cubes (exact,
        mergeable over chunks of samples); *median* is read off the merged histogram."""
        mean = s1 / count
        var = max(s2 / count - mean**2, 0.0)
        m3 = s3 / count - 3 * mean * s2 / count + 2 * mean**3
        return cls(
            count=int(count),
            min=float(vmin),
            max=float(vmax),
            mean=float(mean),
            median=float(median),
            std=float(np.sqrt(var)),
            skewness=float(m3 / var**1.5) if var > 0 else float("nan"),
        )

    def summary(self, label: str) -> str:
        return (
            f"{label:8s} N={self.count}  min={self.min:.3g}  max={self.max:.3g}  "
            f"mean={self.mean:.3g}  med={self.median:.3g}  "
            f"std={self.std:.3g}  skew={self.skewness:.3g}"
        )


def draw_histogram(
    bins: np.typing.NDArray,
    target_counts: np.typing.NDArray,
    pred_counts: np.typing.NDArray,
    t_s: DistStats,
    p_s: DistStats,
    w_dist: float,
    fname: Path,
    title: str,
    varname: str,
    fig_size: tuple | None,
    log_x: bool,
    log_y: bool,
) -> None:
    """Draw and save a histogram figure from its counts: the curves, the ratio and the
    statistics panel.  The per-sample histograms compute the counts from the data
    (``Plotter.plot_histogram``), the across-samples ones sum them over the samples."""
    bin_centers = (bins[:-1] + bins[1:]) / 2
    color_tar = "black"
    color_pred = "#00897B"  # teal / green-blue

    # Create figure with three rows: histogram, ratio, and stats text
    fig = plt.figure(figsize=fig_size or (8, 8), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 0.5])
    ax_hist = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_hist)
    ax_text = fig.add_subplot(gs[2])
    ax_text.set_axis_off()

    # Upper panel: histogram curves
    ax_hist.plot(
        bin_centers, target_counts, alpha=0.7, label="Target", linewidth=1.5, color=color_tar
    )
    ax_hist.plot(
        bin_centers, pred_counts, alpha=0.7, label="Prediction", linewidth=1.5, color=color_pred
    )
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_title(title)
    ax_hist.legend(frameon=False)
    if log_y:
        ax_hist.set_yscale("log")
    ax_hist.grid(True, linestyle="--", alpha=0.5)

    # Lower panel: ratio (prediction / target)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(target_counts > 0, pred_counts / target_counts, np.nan)
    ax_ratio.plot(bin_centers, ratio, linewidth=1.2, color=color_pred)
    ax_ratio.axhline(1.0, linestyle="--", color="gray", linewidth=0.8)
    ax_ratio.set_ylabel("Pred / Target")
    ax_ratio.set_xlabel(f"Variable: {varname}")
    ax_ratio.set_ylim(0, 2)
    ax_ratio.grid(True, linestyle="--", alpha=0.5)

    if log_x:
        ax_hist.set_xscale("log")
        ax_ratio.set_xscale("log")
    ax_ratio.set_xlim(bins[0], bins[-1])

    stat_text = (
        f"Wasserstein distance: {w_dist:.4g}\n{t_s.summary('Target:')}\n{p_s.summary('Pred:')}"
    )
    ax_text.text(
        0.5,
        0.5,
        stat_text,
        ha="center",
        va="top",
        fontsize="x-small",
        family="monospace",
        transform=ax_text.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
    )
    fig.savefig(fname)
    plt.close(fig)


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

    Per (sample, forecast step, channel, region) and field, the target or a prediction tag
    (a member, the mean or the std, as ``select_ensemble_data`` names them in the file
    names), the ``counts`` on the channel's bins (``histogram_bins`` of the channel's
    ``vmin`` / ``vmax`` in *ranges*: the same ``edges`` in every chunk) and the
    ``statistics`` that merge by summing (``_mergeable_statistics``).  Summed over the
    samples, these draw the across-samples histogram of the whole stream
    (``plot_histograms_across_samples``), chunked or not.  ``plot_data`` records the run's
    plot directory in the attrs (``runplot_dir``), where the figures go.
    """
    edges = {c: _bins(da_tars, da_preds, c, ranges.get(c, {}), n_bins) for c in channels}
    fsteps = list(da_tars)
    fields = list(_fields(da_tars[fsteps[0]], da_preds[fsteps[0]], ensemble))
    counts = np.zeros((len(samples), len(fsteps), len(channels), len(regions), len(fields), n_bins))
    statistics = np.zeros(counts.shape[:-1] + (6,))
    for f, fstep in enumerate(fsteps):
        for r, region in enumerate(regions):
            bbox = None if region == "global" else RegionBoundingBox.from_region_name(region)
            tars = da_tars[fstep] if bbox is None else bbox.apply_mask(da_tars[fstep])
            preds = da_preds[fstep] if bbox is None else bbox.apply_mask(da_preds[fstep])
            for k, da in enumerate(_fields(tars, preds, ensemble).values()):
                for c, channel in enumerate(channels):
                    for i, sample in enumerate(samples):
                        v = np.asarray(select_samples(da.sel(channel=channel), [sample])).ravel()
                        v = v[~np.isnan(v)]
                        counts[i, f, c, r, k] = np.histogram(v, bins=edges[channel])[0]
                        statistics[i, f, c, r, k] = _mergeable_statistics(v)
    dims = ("sample", "forecast_step", "channel", "region", "field")
    return xr.Dataset(
        {"counts": (dims + ("bin",), counts), "statistics": (dims + ("stat",), statistics)},
        coords={
            "sample": list(samples),
            "forecast_step": fsteps,
            "channel": list(channels),
            "region": list(regions),
            "field": fields,
            "bin": np.arange(n_bins),
            "stat": ["count", "min", "max", "sum", "sum2", "sum3"],
            "edges": (("channel", "edge"), np.stack([edges[c] for c in channels])),
        },
    )


def _bins(da_tars: dict, da_preds: dict, channel: str, var_range: dict, n_bins: int):
    """The bin edges of *channel*: its configured range, else the range of the data."""
    values = np.concatenate(
        [
            np.asarray(da.sel(channel=channel)).ravel()
            for da in (*da_tars.values(), *da_preds.values())
        ]
    )
    return histogram_bins((var_range.get("vmin"), var_range.get("vmax")), n_bins, values)


def _fields(tars: xr.DataArray, preds: xr.DataArray, ensemble: list) -> dict[str, xr.DataArray]:
    """The target and the prediction per ensemble entry, keyed as the file names tag them."""
    fields = {"target": tars}
    for ens in ensemble:
        if ens in ("mean", "std") or "ens" in preds.dims:
            fields[f"preds_ens_{ens}"] = select_ensemble_data(preds, ens)
        else:
            fields["preds"] = preds
    return fields


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
    prediction tag) under the run's plot directory, the counts and the statistics summed
    over the samples, then their video when the stream's ``data_plots`` ask for one.

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
            style = build_plotter_cfg(global_plotting_opts, stream_cfg)
            apply_font_settings(style)
            runplot_dir = Path(histogram.attrs["runplot_dir"])
            output_dir = runplot_dir / "plots" / stream / "histograms"
            output_dir.mkdir(parents=True, exist_ok=True)
            frames = []
            for fstep in histogram.forecast_step.values:
                frames += _draw_step(
                    histogram.sel(forecast_step=fstep),
                    int(fstep),
                    runplot_dir.name,
                    stream,
                    output_dir,
                    style,
                )
            if "across_samples" in parse_data_plots(stream_cfg.get("plotting")).histogram_videos():
                for video, video_frames in videos_of(sorted(frames), global_plotting_opts).items():
                    _build_single_animation(video_frames, video, duration_of(global_plotting_opts))


def _draw_step(
    step: xr.Dataset, fstep: int, run_id: str, stream: str, output_dir: Path, style: dict
) -> list[Path]:
    """The figures of one forecast step, one per (region, channel, prediction tag): the
    target against the prediction, from the counts summed over the samples.  The frames."""
    frames = []
    for region in step.region.values:
        for channel in step.channel.values:
            bins = step.edges.sel(channel=channel).values
            centers = (bins[:-1] + bins[1:]) / 2
            at = step.sel(region=region, channel=channel)
            t_counts, t_s = _counts_and_dist_stats(at.sel(field="target"), centers)
            for tag in [f for f in step.field.values if f != "target"]:
                p_counts, p_s = _counts_and_dist_stats(at.sel(field=tag), centers)
                name = histogram_name(
                    run_id, tag, "all_samples", None, stream, region, channel, fstep
                )
                fname = output_dir / f"{name}.{style.get('image_format', 'png')}"
                draw_histogram(
                    bins,
                    t_counts,
                    p_counts,
                    t_s,
                    p_s,
                    wd(centers, centers, u_weights=t_counts, v_weights=p_counts),
                    fname,
                    f"{stream}, {channel} : fstep = {fstep:03}",
                    str(channel),
                    style.get("fig_size"),
                    style.get("log_x", False),
                    style.get("log_y", False),
                )
                frames.append(fname)
    return frames


def _counts_and_dist_stats(field: xr.Dataset, centers: np.typing.NDArray):
    """The counts of one field summed over the samples, and its ``DistStats``: the sums from
    the summed ``statistics``, the min and max over the samples, the median read off the
    counts on *centers* (the Wasserstein distance is read off the counts too)."""
    counts = field.counts.sum("sample").values
    summed = field.statistics.sum("sample")
    cumulative = np.cumsum(counts)
    median = (
        float(centers[np.searchsorted(cumulative, cumulative[-1] / 2)])
        if cumulative[-1]
        else np.nan
    )
    stats = DistStats.from_moments(
        float(summed.sel(stat="count")),
        float(field.statistics.sel(stat="min").min()),
        float(field.statistics.sel(stat="max").max()),
        float(summed.sel(stat="sum")),
        float(summed.sel(stat="sum2")),
        float(summed.sel(stat="sum3")),
        median,
    )
    return counts, stats
