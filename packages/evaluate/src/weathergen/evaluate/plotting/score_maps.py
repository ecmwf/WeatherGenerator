# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Score maps: the spatial scores (``MetricResult.spatial``) of one (run, stream) as maps.

They are drawn from scores averaged over samples, so from complete (merged) results.
The drawing itself is done by ``Plotter.scatter_plot``; the videos by ``animations``.
"""

import logging
from pathlib import Path

import matplotlib
import omegaconf as oc
import xarray as xr
from joblib import delayed

from weathergen.evaluate.io.data.io_orchestration import dispatch_parallel, get_num_workers
from weathergen.evaluate.io.io_reader import Reader
from weathergen.evaluate.plotting.plot_utils import apply_font_settings
from weathergen.evaluate.plotting.plotter import Plotter
from weathergen.evaluate.scores.score_orchestration import MetricResult
from weathergen.evaluate.utils.array_utils import common_ranges

_logger = logging.getLogger(__name__)


def plot_score_maps(
    reader: Reader,
    stream: str,
    results: dict[str, dict[str, MetricResult]],
    global_plotting_options: dict | None = None,
) -> None:
    """Plot spatial score maps for all regions and forecast steps.

    The videos of the maps are built from them by ``animations.build_animations``.

    Parameters
    ----------
    reader : Reader
        Reader object containing all info about a particular run.
    stream : str
        Stream name to plot score maps for.
    results : dict[str, dict[str, MetricResult]]
        ``results[region][metric] -> MetricResult`` with the ``spatial`` field filled
        (``calc_scores_per_stream(..., compute_spatial=True)``).
    global_plotting_options : dict | None
        Global plotting options. These can be passed to the plotter and can be used to set options.
    """

    if not reader.is_gridded_data(stream):
        _logger.debug(f"RUN {reader.run_id} - {stream}: Skipping score maps (non-gridded data).")
        return

    # {region: {metric: spatial DataArray(ipoint, forecast_step, channel[, ens])}}
    spatial = {
        region: {m: r.spatial for m, r in by_metric.items() if r.spatial is not None}
        for region, by_metric in results.items()
    }
    spatial = {region: by_metric for region, by_metric in spatial.items() if by_metric}
    if not spatial:
        _logger.warning(f"RUN {reader.run_id} - {stream}: No spatial scores. Skipping score maps.")
        return

    map_dir = reader.runplot_dir / "plots" / stream / "score_maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    _logger.info(f"RUN {reader.run_id} - {stream}: Saving score maps to {map_dir}")

    n_plot_workers = get_num_workers(
        check_process_headroom=True,
        max_workers=reader.eval_cfg.get("max_workers", None),
    )

    cfg = global_plotting_options
    plotter_cfg = {
        "image_format": cfg.get("image_format", "png"),
        "dpi_val": cfg.get("dpi_val", 300),
        "fig_size": cfg.get("fig_size", None),
        "animation_format": cfg.get("animation_format", "mp4"),
        "fps": cfg.get("fps", 2),
        "font_size": cfg.get("font_size"),
        "font_type": cfg.get("font_type"),
    }
    output_basedir = str(reader.runplot_dir)
    apply_font_settings(plotter_cfg)
    run_id = reader.run_id

    # Colour range of each (metric, region, channel) over all forecast steps; the scores
    # are passed as both inputs of common_ranges, which takes targets and predictions.
    score_ranges_dict: dict = {}
    for region, by_metric in spatial.items():
        for metric, da in by_metric.items():
            ranges = common_ranges(
                {0: da}, {0: da}, [str(c) for c in da.channel.values], oc.DictConfig({})
            )
            score_ranges_dict.setdefault(metric, {})[region] = oc.OmegaConf.to_container(ranges)

    regions = list(spatial)
    fsteps = sorted(
        {
            int(f)
            for by_metric in spatial.values()
            for da in by_metric.values()
            for f in da.forecast_step.values
        }
    )
    fstep_tasks: list[dict] = []
    for region, by_metric in spatial.items():
        for fstep in fsteps:
            fstep_tasks.append(
                {
                    "plotter_cfg": plotter_cfg,
                    "score_ranges_dict": score_ranges_dict,
                    "output_basedir": output_basedir,
                    "map_dir": str(map_dir),
                    "stream": stream,
                    "region": region,
                    "spatial_scores": {
                        m: da.sel(forecast_step=fstep, drop=True)
                        for m, da in by_metric.items()
                        if fstep in da.forecast_step.values
                    },
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


def _plot_score_maps_per_stream(
    plotter_cfg: dict,
    score_ranges_dict: dict,
    output_basedir: str,
    map_dir: str,
    stream: str,
    region: str,
    spatial_scores: dict[str, xr.DataArray],
    fstep: int,
    run_id: str = "",
) -> None:
    """Plot 2D score maps for all metrics/channels for one (region, fstep).

    ``spatial_scores[metric]`` are ``DataArray(ipoint, channel[, ens])`` with ``lat``/``lon``
    coordinates on ``ipoint``.
    """

    valid = [
        (metric, result) for metric, result in spatial_scores.items() if "ipoint" in result.dims
    ]
    if not valid:
        return

    ens_metrics = {metric for metric, result in valid if "ens" in result.dims}

    plot_metrics = xr.concat(
        [result for _, result in valid],
        dim="metric",
        coords="minimal",
        combine_attrs="drop_conflicts",
    )
    plot_metrics = plot_metrics.assign_coords(metric=[metric for metric, _ in valid]).compute()

    all_ens = plot_metrics.coords["ens"].values if "ens" in plot_metrics.dims else [None]

    plot_tasks: list[dict] = []
    for metric in plot_metrics.coords["metric"].values:
        metric_has_ens = str(metric) in ens_metrics and "ens" in plot_metrics.dims
        ens_values = all_ens if metric_has_ens else [None]
        for ens_val in ens_values:
            tag = "score_maps" + (f"_ens_{ens_val}" if ens_val is not None else "") + f"_{metric}"
            for channel in plot_metrics.coords["channel"].values:
                sel = {"metric": metric, "channel": channel}
                if ens_val is not None:
                    sel["ens"] = ens_val
                data = plot_metrics.sel(**sel)
                if ens_val is None and "ens" in data.dims:
                    data = data.isel(ens=0, drop=True)
                data = data.squeeze()
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
        _plot_score_map(**t)


def _plot_score_map(
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
