# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# Standard library
import contextlib
import json
import logging
import os
import resource
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Third-party
import matplotlib
import numpy as np
import omegaconf as oc
import xarray as xr
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from tqdm import tqdm

# Local application / package
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


def _score_single_fstep(
    fstep: int,
    tars: xr.DataArray,
    preds: xr.DataArray,
    preds_next: xr.DataArray | None,
    tars_next: xr.DataArray | None,
    climatology: xr.DataArray | None,
    bbox: "RegionBoundingBox",
    metrics: dict,
    group_by_coord: str | None,
) -> tuple[int, xr.DataArray, dict[tuple[int, str], dict]] | None:
    """
    Score all metrics for one fstep in one region. Stateless, thread-safe.

    All inputs are immutable xarray DataArrays. All numpy/xarray operations
    release the GIL during C-level computation, enabling effective threading.

    Parameters
    ----------
    fstep : int
        Forecast step index.
    tars, preds : xr.DataArray
        Target and prediction data for this fstep.
    preds_next, tars_next : xr.DataArray | None
        Next-step data for froct/troct metrics.
    climatology : xr.DataArray | None
        Aligned climatology for this fstep.
    bbox : RegionBoundingBox
        Region bounding box to apply.
    metrics : dict
        Metric name → parameters dict.
    group_by_coord : str | None
        Coordinate to group by (None for gridded, "sample" for scatter).

    Returns
    -------
    (fstep, combined_metrics, metric_attrs) or None if no valid scores.
    """
    if preds.sizes.get("ipoint") == 0:
        return None

    # Apply region mask
    tars, preds, tars_next, preds_next = [
        bbox.apply_mask(x) if x is not None else None for x in (tars, preds, tars_next, preds_next)
    ]

    score_data = VerifiedData(preds, tars, preds_next, tars_next, climatology)

    valid_scores = []
    valid_metric_names = []

    for metric, parameters in metrics.items():
        score = get_score(
            score_data,
            metric,
            agg_dims="ipoint",
            group_by_coord=group_by_coord,
            parameters=parameters,
        )
        if score is not None:
            valid_scores.append(score)
            valid_metric_names.append(metric)

    if not valid_scores:
        return None

    # Preserve attributes before concat drops them
    metric_attrs = {}
    for metric_name, score in zip(valid_metric_names, valid_scores, strict=False):
        if score.attrs:
            metric_attrs[(int(fstep), metric_name)] = score.attrs.copy()

    combined = xr.concat(
        valid_scores,
        dim="metric",
        coords="minimal",
        combine_attrs="drop_conflicts",
    )
    combined = combined.assign_coords(metric=valid_metric_names)
    combined = combined.compute()

    for coord in ["channel", "sample", "ens"]:
        combined = scalar_coord_to_dim(combined, coord)

    return fstep, combined, metric_attrs


def calc_scores_per_stream(
    reader: Reader,
    stream: str,
    regions: list[str],
    metrics_dict: dict,
    output_data: ReaderOutput | None = None,
):
    """
    Calculate scores for a given run and stream using the specified metrics.

    Parameters
    ----------
    reader : Reader
        Reader object containing all info about a particular run.
    stream :
        Stream name to calculate scores for.
    regions :
        List of regions to calculate scores on.
    metrics_dict :
        Dictionary mapping regions to lists of metric names to calculate.
    output_data : ReaderOutput | None
        Pre-loaded data.  When provided, reader.get_data() is skipped — this
        avoids the double-load when data is already loaded for plotting.

    Returns
    -------
    Dictionary containing scores for each metric and stream.

    See Also
    --------
    plot_score_maps_per_stream : Call after this function to produce spatial
        score maps (aggregated over samples rather than spatial points).
    """
    local_scores = {}  # top-level dict: metric -> region -> stream -> run_id

    available_data = reader.check_availability(stream, mode="evaluation")
    fsteps = available_data.fsteps
    samples = available_data.samples
    channels = available_data.channels
    ensemble = available_data.ensemble
    is_gridded_data = reader.is_gridded_data(stream)
    group_by_coord = None if is_gridded_data else "sample"

    if output_data is None:
        output_data = reader.get_data(
            stream,
            fsteps=fsteps,
            samples=samples,
            channels=channels,
            ensemble=ensemble,
        )

    da_preds = output_data.prediction
    da_tars = output_data.target
    fsteps = sorted(list(da_preds.keys()))
    aligned_clim_data = get_climatology(reader, da_tars, stream)

    for region in regions:
        bbox = RegionBoundingBox.from_region_name(region)
        metrics = metrics_dict[region]

        _logger.info(
            f"RUN {reader.run_id} - {stream}: Calculating scores for region {region}"
            f" across {len(fsteps)} fsteps and metrics {list(metrics.keys())}..."
        )
        metric_stream = xr.DataArray(
            np.full(
                (len(samples), len(fsteps), len(channels), len(metrics), len(ensemble)),
                np.nan,
            ),
            coords={
                "sample": samples,
                "forecast_step": fsteps,
                "channel": channels,
                "metric": list(metrics.keys()),
                "ens": ensemble,
            },
        )

        if "lead_time" in da_preds[fsteps[0]].coords:
            metric_stream = metric_stream.assign_coords(
                lead_time=("forecast_step", np.full(len(fsteps), -1, dtype=int))
            )

        # Store metric-specific attributes that get lost during concat
        # Key: (fstep, metric) -> attrs dict
        all_metric_attrs = {}

        # --- Build task arguments for all fsteps (pre-resolve next-step data) ---
        fstep_tasks = []
        for fstep in fsteps:
            tars_fs = da_tars[fstep]
            preds_fs = da_preds[fstep]
            preds_next, tars_next = get_next_data(fstep, da_preds, da_tars, fsteps)
            climatology = aligned_clim_data[fstep] if aligned_clim_data else None
            fstep_tasks.append((fstep, tars_fs, preds_fs, preds_next, tars_next, climatology))

        # --- Execute scoring (threaded or sequential) ---
        num_scoring_threads = int(reader.eval_cfg.get("num_scoring_threads", 12))
        effective_threads = min(num_scoring_threads, len(fstep_tasks))

        if effective_threads > 1 and len(fstep_tasks) > 2:
            with ThreadPoolExecutor(max_workers=effective_threads) as executor:
                futures = {
                    executor.submit(
                        _score_single_fstep,
                        fstep,
                        tars_fs,
                        preds_fs,
                        preds_next,
                        tars_next,
                        climatology,
                        bbox,
                        metrics,
                        group_by_coord,
                    ): fstep
                    for fstep, tars_fs, preds_fs, preds_next, tars_next, climatology in fstep_tasks
                }
                fstep_results = []
                for i, future in enumerate(as_completed(futures), 1):
                    fs = futures[future]
                    result = future.result()
                    if result is not None:
                        fstep_results.append(result)
                    _logger.info(
                        f"RUN {reader.run_id} - {stream}: Scored fstep {fs} for region"
                        f" {region} ({i}/{len(fstep_tasks)})."
                    )
        else:
            fstep_results = []
            for i, (fstep, tars_fs, preds_fs, preds_next, tars_next, climatology) in enumerate(
                tqdm(
                    fstep_tasks,
                    desc=(
                        f"Computing scores for {reader.run_id}"
                        f" - stream {stream} and region {region}"
                    ),
                ),
                1,
            ):
                result = _score_single_fstep(
                    fstep,
                    tars_fs,
                    preds_fs,
                    preds_next,
                    tars_next,
                    climatology,
                    bbox,
                    metrics,
                    group_by_coord,
                )
                if result is not None:
                    fstep_results.append(result)
                _logger.info(
                    f"RUN {reader.run_id} - {stream}: Scored fstep {fstep} for region"
                    f" {region} ({i}/{len(fstep_tasks)})."
                )

        # --- Reassemble results into metric_stream (sequential, deterministic order) ---
        fstep_results.sort(key=lambda r: r[0])

        for fstep, combined_metrics, fstep_attrs in fstep_results:
            all_metric_attrs.update(fstep_attrs)

            criteria = {
                "forecast_step": int(fstep),
                "sample": combined_metrics.sample.values,
                "channel": combined_metrics.channel.values,
                "metric": combined_metrics.metric.values,
            }
            if "ens" in combined_metrics.dims:
                criteria["ens"] = combined_metrics.ens.values

            metric_stream.loc[criteria] = combined_metrics

            # Restore metric-specific coordinates that were dropped by coords="minimal"
            for coord_name in combined_metrics.coords:
                if coord_name in combined_metrics.dims or coord_name in metric_stream.dims:
                    continue
                if coord_name == "lead_time":
                    metric_stream.coords["lead_time"].loc[{"forecast_step": int(fstep)}] = (
                        combined_metrics.coords["lead_time"]
                        .values.astype("timedelta64[h]")
                        .astype(int)
                    )
                else:
                    coord_dims = combined_metrics.coords[coord_name].dims
                    if not all(dim in metric_stream.dims for dim in coord_dims):
                        _logger.debug(
                            f"Skipping coordinate '{coord_name}' with incompatible "
                            f"dimensions {coord_dims} (metric_stream has {metric_stream.dims})"
                        )
                        continue

                    if coord_name not in metric_stream.coords:
                        coord_shape = tuple(len(metric_stream.coords[dim]) for dim in coord_dims)
                        metric_stream = metric_stream.assign_coords(
                            {
                                coord_name: xr.DataArray(
                                    np.full(coord_shape, "", dtype=object),
                                    dims=coord_dims,
                                    coords={dim: metric_stream.coords[dim] for dim in coord_dims},
                                )
                            }
                        )

                    indexers = {dim: criteria[dim] for dim in coord_dims if dim in criteria}
                    metric_stream.coords[coord_name].loc[indexers] = combined_metrics.coords[
                        coord_name
                    ]

        _logger.info(f"Scores for run {reader.run_id} - {stream} calculated successfully.")
        _logger.debug(f"all_metric_attrs keys: {list(all_metric_attrs.keys())}")

        # Build local dictionary for this region
        for metric, parameters in metrics.items():
            metric_data = metric_stream.sel({"metric": metric}).assign_attrs(parameters)
            # Restore metric-specific attributes from all forecast steps
            # Attributes are the same across forecast steps for a given metric
            for (_stored_fstep, stored_metric), attrs in all_metric_attrs.items():
                if stored_metric == metric and attrs:
                    _logger.debug(f"Restoring {len(attrs)} attributes for {metric}")
                    metric_data.attrs.update(attrs)
                    break

            local_scores.setdefault(metric, {}).setdefault(region, {}).setdefault(stream, {})[
                reader.run_id
            ] = metric_data

    return local_scores


def plot_score_maps_per_stream(
    reader: Reader,
    stream: str,
    regions: list[str],
    metrics_dict: dict,
    output_data: "ReaderOutput | None" = None,
) -> None:
    """Plot spatial score maps for all regions and forecast steps.

    This is the public counterpart to :func:`calc_scores_per_stream`.  It
    recomputes scores aggregated over the **sample** dimension (keeping the
    spatial ``ipoint`` dimension) so that the results can be displayed as 2-D
    maps.  Call it after :func:`calc_scores_per_stream` and pass the same
    pre-loaded *output_data* to avoid re-reading from disk.

    All ``(region, fstep)`` combinations are dispatched in parallel via loky.
    The number of outer workers is controlled by ``num_plot_workers`` in the
    eval config (same key used by :func:`plot_data`).

    Parameters
    ----------
    reader : Reader
        Reader object containing all info about a particular run.
    stream : str
        Stream name to plot score maps for.
    regions : list[str]
        List of regions to plot.
    metrics_dict : dict
        Dictionary mapping region names to metric dicts (same shape as for
        :func:`calc_scores_per_stream`).
    output_data : ReaderOutput | None
        Pre-loaded data.  When provided, ``reader.get_data()`` is skipped —
        pass the same object used for scoring to avoid a second I/O round.
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
            stream,
            fsteps=fsteps,
            samples=samples,
            channels=channels,
            ensemble=ensemble,
        )

    da_preds = output_data.prediction
    da_tars = output_data.target
    fsteps = sorted(da_preds.keys())
    aligned_clim_data = get_climatology(reader, da_tars, stream)

    # Resolve worker count once — shared across all fstep/region calls below.
    n_plot_workers = _resolve_num_plot_workers(int(reader.eval_cfg.get("num_plot_workers", 0)))

    # Extract picklable config from reader so loky workers don't need the
    # full Reader object (which may contain file handles / locks).
    cfg = reader.global_plotting_options
    plotter_cfg = {
        "image_format": cfg.get("image_format", "png"),
        "dpi_val": cfg.get("dpi_val", 300),
        "fig_size": cfg.get("fig_size", (8, 10)),
    }
    output_basedir = str(reader.runplot_dir)
    run_id = reader.run_id

    # Build one task per (region, fstep) — all data pre-computed here so
    # workers receive plain DataArrays and dicts only.
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
    """Module-level loky worker: compute scores + plot maps for one (region, fstep).

    Accepts only plain dicts, strings, and DataArrays so that loky can pickle
    this task without needing the full :class:`Reader` object.

    Parameters
    ----------
    plotter_cfg : dict
        Plotter configuration (image_format, dpi_val, fig_size).
    output_basedir : str
        Path to ``reader.runplot_dir``.
    map_dir : str
        Output directory for score-map plots.
    stream : str
        Stream name.
    region : str
        Region name.
    score_data : VerifiedData
        Pre-masked prediction/target data for this (region, fstep).
    metrics : dict
        Metric name → parameter dict.
    fstep : int
        Forecast step index.
    run_id : str
        Run identifier (logging only).
    """
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
    """Plot 2D score maps for all metrics and channels for one (region, fstep).

    Accepts only picklable arguments so it can be called from both the
    parallel loky worker :func:`_score_map_fstep_worker` and directly in
    sequential fallback paths.

    Parameters
    ----------
    plotter_cfg : dict
        Plotter configuration (image_format, dpi_val, fig_size).
    output_basedir : str
        Path to ``reader.runplot_dir``.
    map_dir : str
        Directory where the plots are saved.
    stream : str
        Stream name to plot score maps for.
    region : str
        Region name to plot score maps for.
    score_data : VerifiedData
        Prediction and target stored in the data class.
    metrics : dict
        Metric name → parameter dict.
    fstep : int
        Forecast step to plot.
    run_id : str
        Run identifier used in log messages.
    """
    # TODO: add support for climatology-dependent metrics as well

    preds = score_data.prediction

    # --- Parallel metric computation (threads: xarray/numpy release the GIL) ---
    metric_names = list(metrics.keys())
    metric_params = list(metrics.values())
    score_results: list[xr.DataArray | None] = [None] * len(metric_names)
    with ThreadPoolExecutor(max_workers=min(12, len(metric_names))) as executor:
        future_to_idx = {
            executor.submit(
                get_score,
                score_data,
                m,
                agg_dims="sample",
                parameters=p,
            ): i
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

    # --- Build task list: one entry per (metric, ens, channel) ---
    # Each task is a plain-dict so it is picklable by loky workers.
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

    # --- Sequential scatter plots: fsteps already run in parallel loky workers,
    # so no nested loky is needed here. ---
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
    """Plot a single score-map scatter plot.

    Module-level so it is picklable by loky workers. Each worker creates its
    own :class:`Plotter` instance, keeping matplotlib state isolated.

    Parameters
    ----------
    plotter_cfg : dict
        Plain-dict copy of the plotter configuration (image_format, dpi_val, fig_size).
    output_basedir : str
        Path to ``reader.runplot_dir`` (passed to :class:`Plotter`).
    stream : str
        Stream name.
    data : xr.DataArray
        Pre-selected, squeezed DataArray for this (metric, channel[, ens]).
    map_dir : str
        Output directory for the plot file.
    channel : str
        Variable/channel name.
    region : str
        Region name.
    tag : str
        Filename tag.
    title : str
        Plot title.
    """

    matplotlib.use("Agg")

    plotter = Plotter(plotter_cfg, Path(output_basedir), stream)
    plotter.scatter_plot(data, Path(map_dir), channel, region, tag=tag, title=title)


# ---------------------------------------------------------------------------
#  Parallel plotting helpers
# ---------------------------------------------------------------------------


def _resolve_num_plot_workers(requested: int = 0) -> int:
    """Return a safe number of parallel plot workers.

    Parameters
    ----------
    requested : int
        Value from config (``num_plot_workers``).  ``0`` means auto-detect.

    Returns
    -------
    int
        Number of workers (≥ 1).
    """
    if requested > 0:
        return min(requested, os.cpu_count() or 12)

    # Auto-detect safe parallelism based on process headroom
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
        min_headroom = 64

        if available < min_headroom:
            _logger.info(
                f"Parallel plotting: low process headroom "
                f"({available}/{soft_limit} slots free). Using sequential plotting."
            )
            return 1

        # Be conservative — plotting is memory-heavy (matplotlib + cartopy)
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
    """Plot all maps/histograms for a single (fstep, sample) pair.

    This is a **module-level** function so that it is picklable by loky.
    Each worker creates its own :class:`Plotter` instance (and therefore its
    own matplotlib state), avoiding all thread-safety issues.

    Parameters
    ----------
    plotter_cfg : dict
        Plain-dict copy of the plotter configuration.
    output_basedir : str
        Path to ``reader.runplot_dir``.
    tars : xr.DataArray
        Target data for this fstep (all samples).
    preds : xr.DataArray
        Prediction data for this fstep (all samples).
    bias_data : xr.DataArray | None
        ``preds - tars`` for this fstep (or *None* when bias not requested).
    sample : int | str
        The sample identifier to plot.
    fstep : int | str
        The forecast step identifier.
    stream : str
        Stream name.
    plot_chs : list[str]
        Channels / variables to plot.
    ensemble : list
        Ensemble members to iterate over.
    plot_maps, plot_bias, plot_target, plot_histograms : bool
        Feature flags.
    maps_config, bias_config : dict
        Plain-dict copies of the per-variable colour-range configs.
    """

    matplotlib.use("Agg")  # ensure non-interactive backend in worker

    # Convert plain dicts back to OmegaConf for Plotter compatibility
    maps_cfg = oc.OmegaConf.create(maps_config)
    bias_cfg = oc.OmegaConf.create(bias_config)

    plotter = Plotter(plotter_cfg, Path(output_basedir))

    data_selection = {
        "sample": sample,
        "stream": stream,
        "forecast_step": fstep,
    }

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

    # clean_data_selection is called inside create_maps/create_histograms,
    # but call it explicitly to be safe.
    plotter.clean_data_selection()


def plot_data(
    reader: Reader,
    stream: str,
    global_plotting_opts: dict,
    output_data: ReaderOutput | None = None,
) -> None:
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
    output_data : ReaderOutput | None
        Pre-loaded data.  When provided, reader.get_data() is skipped.
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
        "plot_subtimesteps": reader.get_inference_stream_attr(stream, "tokenize_spacetime", False)
        | plot_settings.get("plot_subtimesteps", False),
    }
    plotter = Plotter(plotter_cfg, reader.runplot_dir)

    available_data = reader.check_availability(stream, mode="plotting")

    # Check if maps should be plotted and handle configuration if provided
    plot_maps = plot_settings.get("plot_maps", False)
    if not isinstance(plot_maps, bool):
        raise TypeError("plot_maps must be a boolean.")

    plot_bias = plot_settings.get("plot_bias", True)
    if not isinstance(plot_bias, bool):
        raise TypeError("plot_bias must be a boolean.")

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

    # -----------------------------------------------------------------
    # Filter pre-loaded data to plotting selection
    # -----------------------------------------------------------------
    # When output_data is shared with scoring, it may contain more fsteps,
    # samples, and channels than the plotting config requests.  Restrict to
    # the plotting-specific subset so we don't plot everything.
    plot_fstep_set = set(available_data.fsteps) if available_data.fsteps is not None else None
    plot_sample_set = set(available_data.samples) if available_data.samples is not None else None
    plot_channel_set = set(available_data.channels) if available_data.channels is not None else None

    # ---- Substep-aware fstep filter ----
    # When the raw I/O splits hourly sub-steps, the output dict may contain
    # *more* fstep keys than the zarr-level set returned by check_availability.
    # For example, 3 zarr fsteps × 6 sub-steps → keys 1..18, but
    # check_availability returns {1, 2, 3}.  Detect this and expand the
    # filter to include all output keys so that sub-steps are not dropped.
    output_fstep_keys = set(da_tars.keys())
    if plot_fstep_set is not None and output_fstep_keys - plot_fstep_set:
        # Output has keys beyond the filter set → sub-step expansion happened.
        # Check if the user requested "all" (original config was None / "all")
        # by seeing if plot_fstep_set matches the zarr-level fstep set.
        zarr_fsteps = set(int(f) for f in reader.get_forecast_steps())
        if plot_fstep_set == zarr_fsteps:
            # "all" was requested — expand to all output keys
            _logger.debug(
                f"Sub-step expansion detected: output has {len(output_fstep_keys)} "
                f"entries vs {len(zarr_fsteps)} zarr fsteps. "
                f"Expanding plotting filter to all output fsteps."
            )
            plot_fstep_set = output_fstep_keys

    # Filter fsteps
    if plot_fstep_set is not None:
        da_tars = {fs: da for fs, da in da_tars.items() if fs in plot_fstep_set}
        da_preds = {fs: da for fs, da in da_preds.items() if fs in plot_fstep_set}

    if not da_tars:
        _logger.info(f"Skipping Plot Data for {stream}. No matching fsteps after filtering.")
        return

    # get common ranges across all run_ids
    if not isinstance(global_plotting_opts.get(stream), oc.DictConfig):
        global_plotting_opts[stream] = oc.DictConfig({})
    maps_config = common_ranges(
        da_tars, da_preds, available_data.channels, global_plotting_opts[stream]
    )
    bias_config = bias_ranges(
        da_tars, da_preds, available_data.channels, global_plotting_opts[stream]
    )

    # Convert OmegaConf to plain dicts for pickling across loky workers
    maps_config_dict = oc.OmegaConf.to_container(maps_config, resolve=True)
    bias_config_dict = oc.OmegaConf.to_container(bias_config, resolve=True)
    output_basedir = str(reader.runplot_dir)

    # Determine parallel workers
    num_plot_workers = _resolve_num_plot_workers(int(reader.eval_cfg.get("num_plot_workers", 0)))

    # Build task list: one entry per (fstep, sample)
    tasks: list[dict] = []
    for (fstep, tars), (_, preds) in zip(da_tars.items(), da_preds.items(), strict=False):
        # Channels available in the data, filtered to plotting selection
        all_chs = list(np.atleast_1d(tars.channel.values))
        plot_chs = (
            [ch for ch in all_chs if ch in plot_channel_set]
            if plot_channel_set is not None
            else all_chs
        )
        if not plot_chs:
            continue

        # Samples available in the data, filtered to plotting selection
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
            Parallel(
                n_jobs=effective_workers,
                backend="loky",
                verbose=2,
            )(delayed(_plot_single_sample)(**task) for task in tasks)

            # Clean up loky workers to free process slots
            with contextlib.suppress(Exception):
                get_reusable_executor().shutdown(wait=True)

        except Exception as exc:
            _logger.warning(
                f"Parallel plotting failed ({type(exc).__name__}: {exc}). "
                f"Falling back to sequential plotting."
            )
            # Clean up loky workers before fallback
            with contextlib.suppress(Exception):
                get_reusable_executor().shutdown(wait=True)

            for task in tqdm(tasks, desc=f"Plotting {run_id} - {stream} (sequential fallback)"):
                _plot_single_sample(**task)
    else:
        # Sequential path (single worker or single task)
        for task in tqdm(tasks, desc=f"Plotting {run_id} - {stream}"):
            _plot_single_sample(**task)

    if plot_animations:
        # Animations must run sequentially after all plots are written.
        # Use a single Plotter for animations (reads generated images).
        plotter = Plotter(plotter_cfg, reader.runplot_dir)
        # Recover plot_chs / plot_samples from the last fstep, filtered to
        # the plotting selection
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
    return


def metric_list_to_json(
    reader: Reader, stream: str, metrics_dict: list[xr.DataArray], regions: list[str]
):
    """
    Write the evaluation results collected in a list of xarray DataArrays for the metrics
    to stream- and metric-specific JSON files.

    Parameters
    ----------
    reader: Reader
        Reader object containing all info about the run_id.
    stream: str
        Stream name.
    metrics_dict: list
        Metrics per stream.
    regions: list
        Region names.
    """
    # stream_loaded_scores['rmse']['nhem']['ERA5']['jjqce6x5']
    reader.metrics_dir.mkdir(parents=True, exist_ok=True)

    for metric, metric_stream in metrics_dict.items():
        for region in regions:
            for run_id, metric_data in metric_stream[region][stream].items():
                # Match the expected filename pattern
                save_path = (
                    reader.metrics_dir
                    / f"{run_id}_{stream}_{region}_{metric}_chkpt{reader.mini_epoch:05d}.json"
                )
                metric_data_dict = metric_data.to_dict()

                if save_path.exists():
                    _logger.info(f"{save_path} already present")

                    with save_path.open("r") as f:
                        data_dict = json.load(f)

                    # Normalize structure
                    if "scores" not in data_dict:
                        data_dict = {"scores": [data_dict]}
                    scores = data_dict.get("scores")

                    # Try to replace existing metric with same attrs
                    for i, existing_score in enumerate(scores):
                        if existing_score["attrs"] == metric_data.attrs:
                            _logger.warning("Metric with same parameters found, replacing")
                            scores[i] = metric_data_dict
                            break
                    else:
                        scores.append(metric_data_dict)
                        _logger.info(f"Appending results to {save_path}")

                else:
                    _logger.info(f"Saving results to new file {save_path}")
                    data_dict = {"scores": [metric_data_dict]}
                with open(save_path, "w") as f:
                    json.dump(data_dict, f, indent=4)

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


############# Utility functions ############


def common_ranges(
    data_tars: list[dict],
    data_preds: list[dict],
    plot_chs: list[str],
    global_plotting_opts_stream: oc.dictconfig.DictConfig,
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
    global_plotting_opts_stream:
        the global plotting configuration for the stream as given by the configuration file, which
        may or may not include predefined ranges for some variables.
    Returns
    -------
    maps_config :
        the global plotting configuration with the ranges added and included for each variable.
    """
    maps_config = global_plotting_opts_stream.copy()
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


def bias_ranges(
    data_tars: dict,
    data_preds: dict,
    plot_chs: list[str],
    global_plotting_opts_stream: oc.dictconfig.DictConfig,
) -> oc.dictconfig.DictConfig:
    """
    Calculate symmetric bias ranges (preds - tars) per variable.

    Parameters
    ----------
    data_tars :
        Dictionary mapping forecast steps to target xarray DataArrays.
    data_preds :
        Dictionary mapping forecast steps to prediction xarray DataArrays.
    plot_chs :
        List of variable (channel) names to compute bias ranges for.
    global_plotting_opts_stream :
        The global plotting configuration for the stream, used as the base config.

    Returns
    -------
    oc.dictconfig.DictConfig
        Per-variable symmetric ranges (vmin = -abs_max, vmax = abs_max) for bias.
    """
    bias_config = global_plotting_opts_stream.copy()
    for var in plot_chs:
        bias_vals = [
            (p - t).sel(channel=var).values
            for t, p in zip(data_tars.values(), data_preds.values(), strict=False)
        ]
        abs_max = float(
            max(abs(np.concatenate(bias_vals).max()), abs(np.concatenate(bias_vals).min()))
        )
        bias_config.update({var: {"vmax": abs_max, "vmin": -abs_max}})
    return bias_config


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


def merge(dst: dict, src: dict) -> dict:
    """
    Recursively merge src into dst.
    Values in src overwrite values in dst.
    Parameters
    ----------
    dst : dict
        Destination dictionary.
    src : dict
        Source dictionary.
    Returns
    -------
    dict
        Merged dictionary.
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def parse_metric_params(metrics):
    """
    Convert a mixed list of str and dict metrics into a dict where the metric
    names are the keys and the values are dicts of parameters for that metric.
    The config might read
        metrics:
        - fbi:
            thresh: 280
        - rmse
        ...
    In python, metrics then looks like
        [{'fbi':{'thresh':280}},'rmse']
    This function converts it to
        {'fbi':{'thresh':280}, 'rmse':{}}
    """
    out = oc.DictConfig({})
    for metric in metrics:
        if isinstance(metric, str):
            out = oc.OmegaConf.merge(out, {metric: {}})
        else:
            out = oc.OmegaConf.merge(out, metric)
    return out
