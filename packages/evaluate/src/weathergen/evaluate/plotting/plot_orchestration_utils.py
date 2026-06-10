import logging

import numpy as np
import pandas as pd
import xarray as xr
from joblib import delayed
from numpy.typing import NDArray

from weathergen.evaluate.io.data.io_orchestration import dispatch_parallel
from weathergen.evaluate.io.io_reader import ReaderOutput
from weathergen.evaluate.scores.score import VerifiedData, get_score
from weathergen.evaluate.scores.score_orchestration import get_next_fstep_data
from weathergen.evaluate.utils.regions import RegionBoundingBox

_logger = logging.getLogger(__name__)


def group_by_init_hour(
    output_data: ReaderOutput,
) -> dict[int, NDArray]:
    """Group sample indices by the hour of day of the initialisation time.

    The initialisation time is taken from the ``init_times`` coordinate
    on the sample dimension (which stores the end of the conditioning window, i.e.
    the forecast reference time).

    Returns only sample index arrays (no data copies) to avoid doubling memory.
    Use ``da.sel(sample=indices)`` on the original data when processing each group.

    Parameters
    ----------
    output_data : ReaderOutput
        Pre-loaded data with ``target`` and ``prediction`` dicts keyed by fstep.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping from hour (0–23) to sample indices belonging to that hour.
    """
    first_tar = next(iter(output_data.target.values()))
    if "init_times" not in first_tar.coords:
        _logger.warning("Cannot group by init hour: 'init_times' coordinate not found.")
        return {}

    init_times = pd.DatetimeIndex(first_tar.init_times.values)
    hours = init_times.hour
    samples = first_tar.sample.values

    grouped: dict[int, NDArray] = {int(hour): samples[hours == hour] for hour in sorted(set(hours))}

    _logger.info(
        f"Grouped {len(samples)} samples into {len(grouped)} init-hour bins: "
        f"{sorted(grouped.keys())}."
    )
    return grouped


def group_by_source_n_points(
    output_data: ReaderOutput,
    stream_filter: str | list[str] | None = None,
    n_bins: int = 30,
) -> tuple[dict[int, NDArray], str]:
    """Group sample indices by binned total source observation count.

    Parameters
    ----------
    output_data : ReaderOutput
        Loaded data with ``source_n_points_<stream>`` coordinates.
    stream_filter : str | list[str] | None
        If given, sum only these source streams. If None, sum all
        available ``source_n_points_*`` coordinates.
    n_bins : int
        Number of quantile bins to create.

    Returns
    -------
    groups : dict[int, np.ndarray]
        Mapping from bin midpoint (total n_points) to sample indices.
    label : str
        Human-readable label describing which streams were summed.
    """
    da = next(iter(output_data.prediction.values()))

    # Collect source_n_points coords
    source_coords = {
        c: da.coords[c].values
        for c in da.coords
        if c.startswith("source_n_points_")
    }
    if not source_coords:
        return {}, ""

    # Filter to requested streams
    if stream_filter is not None:
        if isinstance(stream_filter, str):
            stream_filter = [stream_filter]
        source_coords = {
            c: v
            for c, v in source_coords.items()
            if any(c == f"source_n_points_{s}" for s in stream_filter)
        }

    if not source_coords:
        return {}, ""

    # Sum across source streams (total obs feeding each sample)
    # Exclude -1 sentinels from sum
    n_samples = len(da.sample)
    total_pts = np.zeros(n_samples, dtype=np.int64)
    for arr in source_coords.values():
        valid = arr > 0
        total_pts[valid] += arr[valid]

    # Only bin samples with valid (>0) total points
    valid_mask = total_pts > 0
    if not valid_mask.any():
        return {}, ""

    valid_pts = total_pts[valid_mask]

    _logger.info(
        f"source_n_points binning: {n_samples} samples, {valid_mask.sum()} valid. "
        f"total_pts stats: min={valid_pts.min()}, max={valid_pts.max()}, "
        f"mean={valid_pts.mean():.0f}, unique_values={len(np.unique(valid_pts))}"
    )

    # Bin into quantiles
    bin_edges = np.quantile(valid_pts, np.linspace(0, 1, n_bins + 1))
    bin_edges = np.unique(bin_edges)  # remove duplicates

    _logger.info(
        f"source_n_points binning: requested {n_bins} bins, "
        f"got {len(bin_edges) - 1} after dedup (edges: {bin_edges[:5]}...{bin_edges[-3:]})"
    )

    sample_vals = da.sample.values
    groups: dict[int, NDArray] = {}
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < len(bin_edges) - 2:
            mask = (total_pts >= lo) & (total_pts < hi)
        else:
            mask = (total_pts >= lo) & (total_pts <= hi)
        if mask.any():
            midpoint = int((lo + hi) / 2)
            groups[midpoint] = sample_vals[mask]

    # Build label from stream names
    stream_names = [c.replace("source_n_points_", "") for c in source_coords]
    label = "+".join(sorted(stream_names))

    _logger.info(
        f"Grouped {valid_mask.sum()} valid samples into {len(groups)} n_points bins "
        f"(streams: {label})."
    )
    return groups, label


def get_per_sample_source_n_points(
    output_data: ReaderOutput,
    stream_filter: str | list[str] | None = None,
) -> tuple[NDArray, str]:
    """Get total source n_points per sample (no binning).

    Parameters
    ----------
    output_data : ReaderOutput
        Loaded data with ``source_n_points_<stream>`` coordinates.
    stream_filter : str | list[str] | None
        If given, sum only these source streams. If None, sum all.

    Returns
    -------
    total_pts : NDArray
        Array of shape (n_samples,) with total source obs per sample.
        Entries are 0 for samples without valid source data.
    label : str
        Human-readable label describing which streams were summed.
    """
    da = next(iter(output_data.prediction.values()))

    source_coords = {
        c: da.coords[c].values
        for c in da.coords
        if c.startswith("source_n_points_")
    }
    if not source_coords:
        return np.array([], dtype=np.int64), ""

    if stream_filter is not None:
        if isinstance(stream_filter, str):
            stream_filter = [stream_filter]
        source_coords = {
            c: v
            for c, v in source_coords.items()
            if any(c == f"source_n_points_{s}" for s in stream_filter)
        }

    if not source_coords:
        return np.array([], dtype=np.int64), ""

    n_samples = len(da.sample)
    total_pts = np.zeros(n_samples, dtype=np.int64)
    for arr in source_coords.values():
        valid = arr > 0
        total_pts[valid] += arr[valid]

    stream_names = [c.replace("source_n_points_", "") for c in source_coords]
    label = "+".join(sorted(stream_names))
    return total_pts, label


def _compute_scores_for_fstep(
    region: str,
    fstep: int,
    metric_names: list[str],
    metric_params: list,
    score_data: VerifiedData,
    preds_r: xr.DataArray,
) -> tuple[str, int, list, xr.DataArray, list[str]]:
    """Compute scores for a single (region, fstep) pair (parallelisable worker).

    Returns ``(region, fstep, score_results, preds_r, metric_names)``.
    """
    score_results: list[xr.DataArray | None] = [
        get_score(score_data, m, agg_dims="sample", parameters=p)
        for m, p in zip(metric_names, metric_params, strict=False)
    ]
    return region, fstep, score_results, preds_r, metric_names


def _compute_scores(
    regions: list[str],
    metrics_dict: dict,
    fsteps: list,
    da_preds: dict,
    da_tars: dict,
    aligned_clim_data: dict | None,
    n_workers: int | None = None,
) -> tuple[dict, dict]:
    """Compute scores for all (region, fstep) pairs. Score computation is parallelised across
      (region, fstep) pairs.

    Returns
    -------
    computed : dict[tuple, tuple]
        ``{(region, fstep): (score_results, preds_r, metric_names)}``
    raw_results : list[tuple]
        List of raw results from parallel score computation, each item is a tuple of
        ``(region, fstep, score_results, preds_r, metric_names)``.
    """
    # Build one task per (region, fstep) with pre-applied region masking.
    tasks = []
    for region in regions:
        bbox = RegionBoundingBox.from_region_name(region)
        metrics = metrics_dict[region]
        metric_names = list(metrics.keys())
        metric_params = list(metrics.values())
        for fstep in fsteps:
            tars_fs = da_tars[fstep]
            preds_fs = da_preds[fstep]
            preds_next, tars_next = get_next_fstep_data(fstep, da_preds, da_tars, fsteps)
            climatology = aligned_clim_data[fstep] if aligned_clim_data else None
            tars_r, preds_r, tars_next_r, preds_next_r = [
                bbox.apply_mask(x) if x is not None else None
                for x in (tars_fs, preds_fs, tars_next, preds_next)
            ]
            tasks.append(
                dict(
                    region=region,
                    fstep=fstep,
                    metric_names=metric_names,
                    metric_params=metric_params,
                    score_data=VerifiedData(
                        preds_r, tars_r, preds_next_r, tars_next_r, climatology
                    ),
                    preds_r=preds_r,
                )
            )

    # Compute scores in parallel across (region, fstep) pairs.
    calls = [delayed(_compute_scores_for_fstep)(**t) for t in tasks]
    raw_results = dispatch_parallel(
        calls, n_workers=n_workers, backend="loky", desc="Score computation"
    )

    # Accumulate per-channel colour ranges from the completed results.
    computed: dict[tuple, tuple] = {}
    for region, fstep, score_results, preds_r, metric_names in raw_results:
        computed[(region, fstep)] = (score_results, preds_r, metric_names)

    return computed, raw_results


def _compute_ranges(
    raw_results: list[tuple[str, int, list, xr.DataArray, list[str]]],
) -> dict:
    """Compute colour ranges for each metric/region/channel from the raw score results.

    Returns
    -------
    score_ranges_dict : dict
        ``{metric: {region: {channel: {'vmin': float, 'vmax': float}}}}``
    """
    # Accumulate per-channel colour ranges from the completed results.

    score_ranges_dict: dict = {}
    for region, _, score_results, _, metric_names in raw_results:
        for metric, result in zip(metric_names, score_results, strict=False):
            if result is None or "channel" not in result.coords:
                continue
            score_ranges_dict.setdefault(metric, {}).setdefault(region, {})
            for ch in result.coords["channel"].values:
                vals = result.sel(channel=ch).values.flatten()
                vals = vals[~np.isnan(vals)]
                if vals.size == 0:
                    continue
                ch_key = str(ch)
                vmin, vmax = float(vals.min()), float(vals.max())
                prev = score_ranges_dict[metric][region].get(ch_key)
                score_ranges_dict[metric][region][ch_key] = {
                    "vmin": min(prev["vmin"], vmin) if prev else vmin,
                    "vmax": max(prev["vmax"], vmax) if prev else vmax,
                }

    return score_ranges_dict
