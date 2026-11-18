# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Merging the results of the chunks of one (run, stream) into the results of the whole.

A chunk is one ``_process_stream`` call on a window of samples × forecast steps; its results
are ``results[region][metric] -> MetricResult`` like those of a whole run, plus the colour
ranges of its data and its data statistics (per-sample quantities of the data, for the
plots across samples).  Merging is pure array work on them, no config and no MPI.
"""

import logging

import numpy as np
import xarray as xr

from weathergen.evaluate.scores.score_orchestration import MetricResult

_logger = logging.getLogger(__name__)


def merge_results(
    chunks: list[dict[str, dict[str, MetricResult]]], fsteps: list[list[int]] | None = None
) -> dict[str, dict[str, MetricResult]]:
    """Merge the per-chunk results of one (run, stream).

    Parameters
    ----------
    chunks : list[dict[str, dict[str, MetricResult]]]
        The ``results[region][metric]`` of every chunk, in any order.  The chunks of one
        forecast-step window hold disjoint samples; different windows hold disjoint steps.
    fsteps : list[list[int]] | None
        Per chunk, the forecast steps it is responsible for.  Steps it was scored on beyond
        them, the extra step on the right that ``froct`` / ``troct`` need, are dropped, with
        their per-step attrs.  ``None`` keeps everything.

    Returns
    -------
    dict[str, dict[str, MetricResult]]
        ``results[region][metric]`` of the whole (run, stream), samples and steps sorted.

    Notes
    -----
    - ``score``: the sample chunks of each window are concatenated along ``sample``, then
      the windows along ``forecast_step``.  Exact.
    - attrs: the metric parameters, which must agree between chunks (``ValueError``).
    - ``curves`` (``spatial``, ``by_init_hour``, the spectra of ``psd``, the quantiles of
      ``qq_analysis``): the windows are concatenated along ``forecast_step``; the sample
      chunks of one window merge variable by variable (``_merge_curves``), exactly.
    """
    parts: dict[str, dict[str, list[MetricResult]]] = {}
    for chunk, window in zip(chunks, fsteps or [None] * len(chunks), strict=True):
        for region, by_metric in chunk.items():
            for metric, r in by_metric.items():
                parts.setdefault(region, {}).setdefault(metric, []).append(in_window(r, window))
    return {
        region: {metric: _merge_metric(rs) for metric, rs in by_metric.items()}
        for region, by_metric in parts.items()
    }


def in_window(r: MetricResult, window: list[int] | None) -> MetricResult:
    """*r* restricted to the forecast steps of *window* (the score and the per-step fields)."""
    if window is None:
        return r
    keep = [int(f) for f in r.score.forecast_step.values if int(f) in window]
    if len(keep) == len(r.score.forecast_step):
        return r

    def trim(da):
        return None if da is None else da.sel(forecast_step=keep)

    return MetricResult(r.name, r.region, r.stream, r.run_id, trim(r.score), trim(r.curves))


def _merge_metric(parts: list[MetricResult]) -> MetricResult:
    """One MetricResult from the chunks of one (region, metric)."""
    first = parts[0]
    # the forecast-step windows in step order, each with its sample chunks in sample order
    by_window: dict[tuple[int, ...], list[MetricResult]] = {}
    for r in parts:
        by_window.setdefault(tuple(int(f) for f in r.score.forecast_step.values), []).append(r)
    windows = [
        sorted(by_window[w], key=lambda r: int(r.score.sample.values[0])) for w in sorted(by_window)
    ]
    score = _concat([_concat([r.score for r in w], "sample") for w in windows], "forecast_step")
    score.attrs = _merge_attrs([r.score.attrs for w in windows for r in w], first.name)
    curves = None
    if any(r.curves is not None for r in parts):
        curves = _concat([_merge_curves(w, first.name) for w in windows], "forecast_step")
    return MetricResult(first.name, first.region, first.stream, first.run_id, score, curves)


def _merge_curves(window: list[MetricResult], metric: str) -> xr.Dataset:
    """The curves of one forecast-step window from its sample chunks, variable by variable:
    counts (``*_counts``) add up, the others are means over the chunk's samples and are
    averaged with the counts as weights (``<name>_counts`` when there is one, else the
    chunk's number of samples), which is exact."""
    if len(window) == 1:
        return window[0].curves
    first = window[0].curves
    merged = {}
    for name in first.data_vars:
        parts = [r.curves[name] for r in window]
        if name.endswith("_counts"):
            merged[name] = sum(parts).assign_attrs(first[name].attrs)
        else:
            if f"{name}_counts" in first:
                weights = [r.curves[f"{name}_counts"] for r in window]
            else:
                weights = [xr.DataArray(float(r.score.sizes["sample"])) for r in window]
            total = sum(weights)
            # rmse is the root of a mean of squares: averaged through its square, exactly;
            # a mean of per-sample quantities (mae, bias, a spectrum) directly
            power = 2 if metric == "rmse" else 1
            mean = sum(p**power * w for p, w in zip(parts, weights, strict=True)) / total.where(
                total > 0
            )
            merged[name] = (mean ** (1 / power)).transpose(*first[name].dims)
            merged[name].attrs = first[name].attrs
    _logger.warning(
        f"{metric}: the curves of {len(window)} sample chunks of a window were merged: means "
        "over the samples averaged with the sample counts as weights (exact for means and "
        "rmse; approximate for ratio or correlation scores)."
    )
    return xr.Dataset(merged, attrs=first.attrs)


def merge_statistics(parts: list[dict[str, xr.Dataset]]) -> dict[str, xr.Dataset]:
    """The data statistics of one (run, stream) from those of its chunks, in any order.

    A chunk's statistics are ``{name: Dataset}`` of per-sample quantities with ``sample``
    and ``forecast_step`` dims (``plot_data``: ``timeseries`` is the region means, ...).  Per
    name, the forecast-step windows of each sample window are concatenated along
    ``forecast_step``, then the sample windows along ``sample``, both sorted.  Exact under
    any chunking: every statistic is per (sample, forecast step)."""
    merged = {}
    for name in {name for part in parts for name in part}:
        by_samples: dict[tuple[int, ...], list[xr.Dataset]] = {}
        for ds in [part[name] for part in parts if name in part]:
            by_samples.setdefault(tuple(int(s) for s in ds.sample.values), []).append(ds)
        windows = [_concat(by_samples[w], "forecast_step") for w in sorted(by_samples)]
        merged[name] = _concat(windows, "sample")
    return merged


def _concat(arrays: list, dim: str):
    """Concatenate along *dim* and sort by it; the arrays' other coordinates must agree."""
    if len(arrays) == 1:
        return arrays[0]
    # data_vars is a Dataset-only argument (xarray warns about its changing default)
    kwargs = {"data_vars": "all"} if isinstance(arrays[0], xr.Dataset) else {}
    return xr.concat(arrays, dim=dim, coords="different", compat="equals", **kwargs).sortby(dim)


def _merge_attrs(attrs_list: list[dict], metric: str) -> dict:
    """The metric parameters, which must agree between the chunks."""
    params = attrs_list[0]
    for attrs in attrs_list[1:]:
        if not _same(attrs, params):
            raise ValueError(f"{metric}: parameters differ between chunks: {params}, {attrs}")
    return dict(params)


def _same(a, b) -> bool:
    """Equality that also works for list / array attr values."""
    try:
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))
    except (ValueError, TypeError):
        return a == b
