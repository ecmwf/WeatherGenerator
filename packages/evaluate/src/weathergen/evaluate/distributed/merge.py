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
are ``results[region][metric] -> MetricResult`` like those of a whole run.  Merging is pure
array work on them, no config and no MPI.
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
    - ``spatial`` and ``by_init_hour``: the windows are concatenated along
      ``forecast_step``.  Both are means over the samples, so several sample chunks of one
      window cannot be merged exactly (``ValueError``); a chunked run does not compute them
      under sample chunking.
    - attrs: the metric parameters must agree between chunks (``ValueError``); the per-step
      payloads (``fstep_<n>/...``, ``attr_fsteps``) are the union over the windows, and
      within a window those of the first chunk, with a warning when the chunks disagree
      (``psd`` and the histogram metrics are quantities of one set of samples).
    """
    parts: dict[str, dict[str, list[MetricResult]]] = {}
    for chunk, window in zip(chunks, fsteps or [None] * len(chunks), strict=True):
        for region, by_metric in chunk.items():
            for metric, r in by_metric.items():
                parts.setdefault(region, {}).setdefault(metric, []).append(_in_window(r, window))
    return {
        region: {metric: _merge_metric(rs) for metric, rs in by_metric.items()}
        for region, by_metric in parts.items()
    }


def _in_window(r: MetricResult, window: list[int] | None) -> MetricResult:
    """*r* restricted to the forecast steps of *window* (arrays and per-step attrs)."""
    if window is None:
        return r
    keep = [int(f) for f in r.score.forecast_step.values if int(f) in window]
    if len(keep) == len(r.score.forecast_step):
        return r

    def trim(da):
        return None if da is None else da.sel(forecast_step=keep)

    score = trim(r.score)
    score.attrs = {
        k: v
        for k, v in r.score.attrs.items()
        if not k.startswith("fstep_") or int(k[len("fstep_") :].split("/")[0]) in keep
    }
    attr_fsteps = [f for f in score.attrs.pop("attr_fsteps", []) if f in keep]
    if attr_fsteps:
        score.attrs["attr_fsteps"] = attr_fsteps
    return MetricResult(
        r.name, r.region, r.stream, r.run_id, score, trim(r.spatial), trim(r.by_init_hour)
    )


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

    # means over the samples: one chunk per window, concatenated over the windows
    aggregated = {}
    for field in ("spatial", "by_init_hour"):
        if any(vars(r)[field] is not None for r in parts):
            if any(len(w) > 1 for w in windows):
                raise ValueError(
                    f"{first.name}: the {field} scores of several sample chunks cannot be merged "
                    "(mean over the samples)."
                )
            aggregated[field] = _concat([vars(w[0])[field] for w in windows], "forecast_step")
    return MetricResult(first.name, first.region, first.stream, first.run_id, score, **aggregated)


def _concat(arrays: list[xr.DataArray], dim: str) -> xr.DataArray:
    """Concatenate along *dim* and sort by it; the arrays' other coordinates must agree."""
    if len(arrays) == 1:
        return arrays[0]
    return xr.concat(arrays, dim=dim, coords="different", compat="equals").sortby(dim)


def _merge_attrs(attrs_list: list[dict], metric: str) -> dict:
    """Metric parameters (must agree) plus the union of the per-step payloads."""
    params = {k: v for k, v in attrs_list[0].items() if _is_param(k)}
    merged = dict(params)
    attr_fsteps: set[int] = set()
    warned = False
    for attrs in attrs_list:
        other = {k: v for k, v in attrs.items() if _is_param(k)}
        if not _same(other, params):
            raise ValueError(f"{metric}: parameters differ between chunks: {params}, {other}")
        attr_fsteps.update(int(f) for f in attrs.get("attr_fsteps", []))
        for k, v in attrs.items():
            if not k.startswith("fstep_"):
                continue
            if k not in merged:
                merged[k] = v
            elif not warned and not _same(merged[k], v):
                warned = True
                _logger.warning(
                    f"{metric}: per-step attrs differ between the sample chunks of a window; "
                    "keeping the first chunk's (they are quantities of one set of samples)."
                )
    if attr_fsteps:
        merged["attr_fsteps"] = sorted(attr_fsteps)
    return merged


def _is_param(key: str) -> bool:
    return not key.startswith("fstep_") and key != "attr_fsteps"


def _same(a, b) -> bool:
    """Equality that also works for list / array attr values."""
    try:
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))
    except (ValueError, TypeError):
        return a == b
