# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Scores by initialisation hour of the day.

For every forecast step, the metrics are aggregated over samples and points, grouped by
the hour of day of the samples' initialisation time (``init_times``).  The result shows
how skill depends on the time of day a forecast starts; it is plotted by
``plot_init_hour_summary``.

Scoring goes through ``compute_scores_for_region``, as for the per-sample scores; this
module only attaches the ``init_hour`` coordinate it groups by and stacks the per-step
results into ``MetricResult.by_init_hour``.
"""

import logging

import xarray as xr

from weathergen.evaluate.io.io_reader import Reader, ReaderOutput
from weathergen.evaluate.scores.score_orchestration import (
    MetricResult,
    compute_scores_for_region,
)
from weathergen.evaluate.utils.regions import RegionBoundingBox

_logger = logging.getLogger(__name__)


def compute_scores_by_init_hour(
    reader: Reader,
    stream: str,
    results: dict[str, dict[str, MetricResult]],
    output_data: ReaderOutput,
    metrics_dict: dict,
    climatology: dict | None = None,
) -> None:
    """Fill ``results[region][metric].by_init_hour`` for every region of *results*.

    Parameters
    ----------
    reader : Reader
        Reader of the run (run id and ``max_workers``).
    stream : str
        Stream name.
    results : dict[str, dict[str, MetricResult]]
        Output of ``calc_scores_per_stream``; updated in place.
    output_data : ReaderOutput
        The loaded data the per-sample scores were computed from.
    metrics_dict : dict
        ``metrics_dict[region][metric] -> parameters``, as passed to ``calc_scores_per_stream``.
    climatology : dict | None
        Climatology aligned to the targets (``get_climatology``), for metrics that need it.

    Notes
    -----
    ``by_init_hour`` has dims ``(forecast_step, init_hour, channel[, ens])``.  Gridded
    streams only (the hour is attached to the ``sample`` dimension).  ``psd`` is
    skipped: it is a spectrum over space and is not aggregated over samples.
    """
    if not results or not reader.is_gridded_data(stream):
        _logger.info(f"RUN {reader.run_id} - {stream}: Skipping scores by init hour.")
        return

    fsteps = sorted(output_data.prediction)
    hours = {
        fs: da.init_times.dt.hour.fillna(-1).astype(int).reset_coords(drop=True)
        for fs, da in output_data.prediction.items()
    }
    preds, tars, clim = (
        _with_init_hour(d, hours) for d in (output_data.prediction, output_data.target, climatology)
    )

    for region, by_metric in results.items():
        metrics = {m: p for m, p in metrics_dict[region].items() if m != "psd"}
        if not metrics:
            continue
        per_fstep, _ = compute_scores_for_region(
            reader.run_id,
            stream,
            region,
            preds,
            tars,
            fsteps,
            clim,
            True,
            "init_hour",
            RegionBoundingBox.from_region_name(region),
            metrics,
            reader.eval_cfg.get("max_workers", None),
            ["sample", "ipoint"],
        )
        if not per_fstep:
            continue
        scores = xr.concat(
            [c for _, c, _ in per_fstep], dim="forecast_step", coords="different", compat="equals"
        )
        if "init_hour" not in scores.dims:  # a single init hour: get_score does not group
            scores = scores.expand_dims(init_hour=[int(hours[fsteps[0]].values[0])])
        for metric in scores.metric.values:
            by_metric[str(metric)].by_init_hour = scores.sel(metric=metric, drop=True)


def _with_init_hour(da_dict: dict | None, hours: dict) -> dict | None:
    """Copy of a per-fstep dict with ``hours[fstep]`` attached as the ``init_hour`` coord."""
    if da_dict is None:
        return None
    return {fs: da.assign_coords(init_hour=hours[fs]) for fs, da in da_dict.items()}
