# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Plotting orchestration: the two plotting steps of the evaluation run loop.

The counterpart of ``score_orchestration``.  Field plots (``field_plots``) are drawn from
the loaded data of one (run, stream), and so can be drawn per chunk of samples and forecast
steps; verification plots (``verification_plots``) are drawn from the scores of all runs,
once they are complete.  In between, once all the chunks of a (run, stream) are done, come
its score maps (``score_maps``) and its videos (``animations``), called by name from the
run loop.
"""

from pathlib import Path

import xarray as xr

from weathergen.evaluate.io.io_reader import Reader, ReaderOutput
from weathergen.evaluate.plotting.field_plots import plot_data, plot_timeseries
from weathergen.evaluate.plotting.verification_plots import plot_scores


def plot_stream_data(
    reader: Reader,
    stream: str,
    global_plotting_opts: dict,
    output_data: ReaderOutput | None = None,
) -> dict[str, xr.Dataset]:
    """Field plots of one (run, stream): the maps and histograms are drawn; the plots
    across samples are returned as data statistics, ``{name: Dataset}`` of per-sample
    quantities (``plot_data``), drawn by ``plot_evaluation`` once all runs are done.

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
    return plot_data(reader, stream, global_plotting_opts, output_data=output_data)


def plot_evaluation(
    cfg: dict, scores_dict: dict, init_hour_scores: dict, statistics: dict, summary_dir: Path
) -> None:
    """Plots across all runs: the verification plots from the merged scores, the plots
    across samples from the merged data statistics.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.
    scores_dict : dict
        ``scores_dict[metric][region][stream][run_id] -> DataArray`` (``MetricResult.score``).
    init_hour_scores : dict
        Same nesting with the ``MetricResult.by_init_hour`` arrays; empty unless
        ``init_hour`` is requested.
    statistics : dict
        ``statistics[name][stream][run_id] -> Dataset``, the data statistics of the runs
        (``plot_data``) as ``accumulate_stream_results`` nests them: ``timeseries`` are the
        region means, drawn one figure per initialisation time with every run on it.
    summary_dir : Path
        Directory to write plots to.
    """
    if scores_dict:
        plot_scores(cfg, scores_dict, init_hour_scores, summary_dir)
    if "timeseries" in statistics:
        plot_timeseries(cfg, statistics["timeseries"], summary_dir)
