# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Plotting orchestration: the plotting steps of the evaluation run loop.

The counterpart of ``score_orchestration``.  Three steps, in the order of the run:

1. ``plot_stream_data``: field plots (``field_plots``), drawn from the loaded data of one
   (run, stream), so per chunk of samples and forecast steps.
2. ``plot_stream_results``: score maps (``score_maps``) and all the videos (``animations``)
   of one (run, stream), drawn once its results are complete and all its frames are written.
3. ``plot_evaluation``: verification plots across runs (``verification_plots``), drawn once
   all runs are evaluated.
"""

from pathlib import Path

from weathergen.evaluate.io.io_reader import Reader, ReaderOutput
from weathergen.evaluate.plotting.animations import build_animations
from weathergen.evaluate.plotting.field_plots import plot_data
from weathergen.evaluate.plotting.score_maps import plot_score_maps
from weathergen.evaluate.plotting.verification_plots import plot_scores
from weathergen.evaluate.scores.score_orchestration import MetricResult


def plot_stream_data(
    reader: Reader,
    stream: str,
    global_plotting_opts: dict,
    output_data: ReaderOutput | None = None,
) -> None:
    """Field plots of one (run, stream): maps, histograms and timeseries.

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
    plot_data(reader, stream, global_plotting_opts, output_data=output_data)


def plot_stream_results(
    reader: Reader,
    stream: str,
    results: dict[str, dict[str, MetricResult]],
    global_plotting_opts: dict,
    plot_score_options: dict,
) -> None:
    """Score maps and videos of one (run, stream), from its complete results and frames.

    Parameters
    ----------
    reader : Reader
        Reader object containing all info about the run.
    stream : str
        Stream name.
    results : dict[str, dict[str, MetricResult]]
        ``results[region][metric]`` of the (run, stream); empty when nothing was scored.
    global_plotting_opts : dict
        Global plotting options (applies to all run_ids).
    plot_score_options : dict
        ``get_plot_score_options(...)``: ``plot_score_maps`` and ``plot_score_animations``.
    """
    if results and plot_score_options.get("plot_score_maps", False):
        plot_score_maps(reader, stream, results, global_plotting_options=global_plotting_opts)
    build_animations(reader, stream, results, global_plotting_opts, plot_score_options)


def plot_evaluation(
    cfg: dict, scores_dict: dict, init_hour_scores: dict, summary_dir: Path
) -> None:
    """Verification plots of all runs, from the merged scores.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.
    scores_dict : dict
        ``scores_dict[metric][region][stream][run_id] -> DataArray`` (``MetricResult.score``).
    init_hour_scores : dict
        Same nesting with the ``MetricResult.by_init_hour`` arrays; empty unless
        ``init_hour`` is requested.
    summary_dir : Path
        Directory to write plots to.
    """
    plot_scores(cfg, scores_dict, init_hour_scores, summary_dir)
