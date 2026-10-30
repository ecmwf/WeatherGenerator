# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The chunked evaluation run: every (run, stream) as chunk tasks, merged back.

Runs the chunk tasks one after the other in this process; the MPI master / worker loop over
the same tasks comes on top of it.
"""

import logging

from mlflow.client import MlflowClient

from weathergen.evaluate.distributed.chunking import (
    build_plotting_chunk_tasks,
    build_scoring_chunk_tasks,
    parse_distributed_config,
    windows,
)
from weathergen.evaluate.distributed.merge import merge_results
from weathergen.evaluate.plotting.animations import build_animations
from weathergen.evaluate.plotting.score_maps import plot_score_maps
from weathergen.evaluate.run_evaluation import (
    _process_stream,
    accumulate_stream_results,
    build_stream_tasks,
    cached_scores_and_metrics_to_compute,
    finalise,
    get_reader,
    write_missing_scores,
)
from weathergen.evaluate.scores.score_orchestration import MetricResult

_logger = logging.getLogger(__name__)


def run_chunked(cfg: dict, mlflow_client: MlflowClient | None) -> None:
    """``evaluate_from_config`` with every (run, stream) cut into chunks (``distributed``)."""
    _, sample_size, fstep_size = parse_distributed_config(cfg)
    stream_results = [
        _run_stream_in_chunks(task, sample_size, fstep_size) for task in build_stream_tasks(cfg)
    ]
    finalise(cfg, *accumulate_stream_results(stream_results), mlflow_client)


def _run_stream_in_chunks(
    task: dict, sample_size: int | None, fstep_size: int | None
) -> tuple[str, str, dict[str, dict[str, MetricResult]]]:
    """``_process_stream`` of one (run, stream), as chunk tasks.

    Scoring chunk tasks first (their results merged, the missing scores written), then the
    plotting chunk tasks, then the score maps and videos of the whole (run, stream).  A run
    whose reader does not load data has nothing to chunk and goes through ``_process_stream``
    as it is.
    """
    run_id, run, stream = task["run_id"], task["run"], task["stream"]
    type_ = run.get("type", "zarr")
    if type_ not in ("zarr", "anemoi-target"):
        return _process_stream(**task)
    reader = get_reader(type_, run, run_id, task["private_paths"], task["regions"], task["metrics"])
    stream_cfg = reader.get_stream(stream)
    if not stream_cfg:
        return _process_stream(**task)
    global_plotting_opts = task["global_plotting_opts"]
    plot_score_options = dict(task["plot_score_options"])

    results: dict[str, dict[str, MetricResult]] = {}
    computed: dict[str, dict[str, MetricResult]] = {}
    if stream_cfg.get("evaluation"):
        available = reader.check_availability(stream, mode="evaluation")
        if len(windows(available.samples, sample_size)) > 1:
            # means over the samples: not merged across sample chunks (merge_results)
            for option, name in (
                ("plot_score_maps", "score maps"),
                ("plot_score_init_hour", "init-hour scores"),
            ):
                if plot_score_options.get(option, False):
                    _logger.warning(
                        f"RUN {run_id} - {stream}: {name} are skipped when the samples are chunked."
                    )
                    plot_score_options[option] = False
        task = task | {"plot_score_options": plot_score_options}
        results, metrics_to_compute, recomputable_metrics = cached_scores_and_metrics_to_compute(
            reader, stream, task["regions"], task["metrics"], plot_score_options, True
        )
        if metrics_to_compute:
            tasks, responsible = build_scoring_chunk_tasks(
                task,
                available.samples,
                available.fsteps,
                metrics_to_compute,
                sample_size,
                fstep_size,
            )
            _logger.info(f"RUN {run_id} - {stream}: scoring in {len(tasks)} chunk(s).")
            computed = merge_results([_process_stream(**t)[2] for t in tasks], fsteps=responsible)
            write_missing_scores(reader, stream, computed, recomputable_metrics)
            for region, by_metric in computed.items():
                results.setdefault(region, {}).update(by_metric)

    if stream_cfg.get("plotting"):
        available = reader.check_availability(stream, mode="plotting")
        tasks = build_plotting_chunk_tasks(
            task,
            available.samples,
            available.fsteps,
            available.channels,
            sample_size,
            fstep_size,
        )
        _logger.info(f"RUN {run_id} - {stream}: field plots in {len(tasks)} chunk(s).")
        for t in tasks:
            _process_stream(**t)

    if plot_score_options.get("plot_score_maps", False) and computed:
        plot_score_maps(reader, stream, computed, global_plotting_options=global_plotting_opts)
    build_animations(reader, stream, global_plotting_opts, plot_score_options)
    return run_id, stream, results
