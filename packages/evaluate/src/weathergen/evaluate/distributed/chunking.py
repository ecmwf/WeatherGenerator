# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Cutting one (run, stream) into chunks of samples × forecast steps.

A chunk task is one ``_process_stream(**task, chunk=True)`` call on a copy of the run config
whose stream windows are one chunk; ``_process_stream`` itself knows nothing about chunking.
Scoring chunk tasks cover the stream's ``evaluation`` windows, plotting chunk tasks its
``plotting`` windows.  The config block::

    distributed:
      enabled: false        # auto | true | false
      chunks:
        forecast_step: 10   # window size, or "all"
        sample: all
"""

import logging

import omegaconf as oc

from weathergen.evaluate.plotting.colour_ranges import configured_colour_range

_logger = logging.getLogger(__name__)


def parse_distributed_config(cfg: dict) -> tuple[bool, int | None, int | None]:
    """``(enabled, sample window, forecast-step window)`` of the ``distributed`` block.

    A window of ``None`` means one chunk along that dimension.  ``enabled: auto`` is
    ``False`` here; an MPI run resolves it to ``True`` when it has more than one rank.
    """
    block = cfg.get("distributed") or {}
    enabled = block.get("enabled", False)
    if enabled == "auto":
        enabled = False
    if enabled not in (True, False):
        raise ValueError(f"distributed.enabled must be auto, true or false, got {enabled!r}.")

    chunks = block.get("chunks") or {}
    sizes = []
    for name in ("sample", "forecast_step"):
        size = chunks.get(name, "all")
        if size in ("all", None):
            sizes.append(None)
        elif isinstance(size, int) and size > 0:
            sizes.append(size)
        else:
            raise ValueError(
                f"distributed.chunks.{name} must be a positive window size or 'all', got {size!r}."
            )
    return bool(enabled), sizes[0], sizes[1]


def windows(values: list, size: int | None) -> list[list]:
    """Consecutive windows of at most *size* of the sorted *values*; one window for ``None``."""
    values = sorted(values)
    if size is None:
        return [values]
    return [values[i : i + size] for i in range(0, len(values), size)]


def chunked_run_config(
    run: dict,
    stream: str,
    mode: str,
    samples: list[int],
    fsteps: list[int],
) -> oc.DictConfig:
    """A copy of the run config in which *stream* covers one chunk, for one *mode*.

    The stream's *mode* block (``evaluation`` or ``plotting``) gets ``sample`` /
    ``forecast_step`` set to the chunk's lists and the other block is removed, so a
    ``_process_stream`` call on the copy scores, or plots, exactly this chunk.  *run* is
    not modified.
    """
    cfg = oc.OmegaConf.create(oc.OmegaConf.to_container(run, resolve=True))
    stream_cfg = cfg.streams[stream]
    stream_cfg.pop("plotting" if mode == "evaluation" else "evaluation", None)
    stream_cfg[mode]["sample"] = list(samples)
    stream_cfg[mode]["forecast_step"] = list(fsteps)
    return cfg


def build_scoring_chunk_tasks(
    task: dict,
    samples: list[int],
    fsteps: list[int],
    metrics_to_compute: dict[str, dict],
    sample_size: int | None,
    fstep_size: int | None,
) -> tuple[list[dict], list[list[int]]]:
    """The ``_process_stream`` chunk tasks that score one (run, stream), and their steps.

    Parameters
    ----------
    task:
        The (run, stream) task of ``build_stream_tasks``.
    samples, fsteps:
        The stream's evaluation samples and forecast steps (``check_availability``).
    metrics_to_compute:
        ``{region: {metric: parameters}}`` (``cached_scores_and_metrics_to_compute``).  A
        chunk computes the union of the metrics for all these regions.
    sample_size, fstep_size:
        The chunk windows (``parse_distributed_config``).

    Returns
    -------
    tuple[list[dict], list[list[int]]]
        The tasks, forecast-step-major so that consecutive tasks read the same files, and
        per task the forecast steps it is responsible for.  With ``froct`` / ``troct`` a task
        is scored with the first step of the next window too; ``merge_results`` drops it.
    """
    metrics = {m: p for by_metric in metrics_to_compute.values() for m, p in by_metric.items()}
    next_step = any(m in ("froct", "troct") for m in metrics)
    fstep_windows = windows(fsteps, fstep_size)
    tasks, responsible = [], []
    for i, window in enumerate(fstep_windows):
        scored = window + (
            fstep_windows[i + 1][:1] if next_step and i + 1 < len(fstep_windows) else []
        )
        for sample_window in windows(samples, sample_size):
            tasks.append(
                task
                | {
                    "run": chunked_run_config(
                        task["run"], task["stream"], "evaluation", sample_window, scored
                    ),
                    "regions": list(metrics_to_compute),
                    "metrics": metrics,
                    "chunk": True,
                }
            )
            responsible.append(list(window))
    return tasks, responsible


def build_plotting_chunk_tasks(
    task: dict,
    samples: list[int],
    fsteps: list[int],
    channels: list[str],
    sample_size: int | None,
    fstep_size: int | None,
    colour_ranges: dict[str, dict] | None = None,
) -> list[dict]:
    """The ``_process_stream`` chunk tasks that draw the field plots of one (run, stream).

    Parameters
    ----------
    task:
        The (run, stream) task of ``build_stream_tasks``.
    samples, fsteps, channels:
        The stream's plotting samples, forecast steps and channels (``check_availability``).
    sample_size, fstep_size:
        The chunk windows (``parse_distributed_config``).
    colour_ranges:
        ``data_colour_ranges`` of the whole (run, stream), per channel.  They go into the
        tasks' plotting options for the stream, where the config has not fixed a range, so
        every chunk draws with the same colour ranges.

    Notes
    -----
    The plots across samples (the across-samples histograms, the timeseries) are computed
    per chunk as per-sample statistics of its window and merged by the chunked run, so a
    chunk's ``data_plots`` are the stream's.
    """
    stream = task["stream"]
    sample_windows = windows(samples, sample_size)
    fstep_windows = windows(fsteps, fstep_size)
    # the colour ranges of the whole (run, stream), under the config's own entries
    global_plotting_opts = oc.OmegaConf.create(
        oc.OmegaConf.to_container(task["global_plotting_opts"], resolve=True)
        if isinstance(task["global_plotting_opts"], oc.DictConfig)
        else task["global_plotting_opts"]
    )
    if colour_ranges:
        if stream not in global_plotting_opts:
            global_plotting_opts[stream] = {}
        stream_opts = global_plotting_opts[stream]
        for channel, by_key in colour_ranges.items():
            for key, value in by_key.items():
                if configured_colour_range(stream_opts, channel, key) is None:
                    if channel not in stream_opts:
                        stream_opts[channel] = {}
                    stream_opts[channel][key] = value

    return [
        task
        | {
            "run": chunked_run_config(task["run"], stream, "plotting", sample_window, fstep_window),
            "global_plotting_opts": global_plotting_opts,
            "chunk": True,
        }
        for fstep_window in fstep_windows
        for sample_window in sample_windows
    ]
