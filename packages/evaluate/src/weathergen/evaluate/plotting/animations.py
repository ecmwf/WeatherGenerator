# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Videos assembled from the frames already on disk: field maps / histograms, and score maps.

Only the image files are read, not the data, so the videos of a (run, stream) are built once
all its frames are written, whichever chunks of the run drew them.
"""

import logging
from pathlib import Path

import imageio
from joblib import delayed
from PIL import Image
from tqdm import tqdm

from weathergen.evaluate.io.data.io_orchestration import dispatch_parallel, get_num_workers
from weathergen.evaluate.io.io_reader import Reader
from weathergen.evaluate.plotting.plot_utils import ens_tag, plotter_config
from weathergen.evaluate.plotting.plotter import Plotter
from weathergen.evaluate.scores.score_orchestration import MetricResult
from weathergen.evaluate.utils.config_compat import MAP_KINDS, parse_data_plots

_logger = logging.getLogger(__name__)


def build_animations(
    reader: Reader,
    stream: str,
    results: dict[str, dict[str, MetricResult]],
    global_plotting_opts: dict,
    plot_score_options: dict,
) -> None:
    """Build the videos of one (run, stream) from the frames on disk.

    Map and histogram videos follow the stream's ``data_plots`` config, score-map videos
    ``score_plots: score_animation``.  Nothing is loaded: the frames may come from any
    number of chunks of the run, as long as they are all written when this is called.

    Parameters
    ----------
    reader : Reader
        Reader of the run (plot directory, availability, ``max_workers``).
    stream : str
        Stream name.
    results : dict[str, dict[str, MetricResult]]
        ``results[region][metric]`` of the (run, stream); the metrics with a ``spatial``
        field have score-map frames.
    global_plotting_opts : dict
        Global plotting options (image and animation format, fps, regions, ...).
    plot_score_options : dict
        ``get_plot_score_options(...)``; ``plot_score_animations`` enables score-map videos.
    """
    stream_cfg = reader.get_stream(stream)
    plotter_cfg = plotter_config(
        global_plotting_opts,
        regions=global_plotting_opts.get("regions", stream_cfg.get("regions", ["global"])),
    )
    max_workers = reader.eval_cfg.get("max_workers", None)

    spec = parse_data_plots(stream_cfg.get("plotting", {}))
    map_videos = spec.map_videos()
    hist_videos = spec.histogram_videos()
    available = (
        reader.check_availability(stream, mode="plotting") if map_videos or hist_videos else None
    )
    if available is not None and available.score_availability:
        plotter = Plotter(plotter_cfg, reader.runplot_dir, stream)
        ensemble = list(available.ensemble)
        # The frames carry an ensemble tag unless the ensemble was reduced on loading
        # (``ensemble: [mean]`` or ``[std]`` drop the ens dimension).
        has_ens = any(e not in ("mean", "std") for e in ensemble)
        preds_tags = [ens_tag("preds", e, has_ens) for e in ensemble]
        map_tags = {
            "predictions": preds_tags,
            "target": ["targets"],
            "bias": [ens_tag("bias", e, has_ens) for e in ensemble],
        }
        anim_kw = dict(plotter=plotter, variables=available.channels, max_workers=max_workers)
        for kind in MAP_KINDS:
            if kind in map_videos:
                for tag in map_tags[kind]:
                    _dispatch_field_animations(
                        **anim_kw, samples=available.samples, tag=tag, prefixes=("map",)
                    )

        # Histogram frames are tagged like the predictions they compare.
        hist_samples = (available.samples if "per_sample" in hist_videos else []) + (
            ["all_samples"] if "across_samples" in hist_videos else []
        )
        if hist_samples:
            for tag in preds_tags:
                _dispatch_field_animations(
                    **anim_kw, samples=hist_samples, tag=tag, prefixes=("histogram",)
                )

    if plot_score_options.get("plot_score_animations", False) and results:
        # {region: {metric: spatial DataArray}} of the metrics with score-map frames
        spatial = {
            region: {m: r.spatial for m, r in by_metric.items() if r.spatial is not None}
            for region, by_metric in results.items()
        }
        spatial = {region: by_metric for region, by_metric in spatial.items() if by_metric}
        arrays = [da for by_metric in spatial.values() for da in by_metric.values()]
        if arrays:
            # The frames are tagged per member for the metrics with an ens dim (plot_score_maps),
            # without a member otherwise.
            ens_values: list = sorted(
                {str(e) for da in arrays if "ens" in da.dims for e in da.ens.values}
            )
            if any("ens" not in da.dims for da in arrays):
                ens_values.append(None)
            _dispatch_score_map_animations(
                map_dir=reader.runplot_dir / "plots" / stream / "score_maps",
                plotter_cfg=plotter_cfg,
                run_id=reader.run_id,
                stream=stream,
                metrics=list(dict.fromkeys(m for by_metric in spatial.values() for m in by_metric)),
                regions=list(spatial),
                variables=sorted({str(c) for da in arrays for c in da.channel.values}),
                ens_values=ens_values,
                n_workers=get_num_workers(check_process_headroom=True, max_workers=max_workers),
            )


def _build_single_animation(
    output_dir: Path,
    run_id: str,
    tag: str,
    stream: str,
    region: str | None,
    var: str,
    sample: object,
    image_format: str,
    animation_format: str,
    duration_ms: int,
    prefix: str = "map",
) -> list[str]:
    """Build one animation for a single (region, sample/ens, variable) combination.

    All work is I/O + Pillow — no matplotlib state involved.

    The function scans ``output_dir`` for per-sample map/histogram frames whose filenames follow:

        {prefix}_{run_id}_{tag}_{sample}_{valid_time}_{stream}_{region}_{var}_{fstep:03d}

    Score-map frames follow the same pattern without the sample and valid time
    (``map_{run_id}_score_maps[_ens_{ens}]_{metric}_{stream}_{region}_{var}_{fstep:03d}``):
    pass ``tag="score_maps[_ens_{ens}]_{metric}"`` and ``sample=None``.

    All the frames found are used, in file-name order, which is forecast-step order; the
    frames of one video may have been written by different chunks of the run.

    Returns the list of source frame paths assembled into the animation, or an
    empty list when no frames were found.
    """
    if not output_dir.is_dir():
        return []

    region_part = region if region else ""
    if sample is not None:
        head = "_".join(filter(None, [prefix, run_id, tag, str(sample)]))
    else:
        head = "_".join(filter(None, [prefix, run_id, tag]))
    tail = "_".join(filter(None, [stream, region_part, var]))
    suffix = f".{image_format}"
    image_paths = sorted(
        str(f)
        for f in output_dir.iterdir()
        if f.name.startswith(head + "_") and f.name.endswith(suffix) and f"_{tail}_" in f.name
    )
    if not image_paths:
        return []
    if sample is not None:
        anim_parts = ["animation", run_id, tag, str(sample), stream]
    else:
        anim_parts = ["animation", run_id, tag, stream]
    if region:
        anim_parts.append(region)
    anim_parts.append(var)
    out_path = f"{output_dir / '_'.join(filter(None, anim_parts))}.{animation_format}"

    if animation_format.lower() == "gif":
        images = [Image.open(p).convert("RGB") for p in image_paths]
        # GIF frames are palette-indexed (256 colors max) — this is a hard
        # format limit, so gradients like colorbars can never be truly
        # continuous here.
        palette = images[0].quantize(colors=256, method=Image.Quantize.MAXCOVERAGE)
        frames = [img.quantize(palette=palette, dither=Image.Dither.NONE) for img in images]
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
        for img in images:
            img.close()
    elif animation_format.lower() == "mp4":
        frames = [imageio.imread(p) for p in image_paths]
        fps = 1000 / duration_ms if duration_ms > 0 else 2
        (
            imageio.mimsave(
                out_path,
                frames,
                fps=fps,
                macro_block_size=8,
                ffmpeg_params=["-framerate", str(fps), "-loglevel", "error", "-crf", "18"],
                ffmpeg_log_level="error",
            ),
        )
    else:
        raise ValueError(
            f"Unsupported animation format: {animation_format}. Must be 'gif' or 'mp4'."
        )

    _logger.debug(f"Saved animation to {out_path}")
    return image_paths


def _dispatch_field_animations(
    plotter: Plotter,
    samples: list,
    variables: list[str],
    tag: str,
    max_workers: int | None = None,
    prefixes: tuple[str, ...] = ("map", "histogram"),
) -> list[str]:
    """Build the map / histogram animations in parallel for all (region, sample, variable).

    Animations are built from the frame types in *prefixes* (``"map"``, ``"histogram"``);
    whichever image files exist on disk will be picked up automatically.

    Parameters
    ----------
    plotter : Plotter
        Plotter of the run and stream (used only for config: regions, fps, image_format,
        run_id, stream, frame directories).
    samples, variables, tag
        The samples and channels to animate, and the frame tag (``preds``, ``targets``, ...).
    prefixes : tuple[str, ...]
        Frame types to animate: ``"map"`` and/or ``"histogram"``.

    Returns
    -------
    list[str]
        Paths of all source frames that were assembled into animations.
    """
    duration_ms = int(1000 / plotter.fps) if plotter.fps > 0 else 400

    frame_dirs = {
        "map": plotter.get_map_output_dir(tag),
        "histogram": plotter.get_hist_output_dir(),
    }

    tasks = [
        {
            "output_dir": output_dir,
            "run_id": plotter.run_id,
            "tag": tag,
            "stream": plotter.stream,
            "region": region,
            "var": var,
            "sample": sample,
            "image_format": plotter.image_format,
            "animation_format": plotter.animation_format,
            "duration_ms": duration_ms,
            "prefix": prefix,
        }
        for prefix, output_dir in frame_dirs.items()
        if prefix in prefixes
        for region in plotter.regions
        for sample in samples
        for var in variables
    ]

    calls = [
        delayed(_build_single_animation)(**t)
        for t in tqdm(tasks, desc=f"Creating animations {plotter.stream} {tag}")
    ]
    results = dispatch_parallel(
        calls,
        n_workers=get_num_workers(max_workers=max_workers),
        backend="loky",
        desc="Animations",
    )
    return [p for r in results if r for p in r]


def _dispatch_score_map_animations(
    map_dir: Path,
    plotter_cfg: dict,
    run_id: str,
    stream: str,
    metrics: list[str],
    regions: list[str],
    variables: list[str],
    ens_values: list,
    n_workers: int | None = None,
) -> list[str]:
    """Build score-map animations in parallel for all (metric, region, variable[, ens]) combos.

    Returns the paths of all source frames assembled into animations.
    """
    duration_ms = int(1000 / plotter_cfg["fps"]) if plotter_cfg["fps"] > 0 else 400

    tasks = [
        dict(
            output_dir=map_dir,
            run_id=run_id,
            tag="score_maps" + (f"_ens_{ens_val}" if ens_val is not None else "") + f"_{metric}",
            stream=stream,
            region=region,
            var=var,
            sample=None,
            image_format=plotter_cfg["image_format"],
            animation_format=plotter_cfg["animation_format"],
            duration_ms=duration_ms,
        )
        for metric in metrics
        for region in regions
        for var in variables
        for ens_val in ens_values
    ]

    calls = [delayed(_build_single_animation)(**t) for t in tasks]
    results = dispatch_parallel(
        calls,
        n_workers=n_workers,
        backend="loky",
        desc=f"Score map animations {stream}",
    )
    return [p for r in results if r for p in r]
