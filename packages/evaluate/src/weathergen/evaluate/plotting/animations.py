# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Videos assembled from the frames already on disk: field maps / histograms, and score maps.

Only the image files are read, not the data, so a video can be built wherever all its
frames are available.
"""

import logging
from pathlib import Path

import imageio
from joblib import delayed
from PIL import Image
from tqdm import tqdm

from weathergen.evaluate.io.data.io_orchestration import dispatch_parallel, get_num_workers
from weathergen.evaluate.plotting.plotter import Plotter

_logger = logging.getLogger(__name__)


def _build_single_animation(
    output_dir: Path,
    run_id: str,
    tag: str,
    stream: str,
    region: str | None,
    var: str,
    sample: object,
    fsteps: list,
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

    Returns the list of source frame paths assembled into the animation, or an
    empty list when no (or fewer than two for score maps) frames were found.
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
    fstep_strs = {str(f).zfill(3) for f in fsteps}
    image_paths = sorted(
        str(f)
        for f in output_dir.iterdir()
        if f.name.startswith(head + "_")
        and f.name.endswith(suffix)
        and f"_{tail}_" in f.name
        and f.stem.rsplit("_", 1)[-1] in fstep_strs
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
    plotter: "Plotter",
    samples: list,
    fsteps,
    variables: list[str],
    select: dict,
    tag: str,
    max_workers: int | None = None,
    prefixes: tuple[str, ...] = ("map", "histogram"),
) -> list[str]:
    """Build GIF animations in parallel for all (region, sample, variable) combinations.

    Animations are built from the frame types in *prefixes* (``"map"``, ``"histogram"``);
    whichever image files exist on disk will be picked up automatically.

    Parameters
    ----------
    plotter : Plotter
        Plotter instance (used only for config: regions, fps, image_format, run_id, stream).
    samples, fsteps, variables, select, tag
        Same arguments that ``Plotter.animation`` used to accept.
    prefixes : tuple[str, ...]
        Frame types to animate: ``"map"`` and/or ``"histogram"``.

    Returns
    -------
    list[str]
        Paths of all source frames that were assembled into GIFs.
    """
    plotter.update_data_selection(select)

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
            "fsteps": list(fsteps),
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
    fsteps: list,
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
            fsteps=list(fsteps),
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
