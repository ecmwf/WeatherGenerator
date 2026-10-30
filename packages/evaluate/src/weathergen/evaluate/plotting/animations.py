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
import re
from pathlib import Path

import imageio
from joblib import delayed
from PIL import Image

from weathergen.evaluate.io.data.io_orchestration import dispatch_parallel, get_num_workers
from weathergen.evaluate.io.io_reader import Reader
from weathergen.evaluate.utils.config_compat import parse_data_plots

_logger = logging.getLogger(__name__)


def build_animations(
    reader: Reader, stream: str, global_plotting_opts: dict, plot_score_options: dict
) -> None:
    """Build the videos of one (run, stream) from the frames on disk.

    A video is all the frames of one directory whose names are the same once the valid
    time and the forecast step are removed: frames of one (run, tag, sample, stream,
    region, channel), or of one (run, metric[, member], stream, region, channel) for the
    score maps.  Nothing is loaded, so the frames may come from any number of chunks of
    the run, as long as they are all written when this is called.

    Parameters
    ----------
    reader : Reader
        Reader of the run (plot directory, stream config, ``max_workers``).
    stream : str
        Stream name.
    global_plotting_opts : dict
        Global plotting options: ``image_format``, ``animation_format``, ``fps``.
    plot_score_options : dict
        ``get_plot_score_options(...)``; ``plot_score_animations`` enables score-map videos.
    """
    plot_dir = reader.runplot_dir / "plots" / stream
    spec = parse_data_plots(reader.get_stream(stream).get("plotting", {}))
    hist_videos = spec.histogram_videos()

    # Frame directories with a video option on, as written by Plotter and plot_score_maps.
    map_dirs = {"predictions": "preds*", "target": "targets", "bias": "bias*"}
    frame_dirs = [d for kind in spec.map_videos() for d in (plot_dir / "maps").glob(map_dirs[kind])]
    if hist_videos:
        frame_dirs.append(plot_dir / "histograms")
    if plot_score_options.get("plot_score_animations", False):
        frame_dirs.append(plot_dir / "score_maps")

    image_format = global_plotting_opts.get("image_format", "png")
    animation_format = global_plotting_opts.get("animation_format", "mp4")
    videos: dict[Path, list[Path]] = {}  # video path -> its frames, in forecast-step order
    for frame_dir in frame_dirs:
        for frame in sorted(frame_dir.glob(f"*.{image_format}")):
            # per-sample and across-samples histograms are separate video options
            if frame_dir.name == "histograms":
                kind = "across_samples" if "_all_samples_" in frame.name else "per_sample"
                if kind not in hist_videos:
                    continue
            # {prefix}_{run_id}_{tag}[_{sample}][_{valid_time}]_{stream}_{region}_{var}_{fstep}
            # -> animation_{run_id}_{tag}[_{sample}]_{stream}_{region}_{var}
            stem = re.sub(r"_\d{4}-\d{2}-\d{2}T\d{4}(?=_)", "", frame.stem)
            stem = re.sub(r"_\d{3}$", "", stem).split("_", 1)[1]
            video = frame_dir / f"animation_{stem}.{animation_format}"
            videos.setdefault(video, []).append(frame)

    if not videos:
        return
    fps = global_plotting_opts.get("fps", 2)
    duration_ms = int(1000 / fps) if fps > 0 else 400
    calls = [
        delayed(_build_single_animation)(frames, video, duration_ms)
        for video, frames in videos.items()
    ]
    dispatch_parallel(
        calls,
        n_workers=get_num_workers(max_workers=reader.eval_cfg.get("max_workers", None)),
        backend="loky",
        desc=f"Animations {stream}",
    )


def _build_single_animation(frames: list[Path], out_path: Path, duration_ms: int) -> None:
    """Encode *frames*, in order, as the GIF or mp4 *out_path* (I/O + Pillow / imageio only)."""
    animation_format = out_path.suffix[1:].lower()
    if animation_format == "gif":
        images = [Image.open(p).convert("RGB") for p in frames]
        # GIF frames are palette-indexed (256 colors max) — this is a hard
        # format limit, so gradients like colorbars can never be truly
        # continuous here.
        palette = images[0].quantize(colors=256, method=Image.Quantize.MAXCOVERAGE)
        quantized = [img.quantize(palette=palette, dither=Image.Dither.NONE) for img in images]
        quantized[0].save(
            out_path,
            save_all=True,
            append_images=quantized[1:],
            duration=duration_ms,
            loop=0,
        )
        for img in images:
            img.close()
    elif animation_format == "mp4":
        fps = 1000 / duration_ms if duration_ms > 0 else 2
        imageio.mimsave(
            out_path,
            [imageio.imread(p) for p in frames],
            fps=fps,
            macro_block_size=8,
            ffmpeg_params=["-framerate", str(fps), "-loglevel", "error", "-crf", "18"],
            ffmpeg_log_level="error",
        )
    else:
        raise ValueError(
            f"Unsupported animation format: {animation_format}. Must be 'gif' or 'mp4'."
        )
    _logger.debug(f"Saved animation to {out_path}")
