# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
import re
from collections.abc import Iterable, Sequence
from enum import Enum
from pathlib import Path

import imageio
import matplotlib as mpl
import numpy as np
import xarray as xr
from joblib import delayed
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm

from weathergen.evaluate.io.data.io_orchestration import dispatch_parallel, get_num_workers

_logger = logging.getLogger(__name__)


def apply_font_settings(cfg: dict) -> None:
    """Apply font settings from plotter config to matplotlib rcParams.

    Parameters
    ----------
    cfg : dict
        Plotter configuration dictionary.  Recognised keys:

        - ``font_size``: base font size (default: matplotlib default)
        - ``font_type``: font family, e.g. ``'serif'``, ``'sans-serif'``,
          ``'monospace'`` (default: matplotlib default)
    """
    font_size = cfg.get("font_size")
    font_type = cfg.get("font_type")
    if font_size is not None:
        mpl.rcParams["font.size"] = font_size
    if font_type is not None:
        mpl.rcParams["font.family"] = font_type


class PlotSubdir(str, Enum):
    """Known plot subdirectory names produced by the plotting pipeline.

    Being a ``str`` subclass, members compare and format exactly like plain
    strings (e.g. ``PlotSubdir.line_plots == "line_plots"``), so they can be
    used as drop-in replacements wherever the raw directory name is expected
    (e.g. ``Path(base) / PlotSubdir.line_plots``).
    """

    line_plots = "line_plots"
    ratio_plots = "ratio_plots"
    psd_plots = "psd_plots"
    score_cards = "score_cards"
    bar_plots = "bar_plots"
    qq_plots = "qq_plots"


# Shared helpers
def calculate_average_over_dim(
    x_dim: str, baseline_var: xr.DataArray, data_var: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate average over xarray dimensions that are larger than 1. Those might be the
    forecast-steps or the samples.

    Parameters
    ----------
    x_dim: str
        The dimension for which an average will not be calculated.
    baseline_var: xr.DataArray
        xarray DataArray with the scores of the baseline model for a specific channel/variable
    data_var: xr.DataArray
        xarray DataArray with the scores of the comparison model for a specific channel/variable

    Returns
    -------
    baseline_score: xarray DataArray
        The baseline average scores over the dimensions not specified by x_dim
    model_score: xarray DataArray
        The model average scores over the dimensions not specified by x_dim
    """
    non_zero_dims = [
        dim for dim in baseline_var.dims if dim != x_dim and baseline_var[dim].shape[0] > 1
    ]

    if non_zero_dims:
        _logger.info(f"Found multiple entries for dimensions: {non_zero_dims}. Averaging...")

    baseline_score = baseline_var.mean(
        dim=[dim for dim in baseline_var.dims if dim != x_dim], skipna=True
    )
    model_score = data_var.mean(dim=[dim for dim in data_var.dims if dim != x_dim], skipna=True)

    return baseline_score, model_score


def lower_is_better(metric: str) -> bool:
    """Determine whether lower or higher is better."""
    return metric in {"l1", "l2", "mae", "mse", "rmse", "vrmse", "bias", "crps", "spread"}


def compute_offsets(n, spacing=0.11):
    """Compute symmetric offsets for *n* items centred around zero.

    Parameters
    ----------
    n : int
        Number of items to offset.
    spacing : float
        Distance between consecutive offsets (default ``0.11``).

    Returns
    -------
    np.ndarray
        Array of length *n* with offsets centred at zero.
    """
    idx = np.arange(n)
    return (idx - (n - 1) / 2.0) * spacing


def align_labels(da: xr.DataArray, labels: list[str], x_dim: str) -> xr.DataArray:
    """
    Reindex a DataArray to include all labels in the canonical order.
    Missing variables are filled with NaN.
    """
    labels = np.array(labels, dtype=object)
    return da.reindex({x_dim: labels})


def format_datetime(dt):
    """Format a numpy datetime64 value as a human-readable string.

    Parameters
    ----------
    dt : numpy.datetime64
        Datetime value to format.

    Returns
    -------
    str
        Formatted string in ``'%Y-%m-%d T%H:%M:%S'`` format.
    """
    return dt.astype("datetime64[m]").astype(datetime.datetime).strftime("%Y-%m-%d T%H:%M:%S")


def channel_sort_key(name: str) -> tuple[int, str, int]:
    """
    Sorting key for channel names like 't_850', 'z_500', etc.
    Splits the name into a prefix and a number suffix for sorting.
    """
    m = re.match(r"(.+?)_(\d+)$", name)
    if m:
        prefix, number = m.groups()
        return (0, prefix, int(number))
    else:
        return (1, name, float("inf"))


def clean_label(s: str) -> str:
    """Replace underscores and hyphens with spaces, then strip whitespace.

    Parameters
    ----------
    s : str
        Raw label string (e.g. ``'lead_time'``).

    Returns
    -------
    str
        Cleaned label (e.g. ``'lead time'``).
    """
    return re.sub(r"[_\-]+", " ", s).strip()


def filter_set(items: list, allowed: set | None) -> list:
    """Return *items* filtered to *allowed*, or all items if *allowed* is ``None``."""
    if allowed is None:
        return items
    return [x for x in items if x in allowed]


class DefaultMarkerSize:
    """
    Utility class for managing default configuration values, such as marker sizes
    for various data streams.
    """

    _marker_size_stream = {
        "era5": 2.5,
        "imerg": 0.25,
        "cerra": 0.1,
    }

    _default_marker_size = 0.5

    @classmethod
    def get_marker_size(cls, stream_name: str) -> float:
        """
        Get the default marker size for a given stream name.

        Parameters
        ----------
        stream_name : str
            The name of the stream.

        Returns
        -------
        float
            The default marker size for the stream.
        """
        return cls._marker_size_stream.get(stream_name.lower(), cls._default_marker_size)

    @classmethod
    def list_streams(cls):
        """
        List all streams with defined marker sizes.

        Returns
        -------
        list[str]
            List of stream names.
        """
        return list(cls._marker_size_stream.keys())

    @staticmethod
    def compute_marker_size(marker_size_base: float, scale: bool, lat: NDArray) -> float | NDArray:
        """Return marker sizes, optionally scaled by latitude.

        When *scale* is truthy, markers at higher latitudes are enlarged
        to compensate for the convergence of meridians (HEALPix point
        clustering), keeping visual coverage roughly uniform.

        Parameters
        ----------
        marker_size_base : float
            Base marker size in matplotlib scatter units (pt²).
        scale : bool
            If ``True``, scale marker size by ``1 / cos²(lat)``.
        lat : array-like
            Latitude values (degrees) for every point.

        Returns
        -------
        float or np.ndarray
            Scalar when *scale* is ``False``; array matching *lat* otherwise.
        """
        if not scale:
            return marker_size_base
        return np.clip(
            marker_size_base / np.cos(np.radians(lat)) ** 2,
            a_min=marker_size_base,
            a_max=marker_size_base * 10.0,
        )

    @classmethod
    def auto_marker_size(
        cls,
        n_points: int,
        fig_width_in: float,
        fig_height_in: float,
        stream_default: float,
        scale: bool,
        lat: NDArray,
        *,
        density_threshold: int = 200_000,
        fill_factor: float = 1.8,
        max_size: float = 4.0,
    ) -> float | NDArray:
        """Compute marker size adapting to point density and figure area.

        For dense grids (≥ *density_threshold* points, e.g. n320, CERRA)
        the marker size is derived from the figure area so that points
        fill the globe without white gaps.  Sparser grids (e.g. o96)
        use the stream default as-is.

        Parameters
        ----------
        n_points : int
            Total number of data points to render.
        fig_width_in : float
            Figure width in inches.
        fig_height_in : float
            Figure height in inches.
        stream_default : float
            Default marker size for the stream (from ``get_marker_size``).
        scale : bool
            If ``True``, scale marker size by latitude.
        lat : array-like
            Latitude values (degrees) for every point.
        density_threshold : int
            Minimum point count for auto-sizing to kick in.
        fill_factor : float
            Multiplier for the area-per-point ratio.
        max_size : float
            Upper clamp for the auto-computed base size.

        Returns
        -------
        float or np.ndarray
            Marker size(s) ready to pass to ``ax.scatter(s=...)``.
        """
        if n_points >= density_threshold:
            fig_area_pt2 = (fig_width_in * 72) * (fig_height_in * 72)
            base = float(np.clip(fig_area_pt2 / n_points * fill_factor, 0.05, max_size))
        else:
            base = stream_default

        return cls.compute_marker_size(base, scale, lat)


def collect_streams(runs: dict):
    """Get all unique streams across runs, sorted.

    Parameters
    ----------
    runs : dict
        The dictionary containing all run configs.

    Returns
    -------
    set
        all available streams
    """
    return sorted({s for run in runs.values() for s in run["streams"].keys()})


def collect_channels(scores_dict: dict, metric: str, region: str, runs) -> list[str]:
    """Get all unique channels available for given metric and region across runs.

    Parameters
    ----------
    scores_dict : dict
        The dictionary containing all computed metrics.
    metric: str
        String specifying the metric to plot
    region: str
        String specifying the region to plot
    runs: dict
        Dictionary containing the config for all runs
    Returns
    -------
    list
        returns a list with all available channels
    """
    channels = set()
    if metric not in scores_dict or region not in scores_dict[metric]:
        return []
    for _stream, run_data in scores_dict[metric][region].items():
        for run_id in runs:
            if run_id not in run_data:
                continue
            values = run_data[run_id]["channel"].values
            channels.update([str(x) for x in np.atleast_1d(values)])
    return list(channels)


def _assign_time_coord(selected_data: list[xr.DataArray]) -> tuple[xr.DataArray, str]:
    """Ensure that lead_time coordinate exists in the data array.

    Parameters
    ----------
    selected_data : list[xarray.DataArray]
        The data array to check.

    Returns
    -------
    xarray.DataArray
        The data array with lead_time coordinate ensured.

    time_dim : str
        The name of the time dimension used for x-axis.
    """

    time_dim = "forecast_step"

    for data in selected_data:
        if "forecast_step" not in data.dims and "forecast_step" not in data.coords:
            raise ValueError(
                "forecast_step coordinate not found in data dimensions or coordinates."
            )

        if "lead_time" not in data.coords and "lead_time" not in data.dims:
            _logger.warning(
                "lead_time coordinate not found for all plotted data; "
                "using forecast_step as x-axis."
            )
            return selected_data, time_dim

    # Swap forecast_step with lead_time if all available run_ids have lead_time coord
    time_dim = "lead_time"

    for i, data in enumerate(selected_data):
        lead_time = data.coords["lead_time"]
        forecast_step = data.coords["forecast_step"]

        if (
            lead_time.dims == forecast_step.dims
            and lead_time.shape == forecast_step.shape
            and lead_time.ndim == 1
        ):
            selected_data[i] = data.swap_dims({"forecast_step": "lead_time"})
        else:
            _logger.warning(
                "lead_time coordinate is not compatible with forecast_step for all plotted data; "
                "using forecast_step as x-axis."
            )
            time_dim = "forecast_step"
    return selected_data, time_dim


def create_filename(
    *,
    prefix: Sequence[str] = (),
    middle: Iterable[str] = (),
    suffix: Sequence[str] = (),
    sep: str = "_",
    max_len: int = 255,
):
    """
    Join strings as: prefix + middle + suffix, truncating only `middle`
    to ensure the final string does not exceed max_len.

    Parameters
    ----------
    prefix : Sequence[str]
        Parts that must appear before the truncated section.
    middle : Iterable[str]
        Parts that may be truncated (order preserved).
    suffix : Sequence[str]
        Parts that must appear after the truncated section.
    sep : str
        Separator used for joining.
    max_len : int
        Maximum total length of the joined string.

    Returns
    -------
    str
        The joined string, with only `middle` truncated if necessary.
    """

    pref, mid, suf = map(lambda x: list(map(str, x)), (prefix, middle, suffix))
    fixed = sep.join(pref + suf)
    avail = max_len - len(fixed)

    if mid and pref:
        avail -= len(sep)
    if mid and suf:
        avail -= len(sep)

    truncated_middle, used = [], 0

    for x in mid:
        d = len(x) + (len(sep) if truncated_middle else 0)
        if used + d > avail:
            break
        truncated_middle.append(x)
        used += d

    if len(truncated_middle) < len(mid):
        _logger.warning(
            f"Filename truncated: only {len(truncated_middle)} of {len(mid)} middle parts used "
            f"to keep length <= {max_len}."
        )

    return sep.join(prefix + truncated_middle + suffix)


# ---------------------------------------------------------------------------
# Animations
# ---------------------------------------------------------------------------


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
    plotter: object,
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
