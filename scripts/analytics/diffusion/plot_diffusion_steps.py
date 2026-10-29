# ruff: noqa: T201

"""Plot the diffusion denoising trajectory written in dev mode.

When ``fe_diffusion_log_analytics: true`` is set, ``weathergen.utils.inference_logger`` writes the
per-ODE-step decoded predictions to
``<run>/analytics/diffusion_steps/diffusion_steps.zip`` (see
``weathergen.utils.inference_logger.DiffusionStepsZarrWriter``), one group per validation
sample. Unlike the official validation/test output, denoising steps are kept as their own
explicit ``denoising_step`` array dimension rather than being conflated with real forecast
steps, so this store is independent of ``write_output``/``OutputBatchData`` and cannot be
read by the ``evaluate`` package; this script reads it directly instead.

Maps use the same projection (``ccrs.PlateCarree``) and colormap (``coolwarm``) as the
``evaluate`` package's ``Plotter.scatter_plot``. Plotting for each (sample, stream, channel)
combination is independent and is dispatched across multiple processes (see
``weathergen.evaluate.io.data.io_orchestration.dispatch_parallel``).

For every (sample, stream, channel) found in the store this produces:
  * ``plots/rmse/<sample>_<stream>_<channel>.png``: RMSE and bias vs. denoising step.
  * ``plots/maps/<sample>_<stream>_<channel>.png``: a grid of target/prediction/bias maps
    across the selected denoising steps (default: all), each with its own colorbar.
  * ``plots/maps/targets/``: one map per channel (the target is identical across denoising
    steps, so it is only plotted once).
  * ``plots/maps/pred/`` (and, by default, ``plots/maps/bias/``): one map per denoising
    step, plus (by default) an .mp4 animation across steps, when
    ``--plot-individual-maps`` is enabled (default).

Usage:
    uv run python scripts/analytics/diffusion/plot_diffusion_steps.py --run-id <id>
    uv run python scripts/analytics/diffusion/plot_diffusion_steps.py --path <run_dir> \\
        --channels 2t q_850 --denoising-steps 0 2 4 6 8 --max-workers 8
"""

import argparse
import logging
from pathlib import Path

import cartopy.crs as ccrs
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr
from joblib import delayed
from numpy.typing import NDArray
from zarr.storage import ZipStore

import weathergen.common.config as config
from weathergen.evaluate.io.data.io_orchestration import dispatch_parallel, get_num_workers
from weathergen.utils.inference_logger import get_diffusion_steps_dir

logger = logging.getLogger(__name__)

STORE_NAME = "diffusion_steps"
DEFAULT_CHANNELS = ["2t", "q_850", "z_500"]
# same default projection/colormap as weathergen.evaluate.plotting.plotter.Plotter.scatter_plot
MAP_CMAP = "coolwarm"


def _resolve_store_path(run_id: str | None, path: str | None, mini_epoch: int) -> Path:
    if path is not None:
        base = Path(path)
    else:
        if run_id is None:
            raise ValueError("Either --run-id or --path must be provided.")
        cf = config.load_run_config(run_id=run_id, mini_epoch=mini_epoch, model_path=None)
        base = get_diffusion_steps_dir(cf)
    store_path = base / f"{STORE_NAME}.zip"
    if not store_path.exists():
        raise FileNotFoundError(
            f"No diffusion steps store found at {store_path}. Was the run started with "
            "fe_diffusion_log_analytics=true?"
        )
    return store_path


def _rmse_and_bias(
    pred_channel: NDArray[np.float32], target_channel: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """pred_channel: (denoising_step, ens, point); target_channel: (point,).

    Returns (rmse, bias), each of shape (denoising_step,), averaged over ensemble
    members and points.
    """
    error = pred_channel - target_channel[None, None]
    rmse = np.sqrt(np.mean(error**2, axis=(1, 2)))
    bias = np.mean(error, axis=(1, 2))
    return rmse, bias


def _percentile_range(values: NDArray[np.float32]) -> tuple[float, float]:
    """5th/95th percentile range, matching Plotter.scatter_plot's default map scaling."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [5, 95])
    return float(lo), float(hi)


def _symmetric_range(values: NDArray[np.float32]) -> tuple[float, float]:
    """Symmetric range around 0, for diverging (e.g. bias) maps."""
    finite = values[np.isfinite(values)]
    vmax = float(np.percentile(np.abs(finite), 95)) if finite.size else 1.0
    return -vmax, vmax


def _style_geo_axes(ax) -> None:
    """Match Plotter.scatter_plot's look: coastlines, thin spines, dashed gridlines."""
    try:
        ax.coastlines(linewidth=0.3)
    except Exception:
        logger.warning("Could not add coastlines to plot; continuing without them.")
    ax.set_global()
    ax.gridlines(draw_labels=False, linestyle="--", color="gray", linewidth=0.6, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)


def _scatter_on_map(ax, lon: NDArray[np.float32], lat: NDArray[np.float32], values, vmin, vmax):
    return ax.scatter(
        lon,
        lat,
        c=values,
        vmin=vmin,
        vmax=vmax,
        cmap=MAP_CMAP,
        s=4,
        transform=ccrs.PlateCarree(),
        linewidths=0.0,
        rasterized=True,
    )


def _add_colorbar(fig, ax, artist, label: str) -> None:
    cbar = fig.colorbar(
        artist, ax=ax, fraction=0.03, pad=0.02, shrink=0.6, orientation="horizontal"
    )
    cbar.set_label(label, fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_linewidth(0.3)


def plot_errors(
    pred_channel: NDArray[np.float32],
    target_channel: NDArray[np.float32],
    channel_name: str,
    out_path: Path,
) -> None:
    """pred_channel: (denoising_step, ens, point); target_channel: (point,)."""
    rmse, bias = _rmse_and_bias(pred_channel, target_channel)
    steps = np.arange(pred_channel.shape[0])

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(steps, rmse, "o-", markersize=3, label="RMSE")
    ax.plot(steps, bias, "s-", markersize=3, label="bias")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_title(channel_name)
    ax.set_xlabel("denoising step")
    ax.set_ylabel("error")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved error plot to %s", out_path)


def plot_maps_grid(
    pred_maps: NDArray[np.float32],
    target_map: NDArray[np.float32],
    coords: NDArray[np.float32],
    channel_name: str,
    denoising_steps: list[int],
    out_path: Path,
) -> None:
    """Target/prediction/bias maps side by side across the given denoising steps.

    pred_maps: (denoising_step, point), already averaged over ensemble members.
    target_map: (point,).

    Each panel gets its own colorbar (independent 5th/95th, resp. symmetric, percentile
    range) rather than one shared per row.
    """
    lat, lon = coords[:, 0], coords[:, 1]

    n_cols = len(denoising_steps) + 1
    fig = plt.figure(figsize=(3.4 * n_cols, 7))
    axes = [
        [fig.add_subplot(2, n_cols, row * n_cols + col + 1, projection=ccrs.PlateCarree())
         for col in range(n_cols)]
        for row in range(2)
    ]
    for ax in (a for row in axes for a in row):
        _style_geo_axes(ax)

    vmin, vmax = _percentile_range(target_map)
    artist = _scatter_on_map(axes[0][0], lon, lat, target_map, vmin, vmax)
    axes[0][0].set_title("target", fontsize=8)
    _add_colorbar(fig, axes[0][0], artist, channel_name)
    axes[1][0].set_axis_off()

    for col, step in enumerate(denoising_steps, start=1):
        pred_vmin, pred_vmax = _percentile_range(pred_maps[step])
        artist = _scatter_on_map(axes[0][col], lon, lat, pred_maps[step], pred_vmin, pred_vmax)
        axes[0][col].set_title(f"pred (step {step})", fontsize=8)
        _add_colorbar(fig, axes[0][col], artist, channel_name)

        bias = pred_maps[step] - target_map
        bias_vmin, bias_vmax = _symmetric_range(bias)
        artist = _scatter_on_map(axes[1][col], lon, lat, bias, bias_vmin, bias_vmax)
        axes[1][col].set_title(f"bias (step {step})", fontsize=8)
        _add_colorbar(fig, axes[1][col], artist, "bias")

    fig.suptitle(channel_name)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved map grid to %s", out_path)


def plot_single_map(
    values: NDArray[np.float32],
    coords: NDArray[np.float32],
    title: str,
    out_path: Path,
    symmetric: bool = False,
) -> None:
    lat, lon = coords[:, 0], coords[:, 1]
    vmin, vmax = _symmetric_range(values) if symmetric else _percentile_range(values)

    fig = plt.figure(figsize=(6, 4), dpi=150)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    _style_geo_axes(ax)
    artist = _scatter_on_map(ax, lon, lat, values, vmin, vmax)
    _add_colorbar(fig, ax, artist, title)
    ax.set_title(title, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_animation(image_paths: list[Path], out_path: Path, fps: int = 2) -> None:
    if len(image_paths) < 2:
        logger.debug("Skipping animation %s: need at least 2 frames.", out_path)
        return
    frames = [imageio.imread(str(p)) for p in image_paths]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=fps, ffmpeg_params=["-crf", "18"])
    logger.info("Saved animation to %s", out_path)


def plot_individual_maps(
    pred_maps: NDArray[np.float32],
    target_map: NDArray[np.float32],
    coords: NDArray[np.float32],
    channel_name: str,
    denoising_steps: list[int],
    pred_dir: Path,
    targets_dir: Path,
    bias_dir: Path | None,
    prefix: str,
    animate: bool,
) -> None:
    """One map per denoising step for the prediction (and, optionally, bias); the target is
    identical across denoising steps, so it is plotted once instead.
    """
    plot_single_map(target_map, coords, f"{channel_name} target", targets_dir / f"{prefix}.png")

    pred_paths = []
    for step in denoising_steps:
        step_tag = f"step{step:03d}"

        pred_path = pred_dir / f"{prefix}_{step_tag}.png"
        plot_single_map(pred_maps[step], coords, f"{channel_name} pred (step {step})", pred_path)
        pred_paths.append(pred_path)

        if bias_dir is not None:
            bias_path = bias_dir / f"{prefix}_{step_tag}.png"
            plot_single_map(
                pred_maps[step] - target_map,
                coords,
                f"{channel_name} bias (step {step})",
                bias_path,
                symmetric=True,
            )

    if animate:
        make_animation(pred_paths, pred_dir / f"{prefix}.mp4")


def _process_channel_task(
    pred_channel: NDArray[np.float32],
    target_channel: NDArray[np.float32],
    coords: NDArray[np.float32],
    channel_name: str,
    denoising_steps: list[int],
    prefix: str,
    rmse_dir: Path,
    maps_dir: Path,
    pred_dir: Path,
    targets_dir: Path,
    bias_dir: Path | None,
    plot_individual_maps_enabled: bool,
    animate: bool,
) -> None:
    """Plot everything for one (sample, stream, channel) combination (joblib worker)."""
    matplotlib.use("Agg")
    pred_maps = pred_channel.mean(axis=1)  # average over ens -> (denoising_step, point)

    plot_errors(pred_channel, target_channel, channel_name, rmse_dir / f"{prefix}.png")
    plot_maps_grid(
        pred_maps, target_channel, coords, channel_name, denoising_steps, maps_dir / f"{prefix}.png"
    )
    if plot_individual_maps_enabled:
        plot_individual_maps(
            pred_maps,
            target_channel,
            coords,
            channel_name,
            denoising_steps,
            pred_dir,
            targets_dir,
            bias_dir,
            prefix,
            animate=animate,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-id", default=None, help="Run id to load the config/path from.")
    parser.add_argument(
        "--path", default=None, help="Run directory, as an alternative to --run-id."
    )
    parser.add_argument(
        "--mini-epoch", type=int, default=0, help="Checkpoint mini_epoch used to resolve --run-id."
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=DEFAULT_CHANNELS,
        help="Channels to plot (index or name each), one figure set per channel. "
        f"Default: {DEFAULT_CHANNELS}.",
    )
    parser.add_argument(
        "--denoising-steps",
        type=int,
        nargs="+",
        default=None,
        help="Which denoising steps to include in the map grid and individual maps "
        "(default: all).",
    )
    parser.add_argument(
        "--plot-individual-maps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot one pred map per denoising step (plus the target once) into "
        "plots/maps/pred and plots/maps/targets (default: True).",
    )
    parser.add_argument(
        "--bias-maps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also plot individual bias (pred - target) maps per denoising step into "
        "plots/maps/bias (default: True; requires --plot-individual-maps).",
    )
    parser.add_argument(
        "--animate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build an .mp4 animation across denoising steps from the individual pred "
        "maps (default: True; requires --plot-individual-maps).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: 'plots/' next to the store, with 'rmse/' and "
        "'maps/' subdirectories).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Cap on parallel worker processes (default: auto-detect from Slurm/CPU count).",
    )
    args = parser.parse_args()

    store_path = _resolve_store_path(args.run_id, args.path, args.mini_epoch)
    out_dir = Path(args.output) if args.output else store_path.parent / "plots"
    rmse_dir = out_dir / "rmse"
    maps_dir = out_dir / "maps"
    pred_dir = maps_dir / "pred"
    targets_dir = maps_dir / "targets"
    bias_dir = maps_dir / "bias"

    # Read everything up front (cheap compared to plotting) so the zip store doesn't need
    # to be shared across worker processes; only plain arrays are passed to each task.
    tasks: list[dict] = []
    with ZipStore(str(store_path), mode="r") as store:
        group = zarr.open_group(store=store, mode="r")

        for sample_name in sorted(group.group_keys()):
            sample_group = group[sample_name]
            for stream_name in sorted(sample_group.group_keys()):
                stream_group = sample_group[stream_name]
                predictions = np.asarray(stream_group["predictions"])
                target = np.asarray(stream_group["target"])
                coords = np.asarray(stream_group["coords"])
                channels = list(stream_group.attrs["channels"])

                n_steps = predictions.shape[0]
                denoising_steps = (
                    [s for s in args.denoising_steps if 0 <= s < n_steps]
                    if args.denoising_steps
                    else list(range(n_steps))
                )

                for requested in args.channels:
                    if requested.isdigit():
                        channel_idx = int(requested)
                    elif requested in channels:
                        channel_idx = channels.index(requested)
                    else:
                        # e.g. requested default channels not carried by every stream
                        logger.debug(
                            "Skipping channel '%s' for %s/%s: not in %s",
                            requested,
                            sample_name,
                            stream_name,
                            channels,
                        )
                        continue

                    channel_name = channels[channel_idx]
                    tasks.append(
                        {
                            "pred_channel": predictions[:, :, :, channel_idx],
                            "target_channel": target[:, channel_idx],
                            "coords": coords,
                            "channel_name": channel_name,
                            "denoising_steps": denoising_steps,
                            "prefix": f"{sample_name}_{stream_name}_{channel_name}",
                            "rmse_dir": rmse_dir,
                            "maps_dir": maps_dir,
                            "pred_dir": pred_dir,
                            "targets_dir": targets_dir,
                            "bias_dir": bias_dir if args.bias_maps else None,
                            "plot_individual_maps_enabled": args.plot_individual_maps,
                            "animate": args.animate,
                        }
                    )

    n_workers = get_num_workers(check_process_headroom=True, max_workers=args.max_workers)
    calls = [delayed(_process_channel_task)(**t) for t in tasks]
    dispatch_parallel(calls, n_workers=n_workers, backend="loky", desc="Trajectory plots")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
