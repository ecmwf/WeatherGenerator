# ruff: noqa: T201

"""Plot the diffusion denoising trajectory written in dev mode.

When ``fe_diffusion_dev_mode: true`` is set, ``weathergen.utils.validation_io`` writes the
per-ODE-step decoded predictions to
``<run>/dev/diffusion_trajectory/diffusion_trajectory.zarr`` (see
``weathergen.common.dev_output.DiffusionTrajectoryZarrWriter``), one group per validation
sample. Unlike the official validation/test output, denoising steps are kept as their own
explicit ``denoising_step`` array dimension rather than being conflated with real forecast
steps, so this store is independent of ``write_output``/``OutputBatchData`` and cannot be
read by the ``evaluate`` package; this script reads it directly instead.

For every (sample, stream, channel) found in the store this produces two figures:
  * ``plots/rmse/<sample>_<stream>_<channel>.png``: RMSE and bias vs. denoising step.
  * ``plots/maps/<sample>_<stream>_<channel>.png``: scatter maps of the target and a few
    predictions along the trajectory (plus their error).

Usage:
    uv run python scripts/dev/plot_diffusion_trajectory_dev_eval.py --run-id <id>
    uv run python scripts/dev/plot_diffusion_trajectory_dev_eval.py --path <run_dir> \\
        --channels 2t q_850 --num-map-steps 4
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr
from numpy.typing import NDArray

import weathergen.common.config as config
from weathergen.utils.validation_io import get_dev_trajectory_dir

logger = logging.getLogger(__name__)

STORE_NAME = "diffusion_trajectory"
DEFAULT_CHANNELS = ["2t", "q_850", "z_500"]


def _resolve_store_path(run_id: str | None, path: str | None, mini_epoch: int) -> Path:
    if path is not None:
        base = Path(path)
    else:
        if run_id is None:
            raise ValueError("Either --run-id or --path must be provided.")
        cf = config.load_run_config(run_id=run_id, mini_epoch=mini_epoch, model_path=None)
        base = get_dev_trajectory_dir(cf)
    store_path = base / f"{STORE_NAME}.zarr"
    if not store_path.exists():
        raise FileNotFoundError(
            f"No dev trajectory store found at {store_path}. Was the run started with "
            "fe_diffusion_dev_mode=true?"
        )
    return store_path


def _rmse_and_bias(
    predictions: NDArray[np.float32], target: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """predictions: (denoising_step, ens, point, channel); target: (point, channel).

    Returns (rmse, bias), each of shape (denoising_step, channel), averaged over
    ensemble members and points.
    """
    error = predictions - target[None, None]
    rmse = np.sqrt(np.mean(error**2, axis=(1, 2)))
    bias = np.mean(error, axis=(1, 2))
    return rmse, bias


def plot_errors(
    predictions: NDArray[np.float32],
    target: NDArray[np.float32],
    channel_idx: int,
    channel_name: str,
    out_path: Path,
) -> None:
    rmse, bias = _rmse_and_bias(predictions, target)
    steps = np.arange(predictions.shape[0])

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(steps, rmse[:, channel_idx], "o-", markersize=3, label="RMSE")
    ax.plot(steps, bias[:, channel_idx], "s-", markersize=3, label="bias")
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


def plot_maps(
    predictions: NDArray[np.float32],
    target: NDArray[np.float32],
    coords: NDArray[np.float32],
    channel_idx: int,
    channel_name: str,
    num_map_steps: int,
    out_path: Path,
) -> None:
    n_steps = predictions.shape[0]
    step_idxs = np.unique(np.linspace(0, n_steps - 1, num=min(num_map_steps, n_steps), dtype=int))
    # average over ensemble members for the map view
    pred_maps = predictions[:, :, :, channel_idx].mean(axis=1)
    target_map = target[:, channel_idx]
    lat, lon = coords[:, 0], coords[:, 1]

    vmin, vmax = target_map.min(), target_map.max()
    err_maps = pred_maps[step_idxs] - target_map[None]
    err_abs_max = np.abs(err_maps).max() if err_maps.size else 1.0

    n_cols = len(step_idxs) + 1
    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 6.5))

    axes[0, 0].scatter(lon, lat, c=target_map, s=4, vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0, 0].set_title("target")
    axes[1, 0].set_axis_off()

    for col, step in enumerate(step_idxs, start=1):
        axes[0, col].scatter(
            lon, lat, c=pred_maps[step], s=4, vmin=vmin, vmax=vmax, cmap="viridis"
        )
        axes[0, col].set_title(f"pred (step {step})")
        axes[1, col].scatter(
            lon,
            lat,
            c=pred_maps[step] - target_map,
            s=4,
            vmin=-err_abs_max,
            vmax=err_abs_max,
            cmap="coolwarm",
        )
        axes[1, col].set_title(f"error (step {step})")

    for ax in axes.flat:
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
    fig.suptitle(channel_name)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved map plot to %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        help="Channels to plot (index or name each), one figure pair per channel. "
        f"Default: {DEFAULT_CHANNELS}.",
    )
    parser.add_argument(
        "--num-map-steps", type=int, default=3, help="Number of denoising steps shown in the maps."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: 'plots/' next to the store, with 'rmse/' and "
        "'maps/' subdirectories).",
    )
    args = parser.parse_args()

    store_path = _resolve_store_path(args.run_id, args.path, args.mini_epoch)
    out_dir = Path(args.output) if args.output else store_path.parent / "plots"
    rmse_dir = out_dir / "rmse"
    maps_dir = out_dir / "maps"
    group = zarr.open_group(str(store_path), mode="r")

    for sample_name in sorted(group.group_keys()):
        sample_group = group[sample_name]
        for stream_name in sorted(sample_group.group_keys()):
            stream_group = sample_group[stream_name]
            predictions = np.asarray(stream_group["predictions"])
            target = np.asarray(stream_group["target"])
            coords = np.asarray(stream_group["coords"])
            channels = list(stream_group.attrs["channels"])

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
                prefix = f"{sample_name}_{stream_name}_{channel_name}"
                plot_errors(
                    predictions, target, channel_idx, channel_name, rmse_dir / f"{prefix}.png"
                )
                plot_maps(
                    predictions,
                    target,
                    coords,
                    channel_idx,
                    channel_name,
                    args.num_map_steps,
                    maps_dir / f"{prefix}.png",
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
