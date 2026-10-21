# ruff: noqa: T201

"""Plot diffusion ODE sampling diagnostics written by ScalarSeriesZarrWriter.

Reads the ``diffusion_sampling_diagnostics.zarr`` store produced during inference when
``fe_diffusion_dev_mode: true`` is set (see weathergen.model.diffusion.DiffusionForecastEngine
and weathergen.common.dev_output.ScalarSeriesZarrWriter), and recreates the sampling
diagnostics figure that used to be plotted inline during inference.

Each named array in the store has shape (num_calls, num_ode_steps): one row per
inference call (e.g. per validation sample). By default the most recent call is
plotted; use --call-index to pick another one, or --all to overlay every call.

Usage:
    uv run python scripts/dev/plot_diffusion_sampling_diagnostics.py --run-id <id>
    uv run python scripts/dev/plot_diffusion_sampling_diagnostics.py --path <run_dir> --call-index 3
    uv run python scripts/dev/plot_diffusion_sampling_diagnostics.py --run-id <id> --all
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

import weathergen.common.config as config

logger = logging.getLogger(__name__)

STORE_NAME = "diffusion_sampling_diagnostics"


def _resolve_store_path(run_id: str | None, path: str | None, mini_epoch: int | None) -> Path:
    if path is not None:
        base = Path(path)
    else:
        if run_id is None:
            raise ValueError("Either --run-id or --path must be provided.")
        cf = config.load_run_config(run_id=run_id, mini_epoch=mini_epoch, model_path=None)
        base = config.get_path_run(cf)
    store_path = base / "dev" / STORE_NAME / f"{STORE_NAME}.zarr"
    if not store_path.exists():
        raise FileNotFoundError(
            f"No dev diagnostics store found at {store_path}. Was the run started with "
            "fe_diffusion_dev_mode=true?"
        )
    return store_path


def _plot_call(axes, steps: np.ndarray, series: dict[str, np.ndarray], label: str | None) -> None:
    kwargs = {"label": label} if label else {}

    axes[0].semilogy(steps, series["sigma"], "o-", markersize=3, **kwargs)
    if "sigma_post" in series and not np.array_equal(series["sigma_post"], series["sigma"]):
        axes[0].semilogy(steps, series["sigma_post"], "x--", markersize=4, color="tab:green")
    axes[0].set_ylabel("sigma (noise level)")

    axes[1].plot(steps, series["x_std"], "o-", markersize=3, **kwargs)
    axes[1].plot(steps, series["denoised_std"], "s-", markersize=3)
    axes[1].set_ylabel("std")

    if "l2_to_target" in series and len(series["l2_to_target"]) == len(steps):
        axes[2].plot(steps, series["l2_to_target"], "o-", markersize=3, color="tab:red", **kwargs)
    axes[2].set_ylabel("L2 error to target")

    axes[3].semilogy(steps, series["d_cur_norm"], "o-", markersize=3, **kwargs)
    axes[3].semilogy(steps, series["d_cur_step_norm"], "^-", markersize=3)
    axes[3].set_ylabel("norm (log scale)")

    axes[4].semilogy(steps, series["residual_std"], "s-", markersize=3, color="tab:orange", **kwargs)
    axes[4].set_ylabel("std (log scale)")

    axes[5].plot(steps, series["residual_std"], "s-", markersize=3, color="tab:orange", **kwargs)
    axes[5].set_ylim(0, 1)
    axes[5].set_ylabel("std (clipped to 1)")

    axes[6].semilogy(steps, series["x_std"], "o-", markersize=3, color="tab:blue", **kwargs)
    axes[6].set_ylabel("std (log scale)")


def plot_diagnostics(store_path: Path, call_indices: list[int], out_path: Path) -> None:
    group = zarr.open_group(str(store_path), mode="r")
    num_steps = group["sigma"].shape[1]
    steps = np.arange(num_steps)

    n_plots = 7
    titles = [
        "Sigma schedule",
        "Std of x (noisy state) / denoised estimate",
        "L2 error to target",
        "ODE drift norms",
        "Std(x_post - denoised)",
        "Std(x_post - denoised)  [y <= 1]",
        "Std of x_next over denoising steps",
    ]
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots), sharex=True)

    for call_idx in call_indices:
        # Some series (e.g. "l2_to_target") were only recorded for a subset of calls in
        # older stores, so their row count can be smaller than others'; skip those here
        # rather than failing the whole plot.
        series = {
            key: np.asarray(group[key][call_idx])
            for key in group.array_keys()
            if -group[key].shape[0] <= call_idx < group[key].shape[0]
        }
        label = f"call {call_idx}" if len(call_indices) > 1 else None
        _plot_call(axes, steps, series, label)

    for ax, title in zip(axes, titles, strict=True):
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if len(call_indices) > 1:
            ax.legend(fontsize=8)
    axes[-1].set_xlabel("sampling step")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved sampling diagnostics plot to %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None, help="Run id to load the config/path from.")
    parser.add_argument("--path", default=None, help="Run directory, as an alternative to --run-id.")
    parser.add_argument(
        "--mini-epoch",
        type=int,
        default=0,
        help="Checkpoint mini_epoch used when the run config was saved (default: 0, matching "
        "inference-only runs; use the training checkpoint's mini_epoch otherwise).",
    )
    parser.add_argument(
        "--call-index", type=int, default=-1, help="Which inference call to plot (default: last)."
    )
    parser.add_argument(
        "--all", action="store_true", help="Overlay every recorded call instead of just one."
    )
    parser.add_argument(
        "--output", default=None, help="Output image path (default: <run>/dev/%s/sampling_diagnostics.png)" % STORE_NAME
    )
    args = parser.parse_args()

    store_path = _resolve_store_path(args.run_id, args.path, args.mini_epoch)
    group = zarr.open_group(str(store_path), mode="r")
    num_calls = group["sigma"].shape[0]

    call_indices = list(range(num_calls)) if args.all else [args.call_index % num_calls]
    out_path = (
        Path(args.output)
        if args.output
        else store_path.parent / "sampling_diagnostics.png"
    )

    plot_diagnostics(store_path, call_indices, out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
