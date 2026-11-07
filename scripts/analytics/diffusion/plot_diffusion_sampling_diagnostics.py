# ruff: noqa: T201

"""Plot diffusion ODE sampling diagnostics written by ScalarSeriesJSONLWriter.

Reads the ``diffusion_sampling_diagnostics.jsonl`` file produced during inference when
``fe_diffusion_log_analytics: true`` is set (see weathergen.model.diffusion.DiffusionForecastEngine
and weathergen.utils.inference_logger.ScalarSeriesJSONLWriter), and recreates the sampling
diagnostics figure that used to be plotted inline during inference.

The file holds one JSON line per inference call, each a flat ``dict[str, list[float]]`` of
the named 1D series recorded for that call (e.g. one value per ODE step). By default the
most recent call is plotted; use --call-index to pick another one, or --all to overlay
every call.

Usage:
    uv run python scripts/analytics/diffusion/plot_diffusion_sampling_diagnostics.py --run-id <id>
    uv run python scripts/analytics/diffusion/plot_diffusion_sampling_diagnostics.py --path <run_dir> --call-index 3
    uv run python scripts/analytics/diffusion/plot_diffusion_sampling_diagnostics.py --run-id <id> --all
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
    store_path = base / "analytics" / STORE_NAME / f"{STORE_NAME}.jsonl"
    if not store_path.exists():
        raise FileNotFoundError(
            f"No analytics diagnostics file found at {store_path}. Was the run started with "
            "fe_diffusion_log_analytics=true?"
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


def _read_calls(store_path: Path) -> list[dict[str, list[float]]]:
    """Read every JSON line (one dict of named 1D series per inference call)."""
    with store_path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def plot_diagnostics(calls: list[dict[str, list[float]]], call_indices: list[int], out_path: Path) -> None:
    num_steps = len(calls[call_indices[0]]["sigma"])
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
        call = calls[call_idx]
        # Some series (e.g. "l2_to_target") are only recorded for a subset of calls, so a
        # call's dict may be missing some keys; _plot_call already handles that.
        series = {key: np.asarray(values) for key, values in call.items()}
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
    calls = _read_calls(store_path)
    num_calls = len(calls)

    call_indices = list(range(num_calls)) if args.all else [args.call_index % num_calls]
    out_path = (
        Path(args.output)
        if args.output
        else store_path.parent / "sampling_diagnostics.png"
    )

    plot_diagnostics(calls, call_indices, out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
