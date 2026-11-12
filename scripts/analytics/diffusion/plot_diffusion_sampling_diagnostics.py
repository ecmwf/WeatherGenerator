# ruff: noqa: T201

"""Plot diffusion / flow-matching sampling diagnostics written by ScalarSeriesJSONLWriter.

Reads the ``diffusion_sampling_diagnostics.jsonl`` file produced during inference when
``fe_diffusion_log_analytics: true`` is set (see weathergen.model.edm.EDMForecastEngine,
weathergen.model.flow_matching.FlowMatchingForecastEngine and weathergen.utils.inference_logger.ScalarSeriesJSONLWriter), and recreates the sampling
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


def plot_flow_diagnostics(
    calls: list[dict[str, list[float]]], call_indices: list[int], out_path: Path, skip: int
) -> None:
    """Diagnostic plot of a flow-matching sampling trajectory (FlowMatchingForecastEngine).

    At every path time t we compare the current *noisy state* x_t against the *denoised
    estimate* x0_hat = D_t(x_t) predicted from it:
      1. RMSE to the target:  rmse(x_t, z)  vs  rmse(x0_hat, z)
      2. Standard deviation:  std(x_t)      vs  std(x0_hat)
    Panel 1 requires a reference target and is OMITTED when absent — which is the case in
    rollout mode (``forecast.num_steps>1``), where model.py sets tokens=None after the first
    step. Run with ``forecast.num_steps=1`` to get it.
    x0_hat should approach the target (and the target's std) much earlier than x_t does — x_t
    only gets there at t=1, whereas x0_hat is the model's best guess at the clean latent at
    every t.
    """
    has_target = all("rmse_x_t" in calls[i] for i in call_indices)
    n = 3 if has_target else 2
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.2 * n), sharex=True)

    for call_idx in call_indices:
        series = {key: np.asarray(values) for key, values in calls[call_idx].items()}
        label = f"call {call_idx}: " if len(call_indices) > 1 else ""
        # Drop the first step(s) from the PLOT (the data is still recorded). At t ~ t_eps the
        # denoiser carries a 1/alpha factor for the noise/score parameterisations, so x0_hat and
        # the ODE drift blow up by ~1/t_eps there; left in, that single point dominates the log
        # y-scale and flattens everything else. Set --skip-steps 0 to keep it.
        ts_all = series["t"]
        skip = min(skip, max(len(ts_all) - 2, 0))  # never leave fewer than 2 points
        sl = slice(skip, None)
        steps = np.arange(skip, len(ts_all))
        i = 0

        # 1) RMSE to target: noisy state vs denoised estimate
        if has_target:
            axes[i].semilogy(steps, series["rmse_x_t"][sl], "o-", ms=3, color="tab:blue",
                             label=label + r"rmse($x_t$, $z$)  (noisy state)")
            axes[i].semilogy(steps, series["rmse_x0_hat"][sl], "s-", ms=3, color="tab:red",
                             label=label + r"rmse($\hat{x}_0(x_t)$, $z$)  (denoised estimate)")
            axes[i].set_ylabel("RMSE to target")
            i += 1

        # 2) std: noisy state vs denoised estimate
        axes[i].plot(steps, series["x_t_std"][sl], "o-", ms=3, color="tab:blue",
                     label=label + r"std($x_t$)")
        axes[i].plot(steps, series["x0_hat_std"][sl], "s-", ms=3, color="tab:red",
                     label=label + r"std($\hat{x}_0(x_t)$)")
        if "target_std" in series:
            axes[i].axhline(series["target_std"][0], color="grey", ls="--", lw=0.8,
                            label="target std")
        axes[i].set_ylabel("std")
        axes[i].set_yscale("log")
        i += 1

        # 3) velocity magnitude
        axes[i].semilogy(steps, series["vel_norm"][sl], "^-", ms=3, color="tab:orange")
        axes[i].set_ylabel("||velocity||")

    skipped = f"  [first {skip} step(s) omitted]" if skip else ""
    axes[0].set_title(f"Flow-matching sampling{skipped}")
    for ax in axes:
        ax.grid(True, alpha=0.3, which="both")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    ts = calls[call_indices[0]]["t"]
    axes[-1].set_xlabel(f"sampling step   (path variable: {ts[skip]:.3f} -> {ts[-1]:.3f})")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved flow sampling diagnostics plot to %s", out_path)


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
        "--skip-steps",
        type=int,
        default=1,
        help="Flow matching only: number of initial sampling steps omitted from the plot.",
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

    # flow-matching runs (FlowMatchingForecastEngine) record the path variable "t"
    if "t" in calls[call_indices[0]]:
        plot_flow_diagnostics(calls, call_indices, out_path, args.skip_steps)
    else:
        plot_diagnostics(calls, call_indices, out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
