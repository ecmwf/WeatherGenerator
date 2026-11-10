# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Basic scores for the predictions.npz written by train_latent_decoder.

    python -m weathergen.latent_decoder.evaluate RESULTS_PATH [--plot]

Per channel: number of reports, bias, RMSE, correlation, and RMSE of the constant baseline
(always predicting the mean observation, i.e. the observations' standard deviation). Skill is
1 - RMSE / baseline RMSE: 0 means no better than a constant, 1 means perfect. Also RMSE per
forecast step. --plot writes rmse_vs_fstep.png and scatter.png next to predictions.npz
(needs matplotlib).
"""

import argparse
import logging
from pathlib import Path

import numpy as np

STEP_HOURS = 6
SHOW_FSTEPS = (1, 2, 4, 8, 12, 20, 28, 40)
_logger = logging.getLogger(__name__)


def scores(pred, obs):
    ok = np.isfinite(pred) & np.isfinite(obs)
    p, o = pred[ok], obs[ok]
    if len(o) < 2:
        return dict(n=len(o), bias=np.nan, rmse=np.nan, corr=np.nan, base=np.nan, skill=np.nan)
    rmse = np.sqrt(np.mean((p - o) ** 2))
    base = o.std()
    return dict(
        n=len(o),
        bias=np.mean(p - o),
        rmse=rmse,
        corr=np.corrcoef(p, o)[0, 1],
        base=base,
        skill=1 - rmse / base,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_path", type=Path)
    parser.add_argument("--plot", action="store_true", help="write PNG plots (needs matplotlib)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = _logger.info
    d = np.load(args.results_path / "predictions.npz")
    channels, fstep = [str(c) for c in d["channels"]], d["fstep"]
    fsteps = np.unique(fstep)
    num_inits = len(np.unique(d["init_time"]))
    out(f"{len(fstep)} reports, {num_inits} inits, fsteps {fsteps.min()}..{fsteps.max()}")

    header = ["n", "bias", "rmse", "base", "skill", "corr"]
    widths = [9, 9, 9, 9, 7, 6]
    out(
        "\n"
        + f"{'channel':18s} "
        + " ".join(f"{h:>{w}s}" for h, w in zip(header, widths, strict=True))
    )
    for c, name in enumerate(channels):
        s = scores(d["pred"][:, c], d["obs"][:, c])
        out(
            f"{name:18s} {s['n']:9d} {s['bias']:9.3f} {s['rmse']:9.3f} {s['base']:9.3f}"
            f" {s['skill']:7.3f} {s['corr']:6.3f}"
        )

    shown = [k for k in SHOW_FSTEPS if k in fsteps]
    out("\nRMSE per forecast step (lead time in h)")
    out(f"{'channel':18s} " + " ".join(f"{k * STEP_HOURS:>8d}" for k in shown))
    per_fstep = {}
    for c, name in enumerate(channels):
        per_fstep[name] = [
            scores(d["pred"][fstep == k, c], d["obs"][fstep == k, c])["rmse"] for k in fsteps
        ]
        row = [per_fstep[name][list(fsteps).index(k)] for k in shown]
        out(f"{name:18s} " + " ".join(f"{v:8.3f}" for v in row))

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(channels)
        fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3), squeeze=False)
        for ax, name in zip(axes[0], channels, strict=True):
            ax.plot(fsteps * STEP_HOURS, per_fstep[name], marker=".")
            ax.set(title=name, xlabel="lead time (h)", ylabel="RMSE")
        fig.tight_layout()
        fig.savefig(args.results_path / "rmse_vs_fstep.png", dpi=120)

        fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2), squeeze=False)
        rng = np.random.default_rng(0)
        for c, (ax, name) in enumerate(zip(axes[0], channels, strict=True)):
            p, o = d["pred"][:, c], d["obs"][:, c]
            ok = np.flatnonzero(np.isfinite(p) & np.isfinite(o))
            ok = rng.choice(ok, min(len(ok), 20000), replace=False) if len(ok) else ok
            ax.scatter(o[ok], p[ok], s=1, alpha=0.2)
            if len(ok):
                lo, hi = np.percentile(o[ok], [0.5, 99.5])
                ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
                ax.set(xlim=(lo, hi), ylim=(lo, hi))
            ax.set(title=name, xlabel="observed", ylabel="predicted")
        fig.tight_layout()
        fig.savefig(args.results_path / "scatter.png", dpi=120)
        out(f"\nplots: {args.results_path / 'rmse_vs_fstep.png'}")
        out(f"       {args.results_path / 'scatter.png'}")


if __name__ == "__main__":
    main()
