# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Latent-space RMSE per forecast step, accumulated during (rollout) inference.

The evaluate package computes RMSE-vs-lead-time curves for physical variables from the
zarr output of an inference run. Latents are far too large to write out for that, so the
equivalent latent-space curve is accumulated online here and plotted with the very same
plotting classes from ``weathergen.evaluate`` so that both look alike.
"""

import json
import logging

import numpy as np
import torch
import xarray as xr

from weathergen.common.config import parse_timedelta
from weathergen.evaluate.plotting.line_plots import LinePlots
from weathergen.evaluate.plotting.plot_utils import create_filename
from weathergen.utils.distributed import ddp_average, is_root

logger = logging.getLogger(__name__)

# defaults mirroring the ones the evaluate package uses in plot_summary()
_PLOT_CFG = {
    "image_format": "png",
    "dpi_val": 300,
    "fig_size": (8, 10),
    "log_scale": False,
    "add_grid": True,
    "plot_ensemble": False,
    "baseline": None,
}


class LatentRolloutRMSE:
    """
    Accumulate the RMSE between predicted and encoded ground-truth latents per forecast
    step and plot it against lead time.
    """

    def __init__(self, cf, mode_cfg, device):
        self.run_id = cf.general.run_id
        self.device = device

        time_step = parse_timedelta(mode_cfg.get("forecast", {}).get("time_step", 0))
        self.step_hours = time_step / np.timedelta64(1, "h")

        # squared error sum and element count per forecast step; grown on demand
        self._sum_sq: dict[int, torch.Tensor] = {}
        self._counts: dict[int, torch.Tensor] = {}

    def add(self, step_idx: int, pred: torch.Tensor, truth: torch.Tensor) -> None:
        """
        Accumulate the squared error between a predicted and an encoded truth latent for one
        rollout step of one batch.

        Parameters
        ----------
        step_idx:
            Zero-based rollout step (lead index); step 0 is the forecast of ``t``.
        pred:
            Rolled-out latent, shape ``(N_members, H, D)`` (or ``(1, H, D)``).
        truth:
            Encoded truth latent, shape ``(1, H, D)``; broadcasts over ensemble members so the
            accumulated value is the member-averaged squared error.
        """

        pred = pred.float()
        truth = truth.float()
        if pred.shape[-2:] != truth.shape[-2:]:
            raise ValueError(
                f"Latent prediction shape {tuple(pred.shape)} is incompatible with truth latent "
                f"shape {tuple(truth.shape)} at rollout step {step_idx}."
            )
        diff = pred - truth
        sum_sq = diff.pow(2).sum(dtype=torch.float64)
        count = torch.tensor(float(diff.numel()), device=self.device, dtype=torch.float64)

        if step_idx not in self._sum_sq:
            self._sum_sq[step_idx] = torch.zeros((), device=self.device, dtype=torch.float64)
            self._counts[step_idx] = torch.zeros((), device=self.device, dtype=torch.float64)
        self._sum_sq[step_idx] += sum_sq.to(self.device)
        self._counts[step_idx] += count.to(self.device)

    def plot(self, output_dir) -> None:
        """Reduce across ranks, then write the RMSE-vs-lead-time plot (root rank only)."""

        if not self._sum_sq:
            logger.warning("No latent predictions collected; skipping latent RMSE plot.")
            return

        steps = np.array(sorted(self._sum_sq.keys()))
        sum_sq = ddp_average(torch.stack([self._sum_sq[s] for s in steps]))
        counts = ddp_average(torch.stack([self._counts[s] for s in steps]))
        rmse = torch.sqrt(sum_sq / counts).numpy()

        if not is_root():
            return

        data = xr.DataArray(
            rmse,
            dims=["forecast_step"],
            coords={"forecast_step": steps},
            name="rmse",
        )
        # forecast step k is valid (k+1) * time_step after the last conditioning state
        data = data.assign_coords(lead_time=("forecast_step", (steps + 1) * self.step_hours))
        x_dim = "forecast_step"
        if self.step_hours > 0:
            data = data.swap_dims({"forecast_step": "lead_time"})
            x_dim = "lead_time"

        tag = create_filename(prefix=["rmse", "global"], middle=[self.run_id], suffix=["latent"])

        plotter = LinePlots(_PLOT_CFG, output_dir)
        plotter.plot(
            [data],
            [self.run_id],
            tag=tag,
            x_dim=x_dim,
            y_dim="rmse",
            print_summary=True,
            title="RMSE | latent | z_pre_norm",
        )

        # drop the plotted values next to the figure so the curve can be re-used numerically;
        # "compare_" mirrors the prefix LinePlots.plot() puts on the figure file name
        self._write_json(plotter.out_plot_dir_lines / f"compare_{tag}.json", data)

    def _write_json(self, path, data: xr.DataArray) -> None:
        """Write the plotted curve as JSON, in the same layout the evaluate package uses."""

        data = data.assign_attrs(
            run_id=self.run_id,
            metric="rmse",
            space="latent",
            variable="z_pre_norm",
            step_hours=float(self.step_hours),
        )
        with open(path, "w") as f:
            json.dump(data.to_dict(), f, indent=2)
        logger.info(f"Wrote latent RMSE values to {path}")
