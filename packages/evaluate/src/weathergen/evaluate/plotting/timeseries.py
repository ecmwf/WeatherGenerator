# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The data timeseries: the mean of the prediction and the target over a region, per sample
and forecast step, and its figure, one per initialisation time with every run on it."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from weathergen.evaluate.utils.regions import RegionBoundingBox


class Timeseries:
    """
    Initialize the Timeseries class.

    Parameters
    ----------
    da_preds:
        Dictionary of prediction datasets, keyed by forecast step.
    da_tars:
        Dictionary of target datasets, keyed by forecast step.
    """

    def __init__(self, da_preds: dict[str, xr.Dataset], da_tars: dict[str, xr.Dataset]):
        self.da_preds = da_preds
        self.da_tars = da_tars

    def region_means(self, regions: list[str], channels: list[str]) -> xr.Dataset:
        """The mean of the prediction and the target over each region, per sample and step.

        Parameters
        ----------
        regions:
            Region names (``RegionBoundingBox``); ``global`` is the whole field.
        channels:
            The channels to keep.

        Returns
        -------
        xr.Dataset
            Variables ``prediction`` and ``target`` with dims ``(sample, forecast_step,
            channel[, ens], region)`` (``ens`` on the prediction only, when it has members),
            the ``valid_time`` of every (sample, forecast step) and the ``init_times`` of the
            samples as coordinates.  Small: one number per point of the dims, so a chunk of
            samples and forecast steps returns its share and ``merge.merge_timeseries``
            concatenates the chunks.
        """
        preds, tars = [], []
        for region in regions:
            bbox = None if region == "global" else RegionBoundingBox.from_region_name(region)
            preds_steps, tars_steps = [], []
            for fstep, da_p in self.da_preds.items():
                da_t = self.da_tars[fstep]
                da_p = da_p.sel(channel=channels)
                da_t = da_t.sel(channel=channels)
                if bbox is not None:
                    da_p = bbox.apply_mask(da_p)
                    da_t = bbox.apply_mask(da_t)
                vt = da_p.valid_time.isel(ipoint=0).drop_vars("ipoint", errors="ignore")
                preds_steps.append(da_p.mean(dim="ipoint").assign_coords(valid_time=vt))
                tars_steps.append(da_t.mean(dim="ipoint").assign_coords(valid_time=vt))
            preds.append(
                xr.concat(preds_steps, dim="forecast_step", coords="different", compat="equals")
            )
            tars.append(
                xr.concat(tars_steps, dim="forecast_step", coords="different", compat="equals")
            )
        return xr.Dataset(
            {
                "prediction": xr.concat(preds, dim="region").assign_coords(region=list(regions)),
                "target": xr.concat(tars, dim="region").assign_coords(region=list(regions)),
            }
        ).transpose("sample", "forecast_step", "channel", ..., "region")


def plot_single_timeseries(
    output_dir: str | Path,
    channel: str,
    region: str,
    stream: str,
    init_time: np.datetime64,
    predictions: dict[str, xr.DataArray],
    target: xr.DataArray,
    ens: str | int | None = None,
) -> None:
    """One figure: the region mean of every run's prediction and of the target along the
    valid time of one initialisation time.

    Parameters
    ----------
    output_dir:
        The directory of the figures.
    channel, region, stream, init_time:
        What the figure is of.
    predictions:
        Per run id, the prediction of this (init time, region, channel): dims
        ``(forecast_step[, ens])`` with the ``valid_time`` coordinate.
    target:
        The target, same dims without ``ens``.
    ens:
        The ensemble member to draw, when the predictions have members.
    """
    valid_times = np.atleast_1d(target.valid_time.values)
    has_ens = ens is not None and ens != "mean"

    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(15, 7))
    for run_id, prediction in predictions.items():
        if ens is not None and "ens" in prediction.dims:
            prediction = prediction.sel(ens=ens)
        label = run_id if not has_ens else f"{run_id} (ens {ens})"
        ax.plot(valid_times, np.atleast_1d(prediction.values), label=label)
    ax.plot(valid_times, np.atleast_1d(target.values), label=stream, linestyle="--", color="k")
    init = np.datetime_as_string(np.datetime64(init_time), unit="m")
    fig.suptitle(
        f"Timeseries Average - {region.capitalize()} - {stream} - initialised {init}",
        fontsize="x-large",
        fontweight="bold",
    )
    ax.set_ylabel(channel)
    ax.set_xlabel("Valid Time")
    ax.legend()
    max_ticks = 20
    day_interval = max(1, len(valid_times) // max_ticks)
    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=day_interval))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m-%d"))
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.autofmt_xdate()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fname = f"timeseries_{region}_{stream}_{channel}_{init.replace(':', '')}"
    if has_ens:
        fname += f"_ens_{ens}"
    fig.savefig(out_path / f"{fname}.png", bbox_inches="tight")
    plt.close(fig)
