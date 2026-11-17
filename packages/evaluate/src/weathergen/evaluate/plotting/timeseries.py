# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The data timeseries: the mean of the prediction and the target over a region, per sample
and forecast step, computed per chunk (``Timeseries.region_means``, returned by
``plot_data`` as the ``region_means`` statistic), merged over the chunks
(``merge_statistics``) and drawn once all runs are done (``plot_timeseries``, called by
``plot_evaluation``): one figure per initialisation time with every run on it."""

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from weathergen.evaluate.utils.regions import RegionBoundingBox

_logger = logging.getLogger(__name__)


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


def plot_timeseries(
    cfg: dict, timeseries: dict[str, dict[str, xr.Dataset]], summary_dir: Path
) -> None:
    """The timeseries figures of all runs from their ``region_means``, merged over the
    chunks: one per (stream, region, channel, initialisation time[, ensemble member]), with
    the region mean of every run's prediction in its colour and of the target dashed along
    the valid time, into ``summary_dir/<run_ids>/timeseries/``.

    Parameters
    ----------
    cfg:
        The evaluation config (``run_ids``).
    timeseries:
        ``timeseries[stream][run_id]``: the ``region_means`` of the run's plotting samples,
        as ``accumulate_stream_results`` nests them.
    summary_dir:
        The summary directory of the run.
    """
    output_dir = Path(summary_dir) / "_".join(sorted(cfg.run_ids.keys())) / "timeseries"
    output_dir.mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg")
    n_figures = 0
    for stream, by_run in timeseries.items():
        init_times = sorted({t for ds in by_run.values() for t in ds.init_times.values})
        for init_time in init_times:
            # the sample of this init time in each run that has it
            at = {
                run_id: ds.isel(sample=int(np.flatnonzero(ds.init_times.values == init_time)[0]))
                for run_id, ds in by_run.items()
                if (ds.init_times.values == init_time).any()
            }
            target_run = next(iter(at.values()))
            init = np.datetime_as_string(np.datetime64(init_time), unit="m")
            for region in [str(r) for r in target_run.region.values]:
                for channel in [str(c) for c in target_run.channel.values]:
                    predictions = {
                        run_id: ds.prediction.sel(region=region, channel=channel)
                        for run_id, ds in at.items()
                        if region in ds.region.values and channel in ds.channel.values
                    }
                    target = target_run.target.sel(region=region, channel=channel)
                    valid_times = np.atleast_1d(target.valid_time.values)
                    members = sorted(
                        {
                            e
                            for p in predictions.values()
                            if "ens" in p.dims
                            for e in p.ens.values.tolist()
                        }
                    ) or [None]
                    for ens in members:
                        has_ens = ens is not None and ens != "mean"
                        fig, ax = plt.subplots(figsize=(15, 7))
                        for run_id, prediction in predictions.items():
                            if ens is not None and "ens" in prediction.dims:
                                prediction = prediction.sel(ens=ens)
                            label = run_id if not has_ens else f"{run_id} (ens {ens})"
                            ax.plot(valid_times, np.atleast_1d(prediction.values), label=label)
                        ax.plot(
                            valid_times,
                            np.atleast_1d(target.values),
                            label=stream,
                            linestyle="--",
                            color="k",
                        )
                        fig.suptitle(
                            f"Timeseries Average - {region.capitalize()} - {stream} - "
                            f"initialised {init}",
                            fontsize="x-large",
                            fontweight="bold",
                        )
                        ax.set_ylabel(channel)
                        ax.set_xlabel("Valid Time")
                        ax.legend()
                        day_interval = max(1, len(valid_times) // 20)
                        ax.xaxis.set_major_locator(
                            matplotlib.dates.DayLocator(interval=day_interval)
                        )
                        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m-%d"))
                        ax.grid(True, linestyle="--", alpha=0.5)
                        fig.autofmt_xdate()
                        fname = f"timeseries_{region}_{stream}_{channel}_{init.replace(':', '')}"
                        if has_ens:
                            fname += f"_ens_{ens}"
                        fig.savefig(output_dir / f"{fname}.png", bbox_inches="tight")
                        plt.close(fig)
                        n_figures += 1
    _logger.info(f"Timeseries: {n_figures} figure(s) of {len(timeseries)} stream(s).")
