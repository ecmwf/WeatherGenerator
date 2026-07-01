from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import xarray as xr

from weathergen.evaluate.utils.regions import RegionBoundingBox


class Timeseries:
    """
    Initialize the Timeseries class.

    Parameters
    ----------
    da_preds:
        Dictionary of prediction datasets.
    da_tars:
        Dictionary of target datasets.
    """

    def __init__(self, da_preds: dict[str, xr.Dataset], da_tars: dict[str, xr.Dataset]):
        self.da_preds = da_preds
        self.da_tars = da_tars

    def get_preds_tars_per_region_sample_channel(
        self, region: str, sample: int | str, channel: str
    ) -> tuple[xr.Dataset, xr.Dataset]:
        """Get preds/tars for the given sample/channel from the timeseries data.
        Parameters
        ----------
        region: str
            The region for which to extract data.
        sample: int | str
            The sample for which to extract data.
        channel: str
            The channel for which to extract data.

        Returns
        -------
        tuple[xr.Dataset, xr.Dataset]
            The prediction and target datasets for the given sample and channel.
        """

        preds_steps, tars_steps = [], []
        for da_p, da_t in zip(self.da_preds.values(), self.da_tars.values(), strict=False):
            # Select sample and channel first so lat/lon become 1D (ipoint,)
            da_p = da_p.sel(sample=sample, channel=channel)
            da_t = da_t.sel(sample=sample, channel=channel)
            if region != "global":
                bbox = RegionBoundingBox.from_region_name(region)
                da_p = bbox.apply_mask(da_p)
                da_t = bbox.apply_mask(da_t)
            vt = da_p.valid_time.isel(ipoint=0).drop_vars("ipoint")
            preds_steps.append(da_p.mean(dim="ipoint").assign_coords(valid_time=vt))
            tars_steps.append(da_t.mean(dim="ipoint").assign_coords(valid_time=vt))
        da_preds_ts, da_tars_ts = (
            xr.concat(preds_steps, dim="forecast_step", coords="different", compat="equals"),
            xr.concat(tars_steps, dim="forecast_step", coords="different", compat="equals"),
        )
        return da_preds_ts, da_tars_ts

    def plot_single_timeseries(
        self,
        output_dir: str,
        channel: str,
        sample: int | str,
        stream: str,
        region: str,
        ens: str | int | None = None,
    ) -> None:
        """Plot and save a timeseries figure for one (channel, sample[, ens]) triple."""

        da_preds_ts, da_tars_ts = self.get_preds_tars_per_region_sample_channel(
            region, sample, channel
        )
        has_ens = ens is not None and "ens" in da_preds_ts.dims and ens != "mean"
        if has_ens:
            da_preds_ts = da_preds_ts.sel(ens=ens)
        valid_times = da_tars_ts.valid_time.values

        pred_label = "Prediction" if not has_ens else f"Prediction (ens {ens})"

        matplotlib.use("Agg")
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(valid_times, da_preds_ts.values, label=pred_label)
        ax.plot(valid_times, da_tars_ts.values, label=stream, linestyle="--")
        fig.suptitle(
            f"Timeseries Average - {region.capitalize()}",
            fontsize=13,
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
        out_path = Path(output_dir) / "plots" / stream / "timeseries"
        out_path.mkdir(parents=True, exist_ok=True)
        fname = f"timeseries_{region}_{channel}_sample_{sample}"
        if has_ens:
            fname += f"_ens_{ens}"
        fig.savefig(out_path / f"{fname}.png", bbox_inches="tight")
        plt.close(fig)
