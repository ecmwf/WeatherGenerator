from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import xarray as xr


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

        preds_steps, tars_steps = [], []
        for da_p, da_t in zip(self.da_preds.values(), self.da_tars.values(), strict=False):
            vt = da_p.valid_time.isel(ipoint=0).drop_vars("ipoint")
            preds_steps.append(da_p.mean(dim="ipoint").assign_coords(valid_time=vt))
            tars_steps.append(da_t.mean(dim="ipoint").assign_coords(valid_time=vt))
        self.da_preds_ts, self.da_tars_ts = (
            xr.concat(preds_steps, dim="forecast_step"),
            xr.concat(tars_steps, dim="forecast_step"),
        )

    def get_preds_tars_per_sample_channel(
        self, sample: int | str, channel: str
    ) -> tuple[xr.Dataset, xr.Dataset]:
        """Get preds/tars for the given sample/channel from the timeseries data.
        Parameters
        ----------
        sample: int | str
            The sample for which to extract data.
        channel: str
            The channel for which to extract data.

        Returns
        -------
        tuple[xr.Dataset, xr.Dataset]
            The prediction and target datasets for the given sample and channel.
        """
        da_preds_slice, da_tars_slice = (
            self.da_preds_ts.sel(sample=sample, channel=channel),
            self.da_tars_ts.sel(sample=sample, channel=channel),
        )
        return da_preds_slice, da_tars_slice

    def get_valid_times_per_sample_channel(
        self, sample: int | str, channel: str
    ) -> npt.NDArray[np.datetime64]:
        """Get valid times for the given sample/channel from the timeseries data.
        Parameters
        ----------
        sample: int | str
            The sample for which to extract data.
        channel: str
            The channel for which to extract data.

        Returns
        -------
        npt.NDArray[np.datetime64]
            The array of valid times for the given sample and channel.
        """
        return self.da_tars_ts.sel(sample=sample, channel=channel).valid_time.values

    def get_channels(self) -> list[str]:
        """Get the list of channels from the timeseries data."""
        da_tmp = next(iter(self.da_tars.values()))
        return da_tmp.channel.values

    def get_samples(self) -> list[str]:
        """Get the list of samples from the timeseries data."""
        da_tmp = next(iter(self.da_tars.values()))
        return da_tmp.sample.values

    def plot_single_timeseries(
        self, output_dir: str, channel: str, sample: int | str, stream: str
    ) -> None:
        """Plot and save a timeseries figure for one (channel, sample) pair."""
        da_preds_slice, da_tars_slice = self.get_preds_tars_per_sample_channel(sample, channel)
        valid_times = self.get_valid_times_per_sample_channel(sample, channel)

        matplotlib.use("Agg")
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(valid_times, da_preds_slice.values, label="Prediction")
        ax.plot(valid_times, da_tars_slice.values, label=stream, linestyle="--")
        fig.suptitle(
            f"Timeseries Average \u2013 {self.region_label()}",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_ylabel(channel)
        ax.set_xlabel("Valid Time")
        ax.legend()
        max_ticks = max(4, 15)
        day_interval = max(1, len(valid_times) // max_ticks)
        ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=day_interval))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m-%d"))
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.autofmt_xdate()
        out_path = Path(output_dir) / "plots" / stream / "timeseries"
        out_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_path / f"timeseries_{channel}_sample_{sample}.png",
            bbox_inches="tight",
        )
        plt.close(fig)

    def region_label(self) -> str:
        """Get a human-readable label for the region based on the lat/lon bounds."""

        _first_da = next(iter(self.da_tars.values()))
        lat_min = float(_first_da.lat.min())
        lat_max = float(_first_da.lat.max())
        lon_min = float(_first_da.lon.min())
        lon_max = float(_first_da.lon.max())
        region_label = (
            f"({lat_min:.2f}\u00b0 \u2013 {lat_max:.2f}\u00b0 N, "
            f"{lon_min:.2f}\u00b0 \u2013 {lon_max:.2f}\u00b0 E)"
        )

        return region_label
