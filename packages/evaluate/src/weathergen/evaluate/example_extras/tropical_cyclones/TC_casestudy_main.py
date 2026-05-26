#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "scikit-image>=0.26.0",
#   "scikit-learn>=1.8.0",
#   "xarray",
#   "tqdm",
#   "cartopy",
#   "omegaconf",
#   "netcdf4"
# ]
# ///

"""
This script tracks tropical cyclones in forecast and target
for a single sample, then looks for the tracks corresponding
to a user-selected storm of interest and produces diagnostic
plots for that storm, including the track error, simulated pressure 
and wind speed.

The tracking functionality is also intended for future use in
systematical evaluation of all tropical cyclones in the prediction. 

Before running, export 10u, 10v and msl to netcdf, regridded
to 1°x1° as follows:
uv run export --run-id <INFERENCE_ID> --stream ERA5 \
--output-dir <OUTDIR> --format netcdf --regrid-degree 1 \
--regrid-type regular_ll \
--channel 10u 10v msl
and again with --type target for the target. In TC_config.yml, set inpath to 
<OUTDIR> where the regridded data is.  Make sure that
the timesteps specificed in TC_config.yml are within the simulation. 

Then run this script via 
uv run TC_casestuy_main.py

All parameters including the strom of interest are set in the config.
"""

from functools import cached_property
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cyclone_finder import (
    Cyclone,
    CycloneFinder,
    cyclones_in_ds,
    track2pandas,
    track_cyclones,
    wrap_lon,
)
from cyclone_plots import track_eval_plot, track_snapshots
from omegaconf import OmegaConf


class TcCaseStudy:
    """
    Read the cyclone tracker settings, data paths and the target cyclone
    from config, then find the matched tracks corresponding to that cyclones
    in the prediction and target.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.selected_storm = Cyclone(
            wind=0,
            pressure=0,
            lon=cfg.selected_storm.lon,
            lat=cfg.selected_storm.lat,
            time=np.datetime64(cfg.selected_storm.time),
        )
        self.finder = CycloneFinder(
            sigma=cfg.tracking_params.laplace_size,
            th_laplace=cfg.tracking_params.laplace_threshold,
            th_pressure=cfg.tracking_params.pressure_threshold,
            th_wind=cfg.tracking_params.wind_threshold,
            min_distance=cfg.tracking_params.peak_separation,
        )
        self.outpath = Path(cfg.outpath)

    @cached_property
    def datasets(self):
        infiles = {
            k: f"{self.cfg.inpath}{k}_{self.cfg.init_time}_{self.cfg.runid}_ERA5.nc"
            for k in ("target", "prediction")
        }
        datasets = {
            k: wrap_lon(xr.open_dataset(f)).sel(latitude=slice(self.cfg.latmin, self.cfg.latmax))
            for k, f in infiles.items()
        }
        return datasets

    @cached_property
    def cyclones(self):
        times = self.datasets["target"].valid_time.values
        cyclones = {
            k: [cyclones_in_ds(ds, self.finder, time=t) for t in times]
            for k, ds in self.datasets.items()
        }
        return cyclones

    @cached_property
    def tracks(self):
        tracks = {
            k: track_cyclones(d, self.cfg.tracking_params.merge_distance)
            for k, d in self.cyclones.items()
        }
        return tracks

    @cached_property
    def matched_tracks(self):
        times = self.datasets["target"].valid_time.values
        storm_index = np.argmin(np.abs(times - self.selected_storm.time))
        matched_stroms = {
            k: self.selected_storm.match(x[storm_index]) for k, x in self.cyclones.items()
        }
        matched_tracks = {
            k: track2pandas(d.subset(matched_stroms[k])) for k, d in self.tracks.items()
        }
        return matched_tracks

    def plot(self):
        self.outpath.mkdir(exist_ok=True)
        # evaluation plot
        evalfile = f"{self.outpath}/{self.cfg.runid}_cyclone_{self.cfg.init_time}.png"
        fig, axs = track_eval_plot(self.matched_tracks)
        init_time = self.datasets["target"].forecast_reference_time.values
        fig.suptitle(f"forecast initialized {init_time}")
        plt.savefig(evalfile)

        # example maps
        snapshotfile = f"{self.outpath}/{self.cfg.runid}_cyclone_{self.cfg.init_time}_snapshots.png"
        track_snapshots(self.matched_tracks, self.datasets)
        plt.savefig(snapshotfile)


def main():
    cfg = OmegaConf.load("TC_config.yml")
    casestudy = TcCaseStudy(cfg)
    casestudy.plot()


if __name__ == "__main__":
    main()
