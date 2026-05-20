from cyclone_finder import cyclone_finder, cyclone,track_error, track_cyclones, track2pandas, cyclones_in_ds, wrap_lon
import xarray as xr
import numpy as np
from functools import cached_property
from cyclone_plots import track_eval_plot, track_snapshots
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf

class TC_casestudy():

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.selected_storm = cyclone(
            wind=0, 
            pressure=0, 
            lon=cfg.selected_storm.lon, 
            lat=cfg.selected_storm.lat, 
            time=np.datetime64(cfg.selected_storm.time)
        )
        self.finder = cyclone_finder(
            sigma = cfg.tracking_params.laplace_size,
            th_LoG= cfg.tracking_params.laplace_threshold, 
            th_pressure=cfg.tracking_params.pressure_threshold,
            th_wind=cfg.tracking_params.wind_threshold, 
            min_distance=cfg.tracking_params.peak_separation
        )
        self.outpath = Path(cfg.outpath)
    
    @cached_property
    def datasets(self):
        infiles = { 
            k: f"{self.cfg.inpath}{k}_{self.cfg.init_time}_{self.cfg.runid}_ERA5.nc" 
            for k in ("target","prediction")
        }
        datasets = { 
            k: wrap_lon(xr.open_dataset(f)).sel(latitude=slice(self.cfg.latmin,self.cfg.latmax)) 
            for k,f in infiles.items()
        }
        return datasets

    @cached_property
    def cyclones(self):
        times = self.datasets["target"].valid_time.values
        cyclones =  { 
            k: [ cyclones_in_ds(ds, self.finder, time=t) for t in times ]
            for k,ds in self.datasets.items() 
        }
        return cyclones
    
    @cached_property
    def tracks(self):
        tracks = {
            k: track_cyclones(d, self.cfg.tracking_params.merge_distance) 
            for k,d in self.cyclones.items()
        }
        return tracks

    @cached_property
    def matched_tracks(self):
        times = self.datasets["target"].valid_time.values
        storm_index = np.argmin(np.abs(times - self.selected_storm.time))
        matched_stroms = {
            k: self.selected_storm.match(x[storm_index]) 
            for k,x in self.cyclones.items()
        }
        matched_tracks = {
            k: track2pandas(d.subset(matched_stroms[k]))
            for k,d in self.tracks.items()
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
    casestudy = TC_casestudy(cfg)
    casestudy.plot()

if __name__ == "__main__":
    main()