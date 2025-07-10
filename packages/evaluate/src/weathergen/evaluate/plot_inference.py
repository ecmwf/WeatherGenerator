from pathlib import Path

from typing import List
import dask.array as da
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib as mpl
# import cartopy.crs as ccrs
import numpy as np
from scores import Scores
import itertools
# from scores import Scores
import os

from typing import List

import matplotlib.pyplot as plt
import logging
# from plotting_utils import *
from plotter import Plotter, LinePlots
from omegaconf import OmegaConf
from weathergen.common.io import ZarrIO
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

select = OmegaConf.select
load = OmegaConf.load



if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    cfg = load("plot_config.yaml")

    models = cfg.model_ids
    logging.info(f"Computing scores for {len(models)} models")
    input_dir = cfg.input_dir
    metrics = cfg.metrics

    # to get a structure like: scores_dict[metric][stream][model_id] = plot
    scores_dict = defaultdict(           
                lambda: defaultdict(
                    dict
            )
        )


    for model_id, model  in models.items():

        plotter = Plotter(cfg, model_id)
        
        logging.info(f"MODEL {model_id}: Getting data...")
        fname_zarr = Path(input_dir + f"/{model_id}/validation_epoch{model["epoch"]:05d}_rank{model["rank"]:04d}.zarr")
        
        assert (fname_zarr.exists() and fname_zarr.is_dir()), f"Check Zarr input: {fname_zarr}"

        with ZarrIO(fname_zarr) as zio:
            zio_streams, zio_forecast_steps = zio.streams, zio.forecast_steps
            streams = select(model, "streams").keys() or zio_streams
            for stream in streams: 
                stream_dict = model.streams[stream]
                logging.info(f"MODEL {model_id}: Processing stream {stream}...")

                fsteps    = select(stream_dict, "fsteps")  or zio_forecast_steps
                samples   = select(stream_dict, "samples") or [int(sample) for sample in zio.samples]
                variables = select(stream_dict, "variables")

                da_tars, da_preds = [], []
                
                for fstep in fsteps: 
                    logging.info(f"MODEL {model_id} - {stream}: Processing fstep {fstep}...")
                    da_tars_fs, da_preds_fs = [], []

                    for sample in tqdm(samples, desc=f"Processing samples for {model_id} - {stream} - {fstep}"):

                        out = zio.get_data(sample, stream, fstep)

                        #remove squeeze when ensemble is
                        tars = out.target.as_xarray()
                        preds = out.prediction.as_xarray()
                        da_tars_fs.append(tars)
                        da_preds_fs.append(preds)

                        plotter = plotter.selection(sample, stream, fstep)

                        if model.plot_maps:
                            plotter.map(tars, variables, "target")
                            plotter.map(preds, variables, "preds")

                        if model.plot_histograms:
                            plotter.histogram(tars, preds, variables)
                        
                        plotter = plotter.clean_selection()

                    da_tars_fs  = xr.concat(da_tars_fs , dim="ipoint")
                    da_preds_fs = xr.concat(da_preds_fs, dim="ipoint")

                    da_tars.append(da_tars_fs)
                    da_preds.append(da_preds_fs)

                da_tars  = xr.concat(da_tars , dim="forecasting_step").sel(channel = variables)
                da_preds = xr.concat(da_preds, dim="forecasting_step").sel(channel = variables)

                logging.info(f"MODEL {model_id} - {stream} - {fstep}: Computing scores...")

                #TODO: change ens dims when score class is updated
                score_engine = Scores(da_preds, da_tars, avg_dims=["ipoint"], ens_dim = '')

                #TODO: add option to compute scores only if they're not already in the database
                # cache scores. it should be small
                for metric in metrics:
                    #TODO: is there a better way than a flat dict? 
                    # id_name = "_".join([metric, model_id, stream, str(fstep)])
                    try:
                        #TODO: this needs more thinking as only part of the variables might be already pre-computed. 
                        #with a loop over variables we would loose the advantage of parallel computing. 
                        #possibility: load already computed scores before initialising the data and use e.g. "new_variable" index for the letfovers.
                        #problem: different metrics can be pre-computed for different variables, so not easy to define a "new_variables" list.   
                        scores_dict[metric][stream][model_id] = retrieve_score(model_id, stream, metric, bucket = "XXX")
                    except Exception as e:
                        scores_dict[metric][stream][model_id] = score_engine(metric).compute()




#plot summary
if cfg.fstep_summary_plots:

    logging.info(f"Started creating summary plots..")

    plotter = LinePlots(cfg)
    
    for metric in metrics:
        #get list of streams, independently of model_id 
        #TODO: improve this
        streams_set = set.union(*[set(model["streams"].keys()) for model in models.values()])
        
        #get list of streams, independently of the stream 
        #TODO: improve this
        variables_set = set.union(*[
                        set(stream["variables"])
                        for model in models.values()
                        for stream in model["streams"].values()])

        #TODO: move this into plot_utils so to have more methods
        for stream in streams_set: #loop over streams
            for var in variables_set: #loop over variables
                selected_data = []
                labels = []
                for model_id, data in scores_dict[metric][stream].items():

                    #fill list of plots with one xarray per model_id, if it exists. 
                    if var not in set(data.channel.values):
                        continue
                    selected_data.append(data.sel(channel=var))
                    labels.append(model_id)
                
                name = "_".join([metric, stream, var])
                plotter.plot(selected_data, labels, tag = name, plot_dim="forecast_step")

