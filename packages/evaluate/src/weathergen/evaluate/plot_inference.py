#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "weathergen-evaluate",
#   "weathergen-common",
#   "panel",
#   "omegaconf"
# ]
# [tool.uv.sources]
# weathergen-evaluate = { path = "../../../../../packages/evaluate" }
# ///

from pathlib import Path
import sys
import argparse
import json

from typing import List
import dask.array as da
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib as mpl
import cartopy.crs as ccrs
import numpy as np
from score import VerifiedData, get_score
import itertools
import os

from typing import List
import matplotlib.pyplot as plt
import logging
from plotter import Plotter, LinePlots
from omegaconf import OmegaConf
from weathergen.common.io import ZarrIO
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_logger.addHandler(handler)
_logger.propagate = False  # Prevent double logging if root logger also logs

select = OmegaConf.select
load = OmegaConf.load


def retrieve_score(dir, model_id, stream, metric, epoch, rank):
    #TODO adapt this to JSON output
    json_path = Path(f"{dir}/{metric}_{model_id}_{stream}_epoch{epoch:05d}_rank{rank:04d}.json")
    _logger.info(f"Looking for: {json_path}")
    if json_path.exists():
        with open(json_path, "r") as f:
            data_dict = json.load(f)
            return xr.DataArray.from_dict(data_dict)
    else:
        raise FileNotFoundError

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Fast evaluation of weather generator runs."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="plot_config.yaml",
        help="Path to the configuration file.",
    )

    args = parser.parse_args()

    cfg = load(args.config)
    models = cfg.model_ids
    _logger.info(f"Computing scores for {len(models)} models")
    
    assert select(cfg, "jsons_dir"), "Please provide a path to the directory where the json files are stored or will be saved."
    jsons_dir = Path(cfg.jsons_dir)
    jsons_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(cfg.input_dir)
    metrics = cfg.metrics
    
    # to get a structure like: scores_dict[metric][stream][model_id] = plot
    scores_dict = defaultdict(           
                lambda: defaultdict(
                    dict
            )
        )

    for model_id, model  in models.items():

        plotter = Plotter(cfg, model_id)
        
        _logger.info(f"MODEL {model_id}: Getting data...")
        fname_zarr = input_dir.joinpath(f"{model_id}/validation_epoch{model["epoch"]:05d}_rank{model["rank"]:04d}.zarr")
        
        assert (fname_zarr.exists() and fname_zarr.is_dir()), f"Check Zarr input: {fname_zarr}"

        with ZarrIO(fname_zarr) as zio:
            zio_streams, zio_forecast_steps = zio.streams, zio.forecast_steps
            streams = select(model, "streams").keys() or zio_streams
            for stream in streams: 
                stream_dict = model.streams[stream]
                _logger.info(f"MODEL {model_id}: Processing stream {stream}...")

                fsteps    = select(stream_dict, "fsteps")  or zio_forecast_steps
                samples   = select(stream_dict, "samples") or [int(sample) for sample in zio.samples]
                variables = select(stream_dict, "variables")

                da_tars, da_preds = [], []
                
                for fstep in fsteps: 
                    _logger.info(f"MODEL {model_id} - {stream}: Processing fstep {fstep}...")
                    da_tars_fs, da_preds_fs = [], []

                    for sample in tqdm(samples, desc=f"Processing samples for {model_id} - {stream} - {fstep}"):

                        out = zio.get_data(sample, stream, fstep)

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

                    _logger.debug(
                        f"Concatenating targets and predictions for stream {stream}, forecast_step {fstep}..."
                    )

                    da_tars_fs  = xr.concat(da_tars_fs , dim="ipoint")
                    da_preds_fs = xr.concat(da_preds_fs, dim="ipoint")

                    da_tars.append(da_tars_fs)
                    da_preds.append(da_preds_fs)

                _logger.debug(
                        f"Concatenating targets and predictions along the forecast_step dimension..."
                    )

                da_tars  = xr.concat(da_tars , dim="forecast_step").sel(channel = variables)
                da_preds = xr.concat(da_preds, dim="forecast_step").sel(channel = variables)

                _logger.info(f"MODEL {model_id} - {stream} - {fstep}: Computing scores...")

                score_data = VerifiedData(da_preds, da_tars) 

                for metric in metrics:
                    _logger.info(f"Computing {metric} for {model_id} - {stream}")
                    try:
                        #TODO: this needs more thinking as only part of the variables might be already pre-computed. 
                        #with a loop over variables we would loose the advantage of parallel computing. 
                        #possibility: load already computed scores before initialising the data and use e.g. "new_variable" index for the letfovers.
                        #problem: different metrics can be pre-computed for different variables, so not easy to define a "new_variables" list.   
                        scores_dict[metric][stream][model_id] = retrieve_score(jsons_dir, model_id, stream, metric, model.epoch, model.rank)
                    except Exception as e:
                        scores_dict[metric][stream][model_id] = get_score(score_data, metric, agg_dims=list(cfg.avg_dims)) 
                        #save scores to json
                        save_path = jsons_dir.joinpath(f"{metric}_{model_id}_{stream}_epoch{model.epoch:05d}_rank{model.rank:04d}.json")
                        _logger.info(f"Saving results to {save_path}")
                        with open(save_path, "w") as f:
                            json.dump(scores_dict[metric][stream][model_id].compute().to_dict(), f, indent=4)


#plot summary
if cfg.fstep_summary_plots:

    _logger.info(f"Started creating summary plots..")

    plotter = LinePlots(cfg)
    
    for metric in metrics:
        #get total list of streams
        #TODO: improve this
        streams_set = set.union(*[set(model["streams"].keys()) for model in models.values()])
        
        #get total list of variables
        #TODO: improve this
        variables_set = set.union(*[
                        set(stream["variables"])
                        for model in models.values()
                        for stream in model["streams"].values()])

        #TODO: move this into plot_utils
        for stream in streams_set: #loop over streams
            for var in variables_set: #loop over variables
                selected_data = []
                labels = []
                for model_id, data in scores_dict[metric][stream].items():
                    
                    #fill list of plots with one xarray per model_id, if it exists. 
                    if var not in set(data.channel.values):
                        continue
                    selected_data.append(data.sel(channel=var))
                    labels.append(select(models[model_id], "label") or model_id)

                #if there is data for this stream and variable, plot it
                if selected_data:
                    _logger.info(f"Creating plot for {metric} - {stream} - {var}.")
                    name = "_".join([metric, stream, var])
                    plotter.plot(selected_data, labels, tag = name, x="forecast_step", y = metric)

