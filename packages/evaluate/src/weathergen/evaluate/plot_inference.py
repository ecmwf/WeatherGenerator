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

from collections import defaultdict
from utils import get_data, plot_data, plot_summary

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Fast evaluation of weather generator runs."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the configuration yaml file for plotting. e.g. config/plottig_config.yaml",
    )

    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    models = cfg.model_ids

    _logger.info(f"Detected {len(models)} models")
    
    assert cfg.get("output_scores_dir"), "Please provide a path to the directory where the score files are stored or will be saved."
    out_scores_dir = Path(cfg.output_scores_dir)
    out_scores_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(cfg.model_dir)
    metrics = cfg.evaluation.metrics
    
    # to get a structure like: scores_dict[metric][stream][model_id] = plot
    scores_dict = defaultdict(           
                lambda: defaultdict(
                    dict
            )
        )

    for model_id, model  in models.items():

        plotter = Plotter(cfg, model_id)
        _logger.info(f"MODEL {model_id}: Getting data...")
        
        streams = model["streams"].keys()

        for stream in streams: 
            _logger.info(f"MODEL {model_id}: Processing stream {stream}...")
            
            stream_dict = model["streams"][stream]

            _logger.info(f"MODEL {model_id}: Plotting stream {stream}...")
            plots = plot_data(cfg, model_id, stream, stream_dict)
            
            _logger.info(f"MODEL {model_id} - {stream}: Computing scores...")
           
            if stream_dict.evaluation:
                da_tars, da_preds = get_data(cfg, model_id, stream)
                _logger.info(f"MODEL {model_id} - {stream}: Data retrieved successfully.")
                
                score_data = VerifiedData(da_preds, da_tars) 
                _logger.info(f"MODEL {model_id} - {stream}: Score data engine created successfully.")
                
                for metric in metrics:
                    _logger.info(f"Computing {metric} for {model_id} - {stream}")
                    try:
                        #TODO: this needs more thinking as only part of the variables might be already pre-computed. 
                        #with a loop over variables we would loose the advantage of parallel computing. 
                        #possibility: load already computed scores before initialising the data and use e.g. "new_variable" index for the letfovers.
                        #problem: different metrics can be pre-computed for different variables, so not easy to define a "new_variables" list.   
                        scores_dict[metric][stream][model_id] = retrieve_score(out_scores_dir, model_id, stream, metric, model.epoch, model.rank)
                    except Exception as e:
                        scores_dict[metric][stream][model_id] = get_score(score_data, metric, agg_dims=list(cfg.evaluation.avg_dims)) 
                        #save scores to file
                        save_path = out_scores_dir.joinpath(f"{metric}_{model_id}_{stream}_epoch{model.epoch:05d}_rank{model.rank:04d}.json")
                        _logger.info(f"Saving results to {save_path}")
                        with open(save_path, "w") as f:
                            json.dump(scores_dict[metric][stream][model_id].compute().to_dict(), f, indent=4)


#plot summary
if cfg.summary_plots:

    _logger.info(f"Started creating summary plots..")
    plot_summary(cfg, scores_dict)
    
