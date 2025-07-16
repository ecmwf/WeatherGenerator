# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any
from weathergen.common.io import ZarrIO
import logging
from pathlib import Path
from tqdm import tqdm
import dask.array as da
import xarray as xr
import numpy as np
from plotter import Plotter, LinePlots

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

def get_data(cfg: dict, model_id: str, stream: str, samples: list = None, 
                    fsteps: list = None, variables: list = None) -> tuple:
    """
    Retrieve prediction and target data for a given model from the Zarr store.
    :param cfg: Configuration dictionary containing all information.
    :param model_id: Model identifier.
    :param stream: Stream name to retrieve data for.
    :param samples: List of sample indices to retrieve. If None, all samples are retrieved.
    :param fsteps: List of forecast steps to retrieve. If None, all forecast steps are retrieved.
    :param variables: List of variable names to retrieve. If None, all variables are retrieved.
    :return Tuple of xarray DataArrays for targets and predictions.
    """

    model = cfg.model_ids[model_id]
    model_dir = Path(cfg.get("model_dir"))

    fname_zarr = model_dir.joinpath(f"{model_id}/validation_epoch{model["epoch"]:05d}_rank{model["rank"]:04d}.zarr")
        
    assert (fname_zarr.exists() and fname_zarr.is_dir()), f"Check Zarr input: {fname_zarr}"

    with ZarrIO(fname_zarr) as zio:
        zio_streams, zio_forecast_steps = zio.streams, zio.forecast_steps
        stream_dict = model.streams[stream]
        _logger.info(f"MODEL {model_id}: Processing stream {stream}...")

        fsteps    = stream_dict.get("forecast_step", zio_forecast_steps) if fsteps is None else fsteps 
        samples   = [int(sample) for sample in zio.samples] if samples is None else samples
        variables = stream_dict.get("variables") if variables is None else variables

        da_tars, da_preds = [], []
        
        for fstep in fsteps: 
            _logger.info(f"MODEL {model_id} - {stream}: Processing fstep {fstep}...")
            da_tars_fs, da_preds_fs = [], []

            for sample in tqdm(samples, desc=f"Processing {model_id} - {stream} - {fstep}"):

                out = zio.get_data(sample, stream, fstep)

                tars = out.target.as_xarray()
                preds = out.prediction.as_xarray()
                da_tars_fs.append(tars)
                da_preds_fs.append(preds)
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

        return da_tars, da_preds

def plot_data(cfg: str, model_id: str, stream: str, stream_dict: dict):
    """
    Plot the data for a given model and stream.
    
    :param da_tars: Target data as an xarray DataArray.
    :param da_preds: Prediction data as an xarray DataArray.
    :param model_id: Model identifier.
    :param stream: Stream name.
    :param stream_dict: Dictionary containing stream configuration.
    """

    model = cfg.model_ids[model_id]
    model_dir = Path(cfg.get("model_dir"))

    plot_settings = stream_dict.get("plotting", {})

    if plot_settings is None:
        return
    
    plotter = Plotter(cfg, model_id)

    plot_samples  = plot_settings.get("sample", None)
    plot_fsteps   = stream_dict.get("forecast_step" , None)
    plot_vars     = stream_dict.get("variables", None)

    da_tars, da_preds = get_data(cfg, model_id, stream, plot_samples, plot_fsteps, plot_vars)

    assert np.array_equal(da_tars.sample.values, da_preds.sample.values), f"Samples in targets and predictions do not match for model {model_id} and stream"
    assert np.array_equal(da_tars.forecast_step.values, da_preds.forecast_step.values), f"Forecast steps in targets and predictions do not match for model {model_id} and stream {stream}"
    assert np.array_equal(da_tars.channel.values, da_preds.channel.values), f"Channels in targets and predictions do not match for model {model_id} and stream {stream}"
    
    plot_names = []
    for fstep in da_tars.forecast_step.values:
        for sample in tqdm(da_tars.sample.values, desc=f"Plotting {model_id} - {stream} - fstep {fstep}"):
            plots = []

            select = {"sample": sample, 
                      "stream": stream, 
                      "forecast_step" : fstep}    

            if plot_settings.plot_maps:
                map_tar = plotter.map(da_tars , plot_vars, select, "target")

                map_pred = plotter.map(da_preds, plot_vars, select, "preds")
                plots.extend([map_tar, map_pred])

            if plot_settings.plot_histograms:
                h = plotter.histogram(da_tars, da_preds, plot_vars, select)
                plots.append(h)
            
            plotter = plotter.clean_selection()
           
            plot_names.append(plots)
    
    return plot_names

def retrieve_score(dir: str, model_id: str, stream: str, metric: str, epoch: int, rank: int):
    """
    Retrieve the score for a given model, stream, metric, epoch, and rank from a JSON file.
    :param dir: Directory where the score files are stored.
    :param model_id: Model identifier.
    :param stream: Stream name.
    :param metric: Metric name.
    :param epoch: Epoch number.
    :param rank: Rank number.
    :return: xarray DataArray containing the score.
    """
    
    #TODO adapt this to JSON output
    score_path = Path(f"{dir}/{metric}_{model_id}_{stream}_epoch{epoch:05d}_rank{rank:04d}.json")
    _logger.info(f"Looking for: {score_path}")
    if score_path.exists():
        with open(score_path, "r") as f:
            data_dict = json.load(f)
            return xr.DataArray.from_dict(data_dict)
    else:
        raise FileNotFoundError (f"File {score_path} not found in the archive.")

def plot_summary(cfg: dict, scores_dict: dict):
    """
    Plot summary of the evaluation results.
    This function is a placeholder for future implementation.
    :param cfg: Configuration dictionary containing all information.
    :param models: Dictionary containing model information.
    :param scores_dict: Dictionary containing scores for each model and stream.
    """
    _logger.info("Plotting summary of evaluation results...")
    
    models = cfg.model_ids
    metrics = cfg.evaluation.metrics
    
    plotter = LinePlots(cfg)
    
    for metric in metrics:
        #get total list of streams
        #TODO: improve this
        streams_set = list(sorted(set.union(*[set(model["streams"].keys()) 
                            for model in models.values()])))
        
        #get total list of variables
        #TODO: improve this
        variables_set = list(sorted(set.union(*[
                        set(stream["variables"])
                        for model in models.values()
                        for stream in model["streams"].values()])))

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
                    labels.append(models[model_id].get("label", model_id))

                #if there is data for this stream and variable, plot it
                if selected_data:
                    _logger.info(f"Creating plot for {metric} - {stream} - {var}.")
                    name = "_".join([metric, stream, var])
                    plotter.plot(selected_data, labels, tag = name, x_dim="forecast_step", y_dim = metric)

############# Utility functions ############

def to_list(obj: Any) -> list:
    """
    Convert given object to list if obj is not already a list. Sets are also transformed to a list.

    Parameters
    ----------
    obj : Any
        The object to transform into a list.
    Returns
    -------
    list
        A list containing the object, or the object itself if it was already a list.
    """
    if isinstance(obj, set | tuple):
        obj = list(obj)
    elif not isinstance(obj, list):
        obj = [obj]
    return obj
