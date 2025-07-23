# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any
import logging
from pathlib import Path
from tqdm import tqdm
import json
import xarray as xr
import numpy as np
from omegaconf.listconfig import ListConfig

from weathergen.common.io import ZarrIO
from score import VerifiedData, get_score
from score_utils import to_list
from plotter import Plotter, LinePlots

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

def get_data(
    cfg: dict, 
    model_id: str, 
    stream: str, 
    samples: list[int] = None,          
    fsteps: list[str]  = None, 
    channels: list[str] = None, 
    return_counts: bool = False) -> tuple:
    """
    Retrieve prediction and target data for a given model from the Zarr store.
    :param cfg: Configuration dictionary containing all information.
    :param model_id: Model identifier.
    :param stream: Stream name to retrieve data for.
    :param samples: List of sample indices to retrieve. If None, all samples are retrieved.
    :param fsteps: List of forecast steps to retrieve. If None, all forecast steps are retrieved.
    :param channels: List of channel names to retrieve. If None, all channels are retrieved.
    :param return_counts: If True, also return the number of points per sample.
    :return: Tuple of xarray DataArrays for targets and predictions, and optionally the points per sample.
    """

    model = cfg.model_ids[model_id]
    results_dir = Path(cfg.get("results_dir"))

    fname_zarr = results_dir.joinpath(f"{model_id}/validation_epoch{model["epoch"]:05d}_rank{model["rank"]:04d}.zarr")
        
    if not fname_zarr.exists() or not fname_zarr.is_dir():
        _logger.error(f"Zarr file {fname_zarr} does not exist or is not a directory.")
        raise FileNotFoundError(f"Zarr file {fname_zarr} does not exist or is not a directory.")

    with ZarrIO(fname_zarr) as zio:
        zio_forecast_steps = zio.forecast_steps
        stream_dict = model.streams[stream]
        all_channels = peek_tar_channels(zio, stream, zio_forecast_steps[0])
        _logger.info(f"MODEL {model_id}: Processing stream {stream}...")

        fsteps    = sorted(zio_forecast_steps if fsteps is None else fsteps )
        samples   = sorted([int(sample) for sample in zio.samples] if samples is None else samples)
        channels = channels if channels is not None else stream_dict.get("channels", all_channels) 
        channels = to_list(channels)

        da_tars, da_preds = [], []

        if return_counts:
                points_per_sample = xr.DataArray(
                    np.full((len(fsteps), len(samples)), np.nan),
                    coords={"forecast_step": fsteps, "sample": samples},
                    dims=("forecast_step", "sample"),
                    name=f"points_per_sample_{stream}",
                )
        
        for fstep in fsteps: 
            _logger.info(f"MODEL {model_id} - {stream}: Processing fstep {fstep}...")
            da_tars_fs, da_preds_fs = [], []
            pps = []

            for sample in tqdm(samples, desc=f"Processing {model_id} - {stream} - {fstep}"):

                out = zio.get_data(sample, stream, fstep)
                target, pred = out.target.as_xarray(), out.prediction.as_xarray()
    
                da_tars_fs.append(target.squeeze())
                da_preds_fs.append(pred.squeeze())
                pps.append(len(target.ipoint))

            _logger.debug(
                f"Concatenating targets and predictions for stream {stream}, forecast_step {fstep}..."
            )
            da_tars_fs  = xr.concat(da_tars_fs , dim="ipoint")
            da_preds_fs = xr.concat(da_preds_fs, dim="ipoint")

            if set(channels) != set(all_channels):
                _logger.debug(
                f"Restricting targets and predictions to channels {channels} for stream {stream}..."
                )
                da_tars_fs = da_tars_fs.sel(channel=channels)
                da_preds_fs = da_preds_fs.sel(channel=channels)

            da_tars.append(da_tars_fs)
            da_preds.append(da_preds_fs)
            if return_counts:
                points_per_sample.loc[{"forecast_step": fstep}] = np.array(pps)
        
        #Safer than a list
        da_tars  = {fstep: da for fstep, da in zip(fsteps, da_tars)}
        da_preds = {fstep: da for fstep, da in zip(fsteps, da_preds)}

        assert da_preds.keys() == da_tars.keys(), "Forcast steps in prediction and target do not match. Investigate."

        #sanity checks
        for fstep in da_tars.keys():
            assert np.array_equal(da_tars[fstep].sample.values, da_preds[fstep].sample.values), f"Samples in targets and predictions do not match for model {model_id} and stream"
            assert np.array_equal(da_tars[fstep].forecast_step.values, da_preds[fstep].forecast_step.values), f"Forecast steps in targets and predictions do not match for model {model_id} and stream {stream}"
            assert np.array_equal(da_tars[fstep].channel.values, da_preds[fstep].channel.values), f"Channels in targets and predictions do not match for model {model_id} and stream {stream}"


        if return_counts:
            return da_tars, da_preds, points_per_sample
        else:
            return da_tars, da_preds


def calc_scores_per_stream(
    cfg: dict, 
    model_id: str, 
    stream: str, 
    metrics: list[str]) -> tuple:
    """
    Calculate scores for a given model and stream using the specified metrics.
    :param cfg: Configuration dictionary containing all information for the evaluation.
    :param model_id: Model identifier.
    :param stream: Stream name to calculate scores for.
    :param metrics: List of metric names to calculate.
    :return: Tuple of xarray DataArray containing the scores and the number of points per sample.
    """
    _logger.info(f"MODEL {model_id} - {stream}: Calculating scores for metrics {metrics}...")

    samples     = cfg.evaluation.get("sample", None)
    fsteps      = cfg.evaluation.get("forecast_step", None)
    channels    = cfg.get("channels", None)
    
    if samples == "all":
        samples = None
    
    if fsteps == "all":
        fsteps = None

    da_preds, da_tars, points_per_sample = get_data(cfg, model_id, stream, return_counts=True)

    assert da_preds.keys() == da_tars.keys(), "Forcast steps in prediction and target do not match. Investigate."

    first_da = next(iter(da_tars.values()))

    fsteps   = [int(k) for k in da_tars.keys()]
    #TODO: improve the way we handle samples. This is hacky since we group by sample later. 
    samples  = list(np.atleast_1d(np.unique(first_da.sample.values)))
    channels = list(np.atleast_1d(first_da.channel.values))

    metric_stream = xr.DataArray(
        np.full(
            (len(samples), len(fsteps), len(channels), len(metrics)),
            np.nan,
        ),
        coords={
            "sample": samples,
            "forecast_step": fsteps,
            "channel": channels,
            "metric": metrics,
        },
    )

    for (fstep, tars), (_, preds) in zip(da_tars.items(), da_preds.items() ):

        _logger.debug(f"Verifying data for stream {stream}...")
        score_data = VerifiedData(preds, tars)

        # Build up computation graphs for all metrics
        _logger.debug(
            f"Build computation graphs for metrics for stream {stream}..."
        )

        combined_metrics = [
            get_score(score_data, metric, agg_dims="ipoint", group_by_coord="sample") for metric in metrics
        ]

        combined_metrics = xr.concat(combined_metrics, dim="metric")
        combined_metrics["metric"] = metrics

        _logger.debug(f"Running computation of metrics for stream {stream}...")
        combined_metrics = combined_metrics.compute()

        metric_stream.loc[{"forecast_step": int(fstep)}] = combined_metrics.compute()

        _logger.info(f"Scores for model {model_id} - {stream} calculated successfully.")

    return metric_stream, points_per_sample


def plot_data(
    cfg: str, 
    model_id: str, 
    stream: str, 
    stream_dict: dict):
    """
    Plot the data for a given model and stream.
    
    :param da_tars: Target data as an xarray DataArray.
    :param da_preds: Prediction data as an xarray DataArray.
    :param model_id: Model identifier.
    :param stream: Stream name.
    :param stream_dict: Dictionary containing stream configuration.
    """

    model = cfg.model_ids[model_id]
    results_dir = Path(cfg.get("results_dir"))

    plot_settings = stream_dict.get("plotting", {})

    if not (plot_settings and (plot_settings.plot_maps or plot_settings.plot_histograms)):
        return
    
    plotter = Plotter(cfg, model_id)

    plot_samples  = plot_settings.get("sample", None)
    plot_fsteps   = plot_settings.get("forecast_step", None)
    plot_chs      = stream_dict.get("channels", None)
    
    if plot_fsteps == "all":
        plot_fsteps = None
    
    if plot_samples == "all":
        plot_samples = None

    da_tars, da_preds = get_data(cfg, model_id, stream, plot_samples, plot_fsteps, plot_chs)  
    plot_fsteps = da_tars.keys()

    for (fstep, tars), (_, preds) in zip(da_tars.items(), da_preds.items()): 

        plot_chs = list(np.atleast_1d(tars.channel.values))
        plot_samples = list(np.unique(tars.sample.values))

        plot_names = []

        for sample in tqdm(plot_samples, desc=f"Plotting {model_id} - {stream} - fstep {fstep}"):
            plots = []

            data_selection = {"sample": sample, 
                              "stream": stream, 
                              "forecast_step" : fstep}    

            if plot_settings.plot_maps:
                map_tar = plotter.map(tars, plot_chs, data_selection, "target")

                map_pred = plotter.map(preds, plot_chs, data_selection, "preds")
                plots.extend([map_tar, map_pred])

            if plot_settings.plot_histograms:
                h = plotter.histogram(tars, preds, plot_chs, data_selection)
                plots.append(h)
            
            plotter = plotter.clean_data_selection()
           
            plot_names.append(plots)
    
    return plot_names


def metric_list_to_json(
    metrics_list: list[xr.DataArray],
    npoints_sample_list: list[xr.DataArray],
    streams: list[str],
    metric_dir: Path,
    model_id: str,
    epoch: int,
    rank: int = 0,  # Add rank so it matches filename expectations
):
    """
    Write the evaluation results collected in a list of xarray DataArrays for the metrics
    to stream- and metric-specific JSON files.

    Parameters
    ----------
    metrics_list : list[xr.DataArray]
        Metrics per stream.
    npoints_sample_list : list[xr.DataArray]
        Number of points per sample per stream.
    streams : list[str]
        Stream names.
    metric_dir : Path
        Output directory.
    model_id : str
        Identifier of the inference run.
    epoch : int
        Epoch number.
    rank : int
        Rank ID (default: 0), added to match retrieval expectations.
    """
    assert len(metrics_list) == len(npoints_sample_list) == len(streams), (
        "The lengths of metrics_list, npoints_sample_list, and streams must be the same."
    )

    metric_dir.mkdir(parents=True, exist_ok=True)

    for s_idx, stream in enumerate(streams):
        metrics_stream, npoints_sample_stream = (
            metrics_list[s_idx],
            npoints_sample_list[s_idx],
        )

        for metric in metrics_stream.coords["metric"].values:
            metric_now = metrics_stream.sel(metric=metric)
            # Save as individual DataArray, not Dataset
            metric_now.attrs["npoints_per_sample"] = npoints_sample_stream.values.tolist()
            metric_dict = metric_now.to_dict()

            # Match the expected filename pattern
            save_path = metric_dir / f"{model_id}_{stream}_{metric}_epoch{epoch:05d}.json"

            _logger.info(f"Saving results to {save_path}")
            with open(save_path, "w") as f:
                json.dump(metric_dict, f, indent=4)

    _logger.info(
        f"Saved all results of inference run {model_id} - epoch {epoch:d} successfully to {metric_dir}."
    )


def retrieve_metric_from_json(
    dir: str, 
    model_id: str, 
    stream: str, 
    metric: str, 
    epoch : int, 
    rank  : int = 0):
    """
    Retrieve the score for a given model, stream, metric, epoch, and rank from a JSON file.

    Parameters
    ----------
    dir : str
        Directory where JSON files are stored.
    model_id : str
        Model/run identifier.
    stream : str
        Stream name.
    metric : str
        Metric name.
    epoch : int
        Epoch number.
    rank : int
        Rank ID.

    Returns
    -------
    xr.DataArray
        The metric DataArray.
    """
    score_path = Path(dir) / f"{model_id}_{stream}_{metric}_epoch{epoch:05d}.json"
    _logger.debug(f"Looking for: {score_path}")
    if score_path.exists():
        with open(score_path, "r") as f:
            data_dict = json.load(f)
            return xr.DataArray.from_dict(data_dict)
    else:
        raise FileNotFoundError(f"File {score_path} not found in the archive.")


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
        
        #get total list of channels
        #TODO: improve this
        # TODO: support case where channels are not set in stream-config
        channels_set = list(sorted(set.union(*[
                        set(stream["channels"])
                        for model in models.values()
                        for stream in model["streams"].values()])))

        #TODO: move this into plot_utils
        for stream in streams_set: #loop over streams
            for ch in channels_set: #loop over channels
                selected_data = []
                labels = []
                for model_id, data in scores_dict[metric][stream].items():
                    
                    #fill list of plots with one xarray per model_id, if it exists. 
                    if ch not in set(np.atleast_1d(data.channel.values)):
                        continue
                    selected_data.append(data.sel(channel=ch))
                    labels.append(models[model_id].get("label", model_id))

                #if there is data for this stream and channel, plot it
                if selected_data:
                    _logger.info(f"Creating plot for {metric} - {stream} - {ch}.")
                    name = "_".join([metric, stream, ch])
                    plotter.plot(selected_data, labels, tag = name, x_dim="forecast_step", y_dim = metric)

############# Utility functions ############

def peek_tar_channels(zio: ZarrIO, stream: str, fstep: int = 0) -> list[str]:
    """
    Peek the channels of a target stream in a ZarrIO object.

    Parameters
    ----------
    zio : 
        The ZarrIO object containing the tar stream.
    stream : 
        The name of the tar stream to peek.
    fstep :  
        The forecast step to peek. Default is 0.
    Returns
    -------
    channels : 
        A list of channel names in the tar stream.
    """
    if not isinstance(zio, ZarrIO):
        raise TypeError("zio must be an instance of ZarrIO")

    dummy_out = zio.get_data(0, stream, fstep)
    channels = dummy_out.target.channels
    _logger.debug(f"Peeked channels for stream {stream}: {channels}")

    return channels

def scalar_coord_to_dim(da: xr.DataArray, name: str, axis: int = -1) -> xr.DataArray:
    """
    Convert a scalar coordinate to a dimension in an xarray DataArray.
    If the coordinate is already a dimension, it is returned unchanged.
    
    Parameters
    ----------
    da : xarray.DataArray
        The DataArray to modify.
    name : str
        The name of the coordinate to convert.
    axis : int, optional
        The axis along which to expand the dimension. Default is -1 (last axis).
    Returns
    -------
    xarray.DataArray
        The modified DataArray with the scalar coordinate converted to a dimension.
    """
    if name in da.dims:
        return da  # already a dimension
    if name in da.coords and da.coords[name].ndim == 0:
        val = da.coords[name].item()
        da = da.drop_chs(name)
        da = da.expand_dims({name: [val]}, axis=axis)
    return da
