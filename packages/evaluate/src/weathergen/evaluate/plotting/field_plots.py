# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Plots of the prediction and target fields: maps, histograms and timeseries.

They are drawn from the loaded data of one (run, stream) (``ReaderOutput``), so they can be
drawn per chunk of samples and forecast steps.  The drawing itself is done by ``Plotter``
and ``Timeseries``; the videos of the maps and histograms by ``animations``, once all the
frames are written.
"""

import logging
from pathlib import Path

import matplotlib
import numpy as np
import omegaconf as oc
import xarray as xr
from joblib import delayed

from weathergen.evaluate.io.data.io_orchestration import dispatch_parallel, get_num_workers
from weathergen.evaluate.io.io_reader import Reader, ReaderOutput
from weathergen.evaluate.plotting.plot_utils import apply_font_settings, ens_tag, plotter_config
from weathergen.evaluate.plotting.plotter import Plotter
from weathergen.evaluate.plotting.timeseries import Timeseries
from weathergen.evaluate.utils.array_utils import bias_ranges, common_ranges
from weathergen.evaluate.utils.config_compat import parse_data_plots

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timeseries plots
# ---------------------------------------------------------------------------


def _dispatch_timeseries_plots(
    da_preds: dict,
    da_tars: dict,
    output_dir: str,
    stream: str,
    regions: list[str],
    run_id: str,
    samples: dict,
    channels: dict,
    ensemble: list,
    n_workers: int,
) -> None:
    """Build and dispatch timeseries plot tasks for all (channel, sample[, ens]) triples."""
    data_ts = Timeseries(da_preds, da_tars)
    has_ens = any("ens" in v.dims for v in da_preds.values())
    ens_members = ensemble if has_ens else [None]
    ts_tasks = [
        {
            "output_dir": output_dir,
            "channel": str(channel),
            "sample": sample,
            "stream": stream,
            "region": region,
            "ens": ens,
        }
        for channel in channels
        for sample in samples
        for ens in ens_members
        for region in regions
    ]
    calls = [delayed(data_ts.plot_single_timeseries)(**t) for t in ts_tasks]
    dispatch_parallel(
        calls,
        n_workers=n_workers,
        backend="loky",
        desc=f"Timeseries {run_id} - {stream}",
    )


# ---------------------------------------------------------------------------
# Per-sample map / histogram plots
# ---------------------------------------------------------------------------
def _select_ensemble_data(da: xr.DataArray, ens):
    """Return ensemble member, mean or std view of a DataArray."""
    if "ens" not in da.dims:
        return da

    if ens == "mean":
        return da.mean(dim="ens")

    if ens == "std":
        return da.std(dim="ens")

    return da.sel(ens=ens)


def _plot_single_sample(
    plotter_cfg: dict,
    output_basedir: str,
    tars: xr.DataArray,
    preds: xr.DataArray,
    bias_data: xr.DataArray | None,
    sample: int | str,
    fstep: int | str,
    stream: str,
    plot_chs: list[str],
    ensemble: list,
    plot_maps: bool,
    plot_bias: bool,
    plot_target: bool,
    plot_histograms: bool | str,
    maps_config: dict,
    bias_config: dict,
    std_config: dict,
) -> None:
    """Plot all maps/histograms for a single (fstep, sample) pair (loky worker)."""
    matplotlib.use("Agg")

    maps_cfg = oc.OmegaConf.create(maps_config)
    bias_cfg = oc.OmegaConf.create(bias_config)
    std_cfg = oc.OmegaConf.create(std_config)
    plotter = Plotter(plotter_cfg, Path(output_basedir))

    data_selection = {"sample": sample, "stream": stream, "forecast_step": fstep}

    # Prediction, target and bias maps are enabled independently.
    if plot_target:
        plotter.create_maps_per_sample(tars, plot_chs, data_selection, "targets", maps_cfg)

    # Plot bias once if it doesn't carry an ensemble dimension,
    # otherwise it will be sliced per member inside the loop below.
    bias_has_ens = bias_data is not None and "ens" in bias_data.dims
    if plot_bias and bias_data is not None and not bias_has_ens:
        plotter.create_maps_per_sample(bias_data, plot_chs, data_selection, "bias", bias_cfg)

    has_ens = "ens" in preds.dims
    for ens in ensemble:
        preds_ens = _select_ensemble_data(preds, ens)
        preds_name = ens_tag("preds", ens, has_ens)

        if plot_maps:
            cfg_to_use = std_cfg if ens == "std" else maps_cfg
            plotter.create_maps_per_sample(
                preds_ens, plot_chs, data_selection, preds_name, cfg_to_use
            )

        if plot_bias and bias_has_ens:
            bias_ens = _select_ensemble_data(bias_data, ens)
            plotter.create_maps_per_sample(
                bias_ens, plot_chs, data_selection, ens_tag("bias", ens, has_ens), bias_cfg
            )

        if plot_histograms is True or plot_histograms == "per-sample":
            plotter.create_histograms(
                tars,
                preds_ens,
                plot_chs,
                data_selection,
                preds_name,
                ranges=maps_config,
            )

    plotter.clean_data_selection()


def _plot_histograms_across_samples(
    plotter_cfg: dict,
    output_basedir: str,
    tars: xr.DataArray,
    preds: xr.DataArray,
    bias_data: xr.DataArray | None,
    fstep: int | str,
    stream: str,
    plot_chs: list[str],
    ensemble: list,
    plot_histograms: bool | str,
    maps_config: dict,
    bias_config: dict,
) -> None:
    """Plot histograms across all samples for a single fstep.

    Unlike per-sample histograms, these aggregate all samples together.
    The output filename uses 'global' instead of a sample id and omits the timestep.
    """
    if not (plot_histograms is True or plot_histograms == "across-samples"):
        return

    matplotlib.use("Agg")
    plotter = Plotter(plotter_cfg, Path(output_basedir))

    data_selection = {"sample": "all_samples", "stream": stream, "forecast_step": fstep}

    has_ens = "ens" in preds.dims
    for ens in ensemble:
        preds_ens = _select_ensemble_data(preds, ens)
        plotter.create_histograms(
            tars,
            preds_ens,
            plot_chs,
            data_selection,
            ens_tag("preds", ens, has_ens),
            ranges=maps_config,
        )

    plotter.clean_data_selection()


def _filter_fsteps(
    reader: Reader, da_tars: dict, da_preds: dict, fsteps: list | None
) -> tuple[dict, dict]:
    """Keep the forecast steps requested for plotting.

    When the output is sub-step expanded (more forecast-step keys than the zarr has steps)
    and all zarr steps are requested, all output keys are kept.
    """
    if fsteps is None:
        return da_tars, da_preds

    plot_fstep_set = set(fsteps)
    output_fstep_keys = set(da_tars.keys())
    if output_fstep_keys - plot_fstep_set:
        zarr_fsteps = set(int(f) for f in reader.get_forecast_steps())
        if plot_fstep_set == zarr_fsteps:
            _logger.debug(
                f"Sub-step expansion detected: output has {len(output_fstep_keys)} "
                f"entries vs {len(zarr_fsteps)} zarr fsteps. "
                f"Expanding plotting filter to all output fsteps."
            )
            plot_fstep_set = output_fstep_keys

    return (
        {fs: da for fs, da in da_tars.items() if fs in plot_fstep_set},
        {fs: da for fs, da in da_preds.items() if fs in plot_fstep_set},
    )


def _colour_ranges(
    da_tars: dict, da_preds: dict, channels: list[str], stream_opts: oc.DictConfig
) -> tuple[dict, dict, dict]:
    """Colour ranges of the maps, the bias maps and the ensemble-std maps, per channel.

    Computed over all forecast steps of the data given, unless *stream_opts* (the stream's
    entry of the global plotting options) fixes them.
    """
    range_args = (da_tars, da_preds, channels, stream_opts)
    maps_config_dict = oc.OmegaConf.to_container(common_ranges(*range_args), resolve=True)
    bias_config_dict = oc.OmegaConf.to_container(bias_ranges(*range_args), resolve=True)

    has_ens = any("ens" in da.dims for da in da_preds.values())
    if not has_ens:
        return maps_config_dict, bias_config_dict, maps_config_dict

    std_preds = {fs: da.std(dim="ens") for fs, da in da_preds.items()}
    std_config_dict = oc.OmegaConf.to_container(
        common_ranges(std_preds, std_preds, channels, stream_opts), resolve=True
    )
    for cfg in std_config_dict.values():
        if isinstance(cfg, dict):
            cfg["colormap"] = "YlOrRd"
    return maps_config_dict, bias_config_dict, std_config_dict


def plot_data(
    reader: Reader,
    stream: str,
    global_plotting_opts: dict,
    output_data: ReaderOutput | None = None,
) -> None:
    """Plot prediction/target maps, histograms and timeseries for a given run and stream.

    The frames of the videos are drawn here; the videos themselves are built from them by
    ``animations.build_animations`` once all the frames of the run exist.

    Parameters
    ----------
    reader : Reader
        Reader object containing all info about the run.
    stream : str
        Stream name to plot data for.
    global_plotting_opts : dict
        Global plotting options (applies to all run_ids).
    output_data : ReaderOutput | None
        Pre-loaded data; when provided ``reader.get_data()`` is skipped.
    """
    run_id = reader.run_id
    stream_cfg = reader.get_stream(stream)
    plot_settings = stream_cfg.get("plotting", {})
    spec = parse_data_plots(plot_settings)
    if not spec:
        return

    plotter_cfg = plotter_config(
        global_plotting_opts,
        regions=global_plotting_opts.get("regions", stream_cfg.get("regions", ["global"])),
        log_x=global_plotting_opts.get("log_x", False),
        log_y=global_plotting_opts.get("log_y", False),
        n_bins=global_plotting_opts.get("n_bins", 50),
        plot_subtimesteps=reader.get_inference_stream_attr(stream, "tokenize_spacetime", False)
        | plot_settings.get("plot_subtimesteps", False),
    )
    apply_font_settings(plotter_cfg)

    available_data = reader.check_availability(stream, mode="plotting")
    if not available_data.score_availability:
        _logger.warning(f"RUN {reader.run_id} - {stream}: No plotting config. Skipping plots.")
        return

    # Per-step images are produced for every requested plot (videos are built from them).
    plot_maps = "predictions" in spec.maps
    plot_target = "target" in spec.maps
    plot_bias = "bias" in spec.maps
    plot_timeseries = spec.timeseries
    hist_per_sample = "per_sample" in spec.histograms
    hist_across = "across_samples" in spec.histograms
    if hist_per_sample and hist_across:
        plot_histograms: bool | str = True
    elif hist_per_sample:
        plot_histograms = "per-sample"
    elif hist_across:
        plot_histograms = "across-samples"
    else:
        plot_histograms = False

    model_output = output_data
    if output_data is None:
        empty = [
            name
            for name, val in (
                ("forecast_step", available_data.fsteps),
                ("sample", available_data.samples),
                ("channel", available_data.channels),
            )
            if val is not None and len(val) == 0
        ]
        if empty:
            _logger.warning(
                f"RUN {run_id} - {stream}: none of the requested {', '.join(empty)}(s) "
                f"for plotting exist in the output (available forecast steps: "
                f"{sorted(int(f) for f in reader.get_forecast_steps())[:5]}...). "
                f"Skipping plots."
            )
            return
        model_output = reader.get_data(
            stream,
            samples=available_data.samples,
            fsteps=available_data.fsteps,
            channels=available_data.channels,
            ensemble=available_data.ensemble,
        )

    da_tars = model_output.target
    da_preds = model_output.prediction

    if not da_tars:
        _logger.info(f"Skipping Plot Data for {stream}. Targets are empty.")
        return

    plot_sample_set = set(available_data.samples) if available_data.samples is not None else None
    plot_channel_set = set(available_data.channels) if available_data.channels is not None else None

    output_dir = str(reader.runplot_dir)

    da_tars, da_preds = _filter_fsteps(reader, da_tars, da_preds, available_data.fsteps)
    if not da_tars:
        _logger.info(f"Skipping Plot Data for {stream}. No matching fsteps after filtering.")
        return

    if not isinstance(global_plotting_opts.get(stream), oc.DictConfig):
        global_plotting_opts[stream] = oc.DictConfig({})
    maps_config_dict, bias_config_dict, std_config_dict = _colour_ranges(
        da_tars, da_preds, available_data.channels, global_plotting_opts[stream]
    )

    num_plot_workers = get_num_workers(
        check_process_headroom=True,
        max_workers=reader.eval_cfg.get("max_workers", None),
    )

    tasks: list[dict] = []
    all_samples_tasks: list[dict] = []
    for (fstep, tars), (_, preds) in zip(da_tars.items(), da_preds.items(), strict=False):
        all_chs = list(np.atleast_1d(tars.channel.values))
        plot_chs = (
            [ch for ch in all_chs if ch in plot_channel_set]
            if plot_channel_set is not None
            else all_chs
        )
        if not plot_chs:
            continue

        all_samples = list(np.unique(tars.sample.values))
        plot_samples = (
            [s for s in all_samples if s in plot_sample_set]
            if plot_sample_set is not None
            else all_samples
        )
        if not plot_samples:
            continue

        bias_data = (preds - tars) if plot_bias else None

        all_samples_tasks.append(
            {
                "plotter_cfg": plotter_cfg,
                "output_basedir": output_dir,
                "tars": tars,
                "preds": preds,
                "bias_data": bias_data,
                "fstep": fstep,
                "stream": stream,
                "plot_chs": plot_chs,
                "ensemble": list(available_data.ensemble),
                "plot_histograms": plot_histograms,
                "maps_config": maps_config_dict,
                "bias_config": bias_config_dict,
            }
        )

        for sample in plot_samples:
            tasks.append(
                {
                    "plotter_cfg": plotter_cfg,
                    "output_basedir": output_dir,
                    "tars": tars,
                    "preds": preds,
                    "bias_data": bias_data,
                    "sample": sample,
                    "fstep": fstep,
                    "stream": stream,
                    "plot_chs": plot_chs,
                    "ensemble": list(available_data.ensemble),
                    "plot_maps": plot_maps,
                    "plot_bias": plot_bias,
                    "plot_target": plot_target,
                    "plot_histograms": plot_histograms,
                    "maps_config": maps_config_dict,
                    "bias_config": bias_config_dict,
                    "std_config": std_config_dict,
                }
            )

    _logger.info(
        f"Parallel plotting: dispatching {len(tasks)} (fstep, sample) tasks "
        f"across up to {num_plot_workers} loky workers."
    )
    calls = [delayed(_plot_single_sample)(**task) for task in tasks]
    dispatch_parallel(
        calls, n_workers=num_plot_workers, backend="loky", desc=f"Plotting {run_id} - {stream}"
    )

    if all_samples_tasks:
        _logger.info(
            f"Parallel plotting: dispatching {len(all_samples_tasks)} across-samples "
            f"tasks using up to {num_plot_workers} loky workers."
        )
        as_calls = [delayed(_plot_histograms_across_samples)(**t) for t in all_samples_tasks]
        dispatch_parallel(
            as_calls,
            n_workers=num_plot_workers,
            backend="loky",
            desc=f"Across-samples plots {run_id} - {stream}",
        )

    if plot_timeseries:
        _dispatch_timeseries_plots(
            da_preds=da_preds,
            da_tars=da_tars,
            output_dir=output_dir,
            stream=stream,
            regions=plotter_cfg["regions"],
            run_id=run_id,
            samples=plot_sample_set,
            channels=plot_channel_set,
            ensemble=list(available_data.ensemble),
            n_workers=num_plot_workers,
        )
