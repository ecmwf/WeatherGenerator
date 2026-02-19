# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from pathlib import Path

import numpy as np
import torch
import xarray as xr

import weathergen.common.config as config
import weathergen.common.io as io
from weathergen.common.io import TimeRange, zarrio_writer
from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.evaluate.plotting.plotter import Plotter

_logger = logging.getLogger(__name__)


def _normalize_channel_name(name: str) -> str:
    return str(name).lower().replace("_", "").replace(" ", "")


def _resolve_channel_names(stream_info, raw_channels):
    if not raw_channels:
        return raw_channels
    if isinstance(raw_channels[0], str):
        return list(raw_channels)

    channel_names = None
    if hasattr(stream_info, "val_target_channels") and stream_info.val_target_channels:
        if isinstance(stream_info.val_target_channels[0], str):
            channel_names = list(stream_info.val_target_channels)

    if channel_names is None:
        target_weights = getattr(stream_info, "target_channel_weights", None)
        if isinstance(target_weights, dict):
            channel_names = list(target_weights.keys())

    if channel_names is None:
        channel_weights = getattr(stream_info, "channel_weights", None)
        if isinstance(channel_weights, dict):
            channel_names = list(channel_weights.keys())

    if channel_names is None:
        return [f"ch{idx}" for idx in raw_channels]

    resolved = []
    for idx in raw_channels:
        if 0 <= int(idx) < len(channel_names):
            resolved.append(channel_names[int(idx)])
        else:
            resolved.append(f"ch{idx}")
    return resolved


def write_output(
    cf, val_cfg, batch_size, mini_epoch, batch_idx, dn_data, batch, model_output, target_aux_out
):
    """
    Interface for writing model output
    """

    # TODO: how to handle multiple physical loss terms
    outputs_physical = [
        loss_name
        for i, (loss_name, loss_term) in enumerate(val_cfg.losses.items())
        if loss_term.type == "LossPhysical"
    ]
    assert len(outputs_physical) == 1
    target_aux_out = target_aux_out[outputs_physical[0]]

    # collect all target / prediction-related information
    fp32 = torch.float32
    preds_all, targets_all, targets_coords_all, targets_times_all = [], [], [], []

    timestep_idxs = [0] if len(batch.get_output_idxs()) == 0 else batch.get_output_idxs()
    forecast_offset = timestep_idxs[0]
    targets_lens = []

    # TODO Maybe stopping at forecast_steps explained #1657
    for t_idx in timestep_idxs:
        preds_all += [[]]
        targets_all += [[]]
        targets_coords_all += [[]]
        targets_times_all += [[]]
        targets_lens += [[]]
        for stream_info in cf.streams:
            sname = stream_info["name"]
            # predictions
            preds = model_output.get_physical_prediction(t_idx, sname)
            targets = target_aux_out.physical[t_idx][sname]["target"]

            preds_s, targets_s, t_coords_s, t_times_s = [], [], [], []
            targets_lens[-1] += [[]]

            # handle forcing streams or if sample is empty
            if preds is None:
                # preds are empty so create copy of target and add ensemble dimension
                assert targets[0].shape[0] == 0, "Empty preds but non-empty targets."
                preds = [targets[0].clone().unsqueeze(0)]

            for i_batch, (pred, target) in enumerate(zip(preds, targets, strict=True)):
                # denormalize data if requested and map to storage format
                preds_s += [dn_data(sname, pred).detach().to(fp32).cpu().numpy()]
                targets_s += [dn_data(sname, target).detach().to(fp32).cpu().numpy()]

                # extract original target coords and times from target data
                target_data = target_aux_out.physical[t_idx][sname]
                t_coords_s += [target_data["target_coords"][i_batch].cpu().numpy()]
                t_times_s += [target_data["target_times"][i_batch].astype("datetime64[ns]")]

            targets_lens[-1][-1] += [t.shape[0] for t in targets_s]

            preds_all[-1] += [np.concatenate(preds_s, axis=1)]
            targets_all[-1] += [np.concatenate(targets_s)]
            targets_coords_all[-1] += [np.concatenate(t_coords_s)]
            targets_times_all[-1] += [np.concatenate(t_times_s)]

    #         # TODO: re-enable
    #           if len(idxs_inv) > 0:
    #               pred = pred[:, idxs_inv]
    #               target = target[idxs_inv]
    #               targets_coords_raw[t_idx][i_strm] = targets_coords_raw[t_idx][i_strm][idxs_inv]
    #               targets_times_raw[t_idx][i_strm] = targets_times_raw[t_idx][i_strm][idxs_inv]

    if len(preds_all) == 0:
        _logger.warning("Writing no data since predictions are empty.")
        return

    # collect source information
    sources = []
    for sample in batch.get_source_samples().get_samples():
        sources += [[]]
        for _, stream_data in sample.streams_data.items():
            # TODO: support multiple input steps
            sources[-1] += [stream_data.source_raw[0]]

    sample_idxs = [
        list(sample.streams_data.values())[0].sample_idx
        for sample in batch.get_source_samples().get_samples()
    ]

    # more prep work

    # output stream names to be written, use specified ones or all if nothing specified
    stream_names = [stream.name for stream in cf.streams]
    if val_cfg.get("output").get("streams") is not None:
        output_stream_names = val_cfg.output.streams
    else:
        output_stream_names = stream_names

    output_streams = {name: stream_names.index(name) for name in output_stream_names}
    _logger.debug(f"Using output streams: {output_streams} from streams: {stream_names}")

    target_channels: list[list[str]] = [list(stream.val_target_channels) for stream in cf.streams]
    source_channels: list[list[str]] = [list(stream.val_source_channels) for stream in cf.streams]

    geoinfo_channels = [[] for _ in cf.streams]  # TODO obtain channels

    # calculate global sample indices for this batch by offsetting by sample_start
    sample_start = batch_idx * batch_size

    # write output

    start_date = val_cfg.start_date
    end_date = val_cfg.end_date

    twh = TimeWindowHandler(
        start_date,
        end_date,
        val_cfg.time_window_len,
        val_cfg.time_window_step,
    )
    source_windows = (twh.window(idx) for idx in sample_idxs)
    source_intervals = [TimeRange(window.start, window.end) for window in source_windows]

    data = io.OutputBatchData(
        sources,
        source_intervals,
        targets_all,
        preds_all,
        targets_coords_all,
        targets_times_all,
        targets_lens,
        output_streams,
        target_channels,
        source_channels,
        geoinfo_channels,
        sample_start,
        forecast_offset,
    )
    with zarrio_writer(config.get_path_results(cf, mini_epoch)) as zio:
        for subset in data.items():
            zio.write_zarr(subset)

    # Prepare prediction data for Plotter (scatter plot expects lat/lon coords on ipoint).
    base_plot_dir = config.get_path_run(cf) / "plots" / "validation"
    base_plot_dir.mkdir(parents=True, exist_ok=True)
    plotter = Plotter({"image_format": "png", "dpi_val": 150}, base_plot_dir)
    headline_channels = {"2t", "z500", "q850", "10u", "10v"}

    t_idx = 0
    for stream_idx, stream_info in enumerate(cf.streams):
        stream_name = stream_info["name"]
        preds_stream = preds_all[t_idx][stream_idx]
        coords_stream = targets_coords_all[t_idx][stream_idx]

        if preds_stream.size == 0 or coords_stream.size == 0:
            _logger.warning(f"No prediction data to plot for stream {stream_name}.")
            continue

        # Expected shape is (ens, ipoint, channel). Select first ensemble if present.
        if preds_stream.ndim == 3:
            preds_stream = preds_stream[0]
        elif preds_stream.ndim != 2:
            _logger.warning(
                f"Unsupported prediction shape {preds_stream.shape} for stream {stream_name}."
            )
            continue

        lat = coords_stream[:, 0]
        lon = coords_stream[:, 1]
        channels = _resolve_channel_names(stream_info, target_channels[stream_idx])

        da = xr.DataArray(
            preds_stream,
            dims=("ipoint", "channel"),
            coords={
                "ipoint": np.arange(preds_stream.shape[0]),
                "channel": channels,
                "lat": ("ipoint", lat),
                "lon": ("ipoint", lon),
            },
        )

        plotter.stream = stream_name
        plotter.run_id = config.get_run_id_from_config(cf)
        plotter.fstep = forecast_offset

        selected_channels = [
            ch for ch in channels if _normalize_channel_name(ch) in headline_channels
        ]
        if not selected_channels:
            _logger.warning(
                f"No headline channels available for plotting stream {stream_name}."
            )
            continue

        for varname in selected_channels:
            data = da.sel(channel=varname).dropna(dim="ipoint")
            channel_dir = base_plot_dir / varname
            channel_dir.mkdir(parents=True, exist_ok=True)
            epoch_tag = f"epoch_{mini_epoch:03d}"
            plot_name = plotter.scatter_plot(
                data,
                channel_dir,
                varname=varname,
                regionname="global",
                tag=epoch_tag,
                title=f"{stream_name} - {varname} (fstep {forecast_offset})",
            )
            src = channel_dir / f"{plot_name}.{plotter.image_format}"
            dst = channel_dir / f"{epoch_tag}.{plotter.image_format}"
            if src != dst and src.exists():
                src.replace(dst)