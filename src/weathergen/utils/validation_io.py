# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import numpy as np
import torch

import weathergen.common.config as config
import weathergen.common.io as io
from weathergen.common.io import TimeRange, zarrio_writer
from weathergen.datasets.data_reader_base import TimeWindowHandler

_logger = logging.getLogger(__name__)


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

    window_offset_prediction = val_cfg.get("window_offset_prediction", 0)
    forecast_steps = max(1, val_cfg.get("forecast", {}).get("num_steps", 1))
    targets_lens = []
    # for fstep in range(window_offset_prediction, forecast_steps + 1):
    for fstep in range(window_offset_prediction, forecast_steps):
        preds_all += [[]]
        targets_all += [[]]
        targets_coords_all += [[]]
        targets_times_all += [[]]
        targets_lens += [[]]
        for stream_info in cf.streams:
            sname = stream_info["name"]
            # predictions
            preds = model_output.get_physical_prediction(fstep, stream_info["name"])
            targets = target_aux_out.physical[stream_info["name"]][fstep]["target"]

            preds_s, targets_s, t_coords_s, t_times_s = [], [], [], []
            targets_lens[-1] += [[]]

            for i_batch, (pred, target) in enumerate(zip(preds, targets, strict=True)):
                pred, target = pred.to(fp32), target.to(fp32)

                if not (target.shape[0] > 0 and pred.shape[0] > 0):
                    continue

                # extract data/coords and remove token dimension if it exists
                pred = pred.reshape([pred.shape[0], *target.shape])
                assert pred.shape[1] > 0

                preds_s += [dn_data(sname, pred).detach().cpu().numpy()]
                targets_s += [dn_data(sname, target).detach().cpu().numpy()]

                key = "target_coords"
                t_coords_s += [target_aux_out.physical[sname][fstep][key][i_batch].cpu().numpy()]
                key = "target_times"
                t_times_s += [target_aux_out.physical[sname][fstep][key][i_batch]]

            targets_lens[-1][-1] += [t.shape[0] for t in targets_s]

            preds_all[-1] += [np.concatenate(preds_s, axis=1)]
            targets_all[-1] += [np.concatenate(targets_s)]
            targets_coords_all[-1] += [np.concatenate(t_coords_s)]
            targets_times_all[-1] += [np.concatenate(t_times_s)]

    #         # TODO: re-enable
    #           if len(idxs_inv) > 0:
    #               pred = pred[:, idxs_inv]
    #               target = target[idxs_inv]
    #               targets_coords_raw[fstep][i_strm] = targets_coords_raw[fstep][i_strm][idxs_inv]
    #               targets_times_raw[fstep][i_strm] = targets_times_raw[fstep][i_strm][idxs_inv]

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
        [sdata.sample_idx for _, sdata in sample.streams_data.items()]
        for sample in batch.get_source_samples().get_samples()
    ]
    sample_idxs = [s[0].item() for s in sample_idxs]

    # more prep work

    # output stream names to be written, use specified ones or all if nothing specified
    stream_names = [stream.name for stream in cf.streams]
    if val_cfg.get("output").get("streams") is not None:
        output_stream_names = val_cfg.streams_output
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
        val_cfg.get("window_offset_prediction", 0),
    )
    with zarrio_writer(config.get_path_output(cf, mini_epoch)) as zio:
        for subset in data.items():
            zio.write_zarr(subset)
