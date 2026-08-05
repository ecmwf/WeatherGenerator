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
    preds_coords_all, preds_times_all = [], []

    timestep_idxs = [0] if len(batch.get_output_idxs()) == 0 else batch.get_output_idxs()
    forecast_offset = timestep_idxs[0]
    targets_lens = []
    preds_lens = []

    # TODO Maybe stopping at forecast_steps explained #1657
    for t_idx in timestep_idxs:
        preds_all += [[]]
        targets_all += [[]]
        targets_coords_all += [[]]
        targets_times_all += [[]]
        preds_coords_all += [[]]
        preds_times_all += [[]]
        targets_lens += [[]]
        preds_lens += [[]]
        for sname in cf.streams.keys():
            preds = model_output.get_physical_prediction(t_idx, sname)
            targets = target_aux_out.physical[t_idx][sname]["target"]

            preds_s, targets_s, t_coords_s, t_times_s = [], [], [], []
            preds_coords_s, preds_times_s = [], []

            is_spoof = target_aux_out.physical[t_idx][sname]["is_spoof"][0]
            if is_spoof:
                _logger.debug(
                    f"Stream '{sname}' at t_idx={t_idx} is spoof "
                    "(target time window is outside the dataset range); "
                    "writing model predictions with empty targets."
                )

            # handle forcing streams or if sample is empty
            if preds is None:
                # preds are empty so create copy of target and add ensemble dimension
                assert targets[0].shape[0] == 0, "Empty preds but non-empty targets."
                preds = [target.clone().unsqueeze(0) for target in targets]

            for i_batch, (pred, target) in enumerate(zip(preds, targets, strict=True)):
                target_data = target_aux_out.physical[t_idx][sname]
                t_coords = target_data["target_coords"][i_batch]
                t_times = target_data["target_times"][i_batch]

                idxs_inv = target_aux_out.physical[t_idx][sname]["idxs_inv"][i_batch]
                if idxs_inv is not None and (
                    not isinstance(idxs_inv, torch.Tensor) or idxs_inv.numel() > 0
                ):
                    pred = pred[:, idxs_inv]
                    t_coords = t_coords[idxs_inv]
                    t_times = t_times[idxs_inv]
                    if not is_spoof:
                        target = target[idxs_inv]

                # denormalize predictions
                preds_s += [dn_data(sname, pred.to(fp32)).detach().cpu().numpy()]
                preds_coords_s += [t_coords.cpu().numpy()]
                preds_times_s += [t_times.astype("datetime64[ns]")]

                if is_spoof:
                    # No real observations for this forecast step: write empty target so
                    # the zarr clearly signals missing ground truth, while predictions
                    # (computed by the model at the spoof coordinates) are written in full.
                    n_channels = pred.shape[-1]
                    targets_s += [np.zeros((0, n_channels), dtype=np.float32)]
                    t_coords_s += [np.zeros((0, 2), dtype=np.float32)]
                    t_times_s += [np.array([], dtype="datetime64[ns]")]
                else:
                    # In inference_only mode, target tokens are empty (no channel dim); create a
                    # properly shaped empty array so downstream code handles it gracefully.
                    n_channels = pred.shape[-1]
                    if target.ndim == 1 and target.numel() == 0:
                        target = target.reshape(0, n_channels)
                    targets_s += [dn_data(sname, target.to(fp32)).detach().cpu().numpy()]
                    t_coords_s += [t_coords.cpu().numpy()]
                    t_times_s += [t_times.astype("datetime64[ns]")]

            targets_lens[-1] += [[]]
            targets_lens[-1][-1] += [t.shape[0] for t in t_coords_s]
            preds_lens[-1] += [[]]
            preds_lens[-1][-1] += [p.shape[0] for p in preds_coords_s]

            preds_all[-1] += [np.concatenate(preds_s, axis=1)]
            targets_all[-1] += [np.concatenate(targets_s)]
            targets_coords_all[-1] += [np.concatenate(t_coords_s)]
            targets_times_all[-1] += [np.concatenate(t_times_s)]
            preds_coords_all[-1] += [np.concatenate(preds_coords_s)]
            preds_times_all[-1] += [np.concatenate(preds_times_s)]

    if len(preds_all) == 0 or np.array([p.shape[1] for pp in preds_all for p in pp]).sum() == 0:
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
    stream_names = list(cf.streams.keys())
    stream_infos = list(cf.streams.values())
    if val_cfg.get("output").get("streams") is not None:
        output_stream_names = val_cfg.output.streams
    else:
        output_stream_names = stream_names

    output_streams = {name: stream_names.index(name) for name in output_stream_names}
    _logger.debug(f"Using output streams: {output_streams} from streams: {stream_names}")

    target_channels: list[list[str]] = [list(stream.val_target_channels) for stream in stream_infos]
    source_channels: list[list[str]] = [list(stream.val_source_channels) for stream in stream_infos]

    geoinfo_channels = [[] for _ in stream_infos]  # TODO obtain channels

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
        preds_coords=preds_coords_all,
        preds_times=preds_times_all,
        preds_lens=preds_lens,
    )
    with zarrio_writer(config.get_path_results(cf, mini_epoch)) as zio:
        for subset in data.items():
            zio.write_zarr(subset)
