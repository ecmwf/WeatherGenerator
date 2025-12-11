# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import torch

import weathergen.common.config as config
import weathergen.common.io as io
from weathergen.common.io import TimeRange
from weathergen.datasets.data_reader_base import TimeWindowHandler, str_to_datetime64

_logger = logging.getLogger(__name__)


def write_output(cf, mini_epoch, batch_idx, dn_data, batch, model_output, target_aux_output):
    """
    Interface for writing model output
    """

    # collect all target / prediction-related information
    fp32 = torch.float32
    preds_all, targets_all, targets_coords_all, targets_times_all = [], [], [], []
    for fstep in range(cf.forecast_offset, cf.forecast_steps + 2):
        preds_all += [[]]
        targets_all += [[]]
        targets_coords_all += [[]]
        targets_times_all += [[]]
        for stream_info in cf.streams:
            # predictions
            pred = model_output.get_physical_prediction(fstep, stream_info["name"]).to(fp32)
            target = target_aux_output.physical[stream_info["name"]].target_tokens[fstep].to(fp32)

            if not (target.shape[0] > 0 and pred.shape[0] > 0):
                continue

            # extract data/coords and remove token dimension if it exists
            pred = pred.reshape([pred.shape[0], *target.shape])
            assert pred.shape[1] > 0

            # TODO: the inner lists here should not be needed
            preds_all[-1] += [[dn_data(stream_info["name"], pred).detach().cpu().numpy()]]
            targets_all[-1] += [[dn_data(stream_info["name"], target).detach().cpu().numpy()]]

            sname = stream_info["name"]
            targets_coords_all[-1] += [target_aux_output.physical[sname].target_coords_raw[fstep]]
            targets_times_all[-1] += [target_aux_output.physical[sname].target_times_raw[fstep]]

    #         # TODO: re-enable
    #           if len(idxs_inv) > 0:
    #               pred = pred[:, idxs_inv]
    #               target = target[idxs_inv]
    #               targets_coords_raw[fstep][i_strm] = targets_coords_raw[fstep][i_strm][idxs_inv]
    #               targets_times_raw[fstep][i_strm] = targets_times_raw[fstep][i_strm][idxs_inv]

    # TODO: remove
    targets_lens = [[[t[0].shape[0]] for t in tt] for tt in targets_all]

    # collect source information
    sources = []
    for sample in batch.source_samples:
        sources += [[]]
        for _, stream_data in sample.streams_data.items():
            # TODO: support multiple input steps
            sources[-1] += [stream_data.source_raw[0]]

    sample_idxs = [
        [sdata.sample_idx for _, sdata in sample.streams_data.items()]
        for sample in batch.source_samples
    ]
    sample_idxs = [s[0].item() for s in sample_idxs]

    # more prep work

    stream_names = [stream.name for stream in cf.streams]
    if cf.streams_output is not None:
        output_stream_names = cf.streams_output
    else:
        output_stream_names = None

    if output_stream_names is None:
        output_stream_names = stream_names

    output_streams = {name: stream_names.index(name) for name in output_stream_names}

    _logger.debug(f"Using output streams: {output_streams} from streams: {stream_names}")

    target_channels: list[list[str]] = [list(stream.val_target_channels) for stream in cf.streams]
    source_channels: list[list[str]] = [list(stream.val_source_channels) for stream in cf.streams]

    geoinfo_channels = [[] for _ in cf.streams]  # TODO obtain channels

    # calculate global sample indices for this batch by offsetting by sample_start
    sample_start = batch_idx * cf.batch_size_validation_per_gpu

    # write output

    start_date = str_to_datetime64(cf.start_date_val)
    end_date = str_to_datetime64(cf.end_date_val)

    twh = TimeWindowHandler(start_date, end_date, cf.len_hrs, cf.step_hrs)
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
        cf.forecast_offset,
    )

    with io.ZarrIO(config.get_path_output(cf, mini_epoch)) as writer:
        for subset in data.items():
            writer.write_zarr(subset)
