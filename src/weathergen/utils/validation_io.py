# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import weathergen.common.io as io
import weathergen.utils.config as config

_logger = logging.getLogger(__name__)


def write_output(
    cf,
    epoch,
    batch_idx,
    sources,
    preds_all,
    targets_all,
    targets_coords_all,
    targets_times_all,
    targets_lens,
):
    stream_names = [stream.name for stream in cf.streams]
    if cf.streams_output is not None:
        output_stream_names = cf.streams_output
    elif cf.analysis_streams_output is not None:  # --- to be removed at some point ---
        output_stream_names = cf.analysis_streams_output  # --- to be removed at some point ---
    else:
        output_stream_names = None
        
    if output_stream_names is None:
        output_stream_names = stream_names

    output_streams = {name: stream_names.index(name) for name in output_stream_names}

    _logger.debug(f"Using output streams: {output_streams} from streams: {stream_names}")

    target_channels: list[list[str]] = [list(stream.val_target_channels) for stream in cf.streams]
    source_channels: list[list[str]] = [list(stream.val_source_channels) for stream in cf.streams]

    geoinfo_channels = [[] for _ in cf.streams]  # TODO obtain channels

    # assume: is batch size guarnteed and constant:
    # => calculate global sample indices for this batch by offsetting by sample_start
    sample_start = batch_idx * cf.batch_size_validation_per_gpu

    assert len(stream_names) == len(targets_all[0]), "data does not match number of streams"
    assert len(stream_names) == len(preds_all[0]), "data does not match number of streams"
    assert len(stream_names) == len(sources[0]), "data does not match number of streams"

    data = io.OutputBatchData(
        sources,
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

    with io.ZarrIO(config.get_path_output(cf, epoch)) as writer:
        for subset in data.items():
            writer.write_zarr(subset)
