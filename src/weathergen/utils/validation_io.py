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
import weathergen.common.config as config

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
    if cf.analysis_streams_output is None:
        output_stream_names = [stream.name for stream in cf.streams]
        _logger.info(f"Using all streams as output streams: {output_stream_names}")
    else:
        output_stream_names = [
            stream.name for stream in cf.streams if stream.name in cf.analysis_streams_output
        ]
    _logger.info(f"Using output streams: {output_stream_names}")
    # TODO: streams anemoi `source`, `target` commented out???

    channels: list[list[str]] = [
        list(stream.val_target_channels)
        for stream in cf.streams
        if stream.name in output_stream_names
    ]

    geoinfo_channels = [[] for _ in cf.streams]  # TODO obtain channels

    # assume: is batch size guarnteed and constant:
    # => calculate global sample indices for this batch by offsetting by sample_start
    sample_start = batch_idx * cf.batch_size_validation_per_gpu

    data = io.OutputBatchData(
        sources,
        targets_all,
        preds_all,
        targets_coords_all,
        targets_times_all,
        targets_lens,
        output_stream_names,
        channels,
        geoinfo_channels,
        sample_start,
        cf.forecast_offset,
    )

    with io.ZarrIO(config.get_path_output(cf, epoch)) as writer:
        for subset in data.items():
            writer.write_zarr(subset)
