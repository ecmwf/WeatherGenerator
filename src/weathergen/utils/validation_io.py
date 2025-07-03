# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import weathergen.utils.config as config
import weathergen.utils.io as io

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
    is_output_stream = [stream.name in cf.analysis_streams_output for stream in cf.streams]
    stream_names = [
        stream.name if condition else "" for condition, stream in zip(is_output_stream, cf.streams)
    ]  # TODO: how to correctly handle this => set analysis_streams_output in default config = []
    # => what happens if stream is not in analysis_streams_output?
    # => what happens if analysis_streams_output is none?
    # streams anemoi `source`, `target` commented out???

    # assumption: datasets in a stream share channels
    channels = [list(stream.val_target_channels) for condition, stream in zip(is_output_stream, cf.streams) if condition] # target_channels ??
    geoinfo_channels = [[] for _ in cf.streams] # TODO obtain channels
    # samples = range(batch_idx * batch_size, (batch_idx + 1) * batch_size)

    # assume: is batch size guarnteed and constant?
    sample_start = batch_idx * cf.batch_size_validation
    _logger.info("called validation_io")

    data = io.OutputBatchData(
        sources,
        targets_all,
        preds_all,
        targets_coords_all,
        targets_times_all,
        targets_lens,
        stream_names,
        channels,
        geoinfo_channels,
        sample_start,
        cf.forecast_offset,
    )

    with io.ZarrIO(config.get_path_output(cf, epoch)) as writer:
        for subset in data.items():
            writer.write_zarr(subset)