# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import zarr

import weathergen.utils.config as config
import weathergen.utils.io as io


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
    stream_names = [
        stream.name if stream.name in cf.analysis_streams_output else "" for stream in cf.streams
    ]  # TODO: how to correctly handle this
    # => what happens if stream is not in analysis_streams_output?
    # => what happens if analysis_streams_output is none?
    # streams anemoi `source`, `target` commented out???

    # TODO: right way to query config for stream channels?
    # assumption: datasets in a stream share channels
    channels = [stream.target for stream in cf.streams]
    # samples = range(batch_idx * batch_size, (batch_idx + 1) * batch_size)

    # TODO: is batch size guarnteed and constant?
    sample_start = batch_idx * cf.batch_size_validation

    data = io.OutputBatchData(
        sources,
        preds_all,
        targets_all,
        targets_coords_all,
        targets_times_all,
        targets_lens,
        stream_names,
        channels,
        sample_start,
        cf.forecast_offset,
    )

    with io.ZarrOutput(_get_data_root(cf, epoch)) as writer:
        for subset in data.items():
            writer.write_data(subset)


def _get_data_root(cf, epoch):
    base_path = config.get_path_run(cf)
    fname = f"validation_epoch{epoch:05d}_rank{cf.rank:04d}.zarr"

    store = zarr.DirectoryStore(base_path / fname)
    return zarr.group(store=store), store
