# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections import namedtuple

import numpy as np
import torch
import zarr

import weathergen.utils.config as config
import weathergen.utils.io as io


def _sanitize_stream_name(istr):
    return istr.replace(" ", "_").replace("-", "_").replace(",", "")


Data = namedtuple(
    "Data",
    ["source", "preds", "targets", "target_coords", "target_times", "target_lens", "source_lens"],
)


def write_validation(
    cf,
    epoch,
    sources,
    preds_all,
    targets_all,
    targets_coords_all,
    targets_times_all,
    targets_lens,
):
    if len(cf.analysis_streams_output) == 0:
        return

    data = Data(
        sources, preds_all, targets_all, targets_coords_all, targets_times_all, targets_lens, None
    )
    data_root, store = _get_data_root(cf, epoch)
    streams = [
        stream_info
        for stream_info in cf.streams
        if stream_info["name"] in cf.analysis_streams_output
    ]

    # FIXME: does not include forecast offset
    for forecast_step in range(len(preds_all)):
        # TODO simplify this conditional
        # skip empty entries (e.g. no channels from the sources are used as targets)
        streams_without_empty_entries = (
            stream_info
            for stream_info, k in streams
            if not (
                len(targets_all[forecast_step][k]) == 0
                or len(targets_all[forecast_step][k][0]) == 0
            )
        )
        for k, si in enumerate(streams_without_empty_entries):
            extracted_data = _extract_data(data, forecast_step, k)

            stream_name = _sanitize_stream_name(si["name"])
            group = data_root.require_group(f"{stream_name}/{forecast_step}")

            # TODO: how to avoid the case distinction for write_first
            if _is_first_write(data_root, stream_name, forecast_step):
                _write_first(group, extracted_data)
            else:
                _successive_write(group, extracted_data)

    store.close()


def write_validation_new(
    cf, epoch, batch_idx, sources, preds_all, targets_all, targets_coords_all, targets_times_all, targets_lens
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


def _extract_data(data: Data, fstep, k) -> Data:
    # TODO: this only saves the first batch
    source = data.source[0][k].cpu().detach().numpy()
    source_lens = np.array([source.shape[0]])  # ?
    preds = torch.cat(data.preds[fstep][k], 1).transpose(1, 0).cpu().detach().numpy()
    targets = torch.cat(data.targets[fstep][k], 0).cpu().detach().numpy()
    targets_coords = data.target_coords[fstep][k].numpy()
    targets_times = data.target_times[fstep][k]
    targets_lens = np.array(data.target_lens[fstep][k], dtype=np.int64)
    return Data(source, preds, targets, targets_coords, targets_times, targets_lens, source_lens)


def _is_first_write(data_root, stream_name, forecast_step):
    # TODO: handle more robustly
    if stream_name in data_root.group_keys():
        if f"{forecast_step}" in data_root[stream_name].group_keys():
            return False

    return True


def _write_first(ds_source: zarr.Group, data: Data):
    ds_source.create_dataset("datasources", data=data.source, chunks=(1024, *data.source.shape[1:]))
    ds_source.create_dataset("sources_lens", data=data.source_lens)
    ds_source.create_dataset("preds", data=data.preds, chunks=(1024, *data.preds.shape[1:]))
    ds_source.create_dataset("targets", data=data.targets, chunks=(1024, *data.targets.shape[1:]))
    ds_source.create_dataset(
        "targets_coords", data=data.target_coords, chunks=(1024, *data.target_coords.shape[1:])
    )
    ds_source.create_dataset(
        "targets_times", data=data.target_times, chunks=(1024, *data.target_times.shape[1:])
    )
    ds_source.create_dataset("targets_lens", data=data.target_lens)


def _successive_write(ds_source: zarr.Group, data: Data):
    # appends along the first dimension
    if data.source_lens.sum() > 0:
        ds_source["sources"].append(data.source)
    ds_source["sources_lens"].append(data.source_lens)
    ds_source["preds"].append(data.preds)
    ds_source["targets"].append(data.targets)
    ds_source["targets_coords"].append(data.target_coords)
    ds_source["targets_times"].append(data.target_times)
    ds_source["targets_lens"].append(data.target_lens)


def _get_data_root(cf, epoch):
    base_path = config.get_path_run(cf)
    fname = f"validation_epoch{epoch:05d}_rank{cf.rank:04d}.zarr"

    store = zarr.DirectoryStore(base_path / fname)
    return zarr.group(store=store), store
