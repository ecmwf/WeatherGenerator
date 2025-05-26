# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
from collections import namedtuple

import numpy as np
import torch
import zarr


def sanitize_stream_str(istr):
    return istr.replace(" ", "_").replace("-", "_").replace(",", "")


#################################
def read_validation(cf, epoch, base_path: Path, instruments, forecast_steps, rank=0):
    streams, columns, data = [], [], []

    fname = base_path / f"validation_epoch{epoch:05d}_rank{rank:04d}.zarr"
    store = zarr.DirectoryStore(fname)
    ds = zarr.group(store=store)

    for _ii, stream_info in enumerate(cf.streams):
        n = stream_info["name"]
        if len(instruments):
            if not np.array([r in n for r in instruments]).any():
                continue

        streams += [stream_info["name"]]
        columns.append(ds[f"{sanitize_stream_str(n)}/0"].attrs["cols"])
        data += [[]]

        for fstep in forecast_steps:
            data[-1] += [[]]
            istr = sanitize_stream_str(n)

            data[-1][-1].append(ds[f"{istr}/{fstep}/sources"])
            data[-1][-1].append(ds[f"{istr}/{fstep}/sources_coords"])
            data[-1][-1].append(ds[f"{istr}/{fstep}/preds"])
            data[-1][-1].append(ds[f"{istr}/{fstep}/targets"])
            data[-1][-1].append(ds[f"{istr}/{fstep}/targets_coords"])
            data[-1][-1].append(ds[f"{istr}/{fstep}/sources_lens"])
            data[-1][-1].append(ds[f"{istr}/{fstep}/targets_lens"])
            data[-1][-1].append(~np.isnan(data[-1][-1][3]))

            data[-1][-1].append(np.mean(data[-1][-1][2], axis=1))
            data[-1][-1].append(np.std(data[-1][-1][2], axis=1))

    return streams, columns, data


Data = namedtuple(
    "Data",
    ["source", "preds", "targets", "target_coords", "target_times", "target_lens", "source_lens"],
)


#################################
def write_validation(
    cf,
    base_path: Path,
    rank,
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
    data_root, store = _get_data_root(base_path, rank, epoch)
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
    if data.source_lens.sum() > 0:
        ds_source["sources"].append(data.source)
    ds_source["sources_lens"].append(data.source_lens)
    ds_source["preds"].append(data.preds)
    ds_source["targets"].append(data.targets)
    ds_source["targets_coords"].append(data.target_coords)
    ds_source["targets_times"].append(data.target_times)
    ds_source["targets_lens"].append(data.target_lens)


def _get_data_root(base_path, epoch, rank):
    fname = f"validation_epoch{epoch:05d}_rank{rank:04d}.zarr"

    store = zarr.DirectoryStore(base_path / fname)
    return zarr.group(store=store), store
