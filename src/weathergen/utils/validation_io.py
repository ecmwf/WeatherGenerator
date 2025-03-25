# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path

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
    jac=None,
):
    if len(cf.analysis_streams_output) == 0:
        return

    fname = f"validation_epoch{epoch:05d}_rank{rank:04d}"
    fname += "" if jac is None else "_jac"
    fname += ".zarr"

    store = zarr.DirectoryStore(base_path / fname)
    ds = zarr.group(store=store)

    for fstep in range(len(preds_all)):
        for k, si in enumerate(cf.streams):
            # only store requested streams
            if not np.array([s in si["name"] for s in cf.analysis_streams_output]).any():
                continue

            # skip empty entries (e.g. no channels from the sources are used as targets)
            if len(targets_all[fstep][k]) == 0 or len(targets_all[fstep][k][0]) == 0:
                continue

            # TODO: this only saves the first batch
            source_k = sources[0][k].cpu().detach().numpy()
            source_lens_k = np.array([source_k.shape[0]])
            preds_k = torch.cat(preds_all[fstep][k], 1).transpose(1, 0).cpu().detach().numpy()
            targets_k = torch.cat(targets_all[fstep][k], 0).cpu().detach().numpy()
            targets_coords_k = targets_coords_all[fstep][k].numpy()
            targets_times_k = targets_times_all[fstep][k]
            targets_lens_k = np.array(targets_lens[fstep][k], dtype=np.int64)

            rn = si["name"].replace(" ", "_").replace("-", "_").replace(",", "")

            # TODO: handle more robustly
            write_first = False
            if rn in ds.group_keys():
                if f"{fstep}" not in ds[rn].group_keys():
                    write_first = True
            else:
                write_first = True

            # TODO: how to avoid the case distinction for write_first
            if write_first:
                ds_source = ds.require_group(f"{rn}/{fstep}")
                ds_source.create_dataset(
                    "sources", data=source_k, chunks=(1024, *source_k.shape[1:])
                )
                ds_source.create_dataset("sources_lens", data=source_lens_k)
                ds_source.create_dataset("preds", data=preds_k, chunks=(1024, *preds_k.shape[1:]))
                ds_source.create_dataset(
                    "targets", data=targets_k, chunks=(1024, *targets_k.shape[1:])
                )
                ds_source.create_dataset(
                    "targets_coords",
                    data=targets_coords_k,
                    chunks=(1024, *targets_coords_k.shape[1:]),
                )
                ds_source.create_dataset(
                    "targets_times", data=targets_times_k, chunks=(1024, *targets_times_k.shape[1:])
                )
                ds_source.create_dataset("targets_lens", data=targets_lens_k)
            else:
                rn = rn + f"/{fstep}"
                if source_lens_k.sum() > 0:
                    ds[f"{rn}/sources"].append(source_k)
                ds[f"{rn}/sources_lens"].append(source_lens_k)
                ds[f"{rn}/preds"].append(preds_k)
                ds[f"{rn}/targets"].append(targets_k)
                ds[f"{rn}/targets_coords"].append(targets_coords_k)
                ds[f"{rn}/targets_times"].append(targets_times_k)
                ds[f"{rn}/targets_lens"].append(targets_lens_k)

    store.close()
