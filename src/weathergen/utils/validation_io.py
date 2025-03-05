# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import time

import numpy as np
import torch

import zarr


def sanitize_stream_str(istr):
    return istr.replace(" ", "_").replace("-", "_").replace(",", "")


#################################
def read_validation(cf, epoch, base_path, instruments, forecast_steps, rank=0):

    streams, columns, data = [], [], []

    fname = base_path + "validation_epoch{:05d}_rank{:04d}.zarr".format(epoch, rank)
    store = zarr.DirectoryStore(fname)
    ds = zarr.group(store=store)

    for ii, stream_info in enumerate(cf.streams):

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
    base_path,
    rank,
    epoch,
    cols,
    sources,
    preds_all,
    targets_all,
    targets_coords_all,
    targets_lens,
    jac=None,
):

    if 0 == len(cf.analysis_streams_output):
        return

    fname = base_path + "validation_epoch{:05d}_rank{:04d}".format(epoch, rank)
    fname += "" if jac is None else "_jac"
    fname += ".zarr"

    store = zarr.DirectoryStore(fname)
    ds = zarr.group(store=store)

    for k, si in enumerate(cf.streams):

        # only store requested streams
        if not np.array([s in si["name"] for s in cf.analysis_streams_output]).any():
            continue

        # skip empty entries (e.g. no channels from the sources are used as targets)
        if 0 == len(targets_all[k]) or 0 == len(targets_all[k][0]):
            continue

        # TODO: this only saves the first batch
        source_k = sources[0][k].cpu().detach().numpy()
        source_lens_k = np.array([source_k.shape[0]])
        preds_k = torch.cat(preds_all[k], 1).transpose(1, 0).cpu().detach().numpy()
        targets_k = torch.cat(targets_all[k], 0).cpu().detach().numpy()
        targets_coords_k = torch.cat(targets_coords_all[k], 0).cpu().detach().numpy()
        targets_lens_k = np.array(targets_lens[k], dtype=np.int64)

        fs = cf.forecast_steps
        fs = fs if type(fs) == int else fs[min(epoch, len(fs) - 1)]
        rn = si["name"].replace(" ", "_").replace("-", "_").replace(",", "")

        write_first = False
        if rn in ds.group_keys():
            if f"{fs}" not in ds[rn].group_keys():
                write_first = True
        else:
            write_first = True

        # TODO: how to avoid this
        if write_first:
            ds_source = ds.require_group(f"{rn}/{fs}")
            # column names
            if si["type"] in ["anemoi", "regular", "unstr"]:
                cols_values = np.arange(2, len(cols[k]))
            elif "obs" == si["type"]:
                cols_values = [col[:9] == "obsvalue_" for col in cols[k]]
            else:
                assert False, "Unsuppported stream type"
            ds_source.attrs["cols"] = np.array(cols[k])[cols_values].tolist()
            ds_source.create_dataset(
                "sources", data=source_k, chunks=(1024, *source_k.shape[1:])
            )
            ds_source.create_dataset("sources_lens", data=source_lens_k)
            ds_source.create_dataset(
                "preds", data=preds_k, chunks=(1024, *preds_k.shape[1:])
            )
            ds_source.create_dataset(
                "targets", data=targets_k, chunks=(1024, *targets_k.shape[1:])
            )
            ds_source.create_dataset(
                "targets_coords",
                data=targets_coords_k,
                chunks=(1024, *targets_coords_k.shape[1:]),
            )
            ds_source.create_dataset("targets_lens", data=targets_lens_k)
        else:
            rn = rn + f"/{fs}"
            ds[f"{rn}/sources"].append(source_k)
            ds[f"{rn}/sources_lens"].append(source_lens_k)
            ds[f"{rn}/preds"].append(preds_k)
            ds[f"{rn}/targets"].append(targets_k)
            ds[f"{rn}/targets_coords"].append(targets_coords_k)
            ds[f"{rn}/targets_lens"].append(targets_lens_k)

        if jac is not None:
            ds_source.create_dataset("jacobian", data=jac[k])

    store.close()
