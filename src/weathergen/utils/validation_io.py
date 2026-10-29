# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import concurrent.futures
import logging
import os
import pathlib
import shutil

import astropy_healpix as hp
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

import weathergen.common.config as config
import weathergen.common.io as io
from weathergen.common.io import TimeRange, zarrio_writer
from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.model.engines import LatentState

_logger = logging.getLogger(__name__)


def write_output(
    cf,
    val_cfg,
    batch_size,
    mini_epoch,
    batch_idx,
    dn_data,
    batch,
    model_output,
    target_aux_out,
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

    timestep_idxs = [0] if len(batch.get_output_idxs()) == 0 else batch.get_output_idxs()
    forecast_offset = timestep_idxs[0]
    targets_lens = []

    # TODO Maybe stopping at forecast_steps explained #1657
    for t_idx in timestep_idxs:
        preds_all += [[]]
        targets_all += [[]]
        targets_coords_all += [[]]
        targets_times_all += [[]]
        targets_lens += [[]]
        for sname in cf.streams.keys():
            # handle spoof data: do not write since it might corrupt validation (spoofing invisible
            # there)
            if target_aux_out.physical[t_idx][sname]["is_spoof"][0]:
                targets = target_aux_out.physical[t_idx][sname]["target"]
                # for-loop to make sure we have a consistent number of samples
                preds_s = [np.zeros((1, 0, t.shape[1])) for t in targets]
                targets_s = [np.zeros((0, t.shape[1])) for t in targets]
                t_coords_s = [np.zeros((0, 2)) for t in targets]
                t_times_s = [np.array([]).astype("datetime64[ns]") for t in targets]

            else:
                preds = model_output.get_physical_prediction(t_idx, sname)
                targets = target_aux_out.physical[t_idx][sname]["target"]

                preds_s, targets_s, t_coords_s, t_times_s = [], [], [], []

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
                    if idxs_inv is not None:
                        pred = pred[:, idxs_inv]
                        target = target[idxs_inv]
                        t_coords = t_coords[idxs_inv]
                        t_times = t_times[idxs_inv]

                    # denormalize data if requested and map to storage format
                    preds_s += [dn_data(sname, pred.to(fp32)).detach().cpu().numpy()]
                    targets_s += [dn_data(sname, target.to(fp32)).detach().cpu().numpy()]

                    # extract original target coords and times from target data
                    t_coords_s += [t_coords.cpu().numpy()]
                    t_times_s += [t_times.astype("datetime64[ns]")]

            targets_lens[-1] += [[]]
            targets_lens[-1][-1] += [t.shape[0] for t in targets_s]

            preds_all[-1] += [np.concatenate(preds_s, axis=1)]
            targets_all[-1] += [np.concatenate(targets_s)]
            targets_coords_all[-1] += [np.concatenate(t_coords_s)]
            targets_times_all[-1] += [np.concatenate(t_times_s)]

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

    write_latents = io.LATENT_STREAM in output_stream_names
    output_streams: dict[str, int] = {
        name: stream_names.index(name) for name in output_stream_names if name != io.LATENT_STREAM
    }
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

    latents_all = get_latent_output(batch, model_output) if write_latents else None

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
        latents=latents_all,
        sample_start=sample_start,
        forecast_offset=forecast_offset,
    )

    store_path = config.get_path_results(cf, mini_epoch, batch_idx)

    with zarrio_writer(store_path) as zio:
        for subset in data.items():
            zio.write_zarr(subset)
        # Write latent data directly to zarr store without using OutputItem validation
        if data.latents:
            _write_latent_data_to_zarr(
                zio,
                data,
                cf,
                batch,
                batch_idx,
                batch_size,
            )


def _write_latent_data_to_zarr(zio, data, cf, batch, batch_idx, batch_size):
    """Write latent data directly to zarr store.

    This bypasses OutputItem validation which incorrectly requires source datasets
    for latent-only items.

    Also writes coordinate and time metadata using config healpix coordinates.
    """
    # Calculate sample start index for this batch
    sample_start = batch_idx * batch_size

    # Iterate over latent data
    for t_idx, latents_in_step in enumerate(data.latents):
        for sample_idx_in_batch, latents_in_sample in enumerate(latents_in_step):
            if not latents_in_sample:
                continue

            # Calculate global sample index
            global_sample_idx = sample_start + sample_idx_in_batch

            # Reserve latent step 0 for the initial encoded state.
            group_path = f"{global_sample_idx}/{io.LATENT_STREAM}/{t_idx + 1}"

            npoints = _infer_latent_points_for_metadata(latents_in_sample)
            (
                coords_array,
                geoinfo_array,
                times_array,
                coords_len,
                num_register_tokens,
                num_class_tokens,
            ) = _build_latent_metadata(cf, batch, sample_idx_in_batch, npoints)
            # Collect all attributes upfront so they can be passed to
            # create_group in a single call.  Setting attrs individually
            # after creation causes duplicate zarr.json entries in ZipStore.
            group_attrs: dict[str, int | str] = {}
            if coords_array is not None and times_array is not None:
                group_attrs = {
                    "num_extra_tokens": int(num_register_tokens + num_class_tokens),
                    "num_register_tokens": int(num_register_tokens),
                    "num_class_tokens": int(num_class_tokens),
                    "spatial_points": int(coords_array.shape[0]),
                    "coords_order": "lat_lon",
                }
                if npoints is not None:
                    group_attrs["total_points"] = int(npoints)

            # Create or get group (avoid duplicate entries in ZipStore)
            group = zio.data_root.get(group_path)
            if group is None:
                group = zio.data_root.create_group(group_path, attributes=group_attrs)
            else:
                _logger.debug(f"Latent group already exists at {group_path}, skipping creation.")
            extra_written = False
            latent_names = {_latent_output_name(name) for name in latents_in_sample}
            for latent_name, latent_data in latents_in_sample.items():
                latent_array = np.asarray(latent_data)
                output_name = _latent_output_name(latent_name)
                extra_components, latent_array = _split_extra_tokens(
                    latent_name,
                    latent_array,
                    coords_len,
                    num_register_tokens,
                    num_class_tokens,
                )
                if extra_components is not None and not extra_written:
                    for extra_name, extra_array in extra_components.items():
                        if extra_name in latent_names:
                            continue
                        _write_array(group, extra_name, extra_array)
                        _logger.debug(
                            f"Wrote {extra_name} shape {extra_array.shape} "
                            f"for sample {global_sample_idx}"
                        )
                    extra_written = True

                try:
                    _write_array(group, output_name, latent_array)
                    _logger.debug(
                        f"Wrote latent {output_name} shape {latent_array.shape} "
                        f"for sample {global_sample_idx}"
                    )
                except Exception as e:
                    _logger.warning(
                        f"Failed to write latent {output_name} for sample {global_sample_idx}: {e}"
                    )

            if coords_array is not None and times_array is not None:
                _write_array(group, "coords", coords_array)
                _logger.debug(
                    f"Wrote coords shape {coords_array.shape} for sample {global_sample_idx}"
                )
                _write_array(group, "geoinfo", geoinfo_array)
                _logger.debug(
                    f"Wrote geoinfo shape {geoinfo_array.shape} for sample {global_sample_idx}"
                )
                _write_array(group, "times", times_array)
                _logger.debug(
                    f"Wrote times shape {times_array.shape} for sample {global_sample_idx}"
                )


def _infer_latent_points_for_metadata(latents_for_sample: dict) -> int | None:
    """
    Infer latent spatial length for metadata.
    Prefer tokens if present, else patch tokens, else first available array.
    """
    preferred_keys = ("tokens", "latent_state", "z_pre_norm", "patch_tokens")
    for key in latents_for_sample.keys():
        if any(pref in key for pref in preferred_keys):
            arr = np.asarray(latents_for_sample[key])
            if arr.ndim >= 1:
                return arr.shape[0]

    for latent_data in latents_for_sample.values():
        arr = np.asarray(latent_data)
        if arr.ndim >= 1:
            return arr.shape[0]
    return None


def _write_array(group, name: str, data: npt.NDArray) -> None:
    if name in group:
        # ZipStore cannot truly delete; overwriting creates duplicate entries.
        _logger.debug(f"Array {name} already exists in group, skipping write.")
        return
    group.create_array(name, data=data)


def _latent_output_name(name: str) -> str:
    return {
        "latent_state": "tokens",
        "latent_state_class_token": "class_token",
        "latent_state_register_tokens": "register_tokens",
    }.get(name, name)


def _split_extra_tokens(
    latent_name: str,
    latent_array: npt.NDArray,
    coords_len: int | None,
    num_register_tokens: int,
    num_class_tokens: int,
) -> tuple[dict[str, npt.NDArray] | None, npt.NDArray]:
    num_extra_tokens = num_register_tokens + num_class_tokens
    if (
        coords_len is not None
        and latent_array.ndim >= 1
        and latent_array.shape[0] == coords_len + num_extra_tokens
        and latent_name in ("latent_state", "tokens")
    ):
        extra_components: dict[str, npt.NDArray] = {}
        offset = 0
        extra_components["register_tokens"] = latent_array[offset : offset + num_register_tokens]
        offset += num_register_tokens
        extra_components["class_token"] = latent_array[offset : offset + num_class_tokens]
        return extra_components, latent_array[num_extra_tokens:]
    return None, latent_array


_HEALPIX_COORDS_CACHE: dict[int, tuple[npt.NDArray, npt.NDArray]] = {}


def _get_healpix_coords(cf) -> tuple[npt.NDArray, npt.NDArray] | None:
    if cf is None or not hasattr(cf, "healpix_level"):
        return None
    healpix_level = int(cf.healpix_level)
    cached = _HEALPIX_COORDS_CACHE.get(healpix_level)
    if cached is not None:
        return cached

    num_healpix_cells = 12 * 4**healpix_level
    ipix = np.arange(num_healpix_cells)
    lon, lat = hp.healpix_to_lonlat(ipix, 2**healpix_level, order="nested")
    coords = (lon.to_value("deg"), lat.to_value("deg"))
    _HEALPIX_COORDS_CACHE[healpix_level] = coords
    return coords


def _build_latent_metadata(cf, batch, sample_idx_in_batch, npoints):
    num_register_tokens = int(cf.get("num_register_tokens", 0))
    num_class_tokens = int(cf.get("num_class_tokens", 0))
    num_extra_tokens = num_register_tokens + num_class_tokens

    healpix_coords = _get_healpix_coords(cf)
    if healpix_coords is None or len(healpix_coords) != 2:
        return None, None, None, None, num_register_tokens, num_class_tokens

    lon, lat = healpix_coords
    coords_base = np.stack([lat, lon], axis=1)

    if batch is not None and sample_idx_in_batch < len(batch.get_source_samples().get_samples()):
        sample = batch.get_source_samples().get_samples()[sample_idx_in_batch]
        mask = None
        for meta in sample.meta_info.values():
            if hasattr(meta, "mask") and meta.mask is not None:
                mask = meta.mask
                break
        if mask is not None:
            mask_np = mask.detach().cpu().numpy().astype(bool)
            if mask_np.shape[0] == coords_base.shape[0]:
                coords_base = coords_base[mask_np]

    coords_len = coords_base.shape[0]
    if npoints is not None and npoints not in (coords_len, coords_len + num_extra_tokens):
        return None, None, None, coords_len, num_register_tokens, num_class_tokens

    coords_array = coords_base.astype(np.float32)
    geoinfo_array = np.zeros((coords_len, 0), dtype=np.float32)
    times_array = np.full((coords_len,), np.datetime64("NaT"), dtype="datetime64[ns]")
    return (
        coords_array,
        geoinfo_array,
        times_array,
        coords_len,
        num_register_tokens,
        num_class_tokens,
    )


def get_latent_output(batch, model_output):
    """
    Interface for getting latent states
    """

    # collect latent outputs per forecast step and per sample
    fp32 = torch.float32

    timestep_idxs = [0] if len(batch.get_output_idxs()) == 0 else batch.get_output_idxs()

    sample_idxs = [
        list(sample.streams_data.values())[0].sample_idx
        for sample in batch.get_source_samples().get_samples()
    ]

    latents_all: list[list[dict]] = []
    for t_idx in timestep_idxs:
        latents_all.append([])
        latent_pred = model_output.get_latent_prediction(t_idx)
        n_samples = len(sample_idxs)
        for i_sample in range(n_samples):
            per_sample: dict = {}
            for lname, lval in latent_pred.items():
                if isinstance(lval, LatentState):
                    fields = {
                        "tokens": lval.z_pre_norm,
                        "register_tokens": lval.register_tokens,
                        "class_token": lval.class_token,
                    }
                    for field_name, tensor in fields.items():
                        if tensor is not None:
                            sample_tensor = tensor[i_sample]
                            per_sample[field_name] = sample_tensor.detach().to(fp32).cpu().numpy()
                else:
                    per_sample[lname] = lval[i_sample].detach().to(fp32).cpu().numpy()
            latents_all[-1].append(per_sample)

    return latents_all


# ---------------------------------------------------------------------------
# Temporal averaging of written output. Post-processing on the already written
# full-resolution zarr store: read lazily, bin target/prediction in time per
# sample (= per init time), and write per-bin means. Self-contained in this module.
# ---------------------------------------------------------------------------

_MEAN_OUTPUT_DATASETS = ("target", "prediction")


def compute_time_means(cf, mode_cfg, mini_epoch):
    """
    Write temporally averaged target/prediction output datasets from the output store.

    Opens the full-resolution store written by :func:`write_output`, bins time
    per sample (init times are never pooled) and writes the per-bin mean with
    the bin midpoint as representative time into a sibling ``*_means`` store.
    ``source`` is left untouched.

    Configuration (under ``validation_config.output`` / ``test_config.output``)::

        output:
          frequency: null              # off; or a timedelta (6h, 1D) or weekly/monthly
          delete_full_resolution: false
          time_means_workers: null    # off (sequential); or number of worker processes

    When ``delete_full_resolution`` is true, the original store is removed and
    the means store is renamed to the original filename.
    """
    output_cfg = mode_cfg.get("output", {}) or {}
    freq = output_cfg.get("frequency", None)
    n_workers = output_cfg.get("time_means_workers", None)

    if (freq := _validate_frequency(freq)) is None:
        return

    if (paths := _validate_paths(cf, mini_epoch)) is None:
        return
    else:
        store_path, means_path = paths

    n_workers = _resolve_worker_count(n_workers)

    _write_all_binned_means(store_path, means_path, freq, n_workers)

    if output_cfg.get("delete_full_resolution", False):
        _remove_store(store_path)
        os.replace(means_path, store_path)
        _logger.info(f"Replaced full-resolution store with binned means at {store_path}.")
    else:
        _logger.info(f"Wrote temporally averaged output to {means_path}.")


def _write_all_binned_means(store_path, means_path, freq, n_workers):
    """Bin target/prediction per stream of every sample into the means store."""
    with io.zarrio_reader(store_path) as zio_r, io.zarrio_writer(means_path) as zio_w:
        root = zio_r.data_root
        for sample in list(root.group_keys()):
            sample_grp = root.get(sample)
            for stream in list(sample_grp.group_keys()):
                if stream == io.LATENT_STREAM:
                    continue
                _write_binned_means(
                    zio_w.data_root,
                    f"{sample}/{stream}",
                    sample_grp.get(stream),
                    freq,
                    store_path,
                    n_workers,
                )


def _write_binned_means(zio_root_w, prefix, stream_grp, freq, store_path, n_workers):
    """
    Write per-bin means of target/prediction for one stream.

    A point is written only if it is valid in every output dataset that has data for
    that bin, so target and prediction stay row-aligned even when only one
    of them is NaN-masked (e.g. land points for an ocean variable or stream).
    """
    dataset_bins, dataset_attrs = _accumulate_stream_bins(
        store_path, prefix, stream_grp, freq, n_workers
    )

    if not any(dataset_attrs.values()):
        return

    for bin_ord, present, accs in _resolve_write_bins(dataset_bins, dataset_attrs):
        for dataset, acc in accs.items():
            if acc is not None:
                _write_bin(
                    zio_root_w,
                    f"{prefix}/{bin_ord}/{dataset}",
                    acc,
                    present,
                    dataset_attrs[dataset],
                )


def _accumulate_stream_bins(store_path, prefix, stream_grp, freq, n_workers):
    """
    Accumulate per-bin sum/count for both output datasets of one stream.

    Runs sequentially, reusing the already-open `stream_grp`, unless
    `n_workers` > 1 and there are enough forecast steps to be worth it: the
    forecast steps are then split into contiguous chunks, each accumulated in
    its own worker process (own read-only store handle, safe for both
    ZipStore and LocalStore), and the partial per-bin sums/counts are merged.
    """
    fsteps = sorted(stream_grp.group_keys(), key=int)

    if n_workers <= 1 or len(fsteps) < 2 * n_workers:
        dataset_bins, dataset_attrs = {}, {}
        for dataset in _MEAN_OUTPUT_DATASETS:
            dataset_bins[dataset], dataset_attrs[dataset] = _accumulate_dataset_bins(
                stream_grp, dataset, prefix, freq, fsteps
            )
        return dataset_bins, dataset_attrs
    else:
        return _dispatch_parallel_bins(store_path, prefix, fsteps, freq, n_workers)


def _accumulate_dataset_bins(stream_grp, dataset, prefix, freq, fsteps):
    """
    NaN-aware sum/count accumulation of one dataset into per-bin buckets.

    Each (forecast-step, dataset) group may hold data for multiple valid times
    on a fixed set of points (e.g., intermediate hourly substeps for gridded
    data). Points are grouped by their actual valid time and accumulated into
    the appropriate time bin. This ensures correct binning when intermediate
    times span frequency boundaries (e.g., times straddling a monthly edge).

    Only the per-bin accumulators, never the full-resolution data, are held
    in memory.

    Returns the bins keyed by `_bin_key` and the (first seen) dataset attrs,
    or `({}, None)` if `dataset` has no data at all for this stream.
    """
    bins: dict[object, dict[str, npt.NDArray]] = {}
    attrs = None

    for fstep in fsteps:
        item_grp = stream_grp.get(fstep)
        dsg = item_grp.get(dataset) if item_grp is not None else None
        if dsg is None:
            continue
        times = np.asarray(dsg["times"]).astype("datetime64[ns]")
        if times.size == 0:
            continue
        if attrs is None:
            attrs = dict(dsg.attrs)

        data = np.asarray(dsg["data"], dtype=np.float64)
        coords = np.asarray(dsg["coords"])
        uniq_times = np.unique(times)

        # Process each unique time within this forecast step.
        # For single-time forecast steps, this loop runs once.
        # For multi-time (substeps), each intermediate time is binned separately.
        for unique_time in uniq_times:
            # Mask for points with this unique time
            time_mask = times == unique_time

            # Get bin key for this time
            bin_key, mid = _bin_key(unique_time, freq)

            # Initialize or retrieve accumulator for this bin
            acc = bins.get(bin_key)
            if acc is None:
                acc = bins[bin_key] = {
                    "sum": np.zeros_like(data),
                    "count": np.zeros_like(data),
                    "row_count": np.zeros(data.shape[0]),
                    "coords": coords,
                    "mid": mid,
                }
            else:
                assert acc["sum"].shape == data.shape, (
                    f"{prefix}/{fstep}/{dataset}: inconsistent point/channel count across "
                    "forecast steps in the same time bin."
                )

            # Accumulate only the points with this unique time
            data_for_time = data[time_mask]
            valid = ~np.isnan(data_for_time)
            acc["sum"][time_mask] += np.where(valid, data_for_time, 0.0)
            acc["count"][time_mask] += valid
            acc["row_count"][time_mask] += valid.reshape(valid.shape[0], -1).any(axis=1)

    return bins, attrs


def _dispatch_parallel_bins(store_path, prefix, fsteps, freq, n_workers):
    """Split forecast steps across processes and merge partial per-bin accumulators."""
    chunks = [chunk.tolist() for chunk in np.array_split(fsteps, n_workers) if len(chunk) > 0]
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(chunks)) as pool:
        chunk_results = list(
            pool.map(
                _accumulate_dataset_bins_worker,
                [store_path] * len(chunks),
                [prefix] * len(chunks),
                chunks,
                [freq] * len(chunks),
            )
        )

    dataset_bins: dict[str, dict[object, dict[str, npt.NDArray]]] = {
        d: {} for d in _MEAN_OUTPUT_DATASETS
    }
    dataset_attrs: dict[str, dict | None] = {d: None for d in _MEAN_OUTPUT_DATASETS}
    for chunk_bins, chunk_attrs in chunk_results:
        for dataset in _MEAN_OUTPUT_DATASETS:
            dataset_bins[dataset] = _merge_bins(dataset_bins[dataset], chunk_bins[dataset])
            if dataset_attrs[dataset] is None:
                dataset_attrs[dataset] = chunk_attrs[dataset]

    return dataset_bins, dataset_attrs


def _accumulate_dataset_bins_worker(store_path, prefix, fsteps, freq):
    """Worker entry point: open an independent read handle for one chunk of forecast steps."""
    with io.zarrio_reader(pathlib.Path(store_path)) as zio_r:
        stream_grp = zio_r.data_root
        for part in prefix.split("/"):
            stream_grp = stream_grp.get(part)
        dataset_bins, dataset_attrs = {}, {}
        for dataset in _MEAN_OUTPUT_DATASETS:
            dataset_bins[dataset], dataset_attrs[dataset] = _accumulate_dataset_bins(
                stream_grp, dataset, prefix, freq, fsteps
            )
        return dataset_bins, dataset_attrs


def _merge_bins(bins_a, bins_b):
    """Combine two per-bin accumulator dicts produced from disjoint forecast steps."""
    merged = dict(bins_a)
    for bin_key, acc_b in bins_b.items():
        acc_a = merged.get(bin_key)
        if acc_a is None:
            merged[bin_key] = acc_b
            continue
        acc_a["sum"] += acc_b["sum"]
        acc_a["count"] += acc_b["count"]
        acc_a["row_count"] += acc_b["row_count"]
    return merged


def _resolve_write_bins(dataset_bins, dataset_attrs):
    """[(bin_ord, present, accs) for each bin worth writing], align target/pred."""
    if not any(dataset_attrs.values()):
        return
    bin_keys = sorted({k for bins in dataset_bins.values() for k in bins})
    for bin_ord, bin_key in enumerate(bin_keys):
        accs = {d: dataset_bins[d].get(bin_key) for d in _MEAN_OUTPUT_DATASETS}
        masks = [a["row_count"] > 0 for a in accs.values() if a is not None]
        present = np.logical_and.reduce(masks) if masks else None
        if present is not None and present.any():
            yield bin_ord, present, accs


def _write_bin(zio_root_w, ds_path, acc, present, attrs):
    """Write the NaN-aware mean of one bin, restricted to the `present` points."""
    with np.errstate(invalid="ignore", divide="ignore"):
        counts = np.maximum(acc["count"], 1.0)
        mean = np.where(acc["count"] > 0, acc["sum"] / counts, np.nan)
    mean = mean[present].astype(np.float32)

    group = zio_root_w.get(ds_path)
    if group is None:
        group = zio_root_w.create_group(
            ds_path,
            attributes={
                "channels": attrs.get("channels", []),
                "geoinfo_channels": attrs.get("geoinfo_channels", []),
                "source_interval": attrs.get("source_interval"),
            },
        )
    _write_array(group, "data", mean)
    _write_array(group, "times", np.full(mean.shape[0], acc["mid"], dtype="datetime64[ns]"))
    _write_array(group, "coords", acc["coords"][present])
    _write_array(group, "geoinfo", np.zeros((mean.shape[0], 0), dtype=np.float32))


def _bin_key(t: np.datetime64, freq) -> tuple[object, np.datetime64]:
    """
    Map a single valid time to a (sortable, hashable) bin key and its midpoint.
    """

    t = np.datetime64(t, "ns")
    spec = str(freq).strip().lower()

    if spec == "weekly":
        period = pd.Timestamp(t).to_period("W")
    elif spec == "monthly":
        period = pd.Timestamp(t).to_period("M")
    else:
        raise ValueError(f"Unsupported output.frequency {freq!r}: expected 'weekly' or 'monthly'.")
    start, end = period.start_time, (period + 1).start_time
    mid = (start + (end - start) / 2).to_datetime64()
    return period, mid


def _remove_store(path: pathlib.Path):
    path = pathlib.Path(path)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _validate_frequency(freq) -> str | None:
    """Validate output.frequency; return curated spec, or None to skip computing means."""
    if freq is None or str(freq).strip().lower() in ("", "null", "none"):
        return None
    spec = str(freq).strip().lower()
    if spec not in ("weekly", "monthly"):
        _logger.debug(
            f"Unsupported output.frequency {freq!r}: expected 'weekly' or 'monthly'. "
            "Skipping time means computation."
        )
        return None
    return spec


def _validate_paths(cf, mini_epoch) -> tuple[pathlib.Path, pathlib.Path] | None:
    """Resolve full-resolution and means store paths, clearing any stale means store."""
    store_path = pathlib.Path(config.get_path_results(cf, mini_epoch))
    if not store_path.exists():
        _logger.debug(f"No store at {store_path}; skipping time means computation.")
        return None
    means_path = store_path.with_name(f"{store_path.stem}_means{store_path.suffix}")
    _remove_store(means_path)
    return store_path, means_path


def _resolve_worker_count(configured) -> int:
    """Resolve `time_means_workers`; parallelism is opt-in (default: sequential)."""
    return max(1, int(configured)) if configured else 1
