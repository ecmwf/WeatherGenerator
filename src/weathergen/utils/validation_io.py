# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import numpy as np
import numpy.typing as npt
import torch

import weathergen.common.config as config
import weathergen.common.io as io
from weathergen.common.io import TimeRange, zarrio_writer
from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.datasets.domain import Domain
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

    # The storage stream axis is decoder-qualified; metadata stays native-stream keyed.
    outputs = [
        (decoder, stream)
        for decoder, settings in cf.decoders.items()
        for stream in settings.streams
    ]
    stream_names = [io.physical_output_name(decoder, stream) for decoder, stream in outputs]
    stream_infos = [cf.streams[stream] for _, stream in outputs]
    requested = val_cfg.output.get("streams")
    write_latents = requested is not None and io.LATENT_STREAM in requested
    output_streams = {
        name: i
        for i, (name, (_, source)) in enumerate(zip(stream_names, outputs, strict=True))
        if requested is None or name in requested or source in requested
    }
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
        for decoder, sname in outputs:
            target_data = target_aux_out.physical[t_idx][sname]
            targets = target_data["target"]
            predictions = model_output.get_physical_prediction(t_idx, decoder, sname)
            preds_s, targets_s, t_coords_s, t_times_s = [], [], [], []
            for i_batch, native_target in enumerate(targets):
                prediction = predictions[i_batch] if predictions is not None else None
                if prediction is None:
                    indices = torch.empty(0, dtype=torch.long, device=native_target.device)
                    pred = native_target.new_empty(
                        (cf.streams[sname].pred_head.ens_size, 0, native_target.shape[-1])
                    )
                else:
                    indices, pred = prediction.indices, prediction.values
                if target_data["is_spoof"][i_batch]:
                    indices, pred = indices[:0], pred[:, :0]
                native_indices = indices.detach().cpu().numpy()
                idxs_inv = target_data["idxs_inv"][i_batch]
                if idxs_inv is not None:
                    # idxs_inv maps original -> native order. Rank selected native rows
                    # by their original positions, rather than applying a full permutation
                    # to a shorter decoder subset.
                    original_positions = np.argsort(idxs_inv.detach().cpu().numpy())
                    order = np.argsort(original_positions[native_indices])
                    native_indices = native_indices[order]
                    pred = pred[:, torch.as_tensor(order, device=pred.device)]
                target = native_target[torch.as_tensor(native_indices, device=native_target.device)]
                coords = target_data["target_coords"][i_batch]
                coords = coords[torch.as_tensor(native_indices, device=coords.device)]
                times = target_data["target_times"][i_batch][native_indices]
                preds_s.append(dn_data(sname, pred.float()).detach().cpu().numpy())
                targets_s.append(dn_data(sname, target.float()).detach().cpu().numpy())
                t_coords_s.append(coords.cpu().numpy())
                t_times_s.append(times.astype("datetime64[ns]"))

            targets_lens[-1] += [[]]
            targets_lens[-1][-1] += [t.shape[0] for t in targets_s]

            preds_all[-1] += [np.concatenate(preds_s, axis=1)]
            targets_all[-1] += [np.concatenate(targets_s)]
            targets_coords_all[-1] += [np.concatenate(t_coords_s)]
            targets_times_all[-1] += [np.concatenate(t_times_s)]

    if not write_latents and not any(p.shape[1] for step in preds_all for p in step):
        _logger.warning("Writing no data since predictions are empty.")
        return

    # collect source information
    sources = [
        [sample.streams_data[stream].source_raw[0] for _, stream in outputs]
        for sample in batch.get_source_samples().get_samples()
    ]

    sample_idxs = [
        list(sample.streams_data.values())[0].sample_idx
        for sample in batch.get_source_samples().get_samples()
    ]

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
            group_attrs: dict[str, int | str | list[int]] = {}
            if coords_array is not None and times_array is not None:
                group_attrs = {
                    "num_extra_tokens": int(num_register_tokens + num_class_tokens),
                    "num_register_tokens": int(num_register_tokens),
                    "num_class_tokens": int(num_class_tokens),
                    "spatial_points": int(coords_array.shape[0]),
                    "coords_order": "lat_lon",
                }
                group_attrs["healpix_levels"] = config.get_healpix_levels(cf)
                group_attrs["queries_per_cell"] = int(cf.get("ae_local_num_queries", 1))
                group_attrs["spatial_order"] = "level_cell_query"
                group_attrs["cells_per_level"] = [
                    len(Domain.from_config(cf, level)) for level in config.get_healpix_levels(cf)
                ]
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


def _get_healpix_coords(cf) -> tuple[npt.NDArray, npt.NDArray] | None:
    if cf is None or not hasattr(cf, "streams"):
        return None
    coords = [
        Domain.from_config(cf, level).centres_lonlat() for level in config.get_healpix_levels(cf)
    ]
    num_queries = int(cf.get("ae_local_num_queries", 1))
    return tuple(
        np.concatenate([grid[axis] for grid in coords]).repeat(num_queries) for axis in (0, 1)
    )


def _build_latent_metadata(cf, batch, sample_idx_in_batch, npoints):
    num_queries = int(cf.get("ae_local_num_queries", 1))
    num_register_tokens = int(cf.get("num_register_tokens", 0)) * num_queries
    num_class_tokens = int(cf.get("num_class_tokens", 0)) * num_queries
    num_extra_tokens = num_register_tokens + num_class_tokens

    healpix_coords = _get_healpix_coords(cf)
    if healpix_coords is None or len(healpix_coords) != 2:
        return None, None, None, None, num_register_tokens, num_class_tokens

    lon, lat = healpix_coords
    coords_base = np.stack([lat, lon], axis=1)

    if (
        npoints not in (len(coords_base), len(coords_base) + num_extra_tokens)
        and len(config.get_healpix_levels(cf)) == 1
        and batch is not None
        and sample_idx_in_batch < len(batch.get_source_samples().get_samples())
    ):
        sample = batch.get_source_samples().get_samples()[sample_idx_in_batch]
        mask = None
        for meta in sample.meta_info.values():
            if hasattr(meta, "mask") and meta.mask is not None:
                mask = meta.mask
                break
        if mask is not None:
            mask_np = mask.detach().cpu().numpy().astype(bool).repeat(num_queries)
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
