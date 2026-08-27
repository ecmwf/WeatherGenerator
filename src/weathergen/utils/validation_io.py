# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import astropy_healpix as hp
import numpy as np
import numpy.typing as npt
import torch

import weathergen.common.config as config
import weathergen.common.io as io
from weathergen.common.io import TimeRange, zarrio_writer
from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.model.engines import LatentState

_logger = logging.getLogger(__name__)


def _empty_step(n_samples: int, n_ens: int, n_channels: int):
    """Zero-sized target/prediction entries for a step that carries no data."""
    return (
        [np.zeros((n_ens, 0, n_channels), dtype=np.float32) for _ in range(n_samples)],
        [np.zeros((0, n_channels), dtype=np.float32) for _ in range(n_samples)],
        [np.zeros((0, 2), dtype=np.float32) for _ in range(n_samples)],
        [np.array([]).astype("datetime64[ns]") for _ in range(n_samples)],
    )


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

    # _get_output_length clamps to at least one output step, so this always holds
    assert len(batch.get_output_idxs()) > 0, "Batch carries no output steps."
    forecast_offset = batch.get_output_idxs()[0]

    # the chunk describes which forecast steps it holds, including the leading empty steps
    # that the first chunk keeps so it is indexed by global forecast step
    timestep_idxs = model_output.forecast_steps

    n_samples = len(batch.get_source_samples().get_samples())
    targets_lens = []

    for t_idx in timestep_idxs:
        preds_all += [[]]
        targets_all += [[]]
        targets_coords_all += [[]]
        targets_times_all += [[]]
        targets_lens += [[]]
        for sname in cf.streams.keys():
            chunk_idx = model_output.chunk_idx(t_idx)
            assert model_output.forecast_steps[chunk_idx] == t_idx, (
                f"Prediction at index {chunk_idx} is valid for forecast step "
                f"{model_output.forecast_steps[chunk_idx]}, but the target is valid for {t_idx}."
            )

            n_channels = len(cf.streams[sname].val_target_channels)

            # leading empty steps of the first chunk carry a source but no target/prediction
            if t_idx < forecast_offset:
                preds_s, targets_s, t_coords_s, t_times_s = _empty_step(n_samples, 1, n_channels)

            # handle spoof data: do not write since it might corrupt validation (spoofing invisible
            # there)
            elif target_aux_out.physical[t_idx][sname]["is_spoof"][0]:
                preds = model_output.get_physical_prediction(chunk_idx, sname)
                n_ens = preds[0].shape[0] if preds is not None and len(preds) > 0 else 1
                preds_s, targets_s, t_coords_s, t_times_s = _empty_step(
                    n_samples, n_ens, n_channels
                )

            else:
                preds = model_output.get_physical_prediction(chunk_idx, sname)
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
        forecast_steps=timestep_idxs,
    )

    store_path = config.get_path_results(cf, mini_epoch)

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
