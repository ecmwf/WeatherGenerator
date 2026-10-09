# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cache

import astropy_healpix as hp
import numpy as np
import numpy.typing as npt
import torch

import weathergen.common.config as config
import weathergen.common.io as io
from weathergen.common.io import TimeRange, zarrio_writer
from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.model.engines import LatentState
from weathergen.utils.utils import is_stream_reconstructed

_logger = logging.getLogger(__name__)


def _empty_step(n_samples: int, n_ens: int, n_channels: int):
    """
    Zero-sized target/prediction entries for a step that carries no data.
    """
    return (
        [np.zeros((n_ens, 0, n_channels), dtype=np.float32) for _ in range(n_samples)],
        [np.zeros((0, n_channels), dtype=np.float32) for _ in range(n_samples)],
        [np.zeros((0, 2), dtype=np.float32) for _ in range(n_samples)],
        [np.array([]).astype("datetime64[ns]") for _ in range(n_samples)],
    )


def _extract_one_tstep(
    forecast_offset: int,
    t_idx: int,
    chunk_idx: int,
    n_samples: int,
    sname: str,
    streams,
    model_output,
    batch,
    target_aux_out,
    dn_data,
):
    """
    Extract data for one time step
    """

    fp32 = torch.float32

    n_channels = len(streams[sname].val_target_channels)
    preds = model_output.get_physical_prediction(chunk_idx, sname)
    n_ens = preds[0].shape[0] if preds is not None and len(preds) > 0 else 1

    # handle spoof data: do not write since it might corrupt validation (spoofing invisible
    # there), also handle non-output streams
    not_reconstructed = not is_stream_reconstructed(streams[sname])
    is_spoof = target_aux_out.physical[t_idx][sname]["is_spoof"][0]

    # empty, no predition or spoofed step
    if t_idx < forecast_offset or not_reconstructed or preds is None or is_spoof:
        preds_s, targets_s, t_coords_s, t_times_s = _empty_step(n_samples, n_ens, n_channels)

    else:
        targets = target_aux_out.physical[t_idx][sname]["target"]
        preds_s, targets_s, t_coords_s, t_times_s = [], [], [], []
        # extract prediction and targets for different samples in batch
        for i_batch, (pred, target) in enumerate(zip(preds, targets, strict=False)):
            source_sample = batch.source_samples.samples[i_batch].streams_data[sname]
            t_times = source_sample.target_times_raw[t_idx]
            t_coords = source_sample.target_coords_raw[t_idx]
            idxs_inv = source_sample.idxs_inv[t_idx]

            if len(target) == 0:
                target = torch.zeros((0, n_channels), dtype=torch.float32)

            # invert random reordering from healpix cell-based embedding
            if idxs_inv is not None and len(idxs_inv > 0):
                pred = pred[:, idxs_inv]
                target = target[idxs_inv] if len(target) > 0 else target
                t_coords = t_coords[idxs_inv]
                t_times = t_times[idxs_inv]

            # denormalize data if requested and map to storage format
            preds_s += [dn_data(sname, pred.to(fp32)).detach().cpu().numpy()]
            targets_s += [dn_data(sname, target.to(fp32)).detach().cpu().numpy()]

            # extract original target coords and times from target data
            t_coords_s += [t_coords.cpu().numpy()]
            t_times_s += [t_times.astype("datetime64[ns]")]

    return preds_s, targets_s, t_coords_s, t_times_s


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

    stream_names = list(cf.streams.keys())
    output_stream_names = val_cfg.output.get("streams")
    if output_stream_names is None:
        output_stream_names = stream_names
    output_streams = {
        name: stream_names.index(name) for name in output_stream_names if name != io.LATENT_STREAM
    }
    latents_all = (
        get_latent_output(batch, model_output) if io.LATENT_STREAM in output_stream_names else []
    )
    has_latents = any(sample for step in latents_all for sample in step)

    if output_streams:
        # TODO: how to handle multiple physical loss terms
        outputs_physical = [
            loss_name
            for loss_name, loss_term in val_cfg.losses.items()
            if loss_term.type == "LossPhysical"
        ]
        assert len(outputs_physical) == 1
        target_aux_out = target_aux_out[outputs_physical[0]]

    # collect all target / prediction-related information
    preds_all, targets_all, targets_coords_all, targets_times_all = [], [], [], []
    targets_lens = []

    # _get_output_length clamps to at least one output step, so this always holds
    assert len(batch.get_output_idxs()) > 0, "Batch carries no output steps."
    forecast_offset = batch.get_output_idxs()[0]

    # chunk's ModelOutput includes a leading padding range [0..forecast_offset) so
    # that slot indices equal global forecast step numbers. zarr writing must
    # only emit the steps that this chunk actually computed
    chunk_forecast_offset = model_output.forecast_offset
    timestep_idxs_chunk = [s for s in model_output.forecast_steps if s >= chunk_forecast_offset]

    source_samples = batch.get_source_samples().get_samples()
    n_samples = len(source_samples)

    # Collect physical data only when physical streams were requested.
    physical_steps = timestep_idxs_chunk if output_streams else []
    for t_idx in physical_steps:
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

            preds_s, targets_s, t_coords_s, t_times_s = _extract_one_tstep(
                forecast_offset,
                t_idx,
                chunk_idx,
                n_samples,
                sname,
                cf.streams,
                model_output,
                batch,
                target_aux_out,
                dn_data,
            )

            targets_lens[-1] += [[]]
            targets_lens[-1][-1] += [t.shape[1] for t in preds_s]

            preds_all[-1] += [np.concatenate(preds_s, axis=1)]
            targets_all[-1] += [np.concatenate(targets_s)]
            targets_coords_all[-1] += [np.concatenate(t_coords_s)]
            targets_times_all[-1] += [np.concatenate(t_times_s)]

    has_physical = any(
        step[stream_idx].shape[1] > 0
        for step in preds_all
        for stream_idx in output_streams.values()
    )
    if not has_physical and not has_latents:
        _logger.warning("Writing no data since physical and requested latent outputs are empty.")
        return

    # calculate global sample indices for this batch by offsetting by sample_start
    sample_start = batch_idx * batch_size

    if has_physical:
        # TODO: support multiple input steps
        sources = [
            [stream_data.source_raw[0] for stream_data in sample.streams_data.values()]
            for sample in source_samples
        ]
        sample_idxs = [
            list(sample.streams_data.values())[0].sample_idx for sample in source_samples
        ]
        stream_infos = list(cf.streams.values())
        target_channels = [list(stream.val_target_channels) for stream in stream_infos]
        source_channels = [list(stream.val_source_channels) for stream in stream_infos]
        geoinfo_channels = [[] for _ in stream_infos]  # TODO obtain channels

        twh = TimeWindowHandler(
            val_cfg.start_date,
            val_cfg.end_date,
            val_cfg.time_window_len,
            val_cfg.time_window_step,
        )
        source_windows = (twh.window(idx) for idx in sample_idxs)
        source_intervals = [TimeRange(window.start, window.end) for window in source_windows]

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
            sample_start,
            forecast_offset,
            forecast_steps_override=timestep_idxs_chunk,
        )
    with zarrio_writer(config.get_path_results(cf, mini_epoch)) as zio:
        if has_physical:
            for subset in data.items():
                zio.write_zarr(subset)
        if has_latents:
            # Step 0 is emitted only for a chunk that encoded its input. With
            # offset 0, shift forecast latents to reserve 0 for that initial state.
            latent_steps = [step + int(forecast_offset == 0) for step in timestep_idxs_chunk]
            if model_output.initial_latent is not None:
                latent_steps.insert(0, 0)
            for latent_step, latents_in_step in zip(latent_steps, latents_all, strict=True):
                for sample_idx, latents_in_sample in enumerate(latents_in_step):
                    if latents_in_sample:
                        zio.write_latent(
                            sample_start + sample_idx,
                            latent_step,
                            latents_in_sample,
                            coords=_get_latent_coords(cf, source_samples[sample_idx]),
                            num_register_tokens=int(cf.get("num_register_tokens", 0)),
                            num_class_tokens=int(cf.get("num_class_tokens", 0)),
                        )


@cache
def _get_healpix_coords(healpix_level: int) -> npt.NDArray:
    """Cache latitude/longitude coordinates for each HEALPix level."""
    ipix = np.arange(12 * 4**healpix_level)
    lon, lat = hp.healpix_to_lonlat(ipix, 2**healpix_level, order="nested")
    return np.stack([lat.to_value("deg"), lon.to_value("deg")], axis=1).astype(np.float32)


def _get_latent_coords(cf, sample) -> npt.NDArray | None:
    """Collect HEALPix coordinates and apply the source sample's spatial mask."""
    if "healpix_level" not in cf:
        return None
    coords = _get_healpix_coords(int(cf.healpix_level))
    for meta in sample.meta_info.values():
        if hasattr(meta, "mask") and meta.mask is not None:
            mask = meta.mask.detach().cpu().numpy().astype(bool)
            if mask.shape[0] == coords.shape[0]:
                coords = coords[mask]
            break
    return coords


def get_latent_output(batch, model_output):
    """Collect per-sample arrays, prepending the encoded state when present."""
    n_samples = len(batch.get_source_samples().get_samples())
    latent_preds = [
        model_output.get_latent_prediction(model_output.chunk_idx(step))
        for step in model_output.forecast_steps
        if step >= model_output.forecast_offset
    ]
    if model_output.initial_latent is not None:
        latent_preds.insert(0, {"latent_state": model_output.initial_latent})

    latents_all = []
    for latent_pred in latent_preds:
        tensors = {}
        for name, value in latent_pred.items():
            if name == "posteriors":
                continue
            if isinstance(value, LatentState):
                fields = {
                    "tokens": value.z_pre_norm,
                    "register_tokens": value.register_tokens,
                    "class_token": value.class_token,
                }
            else:
                fields = {name: value}
            tensors.update({key: tensor for key, tensor in fields.items() if tensor is not None})
        per_step = []
        for sample in range(n_samples):
            per_step.append(
                {
                    name: tensor[sample].detach().to(torch.float32).cpu().numpy()
                    for name, tensor in tensors.items()
                }
            )
        latents_all.append(per_step)
    return latents_all
