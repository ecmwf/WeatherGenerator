# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import dataclasses
import logging
import pathlib
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import numpy.typing as npt
import torch

import weathergen.common.config as config
import weathergen.common.io as io
from weathergen.common.io import TimeRange, ZarrWriteConfig, zarrio_writer
from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.model.engines import LatentState
from weathergen.utils.utils import is_stream_reconstructed

_logger = logging.getLogger(__name__)


class _HostCopier:
    """
    Copies tensors to host memory, synchronously or on a separate stream.

    Without a copy stream, a tensor is copied when it is passed in. With one, the call only
    allocates pinned host memory; start() queues all copies on the copy stream behind the
    work queued so far on the current stream, and the host tensors are valid after wait().
    """

    def __init__(self, copy_stream: torch.cuda.Stream | None = None):
        self.copy_stream = copy_stream
        self._pending: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._done: torch.cuda.Event | None = None

    def __call__(self, t: torch.Tensor) -> torch.Tensor | npt.NDArray:
        t = t.detach()
        if self.copy_stream is None or not t.is_cuda:
            return t.cpu().numpy()
        host = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
        self._pending.append((t, host))
        return host

    def start(self) -> None:
        """Queue the copies to host, without waiting for them."""
        if self.copy_stream is None:
            return
        self.copy_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.copy_stream):
            for t, host in self._pending:
                host.copy_(t, non_blocking=True)
                # keep the caching allocator from reusing t before the copy has finished
                t.record_stream(self.copy_stream)
            self._done = torch.cuda.Event()
            self._done.record(self.copy_stream)
        self._pending.clear()

    def wait(self) -> None:
        """Wait until the copies to host have finished."""
        if self._done is not None:
            self._done.synchronize()


def _as_numpy(x):
    """Host tensors to numpy arrays, also inside lists and dicts."""
    if isinstance(x, torch.Tensor):
        return x.numpy()
    if isinstance(x, list):
        return [_as_numpy(v) for v in x]
    if isinstance(x, dict):
        return {k: _as_numpy(v) for k, v in x.items()}
    return x


@dataclasses.dataclass
class StagedOutput:
    """
    Output of a batch (or forecast chunk) whose copies to host are in flight.

    Of the batch, model output and targets it holds only host data that they don't change
    afterwards (the raw sources), so they can be changed or released while it is written.
    """

    host: _HostCopier
    # fstep, stream, sample: data per sample, concatenated over samples when written
    preds: list[list[list]]
    targets: list[list[list]]
    targets_coords: list[list[list]]
    targets_times: list[list[list]]
    latents: list[list[dict]] | None
    # remaining fields of io.OutputBatchData
    batch_data_args: dict
    store_path: pathlib.Path
    write_config: ZarrWriteConfig

    def write(self) -> None:
        """Wait for the copies to host and write the output to the zarr store."""
        self.host.wait()

        def cat(per_step, axis=0):
            return [
                [np.concatenate(_as_numpy(per_sample), axis=axis) for per_sample in per_stream]
                for per_stream in per_step
            ]

        data = io.OutputBatchData(
            targets=cat(self.targets),
            predictions=cat(self.preds, axis=1),
            targets_coords=cat(self.targets_coords),
            targets_times=cat(self.targets_times),
            latents=_as_numpy(self.latents),
            **self.batch_data_args,
        )

        with zarrio_writer(self.store_path, self.write_config) as zio:
            for subset in data.items():
                zio.write_zarr(subset)


class AsyncOutputWriter:
    """
    Writes staged output in a background thread, while the next forecast steps run.

    The copies to host run on a separate stream, so they overlap with the compute on the
    device. One write is in flight at a time, which keeps the writes to the zarr store in
    order and bounds the memory held by staged output.
    """

    def __init__(self, device: torch.device):
        self.copy_stream = torch.cuda.Stream(device) if device.type == "cuda" else None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="output-writer")
        self._in_flight: Future | None = None

    def submit(self, staged: StagedOutput | None) -> None:
        """Write staged output after the previous write has finished."""
        self.wait()
        if staged is not None:
            self._in_flight = self._executor.submit(staged.write)

    def wait(self) -> None:
        """Wait for the write in flight; raises its exception if it failed."""
        in_flight, self._in_flight = self._in_flight, None
        if in_flight is not None:
            in_flight.result()

    def close(self) -> None:
        try:
            self.wait()
        finally:
            self._executor.shutdown()


def _empty_step(n_samples: int, n_ens: int, n_channels: int):
    """Zero-sized target/prediction entries for a step that carries no data."""
    return (
        [np.zeros((n_ens, 0, n_channels), dtype=np.float32) for _ in range(n_samples)],
        [np.zeros((0, n_channels), dtype=np.float32) for _ in range(n_samples)],
        [np.zeros((0, 2), dtype=np.float32) for _ in range(n_samples)],
        [np.array([]).astype("datetime64[ns]") for _ in range(n_samples)],
    )


def _predictions_without_targets(
    batch, sname: str, t_idx: int, preds, n_samples: int, n_channels: int, dn_data, to_host
):
    """
    Predictions of one forecast step and stream at the target points, which have no values.

    The points (raw coords, times and order) come with the target coords of the source
    samples, since inference without targets prepares no target samples.
    """
    fp32 = torch.float32
    n_ens = preds[0].shape[0] if preds is not None and len(preds) > 0 else 1
    samples = batch.get_source_samples().get_samples()
    if preds is None or samples[0].streams_data[sname].target_is_spoof[t_idx]:
        return _empty_step(n_samples, n_ens, n_channels)

    preds_s, targets_s, t_coords_s, t_times_s = [], [], [], []
    for pred, sample in zip(preds, samples, strict=True):
        stream_data = sample.streams_data[sname]
        t_coords = stream_data.target_coords_raw[t_idx]
        t_times = stream_data.target_times_raw[t_idx]

        # invert the reordering of the points by the healpix cell-based embedding
        idxs_inv = stream_data.idxs_inv[t_idx]
        if idxs_inv is not None and len(idxs_inv) > 0:
            pred = pred[:, idxs_inv]
            t_coords = t_coords[idxs_inv]
            t_times = t_times[idxs_inv]

        # copy=True: denormalization is in place and must not change the model output
        preds_s += [to_host(dn_data(sname, pred.to(fp32, copy=True)))]
        targets_s += [np.zeros((0, n_channels), dtype=np.float32)]
        t_coords_s += [to_host(t_coords)]
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
    staged = stage_output(
        cf,
        val_cfg,
        batch_size,
        mini_epoch,
        batch_idx,
        dn_data,
        batch,
        model_output,
        target_aux_out,
    )
    if staged is not None:
        staged.write()


def stage_output(
    cf,
    val_cfg,
    batch_size,
    mini_epoch,
    batch_idx,
    dn_data,
    batch,
    model_output,
    target_aux_out,
    copy_stream: torch.cuda.Stream | None = None,
) -> StagedOutput | None:
    """
    Collect the model output to be written and start its copy to host.

    Everything needed from batch, model_output and target_aux_out is taken here. With
    copy_stream the copies to host are queued on that stream without waiting for them.
    Returns None if there is nothing to write.
    """
    to_host = _HostCopier(copy_stream)

    # TODO: how to handle multiple physical loss terms
    outputs_physical = [
        loss_name
        for i, (loss_name, loss_term) in enumerate(val_cfg.losses.items())
        if loss_term.type == "LossPhysical"
    ]
    assert len(outputs_physical) == 1
    target_aux_out = target_aux_out[outputs_physical[0]]

    # inference without targets: predictions at target points that have no values
    with_targets = val_cfg.get("with_targets", True)

    # collect all target / prediction-related information
    fp32 = torch.float32
    preds_all, targets_all, targets_coords_all, targets_times_all = [], [], [], []

    # _get_output_length clamps to at least one output step, so this always holds
    assert len(batch.get_output_idxs()) > 0, "Batch carries no output steps."
    forecast_offset = batch.get_output_idxs()[0]

    # The chunk's ModelOutput includes a leading padding range [0..forecast_offset) so
    # that slot indices equal global forecast step numbers.  When writing to zarr we must
    # only emit the steps that this chunk actually computed, i.e. steps >= the chunk's own
    # forecast_offset (stored on the ModelOutput), not the batch's global offset.
    chunk_forecast_offset = model_output.forecast_offset
    timestep_idxs = [s for s in model_output.forecast_steps if s >= chunk_forecast_offset]

    n_samples = len(batch.get_source_samples().get_samples())

    # Diffusion inference inflates the model output's fstep dimension to one entry per
    # ODE denoising step (the trajectory). The batch only has the original physical
    # forecast indices, so synthesize a contiguous run of indices starting at the
    # original first index to cover every entry in model_output / target_aux_out.
    n_pred_steps = len(model_output.physical)
    if (
        cf.get("fe_diffusion_model", False)
        and chunk_forecast_offset == 0
        and n_pred_steps > len(timestep_idxs)
    ):
        timestep_idxs = list(range(forecast_offset, forecast_offset + n_pred_steps))

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

            # handle spoof data: do not write since it might corrupt validation (spoofing invisible
            # there)

            # Streams that are not physically reconstructed (forcing, or reconstruct: false
            # JEPA-only targets) have no physical decoder, so there are no predictions to
            # write. They may still carry non-empty targets (used by the teacher), so emit
            # empty per-stream slots to keep the per-stream array alignment used downstream.
            not_reconstructed = not is_stream_reconstructed(cf.streams[sname])

            # leading empty steps of the first chunk carry a source but no target/prediction
            if t_idx < forecast_offset:
                preds_s, targets_s, t_coords_s, t_times_s = _empty_step(n_samples, 1, n_channels)

            elif not with_targets and not not_reconstructed:
                preds_s, targets_s, t_coords_s, t_times_s = _predictions_without_targets(
                    batch,
                    sname,
                    t_idx,
                    model_output.get_physical_prediction(chunk_idx, sname),
                    n_samples,
                    n_channels,
                    dn_data,
                    to_host,
                )

            elif not_reconstructed or target_aux_out.physical[t_idx][sname]["is_spoof"][0]:
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

                    # denormalize data if requested and map to storage format;
                    # copy=True: denormalization is in place and must not change the
                    # model output or the targets, which the loss may still use
                    preds_s += [to_host(dn_data(sname, pred.to(fp32, copy=True)))]
                    targets_s += [to_host(dn_data(sname, target.to(fp32, copy=True)))]

                    # extract original target coords and times from target data
                    t_coords_s += [to_host(t_coords)]
                    t_times_s += [t_times.astype("datetime64[ns]")]

            # datapoints per sample, from the predictions since targets can have no values
            targets_lens[-1] += [[]]
            targets_lens[-1][-1] += [p.shape[1] for p in preds_s]

            # concatenated over samples when written, after the copies to host
            preds_all[-1] += [preds_s]
            targets_all[-1] += [targets_s]
            targets_coords_all[-1] += [t_coords_s]
            targets_times_all[-1] += [t_times_s]

    if sum(n for lens_step in targets_lens for lens in lens_step for n in lens) == 0:
        _logger.warning("Writing no data since predictions are empty.")
        return None

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

    latents_all = get_latent_output(batch, model_output, to_host) if write_latents else None

    to_host.start()

    return StagedOutput(
        host=to_host,
        preds=preds_all,
        targets=targets_all,
        targets_coords=targets_coords_all,
        targets_times=targets_times_all,
        latents=latents_all,
        batch_data_args=dict(
            sources=sources,
            source_intervals=source_intervals,
            targets_lens=targets_lens,
            streams=output_streams,
            target_channels=target_channels,
            source_channels=source_channels,
            geoinfo_channels=geoinfo_channels,
            sample_start=sample_start,
            forecast_offset=forecast_offset,
            forecast_steps=timestep_idxs,
        ),
        store_path=config.get_path_results(cf, mini_epoch),
        write_config=ZarrWriteConfig.from_config(cf),
    )


def get_latent_output(batch, model_output, to_host=None):
    """
    Interface for getting latent states
    """
    to_host = to_host if to_host is not None else _HostCopier()

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
                            per_sample[field_name] = to_host(sample_tensor.to(fp32))
                else:
                    per_sample[lname] = to_host(lval[i_sample].to(fp32))
            latents_all[-1].append(per_sample)

    return latents_all
