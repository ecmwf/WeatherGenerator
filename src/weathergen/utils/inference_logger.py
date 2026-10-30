# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Shared infrastructure for writing "analytic output" diagnostics: data useful for
inspecting/debugging a model (e.g. the diffusion denoising trajectory) that must not
affect the metrics/artifacts of a normal training or validation run.

All writers share a common lifecycle (:class:`AnalyticsOutputWriter`): gated behind an
explicit config flag (disabled by default), rank-0-only, and written incrementally to
disk rather than rendered inline, for later inspection by standalone scripts. The
concrete writers here are inference-only.
"""


from __future__ import annotations

import abc
import json
import logging
import pathlib
import typing
from collections.abc import Mapping, Sequence

import numpy as np
import torch
import zarr
from zarr.storage import ZipStore

from weathergen.common.config import Config, get_path_run
from weathergen.utils.distributed import is_root
from weathergen.utils.utils import is_stream_reconstructed

logger = logging.getLogger(__name__)


class AnalyticsOutputWriter(abc.ABC):
    """
    Base class for all "dev output" writers.

    Owns the parts that are identical across backends: the enable flag, the
    rank-0-only guard, and the on-disk root directory. Subclasses only need to
    implement :meth:`_write`.
    """

    def __init__(
        self,
        cf: Config,
        name: str,
        enabled: bool,
    ):
        """
        Args:
            cf: Run configuration.
            name: Short identifier for this writer, used as the on-disk subdirectory
                name (e.g. "diffusion_sampling_diagnostics").
            enabled: Whether this writer is active. Callers resolve this from their own
                config flag (e.g. ``inference_logging.fe_diffusion_log_analytics`` or
                ``train_logging.collapse_monitoring.enabled``).
        """
        self.cf = cf
        self.name = name
        self.enabled = enabled
        self.root: pathlib.Path = get_path_run(cf) / "analytics" / name

    def write(self, payload: typing.Any) -> None:
        """Persist ``payload`` to disk. No-op when disabled or not on rank 0."""
        if not self.enabled or not is_root():
            return
        self._write(payload)

    @abc.abstractmethod
    def _write(self, payload: typing.Any) -> None:
        """Backend-specific persistence logic. Only called when enabled and on rank 0."""


class ScalarSeriesJSONLWriter(AnalyticsOutputWriter):
    """
    Appends one JSON line per :meth:`write` call to a ``.jsonl`` file, each line a flat
    ``dict[str, list[float]]`` of the named 1D series recorded for that call (e.g. one
    value per diffusion ODE step). Intended for diagnostics gathered once per "event"
    (e.g. once per diffusion inference call); used for the diffusion sampling diagnostics.
    """

    def __init__(
        self,
        cf: Config,
        name: str,
        enabled: bool,
    ) -> None:
        super().__init__(cf, name, enabled)
        self._store_path = self.root / f"{name}.jsonl"

    def _write(self, payload: Mapping[str, Sequence[float]]) -> None:
        record = {
            # e.g. "l2_to_target" is only populated when a reference token is available
            key: [float(v) for v in values]
            for key, values in payload.items()
            if len(values) > 0
        }
        self.root.mkdir(parents=True, exist_ok=True)
        with self._store_path.open("a") as f:
            f.write(json.dumps(record) + "\n")


class DiffusionStepsZarrWriter(AnalyticsOutputWriter):
    """
    Writes the diffusion denoising trajectory decoded in inference mode, one group per
    validation sample ("sample00000", ...) in a zip-backed zarr store, with an explicit
    ``denoising_step`` array dimension.

    The per-sample ``predictions`` array has shape
    ``(denoising_step, ens, point, channel)``, while ``target``/``coords``/``times`` have
    no ``denoising_step`` axis since the real target is the same physical quantity the
    whole trajectory denoises towards.
    Note the output of this writer is independent of ``write_output``/``OutputBatchData``
    and thus not readable by the ``evaluate`` package
    """

    def __init__(
        self,
        cf: Config,
        enabled: bool,
        name: str,
    ) -> None:
        super().__init__(cf, name, enabled)
        self._store_path = self.root / f"{name}.zip"

    def _write(self, payload: tuple[int, Mapping[str, Mapping[str, typing.Any]]]) -> None:
        sample_idx, streams = payload
        self.root.mkdir(parents=True, exist_ok=True)
        # Appends one group per sample to the zip store; channels must be passed as
        # attributes= at creation since entries in this store can't be rewritten afterwards.
        with ZipStore(str(self._store_path), mode="a") as store:
            sample_group = zarr.group(store=store).create_group(f"sample{sample_idx:05d}")
            for stream_name, data in streams.items():
                stream_group = sample_group.create_group(
                    stream_name, attributes={"channels": list(data["channels"])}
                )
                for key in ("predictions", "target", "coords", "times"):
                    # times are stored as int64 ns-since-epoch since zarr has no datetime64 dtype
                    attrs = {"units": "datetime64[ns] (int64 view)"} if key == "times" else None
                    stream_group.create_array(
                        key, data=np.asarray(data[key]), attributes=attrs
                    )


def write_diffusion_steps_output(
    cf,
    val_cfg,
    batch_size,
    mini_epoch,
    batch_idx,
    dn_data,
    batch,
    model_output,
    target_aux_out,
) -> None:
    """
    Write the diffusion denoising trajectory collected in dev mode
    (``inference_logging.fe_diffusion_log_analytics``).

    ``model_output.diffusion_steps_output`` holds one decoded prediction per ODE step. Unlike
    the official validation/test output, denoising steps are kept as an explicit
    ``denoising_step`` array dimension instead of being conflated with real
    forecast steps; the real target (identical for every denoising step, since it is the
    same physical target the whole trajectory denoises towards) is stored once per
    sample/stream. See :class:`DiffusionStepsZarrWriter` and
    ``scripts/analytics/diffusion/plot_diffusion_steps.py``.
    """
    analytics_output = getattr(model_output, "diffusion_steps_output", None)
    if analytics_output is None or cf.stage != "inference":
        return

    outputs_physical = [
        loss_name
        for loss_name, loss_term in val_cfg.losses.items()
        if loss_term.type == "LossPhysical"
    ]
    assert len(outputs_physical) == 1
    t_aux = target_aux_out[outputs_physical[0]]

    fp32 = torch.float32
    # same leading-padding convention as write_output: only steps >= the trajectory's own
    # forecast_offset were actually decoded, the real target is always the list's last entry
    denoising_step_idxs = [
        s for s in analytics_output.forecast_steps if s >= analytics_output.forecast_offset
    ]
    target_data_all = t_aux.physical[-1]

    writer = DiffusionStepsZarrWriter(
        cf,
        enabled=cf.get("inference_logging", {}).get("fe_diffusion_log_analytics", False),
        name="diffusion_steps",
    )
    n_samples = len(batch.get_source_samples().get_samples())
    for i_sample in range(n_samples):
        streams_payload = {}
        for sname in cf.streams.keys():
            if not is_stream_reconstructed(cf.streams[sname]):
                continue
            target_data = target_data_all[sname]
            if target_data["is_spoof"][i_sample]:
                continue

            target = target_data["target"][i_sample]
            t_coords = target_data["target_coords"][i_sample]
            t_times = target_data["target_times"][i_sample]
            idxs_inv = target_data["idxs_inv"][i_sample]

            preds_per_step = []
            for d_idx in denoising_step_idxs:
                chunk_idx = analytics_output.chunk_idx(d_idx)
                preds = analytics_output.get_physical_prediction(chunk_idx, sname)
                # preds are empty (e.g. forcing streams) so use target with an added ens axis
                pred = preds[i_sample] if preds is not None else target.clone().unsqueeze(0)
                if idxs_inv is not None:
                    pred = pred[:, idxs_inv]
                preds_per_step += [dn_data(sname, pred.to(fp32)).detach().cpu().numpy()]

            if idxs_inv is not None:
                target = target[idxs_inv]
                t_coords = t_coords[idxs_inv]
                t_times = t_times[idxs_inv]

            streams_payload[sname] = {
                "predictions": np.stack(preds_per_step, axis=0),
                "target": dn_data(sname, target.to(fp32)).detach().cpu().numpy(),
                "coords": t_coords.cpu().numpy(),
                "times": t_times.astype("datetime64[ns]").astype("int64"),
                "channels": list(cf.streams[sname].val_target_channels),
            }

        if streams_payload:
            sample_idx = batch_idx * batch_size + i_sample
            writer.write((sample_idx, streams_payload))


def get_diffusion_steps_dir(cf) -> pathlib.Path:
    """Directory holding the diffusion-steps analytics output (mirrors get_path_results)."""
    return get_path_run(cf) / "analytics" / "diffusion_steps"


# Dedicated inference writers invoked alongside write_output, each gated by its own dotted
# config flag. Only initialised/called during an actual inference run never during
# training's periodic validation passes. Add new (flag, writer) entries here.
EXTRA_INFERENCE_WRITERS = [
    ("inference_logging.fe_diffusion_log_analytics", write_diffusion_steps_output),
]
