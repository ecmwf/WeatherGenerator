# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Shared infrastructure for writing *development* diagnostics to disk.

"Dev outputs" are data that are useful to inspect while developing/debugging a model
(e.g. the diffusion denoising trajectory, or SSL collapse indicators) but that must not
affect the metrics/artifacts produced by a normal training or validation run. They are:

  * gated behind an explicit config flag (disabled by default),
  * written incrementally to disk instead of rendered inline (keeps model/training code
    free of plotting backends and keeps the hot path cheap), and
  * read and visualised later by small, standalone scripts.

This module defines the common lifecycle (:class:`DevOutputWriter`) and two concrete
backends used across the codebase:

  * :class:`ScalarSeriesZarrWriter` - appends one row per call to a set of named 1D
    series of equal length (e.g. one value per diffusion ODE step). Used for the
    diffusion sampling diagnostics.
  * :class:`ScalarMetricsJSONLWriter` - appends one JSON line per call, each a flat
    ``dict[str, float]``. Thin adapter around ``TrainLogger.log_metrics``, used e.g. for
    SSL collapse monitoring metrics.

A third style of dev output - the diffusion dev-mode trajectory - does not go through a
writer in this module: it reuses ``weathergen.utils.validation_io.write_output`` directly
(with an overridden store path) so it is written with the exact same schema as the
official validation/test output and can be read by the ``evaluate`` package unmodified;
see ``write_dev_trajectory_output``.

Both writers here are intentionally independent of any specific model: they only depend
on `Config` and plain arrays/dicts, so they can be unit tested and reused outside of
model.py / diffusion.py.
"""

from __future__ import annotations

import abc
import logging
import typing
from collections.abc import Mapping, Sequence

import numpy as np
import zarr
from omegaconf import OmegaConf

from weathergen.common.config import Config, get_path_run
from weathergen.utils.distributed import is_root

if typing.TYPE_CHECKING:
    import pathlib

logger = logging.getLogger(__name__)


class DevOutputWriter(abc.ABC):
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
        enabled_flag: str = "fe_diffusion_dev_mode",
    ):
        """
        Args:
            cf: Run configuration.
            name: Short identifier for this writer, used as the on-disk subdirectory
                name (e.g. "diffusion_sampling_diagnostics").
            enabled_flag: Config key gating whether this writer is active, resolved with
                ``OmegaConf.select`` so dotted paths work (e.g.
                ``"train_logging.collapse_monitoring.enabled"``). Defaults to
                ``fe_diffusion_dev_mode``, the shared switch for diffusion dev outputs.
        """
        self.cf = cf
        self.name = name
        self.enabled = bool(OmegaConf.select(cf, enabled_flag, default=False))
        self.root: pathlib.Path = get_path_run(cf) / "dev" / name

    def write(self, payload: typing.Any) -> None:
        """Persist ``payload`` to disk. No-op when disabled or not on rank 0."""
        if not self.enabled or not is_root():
            return
        self._write(payload)

    @abc.abstractmethod
    def _write(self, payload: typing.Any) -> None:
        """Backend-specific persistence logic. Only called when enabled and on rank 0."""


class ScalarSeriesZarrWriter(DevOutputWriter):
    """
    Appends one row per :meth:`write` call to a zarr group of equal-length 1D series.

    Each key in the payload dict becomes a 2D zarr array of shape
    ``(num_calls, series_length)`` that grows by one row per call. Intended for
    diagnostics gathered once per "event" (e.g. once per diffusion inference call),
    such as per-ODE-step scalar traces.
    """

    def __init__(
        self,
        cf: Config,
        name: str,
        enabled_flag: str = "fe_diffusion_dev_mode",
    ) -> None:
        super().__init__(cf, name, enabled_flag)
        self._group: zarr.Group | None = None

    def _open(self) -> zarr.Group:
        if self._group is None:
            self.root.mkdir(parents=True, exist_ok=True)
            self._group = zarr.open_group(str(self.root / f"{self.name}.zarr"), mode="a")
        return self._group

    def _write(self, payload: Mapping[str, Sequence[float]]) -> None:
        group = self._open()
        # Length shared by all series recorded this call (e.g. num_ode_steps). Used to
        # NaN-pad series that are conditionally empty this call (e.g. "l2_to_target" is only
        # populated when a reference token is available) so every key still gets exactly one
        # row per call and row index == call index stays aligned across all keys.
        non_empty_lengths = {len(v) for v in payload.values() if len(v) > 0}
        ref_length = max(non_empty_lengths) if non_empty_lengths else 0

        for key, values in payload.items():
            if len(values) == 0:
                if ref_length == 0:
                    # nothing recorded at all this call; nothing to align against
                    continue
                row = np.full(ref_length, np.nan, dtype=np.float64)
            else:
                row = np.asarray(values, dtype=np.float64)
            if key not in group:
                group.create_array(
                    key, shape=(0, row.shape[0]), chunks=(1, row.shape[0]), dtype=row.dtype
                )
            dset = group[key]
            if dset.shape[1] != row.shape[0]:
                logger.warning(
                    "Skipping dev output series '%s': expected length %d, got %d.",
                    key,
                    dset.shape[1],
                    row.shape[0],
                )
                continue
            dset.resize((dset.shape[0] + 1, dset.shape[1]))
            dset[-1, :] = row


class ScalarMetricsJSONLWriter(DevOutputWriter):
    """
    Appends one JSON line per :meth:`write` call, each a flat ``dict[str, float]``.

    This is a thin :class:`DevOutputWriter` adapter around the append-only metrics file
    already used for scalar training/validation metrics (see
    ``weathergen.utils.train_logger.TrainLogger.log_metrics``), so that scalar dev metrics -
    e.g. SSL collapse indicators - follow the same enable-flag + rank-0-only contract as the
    other dev output writers, without changing how ``CollapseMonitor`` itself computes or
    caches metrics.
    """

    def __init__(
        self,
        cf: Config,
        name: str,
        log_fn,
        enabled_flag: str = "fe_diffusion_dev_mode",
    ) -> None:
        """
        Args:
            log_fn: Callable with the same signature as
                ``TrainLogger.log_metrics(stage, metrics, step=None)``.
        """
        super().__init__(cf, name, enabled_flag)
        self._log_fn = log_fn

    def _write(self, payload: tuple[str, Mapping[str, float]]) -> None:
        stage, metrics = payload
        self._log_fn(stage, metrics)
