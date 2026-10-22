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

This module defines the common lifecycle (:class:`DevOutputWriter`) and three concrete
backends used across the codebase:

  * :class:`ScalarSeriesZarrWriter` - appends one group per call ("call00000", ...) to a
    zip-backed zarr store, each holding the named 1D series recorded for that call (e.g.
    one value per diffusion ODE step). Used for the diffusion sampling diagnostics.
  * :class:`DiffusionTrajectoryZarrWriter` - appends one group per validation sample
    ("sample00000", ...) to a zip-backed zarr store, holding the diffusion denoising
    trajectory with an explicit ``denoising_step`` array dimension (as opposed to
    overloading the real forecast-step dimension used by the official validation/test
    output). Read by ``scripts/dev/plot_diffusion_trajectory_dev_eval.py``.
  * :class:`ScalarMetricsJSONLWriter` - appends one JSON line per call, each a flat
    ``dict[str, float]``. Thin adapter around ``TrainLogger.log_metrics``, used e.g. for
    SSL collapse monitoring metrics.

The two zarr-backed writers store to a ``.zip`` file rather than a plain zarr directory:
a directory store creates many small per-chunk files which can exhaust inode quotas on
HPC filesystems over a long run, whereas a zip archive holds the same data in a single
file. Zip archives only support *appending* new entries, not modifying existing ones in
place, so both writers give every call/sample its own freshly named, immutable group
instead of growing a single shared array in place.

All writers here are intentionally independent of any specific model: they only depend
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
from zarr.storage import ZipStore

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
        enabled_flag: str = "fe_diffusion_log_analytics",
    ):
        """
        Args:
            cf: Run configuration.
            name: Short identifier for this writer, used as the on-disk subdirectory
                name (e.g. "diffusion_sampling_diagnostics").
            enabled_flag: Config key gating whether this writer is active, resolved with
                ``OmegaConf.select`` so dotted paths work (e.g.
                ``"train_logging.collapse_monitoring.enabled"``). Defaults to
                ``fe_diffusion_log_analytics``, the shared switch for diffusion dev outputs.
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
    Appends one group per :meth:`write` call to a zip-backed zarr store, each holding the
    named 1D series recorded for that call (e.g. one value per diffusion ODE step).
    Intended for diagnostics gathered once per "event" (e.g. once per diffusion inference
    call), such as per-ODE-step scalar traces.
    """

    def __init__(
        self,
        cf: Config,
        name: str,
        enabled_flag: str = "fe_diffusion_log_analytics",
    ) -> None:
        super().__init__(cf, name, enabled_flag)
        self._store_path = self.root / f"{name}.zip"
        # lazily resolved: how many calls were already written (e.g. by a previous,
        # resumed run) before this process started
        self._next_call_idx: int | None = None

    def _resolve_next_call_idx(self) -> int:
        if not self._store_path.exists():
            return 0
        with ZipStore(str(self._store_path), mode="r") as store:
            group = zarr.open_group(store=store, mode="r")
            existing = [int(k.removeprefix("call")) for k in group.group_keys()]
        return max(existing, default=-1) + 1

    def _write(self, payload: Mapping[str, Sequence[float]]) -> None:
        if self._next_call_idx is None:
            self._next_call_idx = self._resolve_next_call_idx()

        self.root.mkdir(parents=True, exist_ok=True)
        with ZipStore(str(self._store_path), mode="a") as store:
            call_group = zarr.group(store=store).create_group(f"call{self._next_call_idx:05d}")
            for key, values in payload.items():
                if len(values) == 0:
                    # e.g. "l2_to_target" is only populated when a reference token is available
                    continue
                call_group.create_array(key, data=np.asarray(values, dtype=np.float64))
        self._next_call_idx += 1


class DiffusionTrajectoryZarrWriter(DevOutputWriter):
    """
    Writes the diffusion denoising trajectory decoded in dev mode, one group per validation
    sample, with an explicit ``denoising_step`` array dimension.

    Unlike the official validation/test output (which indexes by real forecast step),
    the per-sample ``predictions`` array here has shape ``(denoising_step, ens,
    point, channel)``; ``target``/``coords``/``times`` have no ``denoising_step``
    axis since the real target is the same physical quantity the whole trajectory denoises
    towards.
    Independent of ``write_output``/``OutputBatchData`` and thus not readable by the
    ``evaluate`` package; see ``scripts/dev/plot_diffusion_trajectory_dev_eval.py``.
    """

    def __init__(
        self,
        cf: Config,
        name: str = "diffusion_trajectory",
        enabled_flag: str = "fe_diffusion_log_analytics",
    ) -> None:
        super().__init__(cf, name, enabled_flag)
        self._store_path = self.root / f"{name}.zip"

    def _write(self, payload: tuple[int, Mapping[str, Mapping[str, typing.Any]]]) -> None:
        sample_idx, streams = payload
        self.root.mkdir(parents=True, exist_ok=True)
        # zip archives only support appending; re-writing the same sample_idx within a
        # run (or across resumed runs) would raise since the group would already exist.
        # Attributes must be set at creation time (attributes=...) rather than assigned
        # afterwards: a zip entry can only be written once per open session, and
        # `.attrs[...] = ...` after the fact would try to rewrite the already-written
        # zarr.json, silently producing a duplicate (and ambiguous-on-read) zip entry.
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
        enabled_flag: str = "fe_diffusion_log_analytics",
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
