# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Residual targets for the physical loss.

Instead of regressing the (normalised) field x(t+k), the model regresses the increment with
respect to the previous step:

    d_k = x_k - x_{k-1}                (in normalised space, same normalisation as the dataset)
    r_k = (d_k - shift_c) / scale_c    (per-channel standardisation of the increment)

The loss (LossPhysical) is unchanged: it simply sees r_k as "target". For output/validation the
full field is rebuilt with a cumulative sum:

    x_hat_k = x_hat_{k-1} + (r_hat_k * scale_c + shift_c),   x_hat_{f-1} := true x_{f-1}

where f is the first output step (forecast.offset), so forecast.offset must be >= 1 and the
sampler must also load the target of step f-1 (data_loading.load_base_target_step: True).

The set of points can differ slightly between steps (e.g. points where the field is undefined
are dropped by the reader). Steps are therefore matched BY COORDINATES. A point without a
counterpart in the previous step gets a NaN residual (to be ignored by the loss); when rebuilding
the full field its missing base state is replaced by 0 (= the normalised mean).

Usage (config of the loss term, in both training and validation/test losses):

    target_and_aux_calc:
      Residual:
        stats_file: /path/to/residual_stats.json   # per-stream, per-channel statistics
        # streams: [ERA5]               # optional: only convert these streams (default: all)
        # center: false                 # also subtract the mean increment (default: false)
        # collect_stats: false          # true => estimate statistics instead of using them
        # max_unmatched_fraction: 0.05  # error if more points than this lack a previous state
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from weathergen.train.target_and_aux_module_base import PhysicalTargetAndAux, TargetAuxOutput

_logger = logging.getLogger(__name__)

_EPS = 1e-6


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _as_index(inv, device):
    return torch.as_tensor(inv, device=device, dtype=torch.long)


def _to_original_order(x: torch.Tensor, inv) -> torch.Tensor:
    # `idxs_inv` is the index such that tokens[idxs_inv] is in the original (dataset) order
    return x if inv is None else x[_as_index(inv, x.device)]


def _coord_keys(xy) -> torch.Tensor:
    """One int64 key per point from (lat, lon), quantised to 1e-4 degrees."""
    xy = xy.detach() if torch.is_tensor(xy) else torch.as_tensor(xy)
    q = torch.round(xy[:, :2].double() * 1e4).long() + 10_000_000
    return q[:, 0] * 100_000_000 + q[:, 1]


def _match_idx(xy_dst, xy_src) -> torch.Tensor | None:
    """
    For every point of `xy_dst`, the row index in `xy_src` with identical coordinates
    (-1 if there is none). Returns None if both have the same points in the same order.
    """
    a = _coord_keys(xy_dst)
    b = _coord_keys(xy_src).to(a.device)
    if a.shape == b.shape and torch.equal(a, b):
        return None
    if b.numel() == 0:
        return torch.full_like(a, -1)
    order = torch.argsort(b)
    b_sorted = b[order]
    pos = torch.searchsorted(b_sorted, a).clamp_(max=b.numel() - 1)
    found = b_sorted[pos] == a
    return torch.where(found, order[pos], torch.full_like(pos, -1))


def _take_rows(src: torch.Tensor, idx: torch.Tensor | None, dim: int, fill: float):
    """Rows of `src` (along `dim`) at `idx`; rows with idx < 0 are set to `fill`."""
    if idx is None:
        return src
    idx = idx.to(src.device)
    valid = idx >= 0
    shape = list(src.shape)
    shape[dim] = idx.numel()
    out = src.new_full(shape, fill)
    sel = [slice(None)] * src.dim()
    sel[dim] = valid
    out[tuple(sel)] = src.index_select(dim, idx[valid])
    return out


class ResidualTargetAndAux(PhysicalTargetAndAux):
    """Physical targets expressed as increments w.r.t. the previous step."""

    def __init__(
        self,
        cf,
        model,
        streams: list[str] | None = None,
        stats_file: str | None = None,
        collect_stats: bool = False,
        center: bool = False,
        max_unmatched_fraction: float = 0.05,
        **kwargs,
    ):
        super().__init__(cf, model)
        self.streams = None if streams is None else list(streams)
        self.stats_file = None if stats_file is None else Path(stats_file)
        self.collect_stats = bool(collect_stats)
        self.center = bool(center)
        self.max_unmatched_fraction = float(max_unmatched_fraction)
        self._warned: set[str] = set()
        self._stats: dict[str, dict] = {}
        self._scale_cache: dict = {}
        self._file_stats = None

        if self.collect_stats:
            if self.stats_file is None:
                raise ValueError("Residual: collect_stats=True requires stats_file (output path).")
        else:
            if self.stats_file is None or not self.stats_file.exists():
                raise FileNotFoundError(
                    f"Residual: stats_file {self.stats_file} not found. Run once with "
                    "collect_stats: true to create it (see run_stats_estimation)."
                )
            self._file_stats = json.loads(self.stats_file.read_text())

    # ---------------------------------------------------------------- statistics
    def _scale_shift(self, stream: str, n_ch: int, like: torch.Tensor):
        if self.collect_stats:
            return like.new_ones(n_ch), like.new_zeros(n_ch)
        key = (stream, like.device, n_ch)
        if key not in self._scale_cache:
            entry = self._file_stats.get(stream)
            if entry is None:
                raise KeyError(f"Residual: no statistics for stream '{stream}' in {self.stats_file}")
            scale = torch.tensor(entry["std" if self.center else "rms"], dtype=like.dtype)
            if scale.numel() != n_ch:
                raise ValueError(
                    f"Residual: stream '{stream}' has {n_ch} target channels but the statistics "
                    f"file has {scale.numel()} (train vs val target channels must match)."
                )
            shift = (
                torch.tensor(entry["mean"], dtype=like.dtype) if self.center else torch.zeros_like(scale)
            )
            self._scale_cache[key] = (scale.clamp_min(_EPS).to(like.device), shift.to(like.device))
        return self._scale_cache[key]

    def _accumulate(self, stream: str, diff: torch.Tensor) -> None:
        d = diff.detach().double()
        ok = torch.isfinite(d)
        d0 = torch.where(ok, d, torch.zeros_like(d))
        st = self._stats.setdefault(stream, {"sum": 0.0, "sumsq": 0.0, "n": 0})
        st["sum"] = st["sum"] + d0.sum(0).cpu()
        st["sumsq"] = st["sumsq"] + (d0 * d0).sum(0).cpu()
        st["n"] = st["n"] + ok.sum(0).cpu()

    def save_stats(self, path: str | Path | None = None) -> None:
        path = Path(path) if path is not None else self.stats_file
        out = {}
        for stream, st in self._stats.items():
            n = st["n"].clamp_min(1).double()
            mean = st["sum"] / n
            out[stream] = {
                "mean": mean.tolist(),
                "std": (st["sumsq"] / n - mean**2).clamp_min(0).sqrt().tolist(),
                "rms": (st["sumsq"] / n).sqrt().tolist(),  # used when center=False
                "n": st["n"].tolist(),
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2))
        _logger.info(f"Residual: wrote statistics for {list(out)} to {path}")

    # ---------------------------------------------------------------- data access
    @staticmethod
    def _gather_step(batch, stream_name: str, step: int):
        """Read target tokens of a step that is not an output step (the step before the first)."""
        tokens, invs, coords = [], [], []
        for sample in batch.samples:
            sd = sample.streams_data[stream_name]
            try:
                tokens.append(sd.target_tokens[step])
                invs.append(sd.idxs_inv[step])
                coords.append(sd.target_coords_raw[step])
            except (IndexError, KeyError, TypeError) as e:
                raise RuntimeError(
                    f"Residual: the target of step {step} (needed as base for step {step + 1}) "
                    f"is not available for stream '{stream_name}'. Use forecast.offset >= 1 and "
                    "data_loading.load_base_target_step: True."
                ) from e
        return tokens, invs, coords

    def _check_unmatched(self, stream: str, step: int, idx) -> None:
        if idx is None:
            return
        frac = (idx < 0).float().mean().item()
        if frac > self.max_unmatched_fraction:
            raise RuntimeError(
                f"Residual: stream '{stream}' step {step}: {frac:.1%} of the points have no "
                f"counterpart at step {step - 1} (max_unmatched_fraction="
                f"{self.max_unmatched_fraction}). The point sets differ too much for residuals."
            )
        if frac > 0 and stream not in self._warned:
            self._warned.add(stream)
            _logger.warning(
                f"Residual: stream '{stream}' step {step}: {frac:.3%} of the points have no "
                f"counterpart at step {step - 1}; their residual is NaN (must be masked by the loss)."
            )

    # ---------------------------------------------------------------- main entry
    def compute(self, bidx, batch, model_params, model) -> TargetAuxOutput:
        targets = super().compute(bidx, batch, model_params, model)
        steps = sorted(targets.output_idxs)

        for stream in list(targets.physical[steps[0]].keys()):
            if self.streams is not None and stream not in self.streams:
                continue
            self._to_residual(batch, targets, stream, steps)
        return targets

    def _to_residual(self, batch, targets: TargetAuxOutput, stream: str, steps: list[int]):
        # original (full-field, normalised) targets of all needed steps
        full = {}
        for step in steps:
            d = targets.physical[step].get(stream)
            if d is None:
                return
            full[step] = (d["target"], d["target_coords"])
        for step in steps:
            if step < 1:
                raise ValueError("Residual targets need forecast.offset >= 1 (no state before 0).")
            if step - 1 not in full:
                tokens, _, coords = self._gather_step(batch, stream, step - 1)
                full[step - 1] = (tokens, coords)

        for step in steps:
            d = targets.physical[step][stream]
            cur, xy_cur = full[step]
            prev, xy_prev = full[step - 1]
            residuals, prev_aligned = [], []
            scale = shift = None

            for i, c in enumerate(cur):
                if c.shape[0] == 0 or d["is_spoof"][i]:
                    residuals.append(c)
                    prev_aligned.append(c)
                    continue
                p = prev[i]
                if p.shape[0] == 0:
                    raise RuntimeError(
                        f"Residual: stream '{stream}': target of step {step - 1} is empty. Set "
                        "data_loading.load_base_target_step: True (and forecast.offset >= 1)."
                    )
                # previous state at the points of the current step, matched by coordinates
                idx = _match_idx(xy_cur[i], xy_prev[i])
                self._check_unmatched(stream, step, idx)
                p_al = _take_rows(p, idx, 0, float("nan"))

                scale, shift = self._scale_shift(stream, c.shape[-1], c)
                diff = c - p_al
                if self.collect_stats:
                    self._accumulate(stream, diff)
                residuals.append((diff - shift) / scale)
                prev_aligned.append(torch.nan_to_num(p_al, nan=0.0))

            d["target_full"] = cur  # true full field (token order)
            d["target_prev_full"] = prev_aligned  # true previous field at this step's points
            d["target"] = residuals  # what the loss sees
            if scale is not None:
                d["residual_scale"] = scale
                d["residual_shift"] = shift


# --------------------------------------------------------------------------------------
# used by validation_io.write_output: residual prediction -> full field
# --------------------------------------------------------------------------------------
def residual_to_full_fields(target_data, i_batch, pred, target, idxs_inv, state, key, coords):
    """
    pred:   [ens, n, C] predicted residual, ALREADY in original point order, float32
    target: [n, C] residual target in original order (only returned unchanged for empty/spoof)
    coords: [n, >=2] (lat, lon) of the points, same order as pred
    state:  dict kept by the caller across forecast steps (holds x_hat_{k-1}); key per sample
    Returns (full-field prediction, full-field target), both normalised and in original order.
    """
    if pred.shape[1] == 0 or target_data["is_spoof"][i_batch]:
        return pred, target
    scale = target_data["residual_scale"].to(pred.device, pred.dtype)
    shift = target_data["residual_shift"].to(pred.device, pred.dtype)
    increment = pred * scale + shift

    prev = state.get(key)
    if prev is None:  # first output step: start from the true previous state
        base = _to_original_order(target_data["target_prev_full"][i_batch], idxs_inv)
        base = base.to(pred.device, pred.dtype).unsqueeze(0)
    else:  # later steps: previous predicted state, matched by coordinates
        prev_pred, prev_xy = prev
        base = _take_rows(prev_pred, _match_idx(coords, prev_xy), 1, 0.0)
    full_pred = base + increment
    # The caller denormalises the returned tensor IN PLACE (dn_data), so the state used by the
    # next step must be a private copy, otherwise it would be denormalised too.
    state[key] = (full_pred.detach().clone(), coords)

    full_target = _to_original_order(target_data["target_full"][i_batch], idxs_inv).clone()
    return full_pred, full_target


# --------------------------------------------------------------------------------------
# one-off statistics estimation (see trainer hook)
# --------------------------------------------------------------------------------------
def run_stats_estimation(trainer, n_batches: int) -> None:
    """Run `n_batches` of the training loader through the calculators in collect_stats mode."""
    from weathergen.train.utils import get_target_idxs_from_cfg
    from weathergen.utils.distributed import is_root

    calcs = {
        name: calc
        for name, calc in trainer.target_and_aux_calculators.items()
        if isinstance(calc, ResidualTargetAndAux) and calc.collect_stats
    }
    if not calcs:
        raise RuntimeError("No Residual calculator with collect_stats: true found.")

    with torch.no_grad():
        for bidx, batch in enumerate(trainer.data_loader):
            if bidx >= n_batches:
                break
            batch.to_device(trainer.device)
            for name, calc in calcs.items():
                idxs = get_target_idxs_from_cfg(trainer.training_cfg, name)
                calc.compute(bidx, batch.get_target_samples(idxs), trainer.model_params, trainer.model)

    if is_root():
        for calc in calcs.values():
            calc.save_stats()