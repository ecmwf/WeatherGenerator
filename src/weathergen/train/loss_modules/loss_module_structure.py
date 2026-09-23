# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Structure-function (variogram) loss for fine-scale fields.

Grid-free, real-space alternative to a power-spectrum loss. It penalises a mismatch between
the predicted and target *second-order structure function*

    S(r) = E[ ( y(x) - y(x+r) )^p ]   as a function of separation distance r,

estimated from random pairs of target points binned by great-circle distance. Matching S(r)
across distance bins is, via Wiener-Khinchin, equivalent to matching the spatial power
spectrum -- but with no FFT and no regular grid, so it applies directly to scattered
per-point output and needs no gridding. A blurry / over-smoothed prediction has too-small
short-range increments, which this loss detects and penalises directly; MSE (a first-order,
pointwise statistic) cannot.

The objective is a per-distance-bin log-ratio, so the *shape* of S(r) is matched and
short-range (high-frequency) bins are not drowned out by the large-scale bins.

Plug-in pattern mirrors the other loss modules (e.g. LossPhysical): a loss *module* rather
than a plain loss-fct, because it needs the raw point coordinates. Select via the training
config with ``target_and_aux_calc: Physical``.

Expected config (under ``training_config.losses``):

.. code-block:: yaml

    "structure": {
        type: LossStructureFunction,
        weight: 0.1,
        target_and_aux_calc: Physical,
        loss_fcts: {
          "struct": {
            target_stream: <stream>,     # a spatially fine target stream
            channels: ['tp'],            # optional: restrict to these target channels (by name);
                                         # omit for all channels. Exclude circular channels
                                         # (e.g. wind direction) -- their 0/360 wrap yields
                                         # spurious increments under |.|^p.
            num_pairs: 262144,
            bin_edges_km: [10, 25, 50, 100, 200],
            increment_power: 2.0,        # 2 = standard structure function; 1 = robust (heavy tails)
            reduce: mean,                # ensemble handling, see below
          },
        },
      }

``reduce`` (ensemble handling; all identical for ens_size=1):
  - ``mean``    : score the ensemble-mean field. Right for a comparability METRIC; for ens>1
                  TRAINING it re-blurs the very structure this loss should reward.
  - ``median``  : per-point sorted middle (mean of the two central members for even K).
  - ``members`` : per-point sort, then compute the loss for every sorted member and average --
                  pushes realistic spatial texture into each member.
  - int         : a single member index (pre-sort). Meaningful only when members have fixed
                  identities.

Note on scale: choose ``bin_edges_km`` to straddle the scales of interest for the *target*
stream; on a spatially coarse stream the sub-cell scales are unresolved, so prefer a fine one.

Note on ``num_pairs`` (IMPORTANT): pairs are sampled uniformly over points, so their distance
distribution follows the domain's pair-distance density -- short separations are RARE. On a
large domain too few pairs land in the short-distance bins (below ``min_pairs_per_bin``), and
those fine-scale bins -- exactly where blur lives -- are then silently skipped. The default
here is a safe floor; the cost is a few gathers plus elementwise ops, negligible next to the
model forward.

Validation-only usage: the loss calculator skips terms with ``weight: 0`` and the validation
config is merged ON TOP of the training config, so adding this block under
``validation_config.losses`` (with any weight > 0) computes it as a validation-only sharpness
metric without affecting training gradients.
"""

import logging
from collections import defaultdict

import torch
from omegaconf import DictConfig

from weathergen.train.loss_modules.loss_module_base import LossModuleBase, LossValues
from weathergen.utils.train_logger import Stage

_logger = logging.getLogger(__name__)


def _haversine_km(coords: torch.Tensor, idx_i: torch.Tensor, idx_j: torch.Tensor) -> torch.Tensor:
    """Great-circle distance (km) between point pairs given (lat, lon) in degrees.

    Args:
        coords: (N, 2) tensor of (latitude, longitude) in degrees.
        idx_i, idx_j: (P,) long index tensors selecting the two endpoints of each pair.
    Returns:
        (P,) distances in km.
    """
    earth_radius_km = 6371.0
    lat = torch.deg2rad(coords[:, 0])
    lon = torch.deg2rad(coords[:, 1])
    lat1, lat2 = lat[idx_i], lat[idx_j]
    dlat = lat2 - lat1
    dlon = lon[idx_j] - lon[idx_i]
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    return 2.0 * earth_radius_km * torch.asin(torch.sqrt(a.clamp(0.0, 1.0)))


def structure_function_loss(
    target: torch.Tensor,
    pred: torch.Tensor,
    coords: torch.Tensor,
    bin_edges_km: list[float],
    num_pairs: int = 8192,
    increment_power: float = 2.0,
    eps: float = 1e-6,
    min_pairs_per_bin: int = 1,
    generator: torch.Generator | None = None,
) -> torch.Tensor | None:
    """Per-channel grid-free structure-function loss for one (reduced) field.

    Samples ``num_pairs`` random point pairs, bins them by great-circle distance into the bins
    defined by ``bin_edges_km``, and matches the predicted vs target mean increment per bin in
    log space: loss = mean_{bin, channel} ( log S_pred - log S_target )^2.

    Args:
        target: (N, C) target values (already restricted to valid finite points).
        pred:   (N, C) predicted values (single field; ensemble already reduced by the caller).
        coords: (N, 2) (lat, lon) in degrees for the same N points.
        bin_edges_km: ascending list of B+1 edges; bins are (e0, e1], ..., (e_{B-1}, e_B].
        num_pairs: number of random pairs to sample.
        increment_power: p in |y_i - y_j|^p (2 = standard; 1 = robust for heavy tails).
        eps: stabiliser inside the log.
        min_pairs_per_bin: skip bins with fewer surviving pairs than this.
        generator: optional RNG for reproducible pair sampling.
    Returns:
        (C,) per-channel loss, or None if too few points/pairs to form any bin.
    """
    n = target.shape[0]
    if n < 2:
        return None

    device = target.device
    channels = target.shape[1]
    num_bins = len(bin_edges_km) - 1
    idx_i = torch.randint(0, n, (num_pairs,), device=device, generator=generator)
    idx_j = torch.randint(0, n, (num_pairs,), device=device, generator=generator)

    # Assign every pair to its distance bin in one pass. bucketize (right=False) returns the
    # first edge >= dist, i.e. bin b spans (edges[b], edges[b+1]]; dist <= edges[0] -> -1 and
    # dist > edges[-1] -> num_bins are both out of range and dropped by the mask below.
    dist = _haversine_km(coords, idx_i, idx_j)
    edges = torch.as_tensor(bin_edges_km, device=device, dtype=dist.dtype)
    bin_id = torch.bucketize(dist, edges) - 1
    valid = (bin_id >= 0) & (bin_id < num_bins) & (idx_i != idx_j)

    idx_i, idx_j, bin_id = idx_i[valid], idx_j[valid], bin_id[valid]

    # increments (target side carries no gradient; pred side does, via the index gather)
    inc_t = (target[idx_i] - target[idx_j]).abs().pow(increment_power)  # [P, C]
    inc_p = (pred[idx_i] - pred[idx_j]).abs().pow(increment_power)  # [P, C]

    # Per-bin mean increment via scatter-add (no Python loop, no per-bin host sync).
    idx_bc = bin_id.unsqueeze(1).expand(-1, channels)  # [P, C]
    counts = torch.zeros(num_bins, device=device, dtype=inc_t.dtype)
    counts.scatter_add_(0, bin_id, torch.ones_like(bin_id, dtype=inc_t.dtype))
    sum_t = torch.zeros(num_bins, channels, device=device, dtype=inc_t.dtype)
    sum_p = torch.zeros(num_bins, channels, device=device, dtype=inc_p.dtype)
    sum_t.scatter_add_(0, idx_bc, inc_t)
    sum_p.scatter_add_(0, idx_bc, inc_p)

    denom = counts.clamp(min=1.0).unsqueeze(1)  # [B, 1]
    per_bin = (torch.log(sum_p / denom + eps) - torch.log(sum_t / denom + eps)).pow(2)  # [B, C]

    # equal-weight mean over bins that met min_pairs_per_bin (masked bins contribute no gradient)
    bin_ok = (counts >= min_pairs_per_bin).unsqueeze(1)  # [B, 1]
    n_ok = bin_ok.sum()
    if int(n_ok) == 0:
        return None
    return (per_bin * bin_ok).sum(0) / n_ok  # [C]


class LossStructureFunction(LossModuleBase):
    """Grid-free structure-function (variogram) loss module. See module docstring."""

    def __init__(
        self,
        cf: DictConfig,
        mode_cfg: DictConfig,
        stage: Stage,
        device: str,
        **loss_fcts,
    ):
        LossModuleBase.__init__(self)
        self.cf = cf
        self.mode_cfg = mode_cfg
        self.stage = stage
        self.device = device
        self.name = "LossStructureFunction"

        # exactly one config key (e.g. "struct")
        cfg_key = next(iter(loss_fcts.keys()))
        cfg = loss_fcts[cfg_key]

        self.target_stream: str = cfg["target_stream"]
        # optional list of target-channel NAMES to restrict to (e.g. ['tp']); None = all
        self.channels: list[str] | None = (
            [str(c) for c in cfg["channels"]] if cfg.get("channels") else None
        )
        self._channel_idx: list[int] | None = None  # resolved lazily from stream channel names
        # see module docstring: uniform pair sampling starves the short-distance bins on large
        # domains, so the default is deliberately high (cost is negligible)
        self.num_pairs: int = int(cfg.get("num_pairs", 262144))
        self.min_pairs_per_bin: int = int(cfg.get("min_pairs_per_bin", 8))
        default_bins = [10, 25, 50, 100, 200]
        self.bin_edges_km: list[float] = [float(x) for x in cfg.get("bin_edges_km", default_bins)]
        self.increment_power: float = float(cfg.get("increment_power", 2.0))
        self.eps: float = float(cfg.get("eps", 1e-6))
        red = cfg.get("reduce", "mean")
        self.reduce = red if red in ("mean", "median", "members") else int(red)

        _logger.info(
            "LossStructureFunction: stream=%s, channels=%s, bins(km)=%s, num_pairs=%d, "
            "power=%.1f, reduce=%s",
            self.target_stream,
            self.channels if self.channels is not None else "all",
            self.bin_edges_km,
            self.num_pairs,
            self.increment_power,
            self.reduce,
        )

    def _resolve_channel_idx(self) -> list[int] | None:
        """Map configured channel names to column indices of the target stream (cached)."""
        if self.channels is None:
            return None
        if self._channel_idx is None:
            stream_info = self.cf.streams[self.target_stream]
            names = list(
                stream_info.val_target_channels
                if self.stage == "val"
                else stream_info.train_target_channels
            )
            self._channel_idx = [names.index(c) for c in self.channels]
        return self._channel_idx

    @staticmethod
    def ensemble_fields(pred: torch.Tensor, reduce) -> list[torch.Tensor]:
        """Resolve the ensemble dim into the field(s) the SF loss scores.

        pred: (ens, N, C). Returns a list of (N, C) fields; the loss is averaged over them.
        For ens_size=1 every mode returns the single member. See the module docstring for the
        semantics of "mean" / "median" / "members" / int.
        """
        k = pred.shape[0]
        if k == 1:
            return [pred[0]]
        if reduce == "mean":
            return [pred.mean(0)]
        if reduce == "median":
            srt = pred.sort(0).values
            if k % 2 == 0:
                return [srt[k // 2 - 1 : k // 2 + 1].mean(0)]
            return [srt[k // 2]]
        if reduce == "members":
            srt = pred.sort(0).values
            return [srt[m] for m in range(k)]
        return [pred[reduce]]

    def compute_loss(self, preds, targets, metadata) -> LossValues:
        _source2target_idxs, output_info, _target2source_idxs, _target_info = metadata

        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        per_step = {}
        ctr_steps = 0
        stream_name = self.target_stream

        for timestep_idx, (preds_cur, target_cur) in enumerate(
            zip(preds.physical, targets.physical, strict=True)
        ):
            if stream_name not in target_cur:
                continue
            preds_batch = preds_cur.get(stream_name, [])
            if not preds_batch:
                continue

            targets_batch = target_cur[stream_name]["target"]
            coords_batch = target_cur[stream_name]["target_coords"]
            targets_params = target_cur[stream_name]["target_metda_data"]
            targets_is_spoof = target_cur[stream_name]["is_spoof"]

            loss_ts = torch.tensor(0.0, device=self.device, requires_grad=True)
            ctr_b = 0
            for pred, pred_params in zip(preds_batch, output_info, strict=True):
                target_idx_native = pred_params.global_params.get("correspondence", -1)
                target_idx = [
                    i
                    for i, t in enumerate(targets_params)
                    if t[stream_name].global_params["idx"] == target_idx_native
                ]
                if len(target_idx) == 0:
                    continue
                assert len(target_idx) == 1
                target_idx = target_idx[0]

                if targets_is_spoof[target_idx]:
                    continue

                target = targets_batch[target_idx]
                coords = coords_batch[target_idx]
                if target.shape[0] == 0 or pred.shape[0] == 0:
                    continue

                pred = pred.reshape([pred.shape[0], *target.shape])  # [ens, N, C]
                fields = self.ensemble_fields(pred, self.reduce)  # list of [N, C]

                # target_coords_raw is produced on CPU by the dataloader; move it first
                coords = coords.to(target.device, non_blocking=True)
                valid = torch.isfinite(target).all(-1) & torch.isfinite(coords).all(-1)
                if int(valid.sum()) < 2:
                    continue

                target_sel = target[valid]
                ch_idx = self._resolve_channel_idx()
                if ch_idx is not None:
                    target_sel = target_sel[:, ch_idx]

                # average the SF loss over the resolved field(s); one independent pair sample
                # per field keeps the same-pair pred/target cancellation within each call
                loss_fields = []
                for field in fields:
                    pred_sel = field[valid]
                    if ch_idx is not None:
                        pred_sel = pred_sel[:, ch_idx]
                    loss_ch = structure_function_loss(
                        target_sel,
                        pred_sel,
                        coords[valid],
                        bin_edges_km=self.bin_edges_km,
                        num_pairs=self.num_pairs,
                        increment_power=self.increment_power,
                        eps=self.eps,
                        min_pairs_per_bin=self.min_pairs_per_bin,
                    )
                    if loss_ch is not None:
                        loss_fields.append(loss_ch.mean())
                if not loss_fields:
                    continue

                loss_ts = loss_ts + torch.stack(loss_fields).mean()
                ctr_b += 1

            if ctr_b > 0:
                loss = loss + loss_ts / ctr_b
                per_step[str(timestep_idx)] = (loss_ts / ctr_b).detach()
                ctr_steps += 1

        if ctr_steps > 0:
            loss = loss / ctr_steps
        elif _logger.isEnabledFor(logging.WARNING):
            _logger.warning(
                "LossStructureFunction: no data for stream '%s' in this batch.", stream_name
            )

        # log layout mirrors LossPhysical: [stream][loss_fct][metric][step]
        reordered = defaultdict(dict)
        reordered[stream_name] = defaultdict(lambda: defaultdict(dict))
        for step_str, val in per_step.items():
            reordered[stream_name]["structure"]["loss"][step_str] = val
        if "structure" in reordered[stream_name]:
            reordered[stream_name]["structure"]["avg"] = loss.detach()

        return LossValues(loss=loss, losses_all=reordered, stddev_all=None)
