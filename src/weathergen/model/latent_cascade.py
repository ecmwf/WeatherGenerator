# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Latent cascade: information exchange BETWEEN the levels of the multi-resolution latent.

Without it the per-level latents are encoded (and forecast) independently. The cascade
couples adjacent levels multigrid-style; one cycle (V-cycle) is

    down passes, coarsest pair first:  every fine cell is updated from the coarse level
    up passes,   finest pair first:    every coarse cell is updated from the fine level

The geometry (which cells exchange information) comes from the exact HEALPix nested
maps of the DomainPyramid:

    down (coarse -> fine): the keys/values of a fine cell are the coarse cells within
                           `down_rings` rings of its parent (0 = the parent alone,
                           1 = parent + 8 neighbours, 2 = up to 25 cells, ...);
    up   (fine -> coarse): the keys/values of a coarse cell are its active children.

Cells outside the other level's domain are left out of the key/value set; a cell with
no key at all (e.g. a coarse cell far from a regional fine level) is left unchanged.

Operator "attention" (default): each pass is `num_blocks` pre-norm transformer
cross-attention blocks,

    x <- x + Attention(LN(x), LN(sources))      (multi-head, qk-norm, varlen flash attention)
    x <- x + MLP(LN(x))                         (Linear -> GELU -> Linear)

with the target cell's `num_queries` tokens as queries and all tokens of its source cells
as keys/values. The blocks reuse the model's MultiCrossAttentionHeadVarlen and MLP.
With down_rings = 0 a fine cell has a single key, so the attention weights are all 1 and
the attention reduces to a linear map of the parent; use down_rings >= 1 for a real
attention in the down pass.

Operators "linear" / "none" (kept for comparison with the original branch) use the parent
alone (down) and the mean of the children (up), with a zero-initialised Linear ("linear")
or a plain addition ("none", not the identity).

Only the spatial cell tokens take part; the register/class tokens are passed through.
With a single-level pyramid there are no pairs and the cascade is a no-op.
"""

import logging
import typing

import numpy as np
import torch
from numpy.typing import NDArray

from weathergen.model.attention import MultiCrossAttentionHeadVarlen
from weathergen.model.layers import MLP
from weathergen.utils.utils import get_dtype

_logger = logging.getLogger(__name__)


class CascadeBlock(torch.nn.Module):
    """
    Pre-norm transformer cross-attention block used by the cascade:

        x <- x + Attention(LN(x), LN(x_kv))
        x <- x + MLP(LN(x))

    Inputs are packed variable-length sequences (flash-attention varlen layout): segment i
    of the queries attends to segment i of the keys/values only.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_factor: float = 2.0,
        dropout_rate: float = 0.0,
        with_qk_lnorm: bool = True,
        norm_type: str = "LayerNorm",
        qk_norm_type: str | None = None,
        norm_eps: float = 1e-5,
        mlp_norm_eps: float = 1e-5,
        attention_dtype=torch.bfloat16,
        with_flash: bool = True,
        zero_init: bool = False,
    ):
        super().__init__()
        self.zero_init = zero_init
        self.attention = MultiCrossAttentionHeadVarlen(
            dim,
            dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            with_residual=True,
            with_qk_lnorm=with_qk_lnorm,
            with_flash=with_flash,
            norm_type=norm_type,
            qk_norm_type=qk_norm_type,
            norm_eps=norm_eps,
            attention_dtype=attention_dtype,
        )
        self.mlp = MLP(
            dim,
            dim,
            num_layers=2,
            hidden_factor=mlp_hidden_factor,
            pre_layer_norm=True,
            dropout_rate=dropout_rate,
            with_residual=True,
            norm_type=norm_type,
            norm_eps=mlp_norm_eps,
        )
        self.init_residual_branches()

    def init_residual_branches(self) -> None:
        """With zero_init, zero the last projection of both residual branches so that the
        block starts as the identity."""
        if self.zero_init:
            torch.nn.init.zeros_(self.attention.proj_out.weight)
            last = self.mlp.layers[-1]
            torch.nn.init.zeros_(last.weight)
            if last.bias is not None:
                torch.nn.init.zeros_(last.bias)

    def forward(self, x, x_kv, x_lens, x_kv_lens):
        x = self.attention(x, x_kv, x_lens, x_kv_lens)
        return self.mlp(x)


class _PairOperator(torch.nn.Module):
    """
    "none" / "linear" operators of one (coarse, fine) pair, on pre-reduced sources:
    down: the parent's latent copied to each fine cell; up: the mean of the active children.
    """

    def __init__(self, dim: int, kind: str = "none"):
        super().__init__()
        self.kind = kind
        if kind == "none":
            self.down_proj = None
            self.up_proj = None
        elif kind == "linear":
            self.down_proj = torch.nn.Linear(dim, dim, bias=False)
            self.up_proj = torch.nn.Linear(dim, dim, bias=False)
            self.init_residual_branches()
        else:
            raise ValueError(f"unknown cascade operator kind={kind!r}")

    def init_residual_branches(self) -> None:
        if self.kind == "linear":
            torch.nn.init.zeros_(self.down_proj.weight)
            torch.nn.init.zeros_(self.up_proj.weight)

    def down(self, source):
        return source if self.kind == "none" else self.down_proj(source)

    def up(self, source):
        return source if self.kind == "none" else self.up_proj(source)


class LatentCascade(torch.nn.Module):
    """
    Runs `num_cycles` V-cycles of coarse <-> fine exchange across all adjacent level pairs.

    Args:
        domain_pyramid    : provides the exact parent/child maps and the ring tables.
        dim               : latent feature dim (ae_global_dim_embed).
        num_queries       : ae_local_num_queries (tokens per cell).
        num_aux           : number of prepended register+class tokens per sample.
        num_cycles        : number of V-cycles; 0 => inert.
        operator          : "attention" | "linear" | "none".
        down_rings        : attention down pass: rings of coarse cells around the parent.
        num_blocks        : attention blocks per pass (and per level pair).
        num_heads, mlp_hidden_factor, dropout_rate, with_qk_lnorm, norm_type, qk_norm_type,
        norm_eps (attention LayerNorms), mlp_norm_eps (MLP LayerNorm),
        attention_dtype, with_flash : attention block settings.
        zero_init         : attention blocks start as the identity (last projection of
                            both residual branches zero-initialised).
    """

    def __init__(
        self,
        domain_pyramid,
        dim: int,
        num_queries: int,
        num_aux: int,
        num_cycles: int = 0,
        operator: str = "attention",
        down_rings: int = 1,
        num_blocks: int = 1,
        num_heads: int = 8,
        mlp_hidden_factor: float = 2.0,
        dropout_rate: float = 0.0,
        with_qk_lnorm: bool = True,
        norm_type: str = "LayerNorm",
        qk_norm_type: str | None = None,
        norm_eps: float = 1e-5,
        mlp_norm_eps: float = 1e-5,
        attention_dtype=torch.bfloat16,
        with_flash: bool = True,
        zero_init: bool = False,
    ):
        super().__init__()
        assert operator in ("attention", "linear", "none"), f"unknown operator {operator!r}"
        assert down_rings >= 0 and num_blocks >= 1
        self.pyramid = domain_pyramid
        self.dim = dim
        self.num_queries = num_queries
        self.num_aux = num_aux
        self.num_cycles = num_cycles
        self.operator_kind = operator
        self.down_rings = down_rings
        self.levels = domain_pyramid.levels

        # adjacent (coarse, fine) pairs, coarse -> fine order
        self.pairs = list(zip(self.levels[:-1], self.levels[1:], strict=True))

        self._ops = torch.nn.ModuleDict()
        if not self.is_inert():
            for lc, lf in self.pairs:
                key = f"{lc}_{lf}"
                if operator == "attention":

                    def blocks():
                        return torch.nn.ModuleList(
                            CascadeBlock(
                                dim,
                                num_heads,
                                mlp_hidden_factor=mlp_hidden_factor,
                                dropout_rate=dropout_rate,
                                with_qk_lnorm=with_qk_lnorm,
                                norm_type=norm_type,
                                qk_norm_type=qk_norm_type,
                                norm_eps=norm_eps,
                                mlp_norm_eps=mlp_norm_eps,
                                attention_dtype=attention_dtype,
                                with_flash=with_flash,
                                zero_init=zero_init,
                            )
                            for _ in range(num_blocks)
                        )

                    self._ops[f"{key}_down"] = blocks()
                    self._ops[f"{key}_up"] = blocks()
                else:
                    self._ops[key] = _PairOperator(dim, operator)

            # geometry as non-persistent buffers (move with .to(device), not checkpointed)
            for name, arr in self._index_arrays().items():
                self.register_buffer(name, torch.as_tensor(arr), persistent=False)
            self._log_connections()

    @classmethod
    def from_config(cls, domain_pyramid, cf, key: str) -> "LatentCascade":
        """Build the cascade configured under `cf[key]` (latent_cascade / forecast_cascade)."""
        c = cf.get(key, None) or {}
        return cls(
            domain_pyramid,
            dim=cf.ae_global_dim_embed,
            num_queries=cf.ae_local_num_queries,
            num_aux=cf.num_register_tokens + cf.num_class_tokens,
            num_cycles=c.get("num_cycles", 0),
            operator=c.get("operator", "attention"),
            down_rings=c.get("down_rings", 1),
            num_blocks=c.get("num_blocks", 1),
            num_heads=c.get("num_heads", 8),
            mlp_hidden_factor=c.get("mlp_hidden_factor", 2.0),
            dropout_rate=c.get("dropout_rate", 0.0),
            with_qk_lnorm=c.get("with_qk_lnorm", True),
            norm_type=cf.get("norm_type", "LayerNorm"),
            qk_norm_type=cf.get("qk_norm_type", None),
            norm_eps=cf.get("norm_eps", 1e-5),
            mlp_norm_eps=cf.get("mlp_norm_eps", 1e-5),
            attention_dtype=get_dtype(cf.get("attention_dtype", "bf16")),
            with_flash=cf.get("with_flash_attention", True),
            zero_init=c.get("zero_init", False),
        )

    # -- geometry ----------------------------------------------------------
    def _index_arrays(self) -> dict[str, NDArray]:
        """
        Index buffers of every pass. For the attention operator, a pass is described by
            tgt : (T,)  compact indices of the target cells that have >= 1 source
            src : (M,)  compact source indices, grouped by target in the order of `tgt`
            cnt : (T,)  number of sources of each target (sum = M)
        """
        out = {}
        for lc, lf in self.pairs:
            key = f"{lc}_{lf}"
            poc = self.pyramid.parent_of_child(lc, lf)  # (n_fine,)
            cop = self.pyramid.child_of_parent(lc, lf)  # (n_coarse, 4**(lf-lc))
            cv = self.pyramid.child_valid(lc, lf)
            if self.operator_kind == "attention":
                rings = self.pyramid.rings(lc, self.down_rings)  # (n_coarse, K)
                src = rings[np.maximum(poc, 0)]  # (n_fine, K)
                valid = (src >= 0) & (poc >= 0)[:, None]
                passes = {"down": (src, valid), "up": (cop, cv)}
                for name, (src_tab, valid_tab) in passes.items():
                    has_src = valid_tab.any(axis=1)
                    out[f"{name}_tgt_{key}"] = np.flatnonzero(has_src).astype(np.int64)
                    out[f"{name}_src_{key}"] = src_tab[valid_tab].astype(np.int64)
                    out[f"{name}_cnt_{key}"] = valid_tab.sum(axis=1)[has_src].astype(np.int64)
            else:
                out[f"poc_{key}"] = poc.astype(np.int64)
                out[f"cop_{key}"] = cop.astype(np.int64)
                out[f"cv_{key}"] = cv.astype(bool)
        return out

    def _log_connections(self) -> None:
        if self.operator_kind != "attention":
            return
        for lc, lf in self.pairs:
            key = f"{lc}_{lf}"
            for name in ("down", "up"):
                cnt = getattr(self, f"{name}_cnt_{key}")
                if cnt.numel() and cnt.device.type != "meta":
                    _logger.info(
                        "LatentCascade hl%d<->hl%d %s: %d target cells, keys per cell %d-%d",
                        lc,
                        lf,
                        name,
                        cnt.numel(),
                        int(cnt.min()),
                        int(cnt.max()),
                    )

    def rebuild_index_buffers(self, device) -> None:
        """
        Recreate the index buffers on `device` from the pyramid. Needed after meta-device
        initialisation (FSDP): the buffers are non-persistent, so neither the checkpoint nor
        to_empty / reset_parameters restores them.
        """
        if self.is_inert():
            return
        for name, arr in self._index_arrays().items():
            setattr(self, name, torch.as_tensor(arr, device=device))

    def init_residual_branches(self) -> None:
        """Re-apply the zero initialisations (Model.reset_parameters resets every Linear)."""
        for module in self.modules():
            if isinstance(module, CascadeBlock | _PairOperator):
                module.init_residual_branches()

    # -- layout helpers ----------------------------------------------------
    def _split(self, latent):
        """(B, (num_aux+n_cells)*q, D) -> aux (B, num_aux*q, D), cells (B, n_cells, q, D)."""
        b, _, d = latent.shape
        aux_t = self.num_aux * self.num_queries
        aux = latent[:, :aux_t, :]
        cells = latent[:, aux_t:, :].reshape(b, -1, self.num_queries, d)
        return aux, cells

    def _merge(self, aux, cells):
        b = cells.shape[0]
        d = cells.shape[-1]
        return torch.cat([aux, cells.reshape(b, -1, d)], dim=1)

    def is_inert(self) -> bool:
        return self.num_cycles == 0 or len(self.pairs) == 0

    # -- exchange ----------------------------------------------------------
    def _attend(self, blocks, target, source, tgt, src, cnt):
        """
        Update the target cells listed in `tgt` with the attention blocks: the queries of
        target cell tgt[i] are its num_queries tokens, its keys/values the tokens of the
        source cells src[sum(cnt[:i]) : sum(cnt[:i+1])]. Other target cells are unchanged.
        """
        b, _, nq, d = target.shape
        num_tgt = tgt.shape[0]
        x = target[:, tgt].reshape(b * num_tgt * nq, d)
        x_kv = source[:, src].reshape(-1, d)
        zero = torch.zeros(1, dtype=torch.int32, device=target.device)
        x_lens = torch.cat(
            [zero, torch.full((b * num_tgt,), nq, dtype=torch.int32, device=zero.device)]
        )
        x_kv_lens = torch.cat([zero, (cnt * nq).to(torch.int32).repeat(b)])
        for block in blocks:
            x = block(x, x_kv, x_lens, x_kv_lens)
        out = target.clone()
        out[:, tgt] = x.reshape(b, num_tgt, nq, d).to(out.dtype)
        return out

    def forward(self, tokens_by_level: dict) -> dict:
        """
        tokens_by_level: {level: latent (B, (num_aux+n_cells)*q, D)}.
        Returns an updated dict (same shapes). Inert -> returns input unchanged.
        """
        if self.is_inert():
            return tokens_by_level

        aux, cells = {}, {}
        for lvl, lat in tokens_by_level.items():
            aux[lvl], cells[lvl] = self._split(lat)

        for _ in range(self.num_cycles):
            # DOWN: coarse -> fine, from the coarsest pair to the finest
            for lc, lf in self.pairs:
                cells[lf] = self._down(lc, lf, cells)
            # UP: fine -> coarse, from the finest pair back to the coarsest
            for lc, lf in reversed(self.pairs):
                cells[lc] = self._up(lc, lf, cells)

        return {lvl: self._merge(aux[lvl], cells[lvl]) for lvl in tokens_by_level}

    def _down(self, lc, lf, cells):
        key = f"{lc}_{lf}"
        if self.operator_kind == "attention":
            buf = [getattr(self, f"down_{n}_{key}") for n in ("tgt", "src", "cnt")]
            return self._attend(self._ops[f"{key}_down"], cells[lf], cells[lc], *buf)
        op = typing.cast(_PairOperator, self._ops[key])
        poc = getattr(self, f"poc_{key}")  # (n_fine,)
        valid = (poc >= 0).view(1, -1, 1, 1)
        parent = cells[lc][:, poc.clamp(min=0)] * valid.to(cells[lc].dtype)
        return cells[lf] + op.down(parent)

    def _up(self, lc, lf, cells):
        key = f"{lc}_{lf}"
        if self.operator_kind == "attention":
            buf = [getattr(self, f"up_{n}_{key}") for n in ("tgt", "src", "cnt")]
            return self._attend(self._ops[f"{key}_up"], cells[lc], cells[lf], *buf)
        op = typing.cast(_PairOperator, self._ops[key])
        cop = getattr(self, f"cop_{key}")  # (n_coarse, K)
        cv = getattr(self, f"cv_{key}")
        fine = cells[lf]
        b, _, nq, d = fine.shape
        n_coarse, k = cop.shape
        gathered = fine[:, cop.clamp(min=0).reshape(-1)].reshape(b, n_coarse, k, nq, d)
        maskf = cv.view(1, n_coarse, k, 1, 1).to(gathered.dtype)
        count = cv.sum(dim=1).clamp(min=1).view(1, -1, 1, 1).to(gathered.dtype)
        pooled = (gathered * maskf).sum(dim=2) / count  # mean over the active children
        return cells[lc] + op.up(pooled)
