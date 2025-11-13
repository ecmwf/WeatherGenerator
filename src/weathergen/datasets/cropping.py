#####################################
# DEPRECATED, MERGED INTO MASKER
#####################################


import logging
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

_logger = logging.getLogger(__name__)


@dataclass
class CropSpec:
    level: int  # data level (healpix_level)
    parent_level: int  # hl_global
    keep_cells: np.ndarray  # [num_cells(level)] bool mask


class Cropper:
    """
    HEALPix-only cropper.

    Global crop (required):
      - hl_global: int (default 0). Must satisfy hl_global < cf.healpix_level.
      - cropping_rate_global: float in [0, 1] (percentage of parents kept at hl_global).
        Optionally, provide 'global_keep_m' (int) instead of a rate.

    Local crops (optional):
      - local_crops: bool (default False)
      - num_local_crops: int (>=1), number of local crops to produce
      - local_crop_frac: float in (0,1], size of each local crop as a fraction of the
        projected global children count
      - local_strategy: "random" | "center" | "left" | "right" |
      - cell_centroids: optional np.ndarray/torch.Tensor of shape [num_cells, 2] with
        (lon_deg, lat_deg) at cf.healpix_level; used for directional strategies.
        If not provided, we fall back to index order and log a warning.

    Output semantics:
      - Keeps (selects) tokens/cells and empties everything else.
      - The same kept subset is applied to source and target (no complement).
      - self.perm_sel stores per-cell boolean vectors (at token granularity).
    """

    def __init__(self, cf):
        self.cropping_rate_global = cf.cropping_rate_global
        self.cropping_rate_sampling = cf.cropping_rate_sampling
        self.cropping_config = cf.cropping_config if cf.cropping_config is not None else {}
        self.dim_time_enc = 6  # hardcoded for now; not used here
        self.healpix_level_data = cf.healpix_level
        self.masking_strategy = cf.get(
            "masking_strategy"
        )  # not used here, needed for the tokenizer
        assert self.healpix_level_data is not None, "cf.healpix_level must be set."
        self.healpix_num_cells = 12 * (4**self.healpix_level_data)

        # define hl_global
        self.hl_global = int(self.cropping_config.get("hl_global", 0))
        # Selectable child level; defaults to data level
        self.hl_child = int(self.cropping_config.get("hl_child", self.healpix_level_data))

        assert self.hl_global < self.hl_child <= self.healpix_level_data, (
            f"Require hl_global < hl_child <= healpix_level; got "
            f"hl_global={self.hl_global}, hl_child={self.hl_child}, level={self.healpix_level_data}"
        )
        # override hook (used to apply global/local views)
        self._override_keep_cells: np.ndarray | None = None

        # RNG & state
        self.rng = np.random.default_rng()
        self.perm_sel: list[np.typing.NDArray] | list[torch.Tensor] = []

    @contextmanager
    def use_keep_cells(self, keep_cells: np.ndarray | None):
        """Temporarily override keep-cells used by mask_source/mask_target."""
        prev = self._override_keep_cells
        self._override_keep_cells = (
            None if keep_cells is None else np.asarray(keep_cells, dtype=bool)
        )
        try:
            yield
        finally:
            self._override_keep_cells = prev

    def make_crops(self, n_local: int, local_frac: float, local_strategy: str = "random"):
        """
        Compute a global crop (parents at hl_global → children at hl_child → expanded to data level)
        and N local crops as strict subsets (runs within the hl_child children), then expanded to
        data level healpix grid.
        Returns: (global_spec, [local_spec, ...])
        """
        Lg = self.hl_global
        Lc = self.hl_child
        Ls = self.healpix_level_data

        # 1) choose parents at Lg
        par_count = 12 * (4**Lg)
        if "global_keep_m" in self.cropping_config:
            m_par = int(self.cropping_config["global_keep_m"])
        else:
            rate = self._get_global_keep_rate()
            m_par = max(0, min(par_count, int(np.round(rate * par_count))))
        if m_par == 0:
            empty = np.zeros(self.healpix_num_cells, dtype=bool)
            return CropSpec(Ls, Lg, empty), []

        if bool(self.cropping_config.get("global_consecutive", False)) and m_par < par_count:
            start = int(self.rng.integers(0, par_count))
            parent_ids = (start + np.arange(m_par)) % par_count
        elif m_par >= par_count:
            parent_ids = np.arange(par_count)
        else:
            parent_ids = np.sort(self.rng.choice(par_count, size=m_par, replace=False))

        # 2) project to hl_child
        kids_per_parent_child = 4 ** (Lc - Lg)
        child_offsets = np.arange(kids_per_parent_child)
        ids_child = (parent_ids[:, None] * kids_per_parent_child + child_offsets).reshape(-1)
        ids_child = np.sort(ids_child)

        # 3) expand to data level if needed
        def expand_to_level(ids_level_from: np.ndarray, L_from: int, L_to: int) -> np.ndarray:
            if L_from == L_to:
                return ids_level_from
            mul = 4 ** (L_to - L_from)
            offs = np.arange(mul)
            return np.sort((ids_level_from[:, None] * mul + offs).reshape(-1))

        ids_data_global = expand_to_level(ids_child, Lc, Ls)
        keep_global = np.zeros(self.healpix_num_cells, dtype=bool)
        keep_global[ids_data_global] = True
        global_spec = CropSpec(Ls, Lg, keep_global)

        # 4) local crops as subsets of ids_child, then expand to data level
        total_children = ids_child.size
        n_local = int(max(1, n_local))
        m_loc = max(1, min(total_children, int(np.round(float(local_frac) * total_children))))
        strategy = str(local_strategy).lower()

        def select_run():
            if total_children == m_loc:
                return ids_child
            if strategy == "center":
                s = (total_children - m_loc) // 2
                return ids_child[s : s + m_loc]
            if strategy == "left":
                return ids_child[:m_loc]
            if strategy == "right":
                return ids_child[-m_loc:]
            s = int(self.rng.integers(0, total_children - m_loc + 1))
            return ids_child[s : s + m_loc]

        local_specs: list[CropSpec] = []
        for k in range(n_local):
            run_child = select_run()
            ids_data_local = expand_to_level(run_child, Lc, Ls)
            keep_local = np.zeros(self.healpix_num_cells, dtype=bool)
            keep_local[ids_data_local] = True
            local_specs.append(CropSpec(Ls, Lg, keep_local))

        return global_spec, local_specs

    def reset_rng(self, rng: np.random.Generator | None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()

    def _get_global_keep_rate(self) -> float:
        rate = float(self.cropping_rate_global)
        if self.cropping_rate_sampling:
            low, high = max(0.0, 0.9 * rate), min(1.0, 1.1 * rate)
            rate = float(self.rng.uniform(low, high))
        return float(np.clip(rate, 0.0, 1.0))

    def mask_source(self, tokenized_data, coords, geoinfos, source) -> list[torch.Tensor]:
        token_lens = [len(t) for t in tokenized_data]
        total_tokens = int(sum(token_lens))

        if total_tokens == 0:
            self.perm_sel = []
            return tokenized_data

        keep_cells = self._override_keep_cells
        if keep_cells is None:
            # No crop override: keep all tokens
            keep_cells = np.ones(self.healpix_num_cells, dtype=bool)
        else:
            keep_cells = np.asarray(keep_cells, dtype=bool)

        if not keep_cells.any():
            self.perm_sel = [np.zeros(tl, dtype=bool) for tl in token_lens]
            return [t[:0] for t in tokenized_data]

        flat_keep = np.repeat(keep_cells, token_lens)
        splits = np.cumsum(token_lens)[:-1]
        self.perm_sel = np.split(flat_keep, splits)

        source_kept = [
            data[p] if len(data) > 0 else data
            for data, p in zip(tokenized_data, self.perm_sel, strict=True)
        ]
        return source_kept

    # Here is mask target we basically just keep the same global view as for source
    # This is different from masking where we keep the complement
    # This is because we may want to have some reconstruction loss on the kept tokens later
    # This can be extended to support different views for the target if needed later
    def mask_target(
        self, target_tokens_cells_nested, coords, geoinfos, source
    ) -> list[torch.Tensor]:
        assert self.perm_sel is not None and len(self.perm_sel) == len(
            target_tokens_cells_nested
        ), "Cropper.perm_sel must be set by keep_source for the same #cells."
        feat_dim = self.dim_time_enc + coords.shape[-1] + geoinfos.shape[-1] + source.shape[-1]

        out: list[torch.Tensor] = []
        for cell_idx, (cc, pp) in enumerate(
            zip(target_tokens_cells_nested, self.perm_sel, strict=True)
        ):
            if len(cc) == 0:
                out.append(torch.empty(0, feat_dim, dtype=coords.dtype, device=coords.device))
                continue
            kept_list = [c for keep, c in zip(pp, cc, strict=False) if keep]
            if kept_list:
                out.append(torch.cat(kept_list))
            else:
                out.append(torch.empty(0, feat_dim, dtype=coords.dtype, device=coords.device))
        return out

    # Functions for compatibility with Masker interface / Not used here
    def set_batch_strategy(self):
        # No-op for cropping mode
        return

    def reset_batch_strategy(self):
        # No-op for cropping mode
        return

    def _select_strategy(self):
        # Not used in cropping; keep for API compatibility
        return "cropping"
