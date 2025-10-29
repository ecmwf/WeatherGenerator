import logging
import numpy as np
import torch

_logger = logging.getLogger(__name__)


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
      - local_strategy: "random" | "center" | "left" | "right" | "top" | "bottom"
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
        self.dim_time_enc = 6     # hardcoded for now; not used here
        self.healpix_level_data = cf.healpix_level
        self.masking_strategy = cf.get("masking_strategy") # not used here, needed for the tokenizer
        assert self.healpix_level_data is not None, "cf.healpix_level must be set."
        self.healpix_num_cells = 12 * (4 ** self.healpix_level_data)


        # Global (parent) level
        self.hl_global = int(self.cropping_config.get("hl_global", 0))   # alternatively hl_mask
        assert self.hl_global < self.healpix_level_data, (
            f"hl_global={self.hl_global} must be < healpix_level={self.healpix_level_data}"
        )

        # RNG & state
        self.rng = np.random.default_rng()   # should be reset from outside
        self.perm_sel: list[np.typing.NDArray] | list[torch.Tensor] = []

        # Optional centroids used by directional local crops
        # self._cell_centroids = self.cropping_config.get("cell_centroids", None)
        # if self._cell_centroids is not None and torch.is_tensor(self._cell_centroids):
        #     self._cell_centroids = self._cell_centroids.detach().cpu().numpy()

    # Utils

    def reset_rng(self, rng: np.random.Generator | None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()

    def _get_global_keep_rate(self) -> float:
        """Per-call keep rate for the global crop (optional jitter)."""
        rate = float(self.cropping_rate_global)
        if self.cropping_rate_sampling:
            low, high = max(0.0, 0.9 * rate), min(1.0, 1.1 * rate)
            rate = float(self.rng.uniform(low, high))
        _logger.debug("[Cropper] global keep-rate (possibly jittered): %.4f", rate)
        return float(np.clip(rate, 0.0, 1.0))


    # Names are kept for compatibility with Masker interface for now
    def mask_source(
        self,
        tokenized_data: list[torch.Tensor],  # per-cell stacked [n_tokens, feat]
        coords: torch.Tensor,
        geoinfos: torch.Tensor,
        source: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Generate global(+optional local) keep mask at cell-level and apply at token-level.
        Sets self.perm_sel (per-cell, per-token KEEP booleans) for use by keep_target().

        Returns list[Tensor], one per cell, with only KEPT tokens.
        """
        _logger.debug("[Cropper.keep_source] cells=%d | tokens_per_cell(head): %s",
                  len(tokenized_data),
                  [len(t) for t in tokenized_data[: min(5, len(tokenized_data))]])
        
        token_lens = [len(t) for t in tokenized_data]
        total_tokens = sum(token_lens)

        if total_tokens == 0:
            self.perm_sel = []
            return tokenized_data

        # cell-level keep
        keep_cells = self._generate_healpix_keep_cells()
        if not keep_cells.any():
            # all dropped → empty per cell selection
            self.perm_sel = [np.zeros(tl, dtype=bool) for tl in token_lens]
            return [t[:0] for t in tokenized_data]

        # expand to per-token boolean
        flat_keep = np.repeat(keep_cells, token_lens)
        splits = np.cumsum(token_lens)[:-1]
        self.perm_sel = np.split(flat_keep, splits)
        kept_tokens_total = int(sum(p.sum() for p in self.perm_sel))
        _logger.debug(
            "[Cropper.keep_source] kept tokens total=%d (by cell head: %s)",
            kept_tokens_total,
            [int(p.sum()) for p in self.perm_sel[: min(5, len(self.perm_sel))]]
        )

        # apply (Keep=True -> keep)
        source_kept = [data[p] if len(data) > 0 else data for data, p in zip(tokenized_data, self.perm_sel, strict=True)]
        return source_kept
    
    def mask_target(
        self,
        target_tokens_cells_nested: list[list[torch.Tensor]],  # per-cell list of token tensors
        coords: torch.Tensor,
        geoinfos: torch.Tensor,
        source: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Apply SAME kept subset to targets. For each cell, concatenate kept token tensors.
        Returns list[Tensor], one per cell. Creates correctly shaped empties if needed.
        """
        assert self.perm_sel is not None and len(self.perm_sel) == len(target_tokens_cells_nested), (
            "Cropper.perm_sel must be set by keep_source for the same #cells."
        )
        
        _logger.debug(
        "[Cropper.keep_target] perm_sel cells=%d | ex first cell keep-vector(len=%d, true=%d)",
        len(self.perm_sel),
        len(self.perm_sel[0]) if self.perm_sel else 0,
        int(self.perm_sel[0].sum()) if self.perm_sel else 0,
        )

        feat_dim = self.dim_time_enc + coords.shape[-1] + geoinfos.shape[-1] + source.shape[-1]
        out: list[torch.Tensor] = []
        for cc, pp in zip(target_tokens_cells_nested, self.perm_sel, strict=True):
            if len(cc) == 0:
                out.append(torch.empty(0, feat_dim, dtype=coords.dtype, device=coords.device))
                continue

            # When cropping, pp is per-token KEEP boolean aligned to the list order.
            kept_list = [c for keep, c in zip(pp, cc, strict=False) if keep]
            if kept_list:
                out.append(torch.cat(kept_list))
            else:
                out.append(torch.empty(0, feat_dim, dtype=coords.dtype, device=coords.device))

        return out

    # The main cropping logic
    def _generate_healpix_keep_cells(self) -> np.ndarray:
        """
        Returns a boolean mask over cells at cf.healpix_level indicating which cells to KEEP.
        Steps:
          1) Global crop at hl_global by sampling parents (percentage or 'global_keep_m').
          2) Project to child cells at healpix_level (NESTED arithmetic).
          3) Optionally refine by building one or more *local crops* (consecutive runs) inside the
             projected global set, using strategy to anchor the run(s).
        """
        Ls = self.healpix_level_data           # child/data level
        Lg = self.hl_global                    # parent/global level
        assert Lg < Ls
        par_count = 12 * (4 ** Lg)
        kids_per_parent = 4 ** (Ls - Lg)

        _logger.debug(
            "[Cropper] Lg=%d -> Ls=%d | par_count=%d | kids_per_parent=%d | cfg=%s",
            Lg, Ls, par_count, kids_per_parent, str(self.cropping_config)
        )
        # ---- 1) choose parents ----
        if "global_keep_m" in self.cropping_config:
            m_par = int(self.cropping_config["global_keep_m"])
        else:
            rate = self._get_global_keep_rate()
            m_par = max(0, min(par_count, int(np.round(rate * par_count))))

        if m_par == 0:
            return np.zeros(self.healpix_num_cells, dtype=bool)

        # Here we simply sample parents uniformly at random.
        # If you want them to be consecutive, add a boolean 'global_consecutive' in config.
        if bool(self.cropping_config.get("global_consecutive", False)) and m_par < par_count:
            start = int(self.rng.integers(0, par_count))
            parent_ids = (start + np.arange(m_par)) % par_count
        elif m_par >= par_count:
            parent_ids = np.arange(par_count)
        else:
            parent_ids = np.sort(self.rng.choice(par_count, size=m_par, replace=False))

        _logger.debug(
        "[Cropper] parents kept: m_par=%d | consecutive=%s | example ids(head): %s",
        m_par, bool(self.cropping_config.get("global_consecutive", False)),
        parent_ids[: min(8, parent_ids.size)]
        )
        # ---- 2) project to children (NESTED contiguous blocks per parent) ----
        child_offsets = np.arange(kids_per_parent)
        all_child_ids = (parent_ids[:, None] * kids_per_parent + child_offsets).reshape(-1)
        all_child_ids = np.sort(all_child_ids)

        if all_child_ids.size == 0:
            return np.zeros(self.healpix_num_cells, dtype=bool)
        
        _logger.debug(
            "[Cropper] projected children: %d | child_ids(head): %s",
            all_child_ids.size, all_child_ids[: min(12, all_child_ids.size)]
        )

        # ---- 3) optional local crops ----
        if not bool(self.cropping_config.get("local_crops", False)):
            # Keep the entire projected global region
            keep_cells = np.zeros(self.healpix_num_cells, dtype=bool)
            keep_cells[all_child_ids] = True
            return keep_cells

        # Prepare union of local crops (as subset of 'all_child_ids')
        total_children = all_child_ids.size
        n_local = int(self.cropping_config.get("num_local_crops", 1))
        frac = float(self.cropping_config.get("local_crop_frac", 1.0))
        m_loc = max(1, min(total_children, int(np.round(frac * total_children))))
        strategy = str(self.cropping_config.get("local_strategy", "random")).lower()

        # Helper to select a **single** consecutive run from the sorted 'all_child_ids'
        def _select_run(strategy_name: str) -> np.ndarray:
            if total_children == m_loc:
                return all_child_ids  # trivial
            if strategy_name == "center":
                start_idx = (total_children - m_loc) // 2
                return all_child_ids[start_idx : start_idx + m_loc]
            if strategy_name == "left":
                return all_child_ids[:m_loc]
            if strategy_name == "right":
                return all_child_ids[-m_loc:]
            if strategy_name in ("top", "bottom"):
                if self._cell_centroids is None:
                    _logger.warning(
                        "Cropper.local_strategy=%s requested but no cell_centroids provided; "
                        "falling back to index-based selection.", strategy_name
                    )
                    return all_child_ids[:m_loc] if strategy_name == "top" else all_child_ids[-m_loc:]
                # Use latitude to anchor selection
                # self._cell_centroids: [num_cells, 2] -> (lon_deg, lat_deg)
                lats = self._cell_centroids[all_child_ids, 1]
                # sort by latitude ascending, pick top or bottom block, then **consecutive by index**
                order = np.argsort(lats)
                sorted_by_lat = all_child_ids[order]
                if strategy_name == "top":     # highest latitudes
                    candidates = sorted_by_lat[-m_loc:]
                else:                           # lowest latitudes
                    candidates = sorted_by_lat[:m_loc]
                # Make it consecutive in index space by taking min..max overlap with all_child_ids
                # If scattered, we just sort and take a tight run around the median index.
                candidates = np.sort(candidates)
                if candidates.size < m_loc:
                    return candidates  # degenerate but fine
                # find a run nearest the median
                med = candidates[candidates.size // 2]
                # find med's position in all_child_ids
                pos = int(np.searchsorted(all_child_ids, med))
                start_idx = max(0, min(total_children - m_loc, pos - m_loc // 2))
                return all_child_ids[start_idx : start_idx + m_loc]
            # default random
            start_idx = int(self.rng.integers(0, total_children - m_loc + 1))
            return all_child_ids[start_idx : start_idx + m_loc]

        kept_union = np.empty(0, dtype=int)
        for _ in range(n_local):
            run = _select_run(strategy)
            kept_union = np.union1d(kept_union, run)

        keep_cells = np.zeros(self.healpix_num_cells, dtype=bool)
        keep_cells[kept_union] = True
        
        _logger.debug(
            "[Cropper] final keep_cells: num_true=%d / %d (%.2f%%)",
            int(keep_cells.sum()), keep_cells.size, 100.0 * float(keep_cells.mean())
        )
        return keep_cells

    # Functions for compatibility with Masker interface / Not used here
    def set_batch_strategy(self):
        """
        Set strategy for this batch.
        Only relevant with combination and same_strategy_per_batch.
        """
        if self.masking_strategy == "combination" and self.same_strategy_per_batch:
            self.current_strategy = self.rng.choice(
                self.masking_strategy_config["strategies"],
                p=self.masking_strategy_config["probabilities"],
            )
            self.batch_strategy_set = True

    def reset_batch_strategy(self):
        """
        Reset for next batch.
        """
        if self.masking_strategy == "combination" and self.same_strategy_per_batch:
            self.current_strategy = None
            self.batch_strategy_set = False

    def _select_strategy(self):
        """
        Select the strategy to use.
        """
        if self.masking_strategy == "combination":
            if self.same_strategy_per_batch:
                assert self.batch_strategy_set, "Must call set_batch_strategy() first"
                return self.current_strategy
            else:
                # Sample new strategy for each stream
                return self.rng.choice(
                    self.masking_strategy_config["strategies"],
                    p=self.masking_strategy_config["probabilities"],
                )
        else:
            # Non-combination strategy, return as is
            return self.masking_strategy
