"""
Unified masker supporting both traditional masking (MAE) and student-teacher (JEPA) modes.

Training modes:
  - "masking": Generate masks for source/target (classic MAE-style).
  - "student_teacher": Generate teacher (global) and student (local) spatial views.

Masking strategies apply to both modes:
  - "random": uniform random token/cell masking
  - "block": contiguous block masking
  - "healpix": hierarchical HEALPix cell masking
  - "channel": per-channel masking
  - "causal": mask recent timesteps
  - "combination": randomly pick from multiple strategies
"""

import logging
import numpy as np
import torch
from weathergen.common.config import Config
from weathergen.datasets.views import ViewMetadata

from contextlib import contextmanager

_logger = logging.getLogger(__name__)


class Masker:
    """
    Unified masker supporting both traditional masking (MAE) and student-teacher (JEPA) modes.
    """

    def __init__(self, cf: Config):
        # Core masking config (used in both modes)
        self.training_mode = cf.get("training_mode", "masking")
        self.masking_rate = cf.masking_rate
        self.masking_strategy = cf.masking_strategy
        self.current_strategy = cf.masking_strategy
        self.masking_rate_sampling = cf.masking_rate_sampling
        self.masking_strategy_config = cf.get("masking_strategy_config", {})
        
        # Student-teacher config (only used if training_mode == "student_teacher")
        self.student_teacher_config = cf.get("student_teacher", {})
        
        # HEALPix grid parameters
        self.healpix_level_data = cf.healpix_level
        self.healpix_num_cells = 12 * (4 ** self.healpix_level_data)
        
        # Encoding dimensions
        self.dim_time_enc = 6
        self.mask_value = 0.0
        
        # State: per-batch mask selection (set by mask_source, used by mask_target)
        self.perm_sel: list[np.ndarray] | None = None
        
        # Per-batch strategy tracking (for "combination" mode)
        self.same_strategy_per_batch = self.masking_strategy_config.get(
            "same_strategy_per_batch", False
        )
        self.batch_strategy_set = False
        
        # Validate config based on mode
        self._validate_config(cf)

        self._override_keep_cells: np.ndarray | None = None  # state for context manager

    @contextmanager
    def use_keep_cells(self, keep_cells: np.ndarray | None):
        """Context manager to apply view-specific cell masks."""
        old_override = self._override_keep_cells
        self._override_keep_cells = keep_cells
        try:
            yield
        finally:
            self._override_keep_cells = old_override

        
    def _validate_config(self, cf: Config):
        """Validate configuration for the selected training mode and strategy."""
        # Check HEALPix masking requirements
        if self.current_strategy == "healpix":
            hl_mask = self.masking_strategy_config.get("hl_mask")
            assert hl_mask is not None, (
                "masking_strategy='healpix' requires 'hl_mask' in masking_strategy_config"
            )
            assert hl_mask < self.healpix_level_data, (
                f"hl_mask={hl_mask} must be < healpix_level={self.healpix_level_data}"
            )
        
        # Check channel masking requirements
        if self.current_strategy == "channel":
            mode = self.masking_strategy_config.get("mode")
            assert mode in ["global", "per_cell"], (
                "masking_strategy='channel' requires 'mode' in ['global', 'per_cell']"
            )
            # Verify source/target channels match across streams
            for stream in cf.streams:
                src_inc = set(stream.get("source_include", []))
                tgt_inc = set(stream.get("target_include", []))
                assert src_inc == tgt_inc, (
                    f"Stream '{stream.get('name')}': source and target channels must match "
                    "for channel masking"
                )
        
        # Validate student-teacher config if in that mode
        if self.training_mode == "student_teacher":
            st_cfg = self.student_teacher_config
            assert "global" in st_cfg, "student_teacher mode requires 'global' config"
            assert "locals" in st_cfg, "student_teacher mode requires 'locals' config"
            
            # Teacher (global) view config
            global_cfg = st_cfg["global"]
            assert "strategy" in global_cfg, "global view requires 'strategy'"
            assert "rate" in global_cfg or "keep_m" in global_cfg, (
                "global view requires 'rate' or 'keep_m'"
            )
            
            # Student (local) views config
            locals_cfg = st_cfg["locals"]
            assert "num_views" in locals_cfg, "locals requires 'num_views' (>=1)"
            assert "strategy" in locals_cfg, "locals requires 'strategy'"
            assert "rate" in locals_cfg, "locals requires 'rate'"
            assert int(locals_cfg["num_views"]) >= 1, "Must have at least 1 local view"

    def reset_rng(self, rng: np.random.Generator | None) -> None:
        """Reset RNG for reproducibility after each epoch."""
        self.rng = rng if rng is not None else np.random.default_rng()

    def set_batch_strategy(self):
        """
        (Only for 'combination' mode with same_strategy_per_batch=True)
        Select one strategy for the entire batch.
        """
        if self.masking_strategy == "combination" and self.same_strategy_per_batch:
            self.current_strategy = self.rng.choice(
                self.masking_strategy_config["strategies"],
                p=self.masking_strategy_config["probabilities"],
            )
            self.batch_strategy_set = True

    def reset_batch_strategy(self):
        """Reset strategy selection for next batch."""
        if self.masking_strategy == "combination" and self.same_strategy_per_batch:
            self.current_strategy = None
            self.batch_strategy_set = False

    def _select_strategy(self) -> str:
        """
        Choose which masking strategy to apply.
        Returns the strategy name (e.g., "random", "healpix", "channel").
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
            return self.masking_strategy

    # =========================================================================
    # STUDENT-TEACHER MODE: Generate teacher (global) and student (local) views
    # =========================================================================

    def make_views(
        self,
        tokenized_data: list[torch.Tensor],
        coords: torch.Tensor,
        geoinfos: torch.Tensor,
        source: torch.Tensor,
    ) -> tuple[ViewMetadata, list[ViewMetadata]]:
        """
        Generate teacher (global) and student (local) views for student-teacher training.
        
        Args:
            tokenized_data: List of token tensors, one per HEALPix cell.
            coords: Coordinate tensor [num_points, coord_dim]
            geoinfos: Geoinfo tensor [num_points, geoinfo_dim]
            source: Source data tensor [num_points, source_channels]
        
        Returns:
            teacher_view: ViewMetadata for the global (teacher) view
            student_views: List of ViewMetadata for each local (student) view
        """
        assert self.training_mode == "student_teacher", (
            "make_views() only available in student_teacher mode"
        )
        
        st_cfg = self.student_teacher_config
        global_cfg = st_cfg["global"]
        locals_cfg = st_cfg["locals"]
        
        # 1) Generate teacher (global) view mask
        teacher_strategy = global_cfg["strategy"]
        teacher_rate = self._get_view_rate(global_cfg)
        teacher_keep_mask = self._generate_view_mask(
            tokenized_data, teacher_strategy, teacher_rate, coords, geoinfos, source, global_cfg
        )
        teacher_view = ViewMetadata(
            view_id="teacher_global",
            keep_mask=teacher_keep_mask,
            strategy=teacher_strategy,
            healpix_level=global_cfg.get("hl_mask", None),
            rate=teacher_rate,
            parent_view_id=None,
        )
        
        # 2) Generate student (local) view masks
        num_locals = int(locals_cfg["num_views"])
        student_strategy = locals_cfg["strategy"]
        student_rate = float(locals_cfg["rate"])
        
        student_views = []
        for i in range(num_locals):
            # Each local view is a subset of the teacher view
            local_keep_mask = self._generate_local_view_mask(
                tokenized_data,
                teacher_keep_mask,
                student_strategy,
                student_rate,
                coords,
                geoinfos,
                source,
            )
            student_views.append(
                ViewMetadata(
                    view_id=f"student_local_{i}",
                    keep_mask=local_keep_mask,
                    strategy=student_strategy,
                    healpix_level=locals_cfg.get("hl_mask", None),
                    rate=student_rate,
                    parent_view_id="teacher_global",
                )
            )
        
        return teacher_view, student_views

    def _get_view_rate(self, view_config: dict) -> float:
        """
        Extract the masking/cropping rate for a view.
        Supports optional rate sampling (jitter).
        """
        if "keep_m" in view_config:
            # Absolute count provided; convert to rate
            keep_m = int(view_config["keep_m"])
            total = self.healpix_num_cells
            rate = float(keep_m) / total
        else:
            rate = float(view_config["rate"])
        
        # Optional: add jitter (±10%)
        if view_config.get("rate_sampling", False):
            low = max(0.0, 0.9 * rate)
            high = min(1.0, 1.1 * rate)
            rate = float(self.rng.uniform(low, high))
        
        return float(np.clip(rate, 0.0, 1.0))

    def _generate_view_mask(
        self,
        tokenized_data: list[torch.Tensor],
        strategy: str,
        rate: float,
        coords: torch.Tensor,
        geoinfos: torch.Tensor,
        source: torch.Tensor,
        config: dict,
    ) -> np.ndarray:
        """
        Generate a cell-level keep mask for a view (teacher or student).
        
        Returns:
            keep_mask: Boolean array [num_healpix_cells] indicating which cells to keep.
        """
        if strategy == "random":
            # Random cell selection
            num_keep = int(np.round(rate * self.healpix_num_cells))
            if num_keep == 0:
                return np.zeros(self.healpix_num_cells, dtype=bool)
            keep_indices = self.rng.choice(self.healpix_num_cells, num_keep, replace=False)
            keep_mask = np.zeros(self.healpix_num_cells, dtype=bool)
            keep_mask[keep_indices] = True
        
        elif strategy == "healpix":
            # Hierarchical HEALPix masking
            keep_mask = self._generate_healpix_cell_mask(rate, config.get("hl_mask"))
        
        elif strategy in ["block", "channel", "causal"]:
            # These strategies operate at token level, not cell level
            # For spatial views, keep all cells and warn
            _logger.warning(
                f"Strategy '{strategy}' is token-level; view mask will keep all cells. "
                "Consider using 'random' or 'healpix' for spatial views."
            )
            keep_mask = np.ones(self.healpix_num_cells, dtype=bool)
        
        else:
            raise ValueError(f"Unknown view strategy: {strategy}")
        
        return keep_mask

    def _generate_local_view_mask(
        self,
        tokenized_data: list[torch.Tensor],
        teacher_mask: np.ndarray,
        strategy: str,
        rate: float,
        coords: torch.Tensor,
        geoinfos: torch.Tensor,
        source: torch.Tensor,
    ) -> np.ndarray:
        """
        Generate a local (student) view mask as a strict subset of the teacher mask.
        
        Args:
            teacher_mask: Boolean array [num_cells] from the teacher view
            strategy: Masking strategy for local view
            rate: Fraction of teacher cells to keep in local view
        
        Returns:
            local_mask: Boolean array [num_cells], subset of teacher_mask
        """
        # Find indices of cells kept in teacher view
        teacher_kept_indices = np.flatnonzero(teacher_mask)
        num_teacher = len(teacher_kept_indices)
        
        if num_teacher == 0:
            # Edge case: teacher is empty, local must also be empty
            return np.zeros(self.healpix_num_cells, dtype=bool)
        
        # Determine how many cells to keep in local view
        num_local = int(np.round(rate * num_teacher))
        num_local = max(1, min(num_teacher, num_local))  # at least 1, at most all teacher cells
        
        if strategy == "random":
            # Random subset of teacher cells
            local_kept_indices = self.rng.choice(teacher_kept_indices, num_local, replace=False)
        
        elif strategy == "center":
            # Contiguous block in the center of teacher cells (in index order)
            start = (num_teacher - num_local) // 2
            local_kept_indices = teacher_kept_indices[start : start + num_local]
        
        elif strategy == "left":
            local_kept_indices = teacher_kept_indices[:num_local]
        
        elif strategy == "right":
            local_kept_indices = teacher_kept_indices[-num_local:]
        
        else:
            # Fallback to random
            local_kept_indices = self.rng.choice(teacher_kept_indices, num_local, replace=False)
        
        # Build mask
        local_mask = np.zeros(self.healpix_num_cells, dtype=bool)
        local_mask[local_kept_indices] = True
        
        return local_mask

    def _generate_healpix_cell_mask(self, rate: float, hl_mask: int | None = None) -> np.ndarray:
        """
        Generate a cell-level mask via hierarchical HEALPix selection.
        Selects parent cells at hl_mask, then marks all their children at data level.
        
        Args:
            rate: Fraction of parent cells to keep
            hl_mask: HEALPix level for parent selection (default from config)
        
        Returns:
            cell_mask: Boolean array [num_healpix_cells]
        """
        if hl_mask is None:
            hl_mask = self.masking_strategy_config.get("hl_mask")
        assert hl_mask is not None and hl_mask < self.healpix_level_data, (
            f"hl_mask ({hl_mask}) must be < healpix_level ({self.healpix_level_data})"
        )
        
        num_parent_cells = 12 * (4 ** hl_mask)
        num_parents_to_keep = int(np.round(rate * num_parent_cells))
        
        if num_parents_to_keep == 0:
            return np.zeros(self.healpix_num_cells, dtype=bool)
        
        # Select parent cells
        parent_ids = self.rng.choice(num_parent_cells, num_parents_to_keep, replace=False)
        
        # Expand to data level
        level_diff = self.healpix_level_data - hl_mask
        num_children_per_parent = 4 ** level_diff
        child_offsets = np.arange(num_children_per_parent)
        child_indices = (parent_ids[:, None] * num_children_per_parent + child_offsets).reshape(-1)
        
        cell_mask = np.zeros(self.healpix_num_cells, dtype=bool)
        cell_mask[child_indices] = True
        
        return cell_mask

    # =========================================================================
    # ORIGINAL MASKING MODE: mask_source, mask_target (MAE-style)
    # =========================================================================

    def mask_source(
        self,
        tokenized_data: list[torch.Tensor],
        coords: torch.Tensor,
        geoinfos: torch.Tensor,
        source: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Apply masking strategy and return the unmasked (source) tokens.
        Sets self.perm_sel which mask_target will use.
        
        This is the main entry point for traditional "masking" mode.
        
        Returns:
            source_data: List of token tensors (unmasked)
        """
        token_lens = [len(t) for t in tokenized_data]
        num_tokens = sum(token_lens)
        
        if num_tokens == 0:
            self.perm_sel = []
            return tokenized_data
        
        # If in a use_keep_cells context, apply cell-level mask first
        if self._override_keep_cells is not None:
            # We're inside a use_keep_cells context
            cell_mask = self._override_keep_cells  # e.g., [True, True, False, True, ...]
            
            # Convert cell-level mask to token-level mask
            # If cell 0 has 5 tokens and cell_mask[0]=True, all 5 tokens are kept
            flat_mask = np.repeat(cell_mask, token_lens)  # [T,T,T,T,T, T,T,T,T,T, ...]
            
            # Store inverted mask for mask_target
            split_indices = np.cumsum(token_lens)[:-1]
            self.perm_sel = np.split(~flat_mask, split_indices)  # True = MASKED
            
            # Return only UNMASKED tokens
            source_data = [
                data[~p]  # Keep where p=False (unmasked)
                for data, p in zip(tokenized_data, self.perm_sel, strict=True)
            ]
            return source_data
        
        # Select strategy
        self.current_strategy = self._select_strategy()
        rate = self._get_sampling_rate()
        
        if rate == 0.0:
            _logger.warning("masking_rate is 0; target will be empty.")
        
        # Generate mask based on strategy
        if self.current_strategy == "random":
            flat_mask = self.rng.uniform(0, 1, num_tokens) < rate
        
        elif self.current_strategy == "block":
            flat_mask = np.zeros(num_tokens, dtype=bool)
            block_size = int(np.round(rate * num_tokens))
            if block_size > 0 and num_tokens > 0:
                start_index = self.rng.integers(0, max(1, num_tokens - block_size + 1))
                flat_mask[start_index : start_index + block_size] = True
        
        elif self.current_strategy == "healpix":
            flat_mask = self._generate_healpix_mask(token_lens, rate)
        
        elif self.current_strategy == "channel":
            mask = self._generate_channel_mask(tokenized_data, rate, coords, geoinfos, source)
        
        elif self.current_strategy == "causal":
            mask = self._generate_causal_mask(tokenized_data, rate, coords, geoinfos, source)
        
        else:
            raise ValueError(f"Unknown masking strategy: {self.current_strategy}")
        
        # Apply mask
        if self.current_strategy == "channel":
            self.perm_sel = mask
            source_data = []
            for data, p in zip(tokenized_data, self.perm_sel, strict=True):
                if len(data) > 0:
                    data[p] = self.mask_value
                    source_data.append(data)
                else:
                    source_data.append(data)
        
        elif self.current_strategy == "causal":
            self.perm_sel = mask
            source_data = [data[~p] if len(data) > 0 else data for data, p in zip(tokenized_data, self.perm_sel, strict=True)]
        
        else:
            split_indices = np.cumsum(token_lens)[:-1]
            self.perm_sel = np.split(flat_mask, split_indices)
            source_data = [data[~p] for data, p in zip(tokenized_data, self.perm_sel, strict=True)]
        
        return source_data

    def mask_target(
        self,
        target_tokens_cells_nested: list[list[torch.Tensor]],
        coords: torch.Tensor,
        geoinfos: torch.Tensor,
        source: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Apply the mask (self.perm_sel) to target tokens.
        In traditional masking mode, perm_sel indicates which tokens were masked (True = masked).
        
        Returns:
            target_data: List of token tensors (masked tokens only)
        """
        assert self.perm_sel is not None, "mask_source must be called first"
        
        feat_dim = self.dim_time_enc + coords.shape[-1] + geoinfos.shape[-1] + source.shape[-1]
        processed_target_tokens = []
        
        for cc, pp in zip(target_tokens_cells_nested, self.perm_sel, strict=True):
            if self.current_strategy == "channel":
                # Channel masking: set unmasked channels to NaN
                selected_tensors = []
                for c, p in zip(cc, pp, strict=True):
                    c[:, (self.dim_time_enc + coords.shape[-1] + geoinfos.shape[-1]):][
                        :, ~p[0, (self.dim_time_enc + coords.shape[-1] + geoinfos.shape[-1]):]
                    ] = torch.nan
                    selected_tensors.append(c)
            
            elif self.current_strategy == "causal":
                # Causal: select only masked timesteps
                selected_tensors = [c for i, c in enumerate(cc) if pp[i]]
            
            else:
                # Default: select tokens where mask is True
                selected_tensors = [c for c, p in zip(cc, pp, strict=True) if p]
            
            if selected_tensors:
                processed_target_tokens.append(torch.cat(selected_tensors))
            else:
                processed_target_tokens.append(
                    torch.empty(0, feat_dim, dtype=coords.dtype, device=coords.device)
                )
        
        return processed_target_tokens

    def _get_sampling_rate(self) -> float:
        """Sample masking rate with optional jitter."""
        if self.masking_rate_sampling:
            rate = np.clip(
                np.abs(self.rng.normal(loc=self.masking_rate, scale=1.0 / (2.5 * np.pi))),
                0.01, 0.99,
            )
        else:
            rate = self.masking_rate
        return rate

    def _generate_healpix_mask(self, token_lens: list[int], rate: float) -> np.ndarray:
        """(Existing method for token-level HEALPix masking in traditional masking mode)"""
        hl_mask = self.masking_strategy_config.get("hl_mask")
        num_parent_cells = 12 * (4 ** hl_mask)
        level_diff = self.healpix_level_data - hl_mask
        num_children_per_parent = 4 ** level_diff
        
        num_parents_to_mask = int(np.round(rate * num_parent_cells))
        if num_parents_to_mask == 0:
            return np.zeros(sum(token_lens), dtype=bool)
        
        parent_ids = self.rng.choice(num_parent_cells, num_parents_to_mask, replace=False)
        child_offsets = np.arange(num_children_per_parent)
        child_indices = (parent_ids[:, None] * num_children_per_parent + child_offsets).reshape(-1)
        
        cell_mask = np.zeros(self.healpix_num_cells, dtype=bool)
        cell_mask[child_indices] = True
        
        flat_mask = np.repeat(cell_mask, token_lens)
        return flat_mask

    def _generate_channel_mask(self, tokenized_data, rate, coords, geoinfos, source):
        """(Existing channel masking logic)"""
        if not tokenized_data:
            return []
        
        num_channels = self.dim_time_enc + coords.shape[-1] + geoinfos.shape[-1] + source.shape[-1]
        num_fixed_channels = self.dim_time_enc + coords.shape[-1] + geoinfos.shape[-1]
        num_data_channels = source.shape[-1]
        mask_count = int(num_data_channels * rate)
        
        tokenized_data_lens = [len(t) for t in tokenized_data]
        tokenized_data_merged = torch.cat(tokenized_data)
        num_tokens = tokenized_data_merged.shape[0]
        token_size = tokenized_data_merged.shape[1]
        
        if self.masking_strategy_config.get("mode") == "global":
            channel_mask = np.zeros(num_channels, dtype=bool)
            m = num_fixed_channels + self.rng.choice(num_data_channels, mask_count, replace=False)
            channel_mask[m] = True
            full_mask = np.zeros_like(tokenized_data_merged).astype(np.bool_)
            full_mask[:, :] = channel_mask
        else:
            channel_mask = np.zeros((token_size, num_tokens, num_channels), dtype=bool)
            nc = (num_tokens, num_data_channels)
            channel_mask[:, :, num_fixed_channels:] = self.rng.uniform(0, 1, nc) < rate
            full_mask = channel_mask.transpose([1, 0, 2])
        
        full_mask = np.split(full_mask, np.cumsum(tokenized_data_lens[:-1]))
        return full_mask

    def _generate_causal_mask(self, tokenized_data, rate, coords, geoinfos, source):
        """(Existing causal masking logic)"""
        if not tokenized_data:
            return []
        
        token_lens = np.array([len(token_data) for token_data in tokenized_data])
        if len(token_lens) == 0:
            return []
        
        num_future_to_mask = (rate * token_lens).astype(int)
        start_mask_indices = np.maximum(1, token_lens - num_future_to_mask)
        mask_valid = token_lens > 1
        start_mask_indices = np.where(mask_valid, start_mask_indices, token_lens)
        
        full_mask = [
            np.concatenate([
                np.zeros(start_idx, dtype=bool),
                np.ones(max(0, token_len - start_idx), dtype=bool),
            ])
            if token_len > 1
            else (np.zeros(1, dtype=bool) if token_len == 1 else np.array([], dtype=bool))
            for token_len, start_idx in zip(token_lens, start_mask_indices, strict=False)
        ]
        
        return full_mask