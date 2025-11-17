import logging
import numpy as np
import torch
from weathergen.common.config import Config
from weathergen.datasets.views import ViewMetadata

from contextlib import contextmanager

_logger = logging.getLogger(__name__)


class Masker:
    """
    Unified masker supporting both traditional masking (MAE) and student-teacher modes.
    Used with training_mode: "masking" and training_mode: "student_teacher".
    Supports multiple masking strategies: random, HEALPix-based, channel-based, and combinations.

    Masking:
    Class to generate masks for token sequences and apply them.
    This class supports different masking strategies and combinations.
    Attributes:	
        masking_rate (float): The base rate at which tokens are masked.	
        masking_strategy (str): The strategy used for masking (e.g., "random",	
        "block", "healpix", "channel").	
        current_strategy (str): The current strategy in use, relevant	
                                when using "combination" strategy.	
        "random" - random masking of tokens at the level of the data	
        "block" - masking out large blocks of tokens in 1D, without spatial meaning	
        "healpix" - masking at the level of HEALPix cells, where all child cells	
                    of a parent cell at a specific HEALpix level are masked	
                    if the parent is masked.	
                    The healpix level must be configured with hl_mask.	
                    e.g. masking_strategy_config = {"hl_mask": 1}	
                    with hl_mask the level for masking that we want to apply	
                    e.g. level 1 very large cells masked	
        "channel" - masking data channels, where channels of the data are masked	
                    can be done per-cell (each cell has different channels masked)	
                    or globally (all have the same channels masked).	
                    e.g. masking_strategy_config = {"mode": "per_cell"} or	
                    {"mode": "global"}	
        "causal" - masking the latest timesteps in each token, according to the masking rate.	
        masking_rate_sampling (bool): Whether to sample the masking rate from a distribution.	
        masking_strategy_config (dict): Configuration for the masking strategy, can include	
                                        additional parameters like "hl_mask", etc.	
                                        specific to the masking strategy. See above.
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
        # number of healpix cells
        self.healpix_num_cells = 12 * (4 ** self.healpix_level_data)
        
        # Encoding dimensions
        self.dim_time_enc = 6
        self.mask_value = 0.0
        
        # Initialize the mask, set to None initially,
        # until it is generated in mask_source.
        self.perm_sel: list[np.typing.NDArray] | None = None
        
        # Per-batch strategy tracking
        self.same_strategy_per_batch = self.masking_strategy_config.get(
            "same_strategy_per_batch", False
        )
        self.batch_strategy_set = False
        
        # Validate config based on mode
        self._validate_config(cf)

        self._override_keep_cells: np.ndarray | None = None  # state for context manager

    @contextmanager
    def use_keep_cells(self, keep_cells: np.ndarray | None):
        """
        Context manager to apply view-specific cell masks.
        Allows us to temporarily override the keep_cells used during masking.
        And hence keep local views as subsets of the generated global view.
        """
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
                "If HEALPix masking, hl_mask must be given in masking_strategy_config."
                )
            assert hl_mask < self.healpix_level_data, (
                f"hl_mask={hl_mask} must be < healpix_level={self.healpix_level_data}"
            )
        
        # Check channel masking requirements
        if self.current_strategy == "channel":
            # Ensure that masking_strategy_config contains either 'global' or 'per_cell'
            assert self.masking_strategy_config.get("mode") in [
                "global",
                "per_cell",
            ], "masking_strategy_config must contain 'mode' key with value 'global' or 'per_cell'."

            # Verify source/target channels match across streams
            for stream in cf.streams:
                # check explicit includes
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

    def reset_rng(self, rng) -> None:
        """
        Reset rng after epoch to ensure proper randomization
        """
        self.rng = rng

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
            return self.masking_strategy

    # NOTE: Here we do the student-teacher mode, where we generate teacher (global) and student (local) views
    # We are focussing really on the iBOT/DINO case first, but some small modifications are needed.
    # We produce the ViewMetadata here, which will then be used to actually cut out the data.
    # Not at all tested for multiple streams.
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
        Supports optional rate sampling.
        To be replaced by actual sampling from a distribution later.
        """
        if "keep_m" in view_config:
            # Absolute count provided; convert to rate
            keep_m = int(view_config["keep_m"])
            total = self.healpix_num_cells
            rate = float(keep_m) / total
        else:
            rate = float(view_config["rate"])
        
        # Sample this. TODO: probably should replace with sampling
        # from a distribution here.
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

    # Here this is iBOT-style, generating strict subset of global view.
    # To be extended to handle JEPA and DINO options, where not necessarily
    # strict subsets.
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

    # Here we are back to the original masking mode, mask_source, mask_target (MAE-style)
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
        # this helps us to keep local views as subsets of global views
        # TODO: check logic here.
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
            assert False, f"Unknown masking strategy: {self.current_strategy}"

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
                # If masking strategy is channel, handle target tokens differently.
                # We don't have Booleans per cell, instead per channel per cell,
                # we set the unmasked channels to NaN so not in loss.
                selected_tensors = []
                for c, p in zip(cc, pp, strict=True):
                    # slightly complicated as the first dimension of c varies with data in the cell.
                    # do not mask the first 8 channels,
                    # and set unmasked channels to nan
                    c[:, (self.dim_time_enc + coords.shape[-1] + geoinfos.shape[-1]):][
                        :, ~p[0, (self.dim_time_enc + coords.shape[-1] + geoinfos.shape[-1]):]
                    ] = torch.nan
                    selected_tensors.append(c)
            elif self.current_strategy == "causal":
                # select only the target times where mask is True
                selected_tensors = [c for i, c in enumerate(cc) if pp[i]]
            else:
                # For other masking strategies, we simply select the tensors where the mask is True.
                selected_tensors = [c for c, p in zip(cc, pp, strict=True) if p]

            # Append the selected tensors to the processed_target_tokens list.
            if selected_tensors:
                processed_target_tokens.append(torch.cat(selected_tensors))
            else:
                processed_target_tokens.append(
                    torch.empty(0, feat_dim, dtype=coords.dtype, device=coords.device)
                )

        return processed_target_tokens

    def _get_sampling_rate(self) -> float:
        """
        Get the sampling, if requested by sampling it itself	
        """

        # if masking_rate_sampling is enabled, sample the rate from a normal distribution.
        if self.masking_rate_sampling:
            rate = np.clip(
                np.abs(self.rng.normal(loc=self.masking_rate, scale=1.0 / (2.5 * np.pi))),
                0.01, 
                0.99,
            )
        else:
            rate = self.masking_rate
        return rate

    def _generate_healpix_mask(self, token_lens: list[int], rate: float) -> np.ndarray:
        """
        Generates a token-level mask based on hierarchical HEALPix cell selection.	
        This method identifies parent cells at a lower resolution (hl_mask) and	
        masks all the child cells (and their corresponding tokens) at the data	
        resolution (hl_data).	
        Args:	
            token_lens (list[int]): A list containing the number of tokens in each cell.	
            rate (float): The desired masking rate, applied to the parent cells.	
        Returns:	
            np.ndarray: A flat boolean array (the token-level mask).	
        """        
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

    def _generate_channel_mask(
        self,	        
        tokenized_data: list[torch.Tensor],	
        rate: float,	
        coords: torch.Tensor,	
        geoinfos: torch.Tensor,	
        source: torch.Tensor,	
    ) -> list[np.typing.NDArray]:
        """
        Generates a channel mask for each cell, handling completely empty tensors.	
        This method is robust against cells represented as 1D tensors of shape [0].	
        Args:	
            tokenized_data (list[torch.Tensor]): A list of tensors. Most will have a shape of	
                                                (dim, num_tokens, num_channels), but some may	
                                            be empty with a shape of (0,), no data in cell	
            rate (float): The desired masking rate for channels.	
            coords (torch.Tensor): The coordinates tensor.	
            geoinfos (torch.Tensor): The geoinfos tensor.	
        Returns:	
            list[np.ndarray]: A list of boolean masks. Each mask corresponds to a tensor	
                            in tokenized_data.	
        """
        if not tokenized_data:
            return []
        
        num_channels = self.dim_time_enc + coords.shape[-1] + geoinfos.shape[-1] + source.shape[-1]
        assert num_channels > 0, "For channel masking, number of channels has to be nonzero."
        num_fixed_channels = self.dim_time_enc + coords.shape[-1] + geoinfos.shape[-1]
        num_data_channels = source.shape[-1]
        mask_count = int(num_data_channels * rate)

        # cat all tokens for efficient processing, split at the end again
        # masks are generated simulatneously for all cells
        
        tokenized_data_lens = [len(t) for t in tokenized_data]
        tokenized_data_merged = torch.cat(tokenized_data)
        num_tokens = tokenized_data_merged.shape[0]
        token_size = tokenized_data_merged.shape[1]
        
        if self.masking_strategy_config.get("mode") == "global":
            # generate global mask
            channel_mask = np.zeros(num_channels, dtype=bool)
            m = num_fixed_channels + self.rng.choice(num_data_channels, mask_count, replace=False)
            channel_mask[m] = True
            full_mask = np.zeros_like(tokenized_data_merged).astype(np.bool_)
            full_mask[:, :] = channel_mask
        else:  # different mask per cell
            # generate all False mask but with swapped token_size and num_tokens dims so that	
            # the masking is constant per token
            channel_mask = np.zeros((token_size, num_tokens, num_channels), dtype=bool)
            # apply masking
            nc = (num_tokens, num_data_channels)
            channel_mask[:, :, num_fixed_channels:] = self.rng.uniform(0, 1, nc) < rate
            # recover correct shape, i.e. swap token_size and num_tokens
            full_mask = channel_mask.transpose([1, 0, 2])
        
        # split across cells again
        full_mask = np.split(full_mask, np.cumsum(tokenized_data_lens[:-1]))
        return full_mask

    def _generate_causal_mask(	    
        self,	        
        tokenized_data: list[torch.Tensor],	
        rate: float,	
        coords: torch.Tensor,	
        geoinfos: torch.Tensor,	
        source: torch.Tensor,	
    ) -> list[np.typing.NDArray]:	
        """	
        Generates a causal mask, masking the latest times	
        in each tokenized_data according to the masking rate.	
        """
        if not tokenized_data:
            return []
        
        token_lens = np.array([len(token_data) for token_data in tokenized_data])
        if len(token_lens) == 0:
            return []
        
        num_future_to_mask = (rate * token_lens).astype(int)
        start_mask_indices = np.maximum(1, token_lens - num_future_to_mask)
        mask_valid = token_lens > 1
        start_mask_indices = np.where(mask_valid, start_mask_indices, token_lens)
        
        # Create masks with list comprehension	
        # Needed to handle variable lengths
        
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
