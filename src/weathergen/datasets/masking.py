import logging

import numpy as np
import torch

from weathergen.common.config import Config
from weathergen.datasets.batch import SampleMetaData

_logger = logging.getLogger(__name__)


class Masker:
    """Class to generate masks for token sequences and apply them.
    This class supports different masking strategies and combinations.

    Attributes:
        masking_rate (float): The base rate at which tokens are masked.
        masking_strategy (str): The strategy used for masking (e.g., "random",
        "block", "healpix", "channel", "causal", "temporal_crop").
        current_strategy (str): The current strategy in use, relevant
                                when using "combination" strategy.

    Supported Masking Strategies:
        "random" - random masking of tokens at the level of the data

        "block" - masking out large blocks of tokens in 1D, without spatial meaning

        "forecast" - all tokens go to target (none kept in source)

        "healpix" - masking at the level of HEALPix cells, where all child cells
                    of a parent cell at a specific HEALpix level are masked
                    if the parent is masked.
                    The healpix level must be configured with hl_mask.
                    e.g. masking_strategy_config = {"hl_mask": 1}
                    with hl_mask the level for masking that we want to apply
                    e.g. level 1 very large cells masked

        "spatial" - spatial masking at cell level using generate_cell_keep_mask.
                    Same spatial mask applied to all timesteps.
                    Config options:
                    - cell_strategy: "random" or "healpix" (default: "random")
                    - hl_mask: required if cell_strategy="healpix"
                    e.g. masking_strategy_config = {"cell_strategy": "healpix", "hl_mask": 3}

        "spatiotemporal" - spatio-temporal masking with different spatial mask per timestep.
                           Each timestep gets independently sampled spatial mask.
                           Config options:
                           - cell_strategy: "random" or "healpix" (default: "random")
                           - hl_mask: required if cell_strategy="healpix"
                           e.g. masking_strategy_config = {"cell_strategy": "healpix", "hl_mask": 3}

        "channel" - masking data channels, where channels of the data are masked
                    can be done per-cell (each cell has different channels masked)
                    or globally (all have the same channels masked).
                    e.g. masking_strategy_config = {"mode": "per_cell"} or
                    {"mode": "global"}

        "causal" - masking the latest timesteps in each token, according to the masking rate.
                   Source gets early timesteps, target gets late timesteps.
                   Useful for autoregressive/causal modeling.
                   No additional config required.

        "temporal_crop" - keeps a specific temporal portion of the data based on crop_direction.
                          Source gets the selected portion, target gets the complement.
                          Required config:
                          - crop_direction: "start", "end", or "middle"
                          - masking_rate is used to determine fraction to keep in source
                          e.g. masking_strategy_config = {"crop_direction": "end"}
                          with masking_rate=0.3 keeps the last 30% of timesteps in source

        masking_rate_sampling (bool): Whether to sample the masking rate from a distribution.
        masking_strategy_config (dict): Configuration for the masking strategy, can include
                                        additional parameters like "hl_mask", "crop_direction", etc.
                                        specific to the masking strategy. See above.
    """

    def __init__(self, cf: Config):
        self.rng = None
        self.masking_rate = cf.masking_rate
        self.masking_strategy = cf.masking_strategy
        self.current_strategy = cf.masking_strategy  # Current strategy in use
        self.masking_rate_sampling = cf.masking_rate_sampling
        # masking_strategy_config is a dictionary that can hold any additional parameters
        self.healpix_level_data = cf.healpix_level
        self.masking_strategy_config = cf.get("masking_strategy_config", {})
        self.perm_sel = None
        self.mask_tokens = None
        self.mask_channels = None

        self.mask_value = 0.0
        self.dim_time_enc = 6

        # number of healpix cells
        self.healpix_num_cells = 12 * (4**self.healpix_level_data)

        # Per-batch strategy tracking
        self.same_strategy_per_batch = self.masking_strategy_config.get(
            "same_strategy_per_batch", False
        )
        self.batch_strategy_set = False

        # Check for required masking_strategy_config at construction time
        if self.current_strategy == "healpix":
            hl_data = self.healpix_level_data
            hl_mask = self.masking_strategy_config.get("hl_mask")
            assert hl_data is not None and hl_mask is not None, (
                "If HEALPix masking, hl_mask must be given in masking_strategy_config."
            )
            assert hl_mask < hl_data, "hl_mask must be less than hl_data for HEALPix masking."

        if self.current_strategy == "channel":
            # Ensure that masking_strategy_config contains either 'global' or 'per_cell'
            assert self.masking_strategy_config.get("mode") in [
                "global",
                "per_cell",
            ], "masking_strategy_config must contain 'mode' key with value 'global' or 'per_cell'."

            # check all streams that source and target channels are identical
            for stream in cf.streams:
                # check explicit includes
                source_include = stream.get("source_include", [])
                target_include = stream.get("target_include", [])
                assert set(source_include) == set(target_include), (
                    "Source and target channels not identical. Required for masking_mode=channel"
                )
                # check excludes
                source_exclude = stream.get("source_exclude", [])
                target_exclude = stream.get("target_exclude", [])
                assert set(source_exclude) == set(target_exclude), (
                    "Source and target channels not identical. Required for masking_mode=channel"
                )

        if self.current_strategy == "temporal_crop":
            # Ensure that crop_direction is specified
            crop_direction = self.masking_strategy_config.get("crop_direction")
            assert crop_direction in ["start", "end", "middle"], (
                "temporal_crop strategy requires 'crop_direction' in masking_strategy_config "
                "with value 'start', 'end', or 'middle'."
            )

        if self.current_strategy in ["spatial", "spatiotemporal"]:
            # Validate spatial/spatiotemporal strategy config
            cell_strategy = self.masking_strategy_config.get("cell_strategy", "random")
            assert cell_strategy in ["random", "healpix"], (
                f"{self.current_strategy} strategy requires 'cell_strategy' to be 'random' or 'healpix', "
                f"got '{cell_strategy}'"
            )
            if cell_strategy == "healpix":
                hl_mask = self.masking_strategy_config.get("hl_mask")
                assert hl_mask is not None and hl_mask < self.healpix_level_data, (
                    f"{self.current_strategy} with cell_strategy='healpix' requires 'hl_mask' "
                    f"in masking_strategy_config and hl_mask < data level {self.healpix_level_data}"
                )

    def reset_rng(self, rng) -> None:
        """
        Reset rng after mini_epoch to ensure proper randomization
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
            # Non-combination strategy, return as is
            return self.masking_strategy

    def mask_source_idxs(
        self,
        idxs_cells,
        idxs_cells_lens,
        keep_mask: np.typing.NDArray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate masks for source data at the index level.

        This method generates boolean masks that determine which tokens/channels
        should be kept in the source (True = KEEP, False = MASK).

        Args:
            idxs_cells: List of lists of token indices per cell
            idxs_cells_lens: List of lists of token lengths per cell
            keep_mask: Optional cell-level keep mask (True = keep cell)

        Return:
            Tuple of (mask_tokens, mask_channels):
                - mask_tokens: np.ndarray[bool] of length num_tokens (True = KEEP in source)
                - mask_channels: np.ndarray[bool] or None for channel masking
        """

        self.mask_tokens, self.mask_channels = None, None

        num_tokens = torch.tensor([len(t) for t in idxs_cells_lens]).sum().item()

        # If there are no tokens, return empty lists.
        if num_tokens == 0:
            return (self.mask_tokens, self.mask_channels)

        # If an explicit keep_mask is provided we bypass strategy selection and directly
        # construct the token-level mask from it. keep_mask expresses cells to KEEP (True=keep).
        # Otherwise fall back to the configured strategy logic.
        if keep_mask is not None:
            assert len(keep_mask) == len(idxs_cells_lens), (
                "keep_mask length does not match number of cells."
            )
            # build token level mask: for each cell replicate the keep flag across its tokens
            token_level_flags: list[np.typing.NDArray] = []
            for km, lens_cell in zip(keep_mask, idxs_cells_lens, strict=True):
                num_tokens_cell = len(lens_cell)
                if num_tokens_cell == 0:
                    continue
                token_level_flags.append(
                    np.ones(num_tokens_cell, dtype=bool)
                    if km
                    else np.zeros(num_tokens_cell, dtype=bool)
                )
            if token_level_flags:
                self.mask_tokens = np.concatenate(token_level_flags)
            else:
                self.mask_tokens = np.array([], dtype=bool)
            return (self.mask_tokens, self.mask_channels)

        # clean strategy selection
        self.current_strategy = self._select_strategy()

        # Set the masking rate.
        rate = self._get_sampling_rate()

        if rate == 0.0:
            _logger.warning(
                "masking_rate is 0. This will result in empty target. The sample will be skipped. "
                + "If this occurs repeatedly the masking settings likely need to be revised."
            )

        # Handle the special case where all tokens are masked (kept in source)
        if rate == 1.0:
            self.mask_tokens = np.ones(num_tokens, dtype=bool)
            return (self.mask_tokens, self.mask_channels)

        # Implementation of different masking strategies
        # Mask semantics: True = KEEP in source (unmasked), False = MASK (goes to target)

        if self.current_strategy == "random":
            # Random masking: randomly select tokens to mask
            # True = KEEP in source, False = mask (for target)
            self.mask_tokens = self.rng.uniform(0, 1, num_tokens) >= rate

        elif self.current_strategy == "block":
            # Block masking: mask a contiguous block of tokens
            self.mask_tokens = np.ones(num_tokens, dtype=bool)  # Start with all kept
            block_size = int(np.round(rate * num_tokens))
            if block_size > 0 and num_tokens > 0:
                start_index = self.rng.integers(0, max(1, num_tokens - block_size + 1))
                self.mask_tokens[start_index : start_index + block_size] = False

        elif self.current_strategy == "forecast":
            # Forecast: all tokens go to target (none kept in source)
            self.mask_tokens = np.zeros(num_tokens, dtype=bool)

        elif self.current_strategy == "healpix":
            # HEALPix hierarchical masking at cell level
            token_lens = [len(t) for t in idxs_cells_lens]
            self.mask_tokens = self._generate_healpix_mask_idxs(token_lens, rate)

        elif self.current_strategy == "channel":
            # Channel masking: mask entire channels rather than tokens
            # For idx-based approach, we still keep all tokens but set mask_channels
            # Note: This is handled differently - all tokens are kept, channels are masked
            self.mask_tokens = np.ones(num_tokens, dtype=bool)
            # Channel masking is applied at the tokenization level, not here
            # We just mark that this is channel masking for downstream processing
            _logger.warning("Channel masking in idx-based approach needs special handling in tokenization")

        elif self.current_strategy == "causal":
            # Causal masking: mask the LATEST timesteps in each cell
            # Source gets early timesteps, target gets late timesteps
            self.mask_tokens = self._generate_causal_mask_idxs(idxs_cells_lens, rate)

        elif self.current_strategy == "temporal_crop":
            # Temporal cropping: keep timesteps based on crop_direction
            crop_direction = self.masking_strategy_config.get("crop_direction", "end")
            temporal_config = {"crop_direction": crop_direction, "rate": 1.0 - rate}

            # _generate_temporal_crop_mask_idxs returns True = KEEP
            # rate here is fraction to MASK, so we invert it for the function
            self.mask_tokens = self._generate_temporal_crop_mask_idxs(
                idxs_cells_lens, temporal_config
            )

        elif self.current_strategy == "spatial":
            # Spatial masking: same spatial mask for all timesteps
            cell_strategy = self.masking_strategy_config.get("cell_strategy", "random")

            # Generate cell-level keep mask using generate_cell_keep_mask
            # Note: rate here is fraction to KEEP in source (inverted from masking_rate semantics)
            keep_rate = 1.0 - rate
            cell_keep_mask = self.generate_cell_keep_mask(
                num_cells=self.healpix_num_cells,
                strategy=cell_strategy,
                rate=keep_rate,
                masking_strategy_config=self.masking_strategy_config,
            )

            # Convert cell-level mask to token-level mask by replicating across tokens
            token_level_flags: list[np.typing.NDArray] = []
            for cell_kept, lens_cell in zip(cell_keep_mask, idxs_cells_lens, strict=True):
                num_tokens_cell = len(lens_cell)
                if num_tokens_cell == 0:
                    continue
                # True = KEEP in source (cell is kept)
                token_level_flags.append(
                    np.ones(num_tokens_cell, dtype=bool)
                    if cell_kept
                    else np.zeros(num_tokens_cell, dtype=bool)
                )

            if token_level_flags:
                self.mask_tokens = np.concatenate(token_level_flags)
            else:
                self.mask_tokens = np.array([], dtype=bool)

        elif self.current_strategy == "spatiotemporal":
            # Spatiotemporal masking: different spatial mask per timestep
            # Each timestep gets an independently generated spatial mask
            cell_strategy = self.masking_strategy_config.get("cell_strategy", "random")
            keep_rate = 1.0 - rate

            # Build token-level mask where each token gets its own spatial mask
            token_level_flags: list[np.typing.NDArray] = []
            for cell_idx, lens_cell in enumerate(idxs_cells_lens):
                num_tokens_cell = len(lens_cell)
                if num_tokens_cell == 0:
                    continue

                # Generate independent spatial mask for each timestep in this cell
                cell_token_masks = []
                for _ in range(num_tokens_cell):
                    # Generate new spatial mask across all cells
                    cell_keep_mask = self.generate_cell_keep_mask(
                        num_cells=self.healpix_num_cells,
                        strategy=cell_strategy,
                        rate=keep_rate,
                        masking_strategy_config=self.masking_strategy_config,
                    )
                    # Extract the keep/mask decision for this specific cell
                    cell_token_masks.append(cell_keep_mask[cell_idx])

                # Convert to boolean array: one mask value per token in this cell
                token_level_flags.append(np.array(cell_token_masks, dtype=bool))

            if token_level_flags:
                self.mask_tokens = np.concatenate(token_level_flags)
            else:
                self.mask_tokens = np.array([], dtype=bool)

        else:
            raise ValueError(f"Unsupported masking strategy: {self.current_strategy}.")

        return (self.mask_tokens, self.mask_channels)

    def mask_targets_idxs(
        self,
        idxs_cells,
        idxs_cells_lens,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate masks for target data at the index level.

        This method uses the masks generated by mask_source_idxs and inverts them
        (except for forecast strategy) to determine which tokens should be in the target.

        Args:
            idxs_cells: List of lists of token indices per cell
            idxs_cells_lens: List of lists of token lengths per cell

        Returns:
            Tuple of (mask_tokens, mask_channels, idxs_ord_inv):
                - mask_tokens: np.ndarray[bool] (True = INCLUDE in target)
                - mask_channels: np.ndarray[bool] or None for channel masking
                - idxs_ord_inv: torch.Tensor for reordering (only for forecast strategy)
        """
        # mask_source_idxs must be called first
        assert self.mask_tokens is not None or self.mask_channels is not None, (
            "mask_source_idxs must be called before mask_targets_idxs"
        )

        idxs_ord_inv = torch.tensor([], dtype=torch.int64)

        if self.current_strategy == "forecast":
            # Forecast strategy: all tokens go to target
            num_tokens = torch.tensor([len(t) for t in idxs_cells_lens]).sum().item()
            self.mask_tokens = np.ones(num_tokens, dtype=bool)

            # Create inverse map for reordering to output data points in same order as input
            idxs_ord = torch.cat([t for tt in idxs_cells for t in tt])
            idxs_ord_inv = torch.argsort(idxs_ord)

        else:
            # For all other masking strategies: target is complement of source
            # Source had True = KEEP, so target has True = was MASKED (now INCLUDE in target)
            if self.mask_tokens is not None:
                self.mask_tokens = ~self.mask_tokens
            if self.mask_channels is not None:
                self.mask_channels = ~self.mask_channels

        return (self.mask_tokens, self.mask_channels, idxs_ord_inv)

    def _get_sampling_rate(self):
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

    def _generate_healpix_mask_idxs(self, token_lens: list[int], rate: float) -> np.typing.NDArray:
        """
        Generates a token-level mask based on hierarchical HEALPix cell selection (idx-based version).

        This method identifies parent cells at a lower resolution (hl_mask) and
        masks all the child cells (and their corresponding tokens) at the data
        resolution (hl_data).

        Args:
            token_lens (list[int]): A list containing the number of tokens in each cell.
            rate (float): The desired masking rate, applied to the parent cells.

        Returns:
            np.ndarray: A flat boolean array (True = KEEP in source, False = MASK for target).
        """

        # hl_mask should be provided in masking_strategy_config
        hl_data = self.healpix_level_data
        hl_mask = self.masking_strategy_config.get("hl_mask")

        assert len(token_lens) == self.healpix_num_cells, (
            f"Expected {self.healpix_num_cells} cells at level {hl_data}, got {len(token_lens)}."
        )

        # Calculate the number of parent cells at the mask level (hl_mask)
        num_parent_cells = 12 * (4**hl_mask)
        level_diff = hl_data - hl_mask
        num_children_per_parent = 4**level_diff

        # Choose parent cells to mask based on the specified rate.
        num_parents_to_mask = int(np.round(rate * num_parent_cells))

        if num_parents_to_mask == 0:
            return np.ones(sum(token_lens), dtype=bool)  # All kept in source

        # Select parent cells to mask
        parent_ids_to_mask = self.rng.choice(num_parent_cells, num_parents_to_mask, replace=False)

        # For each parent ID, calculate the child indices and set them in the mask
        parent_ids = np.asarray(parent_ids_to_mask)
        child_offsets = np.arange(num_children_per_parent)
        child_indices = (parent_ids[:, None] * num_children_per_parent + child_offsets).reshape(-1)

        # set mask list for children (True = MASK, then we'll invert)
        cell_mask = np.zeros(self.healpix_num_cells, dtype=bool)
        cell_mask[child_indices] = True

        # Invert so True = KEEP in source, False = MASK for target
        cell_mask = ~cell_mask

        # Make the cell-level mask flat and apply it to the token lengths.
        # np.repeat repeats each element of `cell_mask` a number of times specified by `token_lens`.
        flat_mask = np.repeat(cell_mask, token_lens)

        return flat_mask

    def _generate_causal_mask_idxs(
        self,
        idxs_cells_lens: list[list[int]],
        rate: float,
    ) -> np.typing.NDArray:
        """
        Generates a causal mask at the index level, masking the latest times
        in each cell according to the masking rate.

        Args:
            idxs_cells_lens: List of lists of token lengths per cell
            rate: Fraction of timesteps to MASK (goes to target)

        Returns:
            np.ndarray: Boolean mask where True = KEEP in source, False = MASK for target
        """
        if not idxs_cells_lens:
            return np.array([], dtype=bool)

        # Extract all lengths at once
        token_lens = np.array([len(lens_cell) for lens_cell in idxs_cells_lens])

        if len(token_lens) == 0:
            return np.array([], dtype=bool)

        # Calculate start indices for masking
        # astype(int) performs floor operation by truncation
        num_future_to_mask = (rate * token_lens).astype(int)
        start_mask_indices = np.maximum(1, token_lens - num_future_to_mask)

        # Handle edge cases
        mask_valid = token_lens > 1  # Only cells with >1 timestep can be masked
        start_mask_indices = np.where(mask_valid, start_mask_indices, token_lens)

        # Create masks with list comprehension
        # True = KEEP in source, False = MASK for target
        full_mask = []
        for token_len, start_idx in zip(token_lens, start_mask_indices, strict=True):
            if token_len > 1:
                # Keep early timesteps, mask late ones
                mask = np.concatenate([
                    np.ones(start_idx, dtype=bool),
                    np.zeros(max(0, token_len - start_idx), dtype=bool),
                ])
            elif token_len == 1:
                mask = np.ones(1, dtype=bool)  # Keep single timestep
            else:
                mask = np.array([], dtype=bool)
            full_mask.append(mask)

        return np.concatenate(full_mask) if full_mask else np.array([], dtype=bool)

    def _generate_temporal_crop_mask_idxs(
        self,
        idxs_cells_lens: list[list[int]],
        temporal_config: dict,
    ) -> np.typing.NDArray:
        """
        Generates a temporal cropping mask at the index level that KEEPS selected timesteps.

        Args:
            idxs_cells_lens: List of lists of token lengths per cell
            temporal_config: Dict with 'crop_direction' ("start", "end", "middle")
                             and 'rate' (fraction of timesteps to KEEP)

        Returns:
            np.ndarray: Boolean mask where True = KEEP in source, False = MASK for target
        """
        if not idxs_cells_lens:
            return np.array([], dtype=bool)

        crop_direction = temporal_config.get("crop_direction", "end")
        rate = temporal_config.get("rate", 0.5)

        assert crop_direction in {"start", "end", "middle"}, (
            f"crop_direction must be 'start', 'end', or 'middle', got {crop_direction}"
        )

        # Extract all lengths at once
        token_lens = np.array([len(lens_cell) for lens_cell in idxs_cells_lens])

        if len(token_lens) == 0:
            return np.array([], dtype=bool)

        # Calculate how many timesteps to keep per cell
        num_to_keep = np.maximum(1, (rate * token_lens).astype(int))

        # Create masks based on crop direction
        full_mask = []
        for token_len, n_keep in zip(token_lens, num_to_keep, strict=True):
            if token_len == 0:
                full_mask.append(np.array([], dtype=bool))
                continue

            # Ensure we don't try to keep more than we have
            n_keep = min(n_keep, token_len)

            # Create mask based on direction (True = KEEP in source)
            mask = np.zeros(token_len, dtype=bool)

            if crop_direction == "start":
                # Keep first n_keep timesteps in source
                mask[:n_keep] = True
            elif crop_direction == "end":
                # Keep last n_keep timesteps in source
                mask[-n_keep:] = True
            else:  # middle
                # Keep middle n_keep timesteps in source
                start_idx = (token_len - n_keep) // 2
                mask[start_idx : start_idx + n_keep] = True

            full_mask.append(mask)

        return np.concatenate(full_mask) if full_mask else np.array([], dtype=bool)

    def build_views_for_stream(
        self,
        num_cells: int,
        teacher_cfg: dict,
        student_cfg: dict,
        relationship: str = "subset",
    ) -> tuple[np.typing.NDArray, list[np.typing.NDArray], list[SampleMetaData]]:
        """
        Construct teacher/student keep masks for a stream.
        SampleMetaData is currently just a dict with the masking params used.
        """

        strat_teacher = teacher_cfg.get("strategy", "random")
        rate_teacher = teacher_cfg.get("rate")
        t_cfg_extra = teacher_cfg.get("masking_strategy_config")

        teacher_keep_mask = self.generate_cell_keep_mask(
            num_cells=num_cells,
            strategy=strat_teacher,
            rate=rate_teacher,
            masking_strategy_config=t_cfg_extra,
        )

        num_views = student_cfg.get("num_views", 1)
        strat_student = student_cfg.get("masking_strategy", student_cfg.get("strategy", "random"))
        rate_student = student_cfg.get("rate")
        s_cfg_extra = student_cfg.get("masking_strategy_config")

        student_keep_masks: list[np.typing.NDArray] = []
        for _ in range(num_views):
            base = self.generate_cell_keep_mask(
                num_cells=num_cells,
                strategy=strat_student,
                rate=rate_student,
                masking_strategy_config=s_cfg_extra,
            )
            if relationship == "subset":
                keep = base & teacher_keep_mask
            elif relationship == "disjoint":
                keep = base & (~teacher_keep_mask)
            else:
                keep = base
            student_keep_masks.append(keep)

        metadata: list[SampleMetaData] = [
            SampleMetaData(
                masking_params=teacher_cfg,
            )
        ]
        for _, _ in enumerate(student_keep_masks):
            metadata.append(
                SampleMetaData(
                    masking_params=student_cfg,
                )
            )

        return teacher_keep_mask, student_keep_masks, metadata

    # ---------------------------------------------------------------------
    # Cell-level keep mask generation (teacher/student view selection)
    # ---------------------------------------------------------------------
    def generate_cell_keep_mask(
        self,
        num_cells: int,
        strategy: str | None = None,
        rate: float | None = None,
        masking_strategy_config: dict | None = None,
        constraint_keep_mask: np.typing.NDArray | None = None,
    ) -> np.typing.NDArray:
        """Generate a boolean keep mask at data healpix level (True = keep cell).

        Parameters
        ----------
        num_cells : int
            Number of cells at data level (should equal 12 * 4**healpix_level).
        strategy : str | None
            Cell selection strategy: currently supports 'random' and 'healpix'. Uses
            instance default if None.
        rate : float | None
            Fraction of parent cells (healpix) or data cells (random) to keep. Falls back
            to instance masking_rate if None.
        masking_strategy_config : dict | None
            Optional override of strategy config (e.g., {'hl_mask': 3}).
        constraint_keep_mask : np.ndarray | None
            Optional boolean mask of allowed cells (True = allowed). Selection will be
            limited to these cells. For subset/disjoint relationships.

        Returns
        -------
        np.ndarray
            Boolean array of shape [num_cells] where True indicates the cell is kept.
        """
        strat = strategy or self.masking_strategy
        cfg = masking_strategy_config or self.masking_strategy_config
        keep_rate = rate if rate is not None else self.masking_rate

        # sample rate if requested (only if explicit rate not provided)
        if rate is None and self.masking_rate_sampling:
            keep_rate = self._get_sampling_rate()

        assert 0.0 <= keep_rate <= 1.0, f"keep_rate out of bounds: {keep_rate}"
        assert num_cells == self.healpix_num_cells, (
            "num_cells inconsistent with configured healpix level."
        )

        if strat not in {"random", "healpix"}:
            raise NotImplementedError(
                f"Cell selection strategy '{strat}' not supported for keep mask generation."
            )

        if strat == "random":
            base_mask = self.rng.uniform(0, 1, num_cells) < keep_rate
        else:  # healpix hierarchical selection
            hl_data = self.healpix_level_data
            hl_mask = cfg.get("hl_mask")
            assert hl_mask is not None and hl_mask < hl_data, (
                "For healpix keep mask generation, cfg['hl_mask'] must be set and < data level."
            )
            num_parent_cells = 12 * (4**hl_mask)
            level_diff = hl_data - hl_mask
            num_children_per_parent = 4**level_diff
            # number of parents to KEEP
            num_parents_to_keep = int(np.round(keep_rate * num_parent_cells))
            if num_parents_to_keep == 0:
                base_mask = np.zeros(num_cells, dtype=bool)
            else:
                parent_ids = self.rng.choice(num_parent_cells, num_parents_to_keep, replace=False)
                child_offsets = np.arange(num_children_per_parent)
                child_indices = (
                    parent_ids[:, None] * num_children_per_parent + child_offsets
                ).reshape(-1)
                base_mask = np.zeros(num_cells, dtype=bool)
                base_mask[child_indices] = True

        # apply constraint if provided (only keep those cells within allowed)
        if constraint_keep_mask is not None:
            assert constraint_keep_mask.shape[0] == num_cells, "constraint_keep_mask wrong shape"
            base_mask = base_mask & constraint_keep_mask

        return base_mask
