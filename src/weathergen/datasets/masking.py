import logging
from dataclasses import dataclass

import numpy as np
import torch

from weathergen.common.config import Config
from weathergen.datasets.batch import SampleMetaData

_logger = logging.getLogger(__name__)


# Convert to torch.bool
def to_bool_tensor(arr):
    return torch.from_numpy(np.asarray(arr)).to(torch.bool)


@dataclass
class MaskingStrategy:
    strategy: str
    config: dict
    num_samples: int


class Masker:
    """Class to generate masks for token sequences and apply them.
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
                   Requires token-level masking via _generate_temporal_mask.
        "temporal_crop" - keeps a specific temporal portion based on crop_direction.
                          Required config: crop_direction ("start", "end", or "middle")
                          Requires token-level masking via _generate_temporal_mask.
        "spatiotemporal" - spatial masking with different mask per timestep.
                           Requires token-level masking via _generate_spatiotemporal_mask.
                           Config: cell_strategy ("random" or "healpix"), hl_mask (for healpix)
        masking_rate_sampling (bool): Whether to sample the masking rate from a distribution.
        masking_strategy_config (dict): Configuration for the masking strategy, can include
                                        additional parameters like "hl_mask", "crop_direction",
                                        "cell_strategy", etc. specific to the masking strategy.
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

        if self.current_strategy == "spatiotemporal":
            # Validate spatiotemporal strategy config
            cell_strategy = self.masking_strategy_config.get("cell_strategy", "random")
            assert cell_strategy in ["random", "healpix"], (
                f"spatiotemporal strategy requires 'cell_strategy' to be 'random' or 'healpix', "
                f"got '{cell_strategy}'"
            )
            if cell_strategy == "healpix":
                hl_mask = self.masking_strategy_config.get("hl_mask")
                assert hl_mask is not None and hl_mask < self.healpix_level_data, (
                    f"spatiotemporal with cell_strategy='healpix' requires 'hl_mask' "
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

    def _generate_healpix_mask(self, token_lens: list[int], rate: float) -> np.typing.NDArray:
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

        rate = self._get_sampling_rate()

        # Choose parent cells to mask based on the specified rate.
        num_parents_to_mask = int(np.round(rate * num_parent_cells))

        if num_parents_to_mask == 0:
            return np.zeros(sum(token_lens), dtype=bool)

        # Select parent cells to mask
        parent_ids_to_mask = self.rng.choice(num_parent_cells, num_parents_to_mask, replace=False)

        # For each parent ID, calculate the child indices and set them in the mask
        parent_ids = np.asarray(parent_ids_to_mask)
        child_offsets = np.arange(num_children_per_parent)
        child_indices = (parent_ids[:, None] * num_children_per_parent + child_offsets).reshape(-1)

        # set mask list for children
        cell_mask = np.zeros(self.healpix_num_cells, dtype=bool)
        cell_mask[child_indices] = True

        # Make the cell-level mask flat and apply it to the token lengths.
        # np.repeat repeats each element of `cell_mask` a number of times specified by `token_lens`.
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

        # masking rate sampling, to be refactored as shared between methods
        rate = self._get_sampling_rate()

        # isolate the number of actual data channels. 6 refers to time.
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

            full_mask = np.zeros_like(tokenized_data_merged).astype(np.bool)
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

        rate = self._get_sampling_rate()

        # Extract all lengths at once
        token_lens = np.array([len(token_data) for token_data in tokenized_data])

        if len(token_lens) == 0:
            return []

        # Calculate start indices for masking
        # astype(int) performs floor operation by truncation
        num_future_to_mask = (rate * token_lens).astype(int)
        start_mask_indices = np.maximum(1, token_lens - num_future_to_mask)

        # Handle edge cases
        mask_valid = token_lens > 1  # Only cells with >1 timestep can be masked
        start_mask_indices = np.where(mask_valid, start_mask_indices, token_lens)

        # Create masks with list comprehension
        # Needed to handle variable lengths
        full_mask = [
            (
                np.concatenate(
                    [
                        np.zeros(start_idx, dtype=bool),
                        np.ones(max(0, token_len - start_idx), dtype=bool),
                    ]
                )
                if token_len > 1
                else (np.zeros(1, dtype=bool) if token_len == 1 else np.array([], dtype=bool))
            )
            for token_len, start_idx in zip(token_lens, start_mask_indices, strict=False)
        ]

        return full_mask

    def build_samples_for_stream(
        self,
        training_mode: str,
        num_cells: int,
        target_cfg: dict,
        source_cfg: dict,
    ) -> tuple[np.typing.NDArray, list[np.typing.NDArray], list[SampleMetaData]]:
        """
        Construct teacher/student keep masks for a stream.
        SampleMetaData is currently just a dict with the masking params used.
        """

        # get source and target configs; target defaults to source config

        source_num_samples = source_cfg.get("num_samples", 1)
        source_strategy = source_cfg.get("masking_strategy", source_cfg.get("strategy", "random"))
        source_masking_params = source_cfg.get("masking_strategy_config")
        relationship = source_cfg.get("relationship", "complement")

        if target_cfg is not None:
            target_num_samples = target_cfg.get("num_samples", 1)
            target_strategy = target_cfg.get("strategy", "random")
            target_masking_params = target_cfg.get("masking_strategy_config")
        else:
            target_strategy = source_strategy
            target_num_samples = source_num_samples
            target_masking_params = source_masking_params
            # # do other relationships make sense
            # assert relationship == "complement"

        assert source_num_samples % target_num_samples == 0, (
            "number of source samples has to be multiple of target samples"
        )

        # translate settings into sampling masks

        # iterate over all target samples
        target_masks: list[np.typing.NDArray] = []
        target_metadata: list[SampleMetaData] = []
        for _ in range(target_num_samples):
            target_masks += [
                self._get_mask(
                    num_cells=num_cells,
                    strategy=target_strategy,
                    target_mask=None,
                    masking_strategy_config=target_masking_params,
                )
            ]
            target_metadata += [SampleMetaData(params=target_cfg)]

        # iterate over all source samples
        source_masks: list[np.typing.NDArray] = []
        source_metadata: list[SampleMetaData] = []
        source_target_mapping = np.zeros(source_num_samples, dtype=np.int32)
        for it in range(source_num_samples):
            source_masks += [
                self._get_mask(
                    num_cells=num_cells,
                    strategy=source_strategy,
                    masking_strategy_config=source_masking_params,
                    target_mask=target_masks[it % target_num_samples],
                    relationship=relationship,
                )
            ]
            source_metadata += [SampleMetaData(params=target_cfg)]
            source_target_mapping[it] = it % target_num_samples

        return (
            (target_masks, target_metadata),
            (source_masks, source_metadata),
            source_target_mapping,
        )

    def _get_mask(
        self,
        num_cells: int,
        strategy: str | None = None,
        rate: float | None = None,
        masking_strategy_config: dict | None = None,
        target_mask: np.typing.NDArray | None = None,
        relationship: str = "subset",
    ) -> np.typing.NDArray:
        """Get effective mask, combining with target mask if specified.

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

        # handle cases where mask is directly derived from target_mask
        if target_mask is not None:
            if relationship == "complement":
                mask = ~target_mask
                return mask

        # get mask
        mask = self._generate_cell_mask(num_cells, strategy, rate, masking_strategy_config)

        # handle cases where mask needs to be combined with target_mask
        if target_mask is not None:
            if relationship == "subset":
                mask = mask & target_mask
            elif relationship == "disjoint":
                mask = mask & (~target_mask)

        return mask

    def _generate_cell_mask(
        self,
        num_cells: int,
        strategy: str | None = None,
        rate: float | None = None,
        masking_strategy_config: dict | None = None,
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

        # get config for mask

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

        # generate cell mask

        if strat == "random":
            mask = self.rng.uniform(0, 1, num_cells) < keep_rate

        elif strat == "forecast" or strat == "causal":
            mask = np.ones(num_cells, dtype=np.bool)

        elif strat == "healpix":
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
                mask = np.zeros(num_cells, dtype=bool)
            else:
                parent_ids = self.rng.choice(num_parent_cells, num_parents_to_keep, replace=False)
                child_offsets = np.arange(num_children_per_parent)
                child_indices = (
                    parent_ids[:, None] * num_children_per_parent + child_offsets
                ).reshape(-1)
                mask = np.zeros(num_cells, dtype=bool)
                mask[child_indices] = True

        else:
            assert False, "Unknown strategy."

        mask = to_bool_tensor(mask)

        return mask

    def _generate_causal_mask_idxs(
        self,
        idxs_cells_lens: list[list[int]],
        rate: float,
    ) -> np.typing.NDArray:
        """
        Generates a causal mask at the index level, masking the latest timesteps
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
        num_future_to_mask = (rate * token_lens).astype(int)
        start_mask_indices = np.maximum(1, token_lens - num_future_to_mask)

        # Handle edge cases
        mask_valid = token_lens > 1
        start_mask_indices = np.where(mask_valid, start_mask_indices, token_lens)

        # Create masks (True = KEEP in source, False = MASK for target)
        full_mask = []
        for token_len, start_idx in zip(token_lens, start_mask_indices, strict=True):
            if token_len > 1:
                # Keep early timesteps, mask late ones
                mask = np.concatenate(
                    [
                        np.ones(start_idx, dtype=bool),
                        np.zeros(max(0, token_len - start_idx), dtype=bool),
                    ]
                )
            elif token_len == 1:
                mask = np.ones(1, dtype=bool)
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

    def _generate_temporal_mask(
        self,
        idxs_cells_lens: list[list[int]],
        strategy: str,
        rate: float,
        masking_strategy_config: dict,
        target_mask: np.typing.NDArray | None = None,
        target_mask_metadata: dict | None = None,
        relationship: str = "subset",
    ) -> np.typing.NDArray:
        """
        Generate temporal mask at token level (after tokenization).

        This method generates masks for temporal strategies (causal, temporal_crop)
        that require knowledge of the temporal structure of tokens.

        Args:
            idxs_cells_lens: List of lists of token lengths per cell (from tokenization)
            strategy: "causal" or "temporal_crop"
            rate: Masking rate (fraction to MASK for target)
            masking_strategy_config: Strategy-specific config (e.g., crop_direction)
            target_mask: Optional token-level target mask for relationship constraints
            target_mask_metadata: Optional metadata if target is also temporal
            relationship: "complement", "subset", or "disjoint"

        Returns:
            np.ndarray: Boolean mask [num_tokens] where True = KEEP in source
        """
        # Handle complement relationship with temporal target
        if target_mask_metadata is not None and relationship == "complement":
            # If target is temporal, we need to generate its mask first then complement
            target_mask = self._generate_temporal_mask(
                idxs_cells_lens,
                target_mask_metadata["strategy"],
                target_mask_metadata["rate"],
                target_mask_metadata["config"],
                None,
                None,
                "subset",
            )

        # Generate base temporal mask based on strategy
        if strategy == "causal":
            # Mask the LATEST timesteps (rate = fraction to MASK)
            mask = self._generate_causal_mask_idxs(idxs_cells_lens, rate)

        elif strategy == "temporal_crop":
            # Keep timesteps based on crop_direction (rate = fraction to MASK)
            temporal_config = {
                "crop_direction": masking_strategy_config.get("crop_direction", "end"),
                "rate": 1.0 - rate,  # Convert from mask rate to keep rate
            }
            mask = self._generate_temporal_crop_mask_idxs(idxs_cells_lens, temporal_config)

        else:
            raise ValueError(f"Unsupported temporal strategy: {strategy}")

        # Apply relationship with target_mask if provided
        if target_mask is not None:
            if relationship == "complement":
                mask = ~target_mask
            elif relationship == "subset":
                mask = mask & target_mask
            elif relationship == "disjoint":
                mask = mask & (~target_mask)

        return mask

    def _generate_spatiotemporal_mask(
        self,
        idxs_cells_lens: list[list[int]],
        cell_strategy: str,
        rate: float,
        masking_strategy_config: dict,
    ) -> np.typing.NDArray:
        """
        Generate spatiotemporal mask where each timestep gets independent spatial mask.

        Args:
            idxs_cells_lens: List of lists of token lengths per cell
            cell_strategy: "random" or "healpix" for spatial mask generation
            rate: Masking rate (fraction to MASK for target)
            masking_strategy_config: Config with cell_strategy details (e.g., hl_mask)

        Returns:
            np.ndarray: Boolean mask [num_tokens] where True = KEEP in source
        """
        keep_rate = 1.0 - rate
        num_cells = len(idxs_cells_lens)

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
                cell_keep_mask_tensor = self._generate_cell_mask(
                    num_cells=num_cells,
                    strategy=cell_strategy,
                    rate=keep_rate,
                    masking_strategy_config=masking_strategy_config,
                )
                cell_keep_mask = cell_keep_mask_tensor.cpu().numpy()
                # Extract the keep/mask decision for this specific cell
                cell_token_masks.append(cell_keep_mask[cell_idx])

            # Convert to boolean array: one mask value per token in this cell
            token_level_flags.append(np.array(cell_token_masks, dtype=bool))

        if token_level_flags:
            return np.concatenate(token_level_flags)
        else:
            return np.array([], dtype=bool)
