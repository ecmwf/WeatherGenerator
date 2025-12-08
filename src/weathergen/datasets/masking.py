import logging

import numpy as np
import torch

from weathergen.common.config import Config

_logger = logging.getLogger(__name__)


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
        masking_rate_sampling (bool): Whether to sample the masking rate from a distribution.
        masking_strategy_config (dict): Configuration for the masking strategy, can include
                                        additional parameters like "hl_mask", etc.
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

        self.mask_value = 0.0
        self.dim_time_enc = 6

        # number of healpix cells
        self.healpix_num_cells = 12 * (4**self.healpix_level_data)

        # Initialize the mask, set to None initially,
        # until it is generated in mask_source.
        self.perm_sel: list[np.typing.NDArray] = None

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

    def mask_source(
        self,
        tokenized_data: list[torch.Tensor],
        coords: torch.Tensor,
        geoinfos: torch.Tensor,
        source: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Receives tokenized data, generates a mask, and returns the source data (unmasked)
        and the permutation selection mask (perm_sel) to be used for the target.

        Args:
            tokenized_data (list[torch.Tensor]): A list of tensors, where each tensor
                                                 represents the tokens for a cell.

        Returns:
            list[torch.Tensor]: The unmasked tokens (model input).
        """

        token_lens = [len(t) for t in tokenized_data]
        num_tokens = sum(token_lens)

        # If there are no tokens, return empty lists.
        if num_tokens == 0:
            return tokenized_data

        # Clean strategy selection
        self.current_strategy = self._select_strategy()

        # Set the masking rate.
        rate = self._get_sampling_rate()

        if rate == 0.0:
            _logger.warning(
                "masking_rate is 0. This will result in empty target. The sample will be skipped. "
                + "If this occurs repeatedtly the masking settings likely need to be revised."
            )

        # Handle the special case where all tokens are masked
        if rate == 1.0:
            token_lens = [len(t) for t in tokenized_data]
            self.perm_sel = [np.ones(tl, dtype=bool) for tl in token_lens]
            source_data = [data[~p] for data, p in zip(tokenized_data, self.perm_sel, strict=True)]
            return source_data

        # Implementation of different masking strategies.
        # Generate a flat boolean mask for random, block, or healpix masking at cell level.
        # Generate a 3D mask to apply to each cell for channel masking.

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

        # apply mask

        # if masking_strategy is channel, we need to handle the masking differently,
        # since p is not 1D Boolean for the list of cells, but 3D to mask the channels in each cell.
        if self.current_strategy == "channel":
            self.perm_sel = mask
            # In the source_data we will set the channels that are masked to 0.0.
            source_data = []
            for data, p in zip(tokenized_data, self.perm_sel, strict=True):
                if len(data) > 0:
                    data[p] = self.mask_value
                    source_data.append(data)
                else:
                    source_data.append(data)

        elif self.current_strategy == "causal":
            # Only select unmasked timesteps
            self.perm_sel = mask
            source_data = []
            for data, p in zip(tokenized_data, self.perm_sel, strict=True):
                source_data.append(data[~p] if len(data) > 0 else data)

        else:
            # Split the flat mask to match the structure of the tokenized data (list of lists)
            # This will be perm_sel, as a class attribute, used to mask the target data.
            split_indices = np.cumsum(token_lens)[:-1]
            self.perm_sel = np.split(flat_mask, split_indices)

            # Apply the mask to get the source data (where mask is False)
            source_data = [data[~p] for data, p in zip(tokenized_data, self.perm_sel, strict=True)]

        return source_data

    def mask_target(
        self,
        target_tokenized_data: list[list[torch.Tensor]],
        coords: torch.Tensor,
        geoinfos: torch.Tensor,
        source: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Applies the permutation selection mask to
        the tokenized data to create the target data.
        Handles cases where a cell has no target
        tokens by returning an empty tensor of the correct shape.

        Args:
            target_tokens_cells (list[list[torch.Tensor]]): List of lists of tensors for each cell.
            coords (torch.Tensor): Coordinates tensor, used to determine feature dimension.
            geoinfos (torch.Tensor): Geoinfos tensor, used to determine feature dimension.
            source (torch.Tensor): Source tensor, used to determine feature dimension.

        Returns:
            list[torch.Tensor]: The target data with masked tokens, one tensor per cell.
        """

        # check that self.perm_sel is set, and not None with an assert statement
        assert self.perm_sel is not None, "Masker.perm_sel must be set before calling mask_target."

        # Pre-calculate the total feature dimension of a token to create
        # correctly shaped empty tensors.

        feature_dim = self.dim_time_enc + coords.shape[-1] + geoinfos.shape[-1] + source.shape[-1]

        processed_target_tokens = []

        # process all tokens used for embedding
        for cc, pp in zip(target_tokenized_data, self.perm_sel, strict=True):
            if len(cc) == 0:  # Skip if there's no target data
                pass

            if self.current_strategy == "channel":
                # If masking strategy is channel, handle target tokens differently.
                # We don't have Booleans per cell, instead per channel per cell,
                # we set the unmasked channels to NaN so not in loss.
                selected_tensors = []
                for c, p in zip(cc, pp, strict=True):
                    # slightly complicated as the first dimension of c varies with data in the cell.
                    # do not mask the first 8 channels,
                    # and set unmasked channels to nan
                    c[:, (self.dim_time_enc + coords.shape[-1] + geoinfos.shape[-1]) :][
                        :, ~p[0, (self.dim_time_enc + coords.shape[-1] + geoinfos.shape[-1]) :]
                    ] = torch.nan
                    selected_tensors.append(c)

            elif self.current_strategy == "causal":
                # select only the target times where mask is True
                if len(cc) == len(pp):
                    selected_tensors = [c for i, c in enumerate(cc) if pp[i]]
                elif len(pp) == 0:
                    selected_tensors = cc
                else:  # If length of target and mask doesn't match, create new mask
                    ratio = np.sum(pp) / len(pp)  # Ratio of masked tokens in source
                    indx = max(1, int(ratio * len(cc)))  # Get the same for target
                    selected_tensors = cc[-indx:]

            elif self.current_strategy == "healpix":
                selected_tensors = (
                    cc if len(pp) > 0 and pp[0] else []
                )  # All tokens inside healpix cell have the same mask

            elif self.current_strategy == "random":
                # For random masking, we simply select the tensors where the mask is True.
                # When there's no mask it's assumed to be False. This is done via strict=False
                selected_tensors = [c for c, p in zip(cc, pp, strict=False) if p]
            else:
                raise NotImplementedError(
                    f"Masking strategy {self.current_strategy} is not supported."
                )

            # Append the selected tensors to the processed_target_tokens list.
            if selected_tensors:
                processed_target_tokens.append(torch.cat(selected_tensors))
            else:
                processed_target_tokens.append(
                    torch.empty(0, feature_dim, dtype=coords.dtype, device=coords.device)
                )

        return processed_target_tokens

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
