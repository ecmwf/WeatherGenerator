import logging

import numpy as np
import torch

_logger = logging.getLogger(__name__)


class Masker:
    """Class to generate masks for token sequences and apply them.
    This class supports different masking strategies and combinations.

    Attributes:
        masking_rate (float): The base rate at which tokens are masked.
        masking_strategy (str): The strategy used for masking (e.g., "random",
        "block", "healpix", "channel").
        masking_rate_sampling (bool): Whether to sample the masking rate from a distribution.
        masking_strategy_config (dict): Configuration for the masking strategy, can include
                                        additional parameters like "hl_data", "hl_mask", etc.
                                        specific to the masking strategy.
    """

    def __init__(
        self,
        masking_rate: float,
        masking_strategy: str,
        masking_rate_sampling: bool,
        masking_strategy_config: dict,
    ):
        self.masking_rate = masking_rate
        self.masking_strategy = masking_strategy
        self.masking_rate_sampling = masking_rate_sampling

        # masking_strategy_config is a dictionary that can hold any additional parameters
        self.masking_strategy_config = masking_strategy_config

        # Initialize the mask, set to None initially,
        # until it is generated in mask_source.
        self.perm_sel: list[np.typing.NDArray] = None

        # Check for required masking_strategy_config at construction time
        if self.masking_strategy == "healpix":
            hl_data = self.masking_strategy_config.get("hl_data")
            hl_mask = self.masking_strategy_config.get("hl_mask")
            assert hl_data is not None and hl_mask is not None, (
                "If HEALPix masking, hl_data and hl_mask must be given in masking_strategy_config."
            )
            assert hl_mask < hl_data, "hl_mask must be less than hl_data for HEALPix masking."

        if self.masking_strategy == "channel":
            # Ensure that masking_strategy_config contains either 'global' or 'per_cell'
            assert (
                "global" in self.masking_strategy_config
                or "per_cell" in self.masking_strategy_config
            ), "Strategy must be 'global' or 'per_cell' in masking_strategy_config."

    def reset_rng(self, rng) -> None:
        """
        Reset rng after epoch to ensure proper randomization
        """
        self.rng = rng

    def mask_source(
        self,
        tokenized_data: list[torch.Tensor],
        coords: torch.Tensor,
        geoinfos: torch.Tensor,
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
            return tokenized_data, []

        # Set the masking rate.
        # Use a local variable rate, so we keep the instance variable intact.
        rate = self.masking_rate

        # If masking_rate_sampling is enabled, sample the rate from a normal distribution.
        if self.masking_rate_sampling:
            rate = np.clip(
                np.abs(self.rng.normal(loc=rate, scale=1.0 / (2.5 * np.pi))),
                0.01,
                0.99,
            )

        if rate == 0.0:
            _logger.warning(
                "masking_rate is 0. This will result in empty target. The sample will be skipped. "
                + "If this occurs repeatedtly the masking settings likely need to be revised."
            )

        # Handle the special case where all tokens are masked
        if rate == 1.0:
            token_lens = [len(t) for t in tokenized_data]
            self.perm_sel = [np.ones(cell, dtype=bool) for cell in token_lens]
            source_data = [data[~p] for data, p in zip(tokenized_data, self.perm_sel, strict=True)]
            return source_data

        # Implementation of different masking strategies.
        # Generate a flat boolean mask for random, block, or healpix masking at cell level.
        # Generate a 3D mask to apply to each cell for channel masking.

        if self.masking_strategy == "random":
            flat_mask = self.rng.uniform(0, 1, num_tokens) < rate

        elif self.masking_strategy == "block":
            flat_mask = np.zeros(num_tokens, dtype=bool)
            block_size = int(np.round(rate * num_tokens))
            if block_size > 0 and num_tokens > 0:
                start_index = self.rng.integers(0, max(1, num_tokens - block_size + 1))
                flat_mask[start_index : start_index + block_size] = True

        elif self.masking_strategy == "healpix":
            flat_mask = self._generate_healpix_mask(token_lens, rate)

        elif self.masking_strategy == "channel":
            mask = self._generate_channel_mask(tokenized_data, rate, coords, geoinfos)

        else:
            assert False, f"Unknown masking strategy: {self.masking_strategy}"

        # if masking_strategy is channel,
        # we need to handle the masking differently,
        # since p is not 1D Boolean for the list of cells,
        # but 3D to mask the channels in each cell.
        if self.masking_strategy == "channel":
            self.perm_sel = mask
            # In the source_data we will set the channels that are masked to 0.0.
            source_data = []
            for data, p in zip(tokenized_data, self.perm_sel, strict=True):
                data[p] = 0.0
                source_data.append(data)
            return source_data

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

        feature_dim = 6 + coords.shape[-1] + geoinfos.shape[-1] + source.shape[-1]

        processed_target_tokens = []

        for cc, pp in zip(target_tokenized_data, self.perm_sel, strict=True):
            if self.masking_strategy == "channel":
                # If masking strategy is channel, handle target tokens differently.
                # We don't have Booleans per cell, instead per channel for cell,
                # we set the unmasked channels to NaN so not in the loss calculation.
                selected_tensors = []
                for c, p in zip(cc, pp, strict=True):

                    # slightly complicated as the first dimension of c varies with data in the cell.
                    c[:, ~p[0, :]] = torch.nan  # Set the channels that are not masked to NaN
                    selected_tensors.append(c)
            
            else:
                # For other masking strategies, we simply select the tensors where the mask is True.
                selected_tensors = [c for c, p in zip(cc, pp, strict=True) if p]

            # Append the selected tensors to the processed_target_tokens list.
            if selected_tensors:
                processed_target_tokens.append(torch.cat(selected_tensors))
            else:
                processed_target_tokens.append(
                    torch.empty(0, feature_dim, dtype=coords.dtype, device=coords.device)
                )

        return processed_target_tokens

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

        # hl_data and hl_mask should be provided in masking_strategy_config
        hl_data = self.masking_strategy_config.get("hl_data")
        hl_mask = self.masking_strategy_config.get("hl_mask")

        num_data_cells = 12 * (4**hl_data)
        assert len(token_lens) == num_data_cells, (
            f"Expected {num_data_cells} cells at level {hl_data}, got {len(token_lens)}."
        )

        # Calculate the number of parent cells at the mask level (hl_mask)
        num_parent_cells = 12 * (4**hl_mask)
        level_diff = hl_data - hl_mask
        num_children_per_parent = 4**level_diff

        # if masking_rate_sampling is enabled, sample the rate from a normal distribution.
        if self.masking_rate_sampling:
            rate = np.clip(
                np.abs(self.rng.normal(loc=rate, scale=1.0 / (2.5 * np.pi))),
                0.01,
                0.99,
            )

        # Choose parent cells to mask based on the specified rate.
        num_parents_to_mask = int(np.round(rate * num_parent_cells))

        if num_parents_to_mask == 0:
            return np.zeros(sum(token_lens), dtype=bool)

        # Mask, and print about what we are doing.
        parent_ids_to_mask = self.rng.choice(num_parent_cells, num_parents_to_mask, replace=False)

        # Now determine which child cells (and their tokens) are masked.
        cell_mask = np.zeros(num_data_cells, dtype=bool)

        # For each parent ID, calculate the child indices and set them in the mask
        parent_ids = np.asarray(parent_ids_to_mask)
        child_offsets = np.arange(num_children_per_parent)
        child_indices = (parent_ids[:, None] * num_children_per_parent + child_offsets).reshape(-1)

        cell_mask[child_indices] = True

        # Make the cell-level mask flat and apply it to the token lengths.
        # np.repeat. It repeats each element of `cell_mask`
        # a number of times specified by `token_lens`.
        flat_mask = np.repeat(cell_mask, token_lens)

        return flat_mask

    def _generate_channel_mask(
        self,
        tokenized_data: list[torch.Tensor],
        rate: float,
        coords: torch.Tensor,
        geoinfos: torch.Tensor,
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

        # determine the number of channels
        num_channels = None
        for t in tokenized_data:
            if t.numel() > 0:
                num_channels = t.shape[2]
                break

        # If all tensors in the batch are empty, return masks matching their shapes.
        if num_channels is None:
            return [np.empty(t.shape, dtype=bool) for t in tokenized_data]

        # masking rate sampling, to be refactored as shared between methods
        if self.masking_rate_sampling:
            rate = np.clip(
                np.abs(self.rng.normal(loc=rate, scale=1.0 / (2.5 * np.pi))),
                0.01,
                0.99,
            )

        # isolate the number of actual data channels. 6 refers to time.
        num_fixed_channels = 6 + coords.shape[-1] + geoinfos.shape[-1]
        num_data_channels = num_channels - num_fixed_channels
        mask_count = int(num_data_channels * rate)

        # generate masks for each cell...TODO: optimise this
        full_mask = []
        is_per_cell = self.masking_strategy_config.get("per_cell", False)
        base_channel_mask_1d = None

        for cell_data in tokenized_data:
            # Check for empty cells or wrong shapes
            if cell_data.ndim != 3 or cell_data.shape[1] == 0:
                # This cell is empty or has an unexpected shape (e.g., [0]).
                # Then create a mask that matches the shape of the cell data.
                full_mask.append(np.empty(cell_data.shape, dtype=bool))
                continue

            # retrieve 1st dimension and token size
            num_1st_dim = cell_data.shape[0]
            num_tokens = cell_data.shape[1]

            # determine the 1D channel mask for the current cell.
            if is_per_cell or base_channel_mask_1d is None:
                channel_mask_1d = np.zeros(num_channels, dtype=bool)
                if mask_count > 0:
                    masked_indices = (
                        self.rng.choice(num_data_channels, mask_count, replace=False)
                        + num_fixed_channels
                    )  # offset by the fixed channels
                    # set randomly chosen masked indices to True
                    channel_mask_1d[masked_indices] = True

                if not is_per_cell:
                    base_channel_mask_1d = channel_mask_1d
            else:
                channel_mask_1d = base_channel_mask_1d

            # Expand the 1D channel mask to the 3D shape: (dim, num_tokens, num_channels).
            cell_mask_3d = np.tile(channel_mask_1d, (num_1st_dim, num_tokens, 1))
            full_mask.append(cell_mask_3d)

        return full_mask
