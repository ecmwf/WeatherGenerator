import logging
import time

import numpy as np
import torch

_logger = logging.getLogger(__name__)


class Masker:
    """Class to generate masks for token sequences and apply them.
    This class supports different masking strategies and combinations.

    Attributes:
        masking_rate (float): The base rate at which tokens are masked.
        masking_strategy (str): The strategy used for masking (e.g., "random",
        "block", "healpix", ...more to be implemented).
        masking_rate_sampling (bool): Whether to sample the masking rate from a distribution.
        rng (np.random.Generator): A random number generator.

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

        # Initialize the random number generator.
        worker_info = torch.utils.data.get_worker_info()
        div_factor = (worker_info.id + 1) if worker_info is not None else 1
        self.rng = np.random.default_rng(int(time.time() / div_factor))

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
        
        # be honest here, what is tokenized_data?
        # It is a list of tensors, where each tensor represents the tokens for a cell.

        #print("Tokenized data shape example:", tokenized_data[0].size() if tokenized_data else "No data")
        
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
                0.0,
                1.0,
            )

        if rate == 0.0:
            _logger.warning(
                "masking_rate is 0. This will result in empty target. The sample will be skipped. "
                + "If this occurs repeatedtly the masking settings likely need to be revised."
            )

        # Handle the special case where all tokens are masked
        if rate == 1.0:
            token_lens = [len(t) for t in tokenized_data]
            self.perm_sel = [np.ones(l, dtype=bool) for l in token_lens]
            source_data = [data[~p] for data, p in zip(tokenized_data, self.perm_sel, strict=True)]
            return source_data

        # Implementation of different masking strategies.
        # Generate a flat boolean mask based on the strategy.

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
            #print("Shape of flat_mask:", flat_mask.shape)

        elif self.masking_strategy == "channel":
            
            mask = self._generate_channel_mask(
                tokenized_data, rate, coords, geoinfos
            )

        else:
            assert False, f"Unknown masking strategy: {self.masking_strategy}"

        # if masking_strategy is channel, we need to handle the masking differently, since p is not 1D
        if self.masking_strategy == "channel":
            
            self.perm_sel = mask
            # For channel masking, we need to handle the masking differently.
            # In the source_data we will set the channels that are masked to 0.0.
            source_data = []
            for data, p in zip(tokenized_data, self.perm_sel, strict=True):
                data[p] = 0.0 # Set the channels that are masked to 0.0
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

        # The following block handles cases
        # where a cell has no target tokens,
        # which would cause an error in torch.cat with an empty list.

        # Pre-calculate the total feature dimension of a token to create
        # correctly shaped empty tensors.

        feature_dim = 6 + coords.shape[-1] + geoinfos.shape[-1] + source.shape[-1]

        processed_target_tokens = []
        
        for cc, pp in zip(target_tokenized_data, self.perm_sel, strict=True):

            if self.masking_strategy == "channel":
                # If the masking strategy is channel, we need to handle the target tokens differently.
                # Since we don't have true or false on a per token basis, instead per channel.
                selected_tensors = []
                for c, p in zip(cc, pp, strict=True):
                    # this slightly complicated as the first dimension of c varies with data in the cell.
                    c[:, ~p[0, :]] = torch.nan  # Set the channels that are not masked to NaN
                    selected_tensors.append(c)
            else:
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
                0.0,
                1.0,
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

    
    def _generate_channel_mask(self, tokenized_data: list[torch.Tensor], rate: float, 
                                coords: torch.Tensor,
                                geoinfos: torch.Tensor) -> list[np.typing.NDArray]:
        """
        Generates a mask for each channel in the tokenized data based on the specified rate.

        Args:
            tokenized_data (list[torch.Tensor]): A list of tensors, where each tensor represents
                                                 the tokens for a cell.
            rate (float): The desired masking rate for each channel.

        Returns:
            list[np.ndarray]: A list of boolean arrays, where each array corresponds to a channel
                              and indicates which tokens are masked.
        """
        
        # Calculate the number of channels from the first tensor
        num_channels = tokenized_data[0].shape[-1]

        # Check if all tensors have the same number of channels
        for t in tokenized_data:
            if t.shape[-1] != num_channels:
                assert False, (
                    "All tensors in tokenized_data must have the same number of channels."
                    f" Expected {num_channels}, but found {t.shape[-1]}."
                )

        # if masking_rate_sampling is enabled, sample the rate from a normal distribution.
        if self.masking_rate_sampling:
            rate = np.clip(
                np.abs(self.rng.normal(loc=rate, scale=1.0 / (2.5 * np.pi))),
                0.0,
                1.0,
            )
        
        # this should be something like 6 + coords.shape[-1] + geoinfos.shape[-1] rather than -8 ?
        mask_count = int((num_channels - (6 + coords.shape[-1] + geoinfos.shape[-1])) * rate)

        # Create a 1D boolean array for channels to mask (True means mask)
        channel_mask_1d = np.zeros(num_channels, dtype=bool)
                
        # We ensure the first channels, of time, coords and geoinfos channels are not masked        
        if mask_count > 0:
            masked_channel_indices = np.random.choice(
                    (num_channels - (6 + coords.shape[-1] + geoinfos.shape[-1])), mask_count, replace=False
                ) + (6 + coords.shape[-1] + geoinfos.shape[-1])  # Offset by time, coords, geoinfos to skip these chans
            channel_mask_1d[masked_channel_indices] = True
        
        # our tokens have shape (X, token_size, num_channels), so expand to 3D
        channel_mask_3d = channel_mask_1d[np.newaxis, np.newaxis, :]

        # repeat the mask for each token_size in the tokenized data
        # so the second axis has shape token_size
        channel_mask_3d_token_size = np.repeat(channel_mask_3d, tokenized_data[0].shape[1], axis=1)

        # based on the masking_strategy_config, we will generate the full mask
        # currently the options are global or per_cell channel masking
        if self.masking_strategy_config.get("global"):
            # If global_masking is True, 
            # we generate the same channel mask for all cells.
            full_mask = [channel_mask_3d_token_size] * len(tokenized_data)
            #print("Number of masked channels per cell:", np.sum(full_mask[0], axis=-1))
             
        elif self.masking_strategy_config.get("per_cell"):
            
            # This is ugly, and should be refactored.
            # Probably should be done above with channel_mask_1d.
            # Create a list to store the permuted masks
            permuted_masks = []

            # Get the length of channels to permute
            # the number of data channels minus time, coords and geoinfos channels
            permute_length = channel_mask_3d_token_size.shape[2] - (6 + coords.shape[-1] + geoinfos.shape[-1])

            for mask_idx in range(len(tokenized_data)):
                # Create a copy of the original full_mask to modify for each token
                current_mask = channel_mask_3d_token_size.copy()

                # Generate a perturbation to applied to the channels.
                permutation_indices = np.random.permutation(permute_length)

                # Iterate through the first two dimensions (1 and 32)
                for i in range(current_mask.shape[0]): # Loop through the first dimension
                    for j in range(current_mask.shape[1]): # Loop through the second dimension, token_size
                        # Extract the slice of the third dimension we want to permute
                        slice_to_permute = current_mask[i, j, (6 + coords.shape[-1] + geoinfos.shape[-1]):]

                        # Permute the slice using permutation_indices
                        permuted_slice = slice_to_permute[permutation_indices]

                        # Assign the permuted slice back to the current_mask
                        current_mask[i, j, (6 + coords.shape[-1] + geoinfos.shape[-1]):] = permuted_slice

                # Append the modified current_mask to our list
                permuted_masks.append(current_mask)

            full_mask = permuted_masks         
        
        # else if there is no masking_strategy_config, 
        # return an error with an assert statement
        else:
            assert False, "Masking strategy must specify either 'global' or 'per_cell' in masking_strategy_config."

        # assert that the first 8 channels are not masked
        assert np.all(~full_mask[0][0, 0, :8]), (
            "The first 8 channels should not be masked. "
            "This is likely due to an error in the masking strategy configuration."
        )

        return full_mask
