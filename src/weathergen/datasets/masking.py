import time

import numpy as np
import torch


class Masker:
    """Class to generate masks for token sequences and apply them.
    This class supports different masking strategies and combinations.

    Attributes:
        masking_rate (float): The base rate at which tokens are masked.
        masking_strategy (str): The strategy used for masking (e.g., "random",
        "block", MORE TO BE IMPLEMENTED...).
        masking_rate_sampling (bool): Whether to sample the masking rate from a distribution.
        rng (np.random.Generator): A random number generator.

    """

    def __init__(
        self,
        masking_rate: float,
        masking_strategy: str,
        masking_rate_sampling: bool,
        # NOTE: adding strategy_kwargs to allow for strategy-specific configurations
        # e.g., for healpix strategy, we might need hl_data and hl_mask parameters
        # or for different strategies, we might need different parameters?
        strategy_kwargs: dict,
    ):
        self.masking_rate = masking_rate
        self.masking_strategy = masking_strategy
        self.masking_rate_sampling = masking_rate_sampling

        # NOTE: strategy_kwargs is a dictionary that can hold any additional parameters
        self.strategy_kwargs = strategy_kwargs or {}

        # Initialize the random number generator.
        worker_info = torch.utils.data.get_worker_info()
        div_factor = (worker_info.id + 1) if worker_info is not None else 1
        self.rng = np.random.default_rng(int(time.time() / div_factor))

        # Initialize the mask, set to None initially,
        # until it is generated in mask_source.
        self.perm_sel: list[np.typing.NDArray] = None

    def mask_source(
        self,
        tokenized_data: list[torch.Tensor],
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

        # print("Length of each token t in tokenized_data:", token_lens)
        print("Number of tokens in the batch:", num_tokens)

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

        else:
            assert False, f"Unknown masking strategy: {self.masking_strategy}"

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
        Applies the permutation selection mask to the tokenized data to create the target data.
        Handles cases where a cell has no target tokens by returning an empty tensor of the correct shape.

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
            selected_tensors = [c for c, p in zip(cc, pp, strict=True) if p]
            if selected_tensors:
                processed_target_tokens.append(torch.cat(selected_tensors))
            else:
                processed_target_tokens.append(
                    torch.empty(0, feature_dim, dtype=coords.dtype, device=coords.device)
                )

        return processed_target_tokens

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

        print("Generating HEALPix mask...")

        # NOTE: hl_data and hl_mask are expected to be provided in strategy_kwargs?
        hl_data = self.strategy_kwargs.get("hl_data")
        hl_mask = self.strategy_kwargs.get("hl_mask")

        print(
            f"HEALPix level of data, and chosen masking level: hl_data={hl_data}, hl_mask={hl_mask}"
        )

        # NOTE: just for demonstration purposes, using hardcoded values
        # hl_data = 5
        # hl_mask = 1

        if hl_data is None or hl_mask is None:
            assert False, "HEALPix levels hl_data and hl_mask must be provided in strategy_kwargs."

        if hl_mask >= hl_data:
            assert False, "hl_mask must be less than hl_data for HEALPix masking."

        num_data_cells = 12 * (4**hl_data)
        if len(token_lens) != num_data_cells:
            assert False, (
                f"Expected {num_data_cells} data cells at level {hl_data}, but got {len(token_lens)}."
            )

        # Calculate the number of parent cells at the mask level (hl_mask)
        num_parent_cells = 12 * (4**hl_mask)
        level_diff = hl_data - hl_mask
        num_children_per_parent = 4**level_diff

        # print(f"[HEALPix Setup] Data Level (hl_data): {hl_data} ({num_data_cells} cells)")
        # print(f"[HEALPix Setup] Mask Level (hl_mask): {hl_mask} ({num_parent_cells} parent cells)")
        # print(f"[HEALPix Setup] Each parent cell at L{hl_mask} contains {num_children_per_parent} child cells at L{hl_data}.")

        # Choose parent cells to mask based on the specified rate.
        num_parents_to_mask = int(np.round(rate * num_parent_cells))
        if num_parents_to_mask == 0:
            # print("[HEALPix Masking] Masking rate is too low. No parent cells were selected to be masked.")
            return np.zeros(sum(token_lens), dtype=bool)

        # Mask, and print about what we are doing.
        parent_ids_to_mask = self.rng.choice(num_parent_cells, num_parents_to_mask, replace=False)
        # print(f"[HEALPix Masking] Based on rate {rate:.2f}, selected {num_parents_to_mask}/{num_parent_cells} parent cells to mask.")
        # print(f"[HEALPix Masking] Parent IDs selected: {parent_ids_to_mask}")

        # Now determine which child cells (and their tokens) are masked.
        # This is cells.
        cell_mask = np.zeros(num_data_cells, dtype=bool)
        # print("[HEALPix Masking] Mapping parent cells to child cell indices:")
        for parent_id in parent_ids_to_mask:
            start_child_idx = parent_id * num_children_per_parent
            end_child_idx = start_child_idx + num_children_per_parent
            # print(f"  - Parent {parent_id} (L{hl_mask}) -> Child indices {start_child_idx}-{end_child_idx-1} (L{hl_data})")
            cell_mask[start_child_idx:end_child_idx] = True

        # Make the cell-level mask flat and apply it to the token lengths.
        # np.repeat. It repeats each element of `cell_mask`
        # a number of times specified by `token_lens`.
        # print("[HEALPix Masking] Expanding cell-level mask to token-level mask.")
        flat_mask = np.repeat(cell_mask, token_lens)

        # Print the number of masked tokens and the total number of tokens.
        num_masked_tokens = np.sum(flat_mask)
        total_tokens = len(flat_mask)
        # print(f"[HEALPix Masking] Complete. Masked {num_masked_tokens}/{total_tokens} tokens ({num_masked_tokens/total_tokens:.2%}).\n")

        return flat_mask
