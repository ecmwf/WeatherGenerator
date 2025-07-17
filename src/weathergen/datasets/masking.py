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
        "block", MORE TO BE IMPLEMENTED...).
        masking_rate_sampling (bool): Whether to sample the masking rate from a distribution.
        rng (np.random.Generator): A random number generator.

    """

    def __init__(
        self,
        masking_rate: float,
        masking_strategy: str,
        masking_rate_sampling: bool,
    ):
        self.masking_rate = masking_rate
        self.masking_strategy = masking_strategy
        self.masking_rate_sampling = masking_rate_sampling

        # Initialize the mask, set to None initially,
        # until it is generated in mask_source.
        self.perm_sel: list[np.typing.NDArray] = None

    def reset_rng(self, rng) -> None:
        """
        Reset rng after epoch to ensure proper randomization
        """
        self.rng = rng

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
            self.perm_sel = [np.ones(tl, dtype=bool) for tl in token_lens]
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
        Handles cases where a cell has no target tokens by returning an empty tensor of the correct
        shape.

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
