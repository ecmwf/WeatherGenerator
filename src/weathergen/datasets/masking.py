import numpy as np
import torch


class Masker:
    """Class to generate boolean masks for token sequences and apply them.
    This class supports different masking strategies and combinations.
    Attributes:
        masking_rate (float): The base rate at which tokens are masked.
        masking_strategy (str): The strategy used for masking (e.g., "random", "block", MORE TO BE IMPLEMENTED...).
        TO BE IMPLEMENTED: masking_combination (str): The strategy for combining masking strategies through training (e.g., "sequential").
        masking_rate_sampling (bool): Whether to sample the masking rate from a distribution.
    """

    def __init__(
        self,
        masking_rate: float,
        masking_strategy: str = "random",
        masking_combination: str = "global",
        masking_rate_sampling: bool = False,
    ):
        self.masking_rate = masking_rate
        self.masking_strategy = masking_strategy
        self.masking_combination = masking_combination
        self.masking_rate_sampling = masking_rate_sampling

    def mask(
        self,
        tokenized_data: list[torch.Tensor],
        rng: np.random.Generator,
        masking_rate: float = None,
    ) -> tuple[list[torch.Tensor], list[np.ndarray]]:
        """
        Receives tokenized data, generates a mask, and returns the source data (unmasked)
        and the permutation selection mask (perm_sel) to be used for the target.

        Args:
            tokenized_data (list[torch.Tensor]): A list of tensors, where each tensor
                                                 represents the tokens for a cell.
            rng (np.random.Generator): A random number generator.
            masking_rate (float, optional): Override for the instance's masking_rate.
                                            Defaults to None.

        Returns:
            tuple: A tuple containing:
                - source_data (list[torch.Tensor]): The unmasked tokens (model input).
                - perm_sel (list[np.ndarray]): The boolean mask for each cell. The `True`
                                               values indicate the tokens that are masked
                                               out and will become the target.
        """
        token_lens = [len(t) for t in tokenized_data]
        num_tokens = sum(token_lens)

        # If there are no tokens, return empty lists.
        if num_tokens == 0:
            return tokenized_data, []

        # Determine the masking rate to use for this call
        rate = self.masking_rate if masking_rate is None else masking_rate

        if self.masking_rate_sampling:
            rate = np.clip(
                np.abs(rng.normal(loc=rate, scale=1.0 / (2.5 * np.pi))),
                0.0,
                1.0,
            )

        # Generate a flat boolean mask based on the strategy
        if self.masking_strategy == "random":
            flat_mask = rng.uniform(0, 1, num_tokens) < rate

        elif self.masking_strategy == "block":
            flat_mask = np.zeros(num_tokens, dtype=bool)
            block_size = int(np.round(rate * num_tokens))
            if block_size > 0 and num_tokens > 0:
                start_index = rng.integers(0, max(1, num_tokens - block_size + 1))
                flat_mask[start_index : start_index + block_size] = True
        else:
            assert False, f"Unknown masking strategy: {self.masking_strategy}"

        # Ensure the mask is not degenerate (all True or all False) if there's more than one token.
        if num_tokens > 1:
            if flat_mask.all():
                flat_mask[rng.integers(low=0, high=num_tokens)] = False
            elif not flat_mask.any():
                flat_mask[rng.integers(low=0, high=num_tokens)] = True

        # Split the flat mask to match the structure of the tokenized data (list of lists)
        # This will be our perm_sel
        split_indices = np.cumsum(token_lens)[:-1]
        perm_sel = np.split(flat_mask, split_indices)

        # Apply the mask to get the source data (where mask is False)
        source_data = [data[~p] for data, p in zip(tokenized_data, perm_sel, strict=True)]

        return source_data, perm_sel


class Masker_archived:
    """Class to generate boolean masks for token sequences.
    This class supports different masking strategies and combinations.
    Attributes:
        masking_rate (float): The rate at which tokens are masked.
        masking_strategy (str): The strategy used for masking (e.g., "random").
        masking_combination (str): The combination strategy for masking (e.g., "global").
    """

    def __init__(
        self,
        masking_rate: float,
        masking_strategy: str = "random",
        masking_combination: str = "global",
        masking_rate_sampling: bool = False,
    ):
        self.masking_rate = masking_rate
        self.masking_strategy = masking_strategy
        self.masking_combination = masking_combination
        self.masking_rate_sampling = masking_rate_sampling

    def mask(
        self, num_tokens: int, rng: np.random.Generator, masking_rate: float = None
    ) -> np.ndarray:
        """Return a boolean mask array of length num_tokens."""
        # Use the provided masking_rate if given, else the default
        rate = self.masking_rate if masking_rate is None else masking_rate

        # If masking_rate_sampling is True, sample the masking rate from a normal distribution
        if self.masking_rate_sampling:
            rate = np.clip(
                np.abs(rng.normal(loc=rate, scale=1.0 / (2.5 * np.pi))),
                0.0,
                1.0,
            )

        # Here we implement various masking strategies, which can be extended.
        # For now, we only implement "random" and "block" masking strategies.

        if self.masking_strategy == "random":
            mask = rng.uniform(0, 1, num_tokens) < rate

        elif self.masking_strategy == "block":
            mask = np.zeros(num_tokens, dtype=bool)
            block_size = int(np.round(rate * num_tokens))
            if block_size > 0 and num_tokens > 0:
                start = rng.integers(0, num_tokens - block_size + 1)
                mask[start : start + block_size] = True

        # Change to asserts as we do not catch errors atm
        else:
            assert False, f"Unknown masking strategy: {self.masking_strategy}"

        # Ensure not all True or all False
        if not mask.any():
            mask[rng.integers(low=0, high=len(mask))] = True
        if mask.all():
            mask[rng.integers(low=0, high=len(mask))] = False

        return mask
