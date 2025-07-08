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
        strategy_kwargs: dict,
    ):
        self.masking_rate = masking_rate
        self.masking_strategy = masking_strategy
        self.masking_rate_sampling = masking_rate_sampling

        # strategy_kwargs is a dictionary that can hold any additional parameters
        self.strategy_kwargs = strategy_kwargs

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
            # so the masking strategy is channel masking,
            # which means we generate a mask for each channel...
            # of the tokenized data.
            # NOTE: probably shouldn't call this flat_mask,
            # as it is not flat, but a list of masks for each channel.
            not_flat_mask = self._generate_channel_mask(tokenized_data, rate)

            # quick hack to do this, as we have a list of masks for each channel,
            # we need different logic to what we have for the flat mask

            #split_indices = np.cumsum(token_lens)[:-1]
            #self.perm_sel = np.split(not_flat_mask, split_indices)
            
            #print("What is the type of perm_sel?", type(self.perm_sel))
            #print("What is actually the shape of the first element of perm_sel in channel?", self.perm_sel[0].shape)

            # make a list of not_flat_mask of the same length of tokenized_data, with the same not_flat_mask
            # so that we can use it to mask the source data.

            self.perm_sel = [not_flat_mask] * len(tokenized_data)
            #print("Computed the not_flat_mask for channel masking, it is a list of tensors.")

            #for data, p in zip(tokenized_data, self.perm_sel, strict=True):
            #    print("In the source...")
            #    print("Data shape:", len(data), "Mask shape:", len(p), "Shape of first element of data:", data[0].shape, "Shape of first element of mask:", p[0].shape)
                #print("Some zipped item:", zip(tokenized_data, not_flat_mask, strict=True)[0])

            # set some values of data to be 0.0, depending on the mask

            source_data = []
            for data, p in zip(tokenized_data, self.perm_sel, strict=True):
                #print("Data shape before masking:", data.shape)
                #print("First 20 values before masking:", data[0, 0, :20])
                data[:, :, p[0, 0, :]] = 0.0
                #print("Data shape after masking:", data.shape)
                #print("First 20 values after masking:", data[0, 0, :20])
                source_data.append(data)

            #source_data = [data[:, :, ~p[0, 0, :]] for data, p in zip(tokenized_data, self.perm_sel, strict=True)]

            #print("Shape of first element of tokenized_data:", tokenized_data[0].shape if tokenized_data else "No data")
            #print("Shape of first element of source_data:", source_data[0].shape if source_data else "No data")

            #print("I just computed the source_data for channel masking, it is a list of tensors.")

            return source_data

        else:
            assert False, f"Unknown masking strategy: {self.masking_strategy}"

        # Split the flat mask to match the structure of the tokenized data (list of lists)
        # This will be perm_sel, as a class attribute, used to mask the target data.
        split_indices = np.cumsum(token_lens)[:-1]
        self.perm_sel = np.split(flat_mask, split_indices)

        #print("What is the type of perm_sel?", type(self.perm_sel))
        #print("Length of perm_sel:", [len(self.perm_sel)])
        #print("What is the shape of the first element of perm_sel?", self.perm_sel[0].shape)
        #print("What is actually the first element of perm_sel?", self.perm_sel[0])

        #print("What is the type of tokenized_data?", type(tokenized_data))
        #print("Length of tokenized_data:", len(tokenized_data))
        #print("What is the shape of the first element of tokenized_data?", tokenized_data[0].shape if tokenized_data else "No data")

        # Apply the mask to get the source data (where mask is False)
        

        
        source_data = [data[~p] for data, p in zip(tokenized_data, self.perm_sel, strict=True)]

        #print("***********************************")
        #print([data.shape for data, p in zip(tokenized_data, self.perm_sel, strict=True)])
        #print([p.shape for data, p in zip(tokenized_data, self.perm_sel, strict=True)])
        #print("***********************************")

        #print("Length of source_data:", len(source_data))
        #print("Shape of first element of source_data:", source_data[0].shape if source_data else "No data")

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

        #print("What is the feature dimension?", feature_dim)

        processed_target_tokens = []

        #print("Length of target_tokenized_data:", len(target_tokenized_data))
        #print("Length of first element of target_tokenized_data:", len(target_tokenized_data[0]) if target_tokenized_data else "No data")
        #print("Target tokenized data shape of first element:", target_tokenized_data[0][0].shape if target_tokenized_data else "No data")

        if self.masking_strategy == "channel":

            # If the masking strategy is channel, we need to handle the target tokens differently.
            # We will apply the same channel mask to each tensor in target_tokenized_data.
            for cc, pp in zip(target_tokenized_data, self.perm_sel, strict=True):
                
                #print("Length of cc:", len(cc))
                #print("Length of pp:", len(pp))
                #print("cc", cc)
                #print("pp", pp)
                # cc is a list containing one tensor
                # pp is a list containing one tensor, which is a boolean mask for the channels
                #print("Shape of c:", [c.shape for c in cc[0:10]])
                #print("Shape of c:", [c for c in cc[0:10]])
                #print("p shape:", [p.shape for p in pp[0:10]])
                # assert that the first and second rows of pp are the same values                
                #for p in pp:
                #     assert all(p[0, :] == p[1, :]), "The first and second rows of pp must be the same values for channel masking."
                #    assert all(p[0, :] == p[2, :]), "The first and third rows of pp must be the same values for channel masking."


                # select the first 8 channels from each tensor in cc 
                # actually, I don't think we want these first information channels for the target tokens?
                
                #first_8_channels = [c[:, :(6 + coords.shape[-1] + geoinfos.shape[-1])] for c in cc]
                #print("Shape of first_8_channels:", first_8_channels[0].shape if first_8_channels else "No data")
                #selected_tensors = [c[:, p[0, :]] for c, p in zip(cc, pp, strict=True)]
                #print("Shape of selected_tensors:", [s.shape for s in selected_tensors[0:10]])


                selected_tensors = []
                for c, p in zip(cc, pp, strict=True):
                    c[:, ~p[0, :]] = np.nan  # Set the channels that are not masked to NaN
                    selected_tensors.append(c)


                # concatenate first_8_channels and selected_tensors along the last dimension
                #selected_tensors = [torch.cat((c, fc), dim=-1) for c, fc in zip(first_8_channels, selected_tensors, strict=True)]

                #print("Shape of selected_tensors after concatenation:", [s.shape for s in selected_tensors[0:10]])

                if selected_tensors:
                    #print("Length of selected_tensors:", len(selected_tensors))
                    #print("Shape of first element of selected_tensors:", [s.shape for s in selected_tensors[0:10]])
                    processed_target_tokens.append(torch.cat(selected_tensors))
                else:
                    print("I SHOULD NOT BE HERE")
                    processed_target_tokens.append(
                        torch.empty(0, feature_dim, dtype=coords.dtype, device=coords.device)
                    )

            #print("Processed target tokens shape:", [t.shape for t in processed_target_tokens[0:10]])            
            
            return processed_target_tokens

        
        
        for cc, pp in zip(target_tokenized_data, self.perm_sel, strict=True):

            #print("Shape of c:", [c.shape for c in cc[0:10]])
            #print("p:", [p for p in pp[0:10]])

            selected_tensors = [c for c, p in zip(cc, pp, strict=True) if p]
            #print("Length of selected_tensors:", len(selected_tensors))
            #print("Shape of first element of selected_tensors:", [s.shape for s in selected_tensors[0:10]])
            if selected_tensors:
                processed_target_tokens.append(torch.cat(selected_tensors))
            else:
                processed_target_tokens.append(
                    torch.empty(0, feature_dim, dtype=coords.dtype, device=coords.device)
                )

        #print("Processed target tokens shape:", [t.shape for t in processed_target_tokens[0:10]])
            
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

        # hl_data and hl_mask should be provided in strategy_kwargs
        hl_data = self.strategy_kwargs.get("hl_data")
        hl_mask = self.strategy_kwargs.get("hl_mask")

        if hl_data is None or hl_mask is None:
            assert False, (
                "If masking with HEALPix, hl_data and hl_mask must be provided in strategy_kwargs."
            )

        if hl_mask >= hl_data:
            assert False, "hl_mask must be less than hl_data for HEALPix masking."

        num_data_cells = 12 * (4**hl_data)
        if len(token_lens) != num_data_cells:
            assert False, (
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

    
    def _generate_channel_mask(self, tokenized_data: list[torch.Tensor], rate: float) -> list[np.typing.NDArray]:
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



        #print(f"Number of channels in tokenized data: {num_channels}")
        
        # this should be something like 6 + coords.shape[-1] + geoinfos.shape[-1] ?
        mask_count = int((num_channels - 8) * rate)

        #print("What is the mask_count?", mask_count)
        
        #if mask_count == 0 and self.masking_ratio > 0: # Ensure at least one if ratio is > 0
        #    mask_count = 1 if num_channels > 0 else 0

        # Create a 1D boolean array for channels to mask (True means mask)
        channel_mask_1d = np.zeros(num_channels, dtype=bool)
        
        # print("Channel mask 1d:", channel_mask_1d)
        
        # make sure the first 8 entries False, as they are not masked
        # this should be something like 6 + coords.shape[-1] + geoinfos.shape[-1]
        channel_mask_1d[:8] = False
        # print("Channel mask 1d after setting first 8 entries to False:", channel_mask_1d)
        
        if mask_count > 0:
            masked_channel_indices = np.random.choice(
                    (num_channels-8), mask_count, replace=False
                ) + 8  # Offset by 8 to skip the first 8 channels
            # print("Masked channel indices:", masked_channel_indices)    

            channel_mask_1d[masked_channel_indices] = True

            # print("Channel mask 1d after masking:", channel_mask_1d)

        
        full_mask = channel_mask_1d[np.newaxis, np.newaxis, :]

        # repeat the mask for each token_size in the tokenized data
        # so the second axis needs shape token_size
        full_mask = np.repeat(full_mask, tokenized_data[0].shape[1], axis=1)

        #print("Shape of full_mask after repeating to try to extend to the right shape for token_size:", full_mask.shape)

        return full_mask
