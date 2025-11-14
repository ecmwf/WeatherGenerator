# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import torch

from weathergen.common.io import IOReaderData
from weathergen.datasets.masking import Masker
from weathergen.datasets.tokenizer import Tokenizer
from weathergen.datasets.tokenizer_utils import (
    encode_times_source,
    encode_times_target,
    tokenize_apply_mask_source,
    tokenize_apply_mask_target,
    tokenize_space,
    tokenize_spacetime,
)


class TokenizerMasking(Tokenizer):
    def __init__(self, healpix_level: int, masker: Masker):
        super().__init__(healpix_level)
        self.masker = masker

    def reset_rng(self, rng) -> None:
        """
        Reset rng after epoch to ensure proper randomization
        """
        self.masker.reset_rng(rng)
        self.rng = rng

    def batchify_source(
        self,
        stream_info: dict,
        rdata: IOReaderData,
        time_win: tuple,
    ):
        token_size = stream_info["token_size"]
        stream_id = stream_info["stream_id"]
        assert token_size is not None, "stream did not specify token_size"
        is_diagnostic = stream_info.get("diagnostic", False)

        # return empty if there is no data or we are in diagnostic mode
        if is_diagnostic or rdata.data.shape[1] == 0 or len(rdata.data) < 2:
            source_tokens_cells = [torch.tensor([])]
            source_tokens_lens = torch.zeros([self.num_healpix_cells_source], dtype=torch.int32)
            source_centroids = [torch.tensor([])]
            return (source_tokens_cells, source_tokens_lens, source_centroids)

        # create tokenization index
        tok = tokenize_spacetime if stream_info.get("tokenize_spacetime", False) else tokenize_space
        idxs_cells, idxs_cells_lens = tok(rdata, token_size, self.hl_source, pad_tokens=True)

        (mask_tokens, mask_channels) = self.masker.mask_source_idxs(
            idxs_cells, idxs_cells_lens, rdata
        )

        source_tokens_cells, source_tokens_lens = tokenize_apply_mask_source(
            idxs_cells,
            idxs_cells_lens,
            mask_tokens,
            mask_channels,
            stream_id,
            rdata,
            time_win,
            self.hpy_verts_rots_source[-1],
            encode_times_source,
        )

        # if source_tokens_lens.sum() > 0:
        #     source_centroids = self.compute_source_centroids(source_tokens_cells)
        # else:
        # TODO: remove completely?
        source_centroids = [torch.tensor([])]

        return (source_tokens_cells, source_tokens_lens, source_centroids)

    def batchify_target(
        self,
        stream_info: dict,
        sampling_rate_target: float,
        rdata: IOReaderData,
        time_win: tuple,
    ):
        token_size = stream_info["token_size"]

        # create tokenization index
        tok = tokenize_spacetime if stream_info.get("tokenize_spacetime", False) else tokenize_space
        idxs_cells, idxs_cells_lens = tok(rdata, token_size, self.hl_source, pad_tokens=False)

        (mask_tokens, mask_channels) = self.masker.mask_targets_idxs(
            idxs_cells, idxs_cells_lens, rdata
        )
        # mask_tokens = ~self.mask_tokens
        # # TODO
        # # mask_channels = ~self.mask_channels if self.mask_channels is not None
        # # else self.mask_channels
        # mask_channels = self.mask_channels

        data, datetimes, coords, coords_local, coords_per_cell = tokenize_apply_mask_target(
            self.hl_target,
            idxs_cells,
            idxs_cells_lens,
            mask_tokens,
            mask_channels,
            rdata,
            time_win,
            self.hpy_verts_rots_target,
            self.hpy_verts_local_target,
            self.hpy_nctrs_target,
            encode_times_target,
        )

        # TODO, TODO, TODO: max_num_targets
        # max_num_targets = stream_info.get("max_num_targets", -1)

        return (data, datetimes, coords, coords_local, coords_per_cell)

    def sample_tensors_uniform_vectorized(
        self, tensor_list: list, lengths: list, max_total_points: int
    ):
        """
        This function randomly selects tensors up to a maximum number of total points

        tensor_list: List[torch.tensor] the list to select from
        lengths: List[int] the length of each tensor in tensor_list
        max_total_points: the maximum number of total points to sample from
        """
        if not tensor_list:
            return [], 0

        # Create random permutation
        perm = self.rng.permutation(len(tensor_list))

        # Vectorized cumulative sum
        cumsum = torch.cumsum(lengths[perm], dim=0)

        # Find cutoff point
        valid_mask = cumsum <= max_total_points
        if not valid_mask.any():
            return [], 0

        num_selected = valid_mask.sum().item()
        perm = torch.tensor(perm)
        selected_indices = perm[:num_selected]
        selected_indices = torch.zeros_like(perm).scatter(0, selected_indices, 1)

        selected_tensors = [
            t if mask.item() == 1 else t[:0]
            for t, mask in zip(tensor_list, selected_indices, strict=False)
        ]

        return selected_tensors
