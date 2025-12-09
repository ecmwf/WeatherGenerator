# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
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


def readerdata_to_torch(rdata: IOReaderData) -> IOReaderData:
    """
    Convert data, coords, and geoinfos to torch tensor
    """
    if type(rdata.coords) is not torch.Tensor:
        rdata.coords = torch.tensor(rdata.coords)
    if type(rdata.geoinfos) is not torch.Tensor:
        rdata.geoinfos = torch.tensor(rdata.geoinfos)
    if type(rdata.data) is not torch.Tensor:
        rdata.data = torch.tensor(rdata.data)

    return rdata


class TokenizerMasking(Tokenizer):
    def __init__(self, healpix_level: int, masker: Masker):
        super().__init__(healpix_level)
        self.masker = masker
        self.rng = None
        self.token_size = None

    def reset_rng(self, rng) -> None:
        """
        Reset rng after mini_epoch to ensure proper randomization
        """
        self.masker.reset_rng(rng)
        self.rng = rng

    def get_tokens_windows(self, stream_info, data, pad_tokens):
        """
        Tokenize data (to amortize over the different views that are generated)

        """

        tok_spacetime = stream_info.get("tokenize_spacetime", False)
        tok = tokenize_spacetime if tok_spacetime else tokenize_space
        hl = self.healpix_level
        token_size = stream_info["token_size"]

        tokens = []
        for rdata in data:
            idxs_cells, idxs_cells_lens = tok(
                readerdata_to_torch(rdata), token_size, hl, pad_tokens
            )
            tokens += [(idxs_cells, idxs_cells_lens)]

        return tokens

    def cell_to_token_mask(self, idxs_cells, idxs_cells_lens, mask):
        """ """

        mask_tokens, mask_channels = None, None
        num_tokens = torch.tensor([len(t) for t in idxs_cells_lens]).sum().item()

        # If there are no tokens, return empty lists.
        if num_tokens == 0:
            return (mask_tokens, mask_channels)

        # TODO, TODO, TODO: use np.repeat
        # https://stackoverflow.com/questions/26038778/repeat-each-values-of-an-array-different-times
        # build token level mask: for each cell replicate the keep flag across its tokens
        token_level_flags: list[np.typing.NDArray] = []
        for km, lens_cell in zip(mask, idxs_cells_lens, strict=True):
            num_tokens_cell = len(lens_cell)
            if num_tokens_cell == 0:
                continue
            token_level_flags.append(
                np.ones(num_tokens_cell, dtype=bool)
                if km
                else np.zeros(num_tokens_cell, dtype=bool)
            )
        if token_level_flags:
            mask_tokens = np.concatenate(token_level_flags)
        else:
            mask_tokens = np.array([], dtype=bool)

        return (mask_tokens, mask_channels)

    def get_source(
        self,
        stream_info: dict,
        rdata: IOReaderData,
        idxs_cells_data,
        time_win: tuple,
        cell_mask: torch.Tensor,
    ):
        stream_id = stream_info["stream_id"]
        is_diagnostic = stream_info.get("diagnostic", False)

        # return empty if there is no data or we are in diagnostic mode
        if is_diagnostic or rdata.data.shape[1] == 0 or len(rdata.data) < 2:
            source_tokens_cells = [torch.tensor([])]
            source_tokens_lens = torch.zeros([self.num_healpix_cells_source], dtype=torch.int32)
            mask_state = {
                "strategy": self.masker.current_strategy,
                "mask_tokens": None,
                "mask_channels": None,
            }
            return (source_tokens_cells, source_tokens_lens, mask_state)

        # create tokenization index
        (idxs_cells, idxs_cells_lens) = idxs_cells_data

        # select strategy from XXX depending on stream and if student or teacher

        (mask_tokens, mask_channels) = self.cell_to_token_mask(
            idxs_cells, idxs_cells_lens, cell_mask
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

        # capture per-view mask state to later produce consistent targets
        mask_state = {
            "strategy": None,  # self.masker.current_strategy,
            "mask_tokens": mask_tokens,
            "mask_channels": mask_channels,
        }

        return (source_tokens_cells, source_tokens_lens, mask_state)

    # batchify_target_for_view now unified into batchify_target via optional mask_state

    def get_target(
        self,
        stream_info: dict,
        sampling_rate_target: float,
        rdata: IOReaderData,
        token_data,
        time_win: tuple,
        mask_state: dict | None = None,
    ):
        # TODO: remove

        # create tokenization index
        (idxs_cells, idxs_cells_lens) = token_data

        # Apply per-view mask state if provided
        if mask_state is not None:
            self.masker.current_strategy = mask_state.get("strategy", self.masker.masking_strategy)
            self.masker.mask_tokens = mask_state.get("mask_tokens")
            self.masker.mask_channels = mask_state.get("mask_channels")

        (mask_tokens, mask_channels, idxs_ord_inv) = self.masker.mask_targets_idxs(
            idxs_cells,
            idxs_cells_lens,
        )

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

        return (data, datetimes, coords, coords_local, coords_per_cell, idxs_ord_inv)

    def get_target_coords(
        self,
        stream_info: dict,
        rdata: IOReaderData,
        token_data,
        time_win: tuple,
        cell_mask,
        # mask_state: dict | None = None,
    ):
        # create tokenization index
        (idxs_cells, idxs_cells_lens) = token_data

        (mask_tokens, mask_channels) = self.cell_to_token_mask(
            idxs_cells, idxs_cells_lens, cell_mask
        )

        # TODO: split up
        _, _, _, coords_local, coords_per_cell = tokenize_apply_mask_target(
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

        # selection = self._select_target_subset(stream_info, coords_local.shape[0])

        # if selection is not None and coords_local.numel() > 0:
        #     # use nice index_select method
        #     coords_local = coords_local.index_select(0, selection.to(coords_local.device))

        # # coords_per_cell is trickier
        # if selection is not None and coords_per_cell.numel() > 0:
        #     total_points = int(coords_per_cell.sum().item())
        #     if total_points == 0:
        #         coords_per_cell = torch.zeros_like(coords_per_cell)
        #     else:
        #         cell_ids = torch.repeat_interleave(
        #             torch.arange(coords_per_cell.shape[0], dtype=torch.long),
        #             coords_per_cell.to(torch.long),
        #         )
        #         if cell_ids.numel() == 0:
        #             coords_per_cell = torch.zeros_like(coords_per_cell)
        #         else:
        #             new_counts = torch.bincount(
        #                 cell_ids[selection.to(cell_ids.device)],
        #                 minlength=coords_per_cell.shape[0],
        #             )
        #             coords_per_cell = new_counts.to(dtype=coords_per_cell.dtype)

        # pass the selection back for use in get_target_values
        return (coords_local, coords_per_cell)

    def get_target_values(
        self,
        stream_info: dict,
        rdata: IOReaderData,
        token_data,
        time_win: tuple,
        cell_mask,
        # mask_state: dict | None = None,
        # selection: torch.Tensor | None = None,
    ):
        # create tokenization index
        (idxs_cells, idxs_cells_lens) = token_data

        (mask_tokens, mask_channels) = self.cell_to_token_mask(
            idxs_cells, idxs_cells_lens, cell_mask
        )

        data, datetimes, coords, _, _ = tokenize_apply_mask_target(
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

        # if selection is None:
        #     selection = self._select_target_subset(stream_info, data.shape[0])

        # if selection is not None and data.numel() > 0:
        #     device_sel = selection.to(data.device)
        #     data = data.index_select(0, device_sel)
        #     coords = coords.index_select(0, device_sel)
        #     if idxs_ord_inv.numel() > 0:
        #         idxs_ord_inv = idxs_ord_inv.index_select(0, device_sel)

        #     # datetimes is numpy here
        #     np_sel = selection.cpu().numpy()
        #     datetimes = datetimes[np_sel]

        # TODO: shuffling

        # TODO: idxs_ord_inv
        idxs_ord_inv = None

        # selection not passed on, we call get_target_coords first
        return (data, datetimes, coords, idxs_ord_inv)

    def _select_target_subset(
        self,
        stream_info: dict,
        num_points: int,
    ) -> torch.Tensor | None:
        max_num_targets = stream_info.get("max_num_targets", -1)

        if max_num_targets is None or max_num_targets <= 0 or num_points <= max_num_targets:
            return None

        rng = getattr(self, "rng", None)
        if rng is None:
            rng = np.random.default_rng()
            self.rng = rng

        selected = np.sort(rng.choice(num_points, max_num_targets, replace=False))

        return torch.from_numpy(selected).to(torch.long)

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
