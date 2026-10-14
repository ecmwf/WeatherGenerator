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
from weathergen.datasets.batch import SampleMetaData
from weathergen.datasets.healpix_geometry import HealpixGeometry
from weathergen.datasets.masking import Masker
from weathergen.datasets.stream_data import DecoderTargets
from weathergen.datasets.tokenizer_utils import (
    encode_times_source,
    encode_times_target,
    get_target_coords_local,
    hpy_cell_splits,
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


class TokenizerMasking:
    def __init__(self, masker: Masker):
        self.masker = masker
        self.rng = None
        self.token_size = None
        self.size_time_embedding = 6
        self._geometries: dict[int, HealpixGeometry] = {}

    def get_size_time_embedding(self) -> int:
        return self.size_time_embedding

    def get_geometry(self, healpix_level: int) -> HealpixGeometry:
        """Return read-only geometry cached by level for this tokenizer."""
        if healpix_level not in self._geometries:
            self._geometries[healpix_level] = HealpixGeometry(
                healpix_level, self.masker.domains.get(healpix_level)
            )
        return self._geometries[healpix_level]

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
        healpix_level = stream_info["healpix_level"]
        token_size = stream_info["token_size"]

        tokens = []
        for rdata in data:
            # skip empty data
            if rdata.is_empty():
                tokens += [(None, None)]
                continue
            # tokenize data
            idxs_cells, idxs_cells_lens = tok(
                readerdata_to_torch(rdata),
                token_size,
                healpix_level,
                pad_tokens,
                domain=self.masker.domains.get(healpix_level),
            )
            tokens += [(idxs_cells, idxs_cells_lens)]

        return tokens

    def build_samples_for_stream(
        self,
        training_mode: str,
        stream_info: dict,
    ) -> tuple[np.typing.NDArray, list[np.typing.NDArray], list[SampleMetaData]]:
        """
        Create masks for samples
        """
        return self.masker.build_samples_for_stream(training_mode, stream_info)

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
        # create tokenization index
        (idxs_cells, idxs_cells_lens) = idxs_cells_data

        # select strategy from XXX depending on stream and if student or teacher

        (mask_tokens, mask_channels) = self.cell_to_token_mask(
            idxs_cells, idxs_cells_lens, cell_mask
        )
        geometry = self.get_geometry(stream_info["healpix_level"])

        source_tokens_cells, source_tokens_lens = tokenize_apply_mask_source(
            idxs_cells,
            idxs_cells_lens,
            mask_tokens,
            mask_channels,
            stream_info["stream_id"],
            rdata,
            time_win,
            geometry.hpy_verts_rots[-1],
            encode_times_source,
        )

        return (source_tokens_cells, source_tokens_lens)

    def get_decoder_targets(
        self,
        stream_info: dict,
        rdata: IOReaderData,
        token_data,
        time_win: tuple,
        cell_mask,
        healpix_level: int,
    ) -> DecoderTargets:
        """Regroup retained native targets on a decoder grid, without changing native rows."""
        geometry = self.get_geometry(healpix_level)
        coord_dim = rdata.geoinfos.shape[-1] + self.size_time_embedding + 5 * (3 * 5) + 3 * 8
        idxs_cells, idxs_cells_lens = token_data
        mask_tokens, _ = self.cell_to_token_mask(idxs_cells, idxs_cells_lens, cell_mask)
        if mask_tokens is None:
            return DecoderTargets.empty(geometry.num_healpix_cells, coord_dim)
        native_tokens = [indices for cell in idxs_cells for indices in cell]
        retained = [
            indices for indices, keep in zip(native_tokens, mask_tokens, strict=True) if keep
        ]
        if not retained:
            return DecoderTargets.empty(geometry.num_healpix_cells, coord_dim)
        native_indices = torch.cat(retained)
        cells, _, _ = hpy_cell_splits(rdata.coords[native_indices], healpix_level, geometry.domain)
        lengths = torch.tensor([len(cell) for cell in cells], dtype=torch.int32)
        indices = torch.from_numpy(np.concatenate(cells))
        if indices.numel() == 0:
            return DecoderTargets.empty(geometry.num_healpix_cells, coord_dim)
        raw_indices = native_indices[indices]
        coords = get_target_coords_local(
            stream_info["stream_id"],
            healpix_level,
            lengths,
            rdata.coords[raw_indices],
            rdata.geoinfos[raw_indices],
            encode_times_target(np.atleast_1d(rdata.datetimes[raw_indices]), time_win),
            geometry.hpy_verts_rots,
            geometry.hpy_verts_local,
            geometry.hpy_nctrs,
        )
        return DecoderTargets(coords, lengths, indices)

    def get_target_values(
        self,
        rdata: IOReaderData,
        token_data,
        cell_mask,
    ):
        # create tokenization index
        (idxs_cells, idxs_cells_lens) = token_data

        (mask_tokens, mask_channels) = self.cell_to_token_mask(
            idxs_cells, idxs_cells_lens, cell_mask
        )
        data, datetimes, coords = tokenize_apply_mask_target(
            idxs_cells, mask_tokens, mask_channels, rdata
        )

        idxs_ord_inv = None
        if data.numel() > 0:
            # Invert only the retained tokens, matching the returned target rows.
            tokens = [idxs for cell in idxs_cells for idxs in cell]
            idxs_flat = torch.cat(
                [idxs for idxs, keep in zip(tokens, mask_tokens, strict=True) if keep]
            )
            # compute indices for inversion
            _, idxs_ord_inv = torch.sort(idxs_flat)

        return (data, datetimes, coords, idxs_ord_inv)
