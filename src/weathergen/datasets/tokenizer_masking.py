# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import hashlib
import logging
import os
from dataclasses import dataclass

import numpy as np
import torch

from weathergen.common.io import IOReaderData
from weathergen.datasets.batch import SampleMetaData
from weathergen.datasets.masking import Masker
from weathergen.datasets.tokenizer import Tokenizer
from weathergen.datasets.tokenizer_utils import (
    encode_times_source,
    encode_times_target,
    target_coords_from_template,
    tokenize_apply_mask_source,
    tokenize_apply_mask_target,
    tokenize_space,
    tokenize_spacetime,
)

_logger = logging.getLogger(__name__)


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


def grid_cache_enabled(option: bool | None = None) -> bool:
    """
    Grid caches are on by default. Switch them off with the config option
    `data_loading.grid_cache=False` (or the environment variable WEATHERGEN_GRID_CACHE=0, which
    is not forwarded to slurm jobs by the launcher) to restore the uncached behaviour.
    """
    if option is not None:
        return bool(option)
    return os.environ.get("WEATHERGEN_GRID_CACHE", "1") != "0"


def grid_cache_verify(option: bool | None = None) -> bool:
    """
    `data_loading.grid_cache_verify=True` (or WEATHERGEN_GRID_CACHE_VERIFY=1): on every cache hit
    also compute the regular result and raise if it is not bitwise identical. Slow; for validating
    a run on real data.
    """
    if option is not None:
        return bool(option)
    return os.environ.get("WEATHERGEN_GRID_CACHE_VERIFY", "0") == "1"


def _assert_identical(what: str, cached, regular) -> None:
    """Raise AssertionError unless the two (nested) results are bitwise identical."""
    if cached is None or regular is None:
        assert cached is None and regular is None, f"grid cache mismatch in {what}: None"
    elif isinstance(cached, torch.Tensor):
        assert isinstance(regular, torch.Tensor), f"grid cache mismatch in {what}: type"
        assert cached.dtype == regular.dtype and cached.shape == regular.shape, (
            f"grid cache mismatch in {what}: dtype/shape {cached.dtype}{tuple(cached.shape)} vs "
            f"{regular.dtype}{tuple(regular.shape)}"
        )
        assert torch.equal(cached, regular), f"grid cache mismatch in {what}: values"
    elif isinstance(cached, (list, tuple)):
        assert len(cached) == len(regular), f"grid cache mismatch in {what}: length"
        for i, (c, r) in enumerate(zip(cached, regular, strict=True)):
            _assert_identical(f"{what}[{i}]", c, r)
    else:
        c, r = np.asarray(cached), np.asarray(regular)
        assert c.dtype == r.dtype and c.shape == r.shape and np.array_equal(c, r), (
            f"grid cache mismatch in {what}: values"
        )


def _is_identical(cached, regular) -> bool:
    try:
        _assert_identical("confirm", cached, regular)
    except AssertionError:
        return False
    return True


def _geometry_digest(
    coords: torch.Tensor,
    coords_local: torch.Tensor,
    coords_per_cell: torch.Tensor,
    idxs_ord_inv: torch.Tensor,
    n_times: int,
    n_geoinfos: int,
) -> bytes:
    """
    Hash of the cacheable part of get_target_coords: everything except the time and geoinfo
    columns of coords_local, which are rewritten every step.
    """
    h = hashlib.sha256()
    for tensor in (coords, coords_per_cell, idxs_ord_inv):
        x = tensor.detach().contiguous().cpu().numpy()
        h.update(x.dtype.str.encode())
        h.update(np.asarray(x.shape, dtype=np.int64).tobytes())
        h.update(x.tobytes())
    cl = coords_local.detach().contiguous()
    varying = slice(1, 1 + n_times + n_geoinfos)
    keep = torch.cat([cl[..., :1], cl[..., varying.stop :]], dim=-1)
    x = keep.cpu().numpy()
    h.update(x.dtype.str.encode())
    h.update(np.asarray(x.shape, dtype=np.int64).tobytes())
    h.update(x.tobytes())
    return h.digest()


@dataclass
class _TokensCacheEntry:
    """Tokenization of the last window of a stream (valid for windows with equal coords)."""

    coords: torch.Tensor
    idxs_cells: list
    idxs_cells_lens: list


@dataclass
class _TargetCoordsCacheEntry:
    """
    Geometry dependent part of the target coordinates of a stream. It is valid as long as the
    tokenization (idxs_cells / idxs_cells_lens objects), the token mask and the coordinates are the
    same; only datetimes and geoinfos can change between windows.
    """

    # inputs the entry is valid for
    idxs_cells: list
    idxs_cells_lens: list
    mask_tokens: np.typing.NDArray
    coords_raw: torch.Tensor
    # outputs that only depend on the above
    idxs_data: torch.Tensor | None = None
    coords: torch.Tensor | None = None
    masked_points_per_cell: torch.Tensor | None = None
    idxs_ord_inv: torch.Tensor | None = None
    # result of get_target_coords_local of an earlier window and the number of
    # (geoinfo, time) columns it was built with
    coords_local: torch.Tensor | None = None
    num_geoinfos: int = -1
    num_times: int = -1
    # digest of the geometry of the first window; compared with the second matching window
    geometry_digest: bytes | None = None


class TokenizerMasking(Tokenizer):
    def __init__(
        self,
        healpix_level: int,
        masker: Masker,
        grid_cache_option: bool | None = None,
        grid_cache_verify_option: bool | None = None,
    ):
        super().__init__(healpix_level)
        self.masker = masker
        self.rng = None
        self.token_size = None
        # caches for streams with a fixed grid, see get_tokens_windows and get_target_coords
        self._grid_cache_enabled = grid_cache_enabled(grid_cache_option)
        self._grid_cache_verify = grid_cache_verify(grid_cache_verify_option)
        self._tokens_cache: dict[tuple, _TokensCacheEntry] = {}
        self._target_coords_cache: dict[int, _TargetCoordsCacheEntry] = {}
        # keys that passed the window-0 vs window-1 check; keys we will not cache again
        self._tokens_confirmed: set[tuple] = set()
        self._tokens_disabled: set[tuple] = set()
        self._target_coords_disabled: set[int] = set()

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
            # skip empty data
            if rdata.is_empty():
                tokens += [(None, None)]
                continue
            # tokenize data
            idxs_cells, idxs_cells_lens = tok(
                readerdata_to_torch(rdata), token_size, hl, pad_tokens
            )
            tokens += [(idxs_cells, idxs_cells_lens)]

        return tokens

    def build_samples_for_stream(
        self,
        training_mode: str,
        num_cells: int,
        stream_info: dict,
    ) -> tuple[np.typing.NDArray, list[np.typing.NDArray], list[SampleMetaData]]:
        """
        Create masks for samples
        """
        return self.masker.build_samples_for_stream(training_mode, num_cells, stream_info)

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

        source_tokens_cells, source_tokens_lens = tokenize_apply_mask_source(
            idxs_cells,
            idxs_cells_lens,
            mask_tokens,
            mask_channels,
            stream_info["stream_id"],
            rdata,
            time_win,
            self.hpy_verts_rots_source[-1],
            encode_times_source,
        )

        return (source_tokens_cells, source_tokens_lens)

    def get_target_coords(
        self,
        stream_info: dict,
        rdata: IOReaderData,
        token_data,
        time_win: tuple,
        cell_mask,
    ):
        # create tokenization index
        (idxs_cells, idxs_cells_lens) = token_data

        (mask_tokens, mask_channels) = self.cell_to_token_mask(
            idxs_cells, idxs_cells_lens, cell_mask
        )

        # TODO: split up
        _, datetimes, coords, coords_local, coords_per_cell = tokenize_apply_mask_target(
            stream_info["stream_id"],
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

        idxs_ord_inv = None
        if coords.numel() > 0:
            # flatten per-token indices into one flat list
            idxs_flat = torch.cat([idxs for idxs_cell in idxs_cells for idxs in idxs_cell])
            # compute indices for inversion
            _, idxs_ord_inv = torch.sort(idxs_flat)

        return (datetimes, coords, coords_local, coords_per_cell, idxs_ord_inv)

    def get_target_values(
        self,
        stream_info: dict,
        rdata: IOReaderData,
        token_data,
        time_win: tuple,
        cell_mask,
    ):
        # create tokenization index
        (idxs_cells, idxs_cells_lens) = token_data

        (mask_tokens, mask_channels) = self.cell_to_token_mask(
            idxs_cells, idxs_cells_lens, cell_mask
        )

        data, datetimes, coords, _, _ = tokenize_apply_mask_target(
            stream_info["stream_id"],
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

        return (data, datetimes, coords)
