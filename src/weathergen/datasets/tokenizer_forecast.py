# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from functools import partial

import numpy as np
import torch

from weathergen.common.io import IOReaderData
from weathergen.datasets.tokenizer import Tokenizer
from weathergen.datasets.tokenizer_utils import (
    encode_times_source,
    encode_times_target,
    hpy_cell_splits,
    tokenize_window_space,
    tokenize_window_spacetime,
)
from weathergen.datasets.utils import (
    get_target_coords_local_ffast,
)


class TokenizerForecast(Tokenizer):
    def __init__(self, healpix_level: int):
        super().__init__(healpix_level)
        self.rng = None

    def reset_rng(self, rng) -> None:
        """
        Reset rng after mini_epoch to ensure proper randomization
        """
        self.rng = rng

    def batchify_source(
        self,
        stream_info: dict,
        rdata: IOReaderData,
        time_win: tuple,
        normalize_coords,
    ):
        token_size = stream_info["token_size"]
        is_diagnostic = stream_info.get("diagnostic", False)
        tokenize_spacetime = stream_info.get("tokenize_spacetime", False)

        tokenize_window = partial(
            tokenize_window_spacetime if tokenize_spacetime else tokenize_window_space,
            time_win=time_win,
            token_size=token_size,
            hl=self.hl_source,
            hpy_verts_rots=self.hpy_verts_rots_source[-1],
            n_coords=normalize_coords,
            enc_time=encode_times_source,
        )

        source_tokens_cells = [torch.tensor([])]
        source_centroids = [torch.tensor([])]
        source_tokens_lens = torch.zeros([self.num_healpix_cells_source], dtype=torch.int32)

        if is_diagnostic or rdata.data.shape[1] == 0 or len(rdata.data) < 2:
            return (source_tokens_cells, source_tokens_lens, source_centroids)

        # TODO: properly set stream_id; don't forget to normalize
        source_tokens_cells = tokenize_window(
            0,
            rdata.coords,
            rdata.geoinfos,
            rdata.data,
            rdata.datetimes,
        )

        source_tokens_cells = [
            torch.stack(c) if len(c) > 0 else torch.tensor([]) for c in source_tokens_cells
        ]

        source_tokens_lens = torch.tensor([len(s) for s in source_tokens_cells], dtype=torch.int32)
        if source_tokens_lens.sum() > 0:
            source_centroids = self.compute_source_centroids(source_tokens_cells)

        return (source_tokens_cells, source_tokens_lens, source_centroids)

    def batchify_target(
        self,
        stream_info: dict,
        sampling_rate_target: float,
        rdata: IOReaderData,
        time_win: tuple,
    ):
        target_tokens = torch.zeros([self.num_healpix_cells_target], dtype=torch.int32)
        target_coords = torch.zeros([self.num_healpix_cells_target], dtype=torch.int32)
        target_tokens_lens = torch.zeros([self.num_healpix_cells_target], dtype=torch.int32)

        sampling_rate_target = stream_info.get("sampling_rate_target", sampling_rate_target)
        if sampling_rate_target < 1.0:
            mask = self.rng.uniform(0.0, 1.0, rdata.data.shape[0]) < sampling_rate_target
            rdata.coords = rdata.coords[mask]
            rdata.geoinfos = rdata.geoinfos[mask]
            rdata.data = rdata.data[mask]
            rdata.datetimes = rdata.datetimes[mask]

        # TODO: currently treated as empty to avoid special case handling
        if len(rdata.data) < 2:
            return (target_tokens, target_coords, torch.tensor([]), torch.tensor([]))

        # compute indices for each cell
        hpy_idxs_ord_split, _, _, _ = hpy_cell_splits(rdata.coords, self.hl_target)

        # TODO: expose parameter
        with_perm_target = True
        if with_perm_target:
            hpy_idxs_ord_split = [
                idx[self.rng.permutation(len(idx))[: int(len(idx))]] for idx in hpy_idxs_ord_split
            ]

        # helper variables to split according to cells
        idxs_ord = np.concatenate(hpy_idxs_ord_split)
        ll = np.cumsum(np.array([len(a) for a in hpy_idxs_ord_split]))[:-1]

        # compute encoding of time
        times_reordered = rdata.datetimes[idxs_ord]
        times_reordered_enc = encode_times_target(times_reordered, time_win)

        # reorder and split all relevant information based on cells
        target_tokens = np.split(rdata.data[idxs_ord], ll)
        coords_reordered = rdata.coords[idxs_ord]
        target_coords = np.split(coords_reordered, ll)
        target_coords_raw = np.split(coords_reordered, ll)
        target_geoinfos = np.split(rdata.geoinfos[idxs_ord], ll)
        target_times_raw = np.split(times_reordered, ll)
        target_times = np.split(times_reordered_enc, ll)

        target_tokens_lens = torch.tensor([len(s) for s in target_tokens], dtype=torch.int32)

        # compute encoding of target coordinates used in prediction network
        if target_tokens_lens.sum() > 0:
            target_coords = get_target_coords_local_ffast(
                self.hl_target,
                target_coords,
                target_geoinfos,
                target_times,
                self.hpy_verts_rots_target,
                self.hpy_verts_local_target,
                self.hpy_nctrs_target,
            )
            target_coords.requires_grad = False
            target_coords = list(target_coords.split(target_tokens_lens.tolist()))

        return (target_tokens, target_coords, target_coords_raw, target_times_raw)
