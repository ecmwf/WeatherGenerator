# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import time
import warnings
from functools import partial

import astropy_healpix as hp
import numpy as np
import torch

from weathergen.datasets.masking import Masker
from weathergen.datasets.tokenizer_utils import (
    arc_alpha,
    encode_times_source,
    encode_times_target,
    tokenize_window_space,
    tokenize_window_spacetime,
)
from weathergen.datasets.utils import (
    get_target_coords_local_ffast,
    healpix_verts_rots,
    locs_to_cell_coords_ctrs,
    r3tos2,
)
from weathergen.utils.logger import init_loggers


class TokenizerMasking:
    def __init__(self, healpix_level: int, masker: Masker):
        ref = torch.tensor([1.0, 0.0, 0.0])

        self.hl_source = healpix_level
        self.hl_target = healpix_level

        self.masker = masker

        self.num_healpix_cells_source = 12 * 4**self.hl_source
        self.num_healpix_cells_target = 12 * 4**self.hl_target

        verts00, verts00_Rs = healpix_verts_rots(self.hl_source, 0.0, 0.0)
        verts10, verts10_Rs = healpix_verts_rots(self.hl_source, 1.0, 0.0)
        verts11, verts11_Rs = healpix_verts_rots(self.hl_source, 1.0, 1.0)
        verts01, verts01_Rs = healpix_verts_rots(self.hl_source, 0.0, 1.0)
        vertsmm, vertsmm_Rs = healpix_verts_rots(self.hl_source, 0.5, 0.5)
        self.hpy_verts = [
            verts00.to(torch.float32),
            verts10.to(torch.float32),
            verts11.to(torch.float32),
            verts01.to(torch.float32),
            vertsmm.to(torch.float32),
        ]
        self.hpy_verts_Rs_source = [
            verts00_Rs.to(torch.float32),
            verts10_Rs.to(torch.float32),
            verts11_Rs.to(torch.float32),
            verts01_Rs.to(torch.float32),
            vertsmm_Rs.to(torch.float32),
        ]

        verts00, verts00_Rs = healpix_verts_rots(self.hl_target, 0.0, 0.0)
        verts10, verts10_Rs = healpix_verts_rots(self.hl_target, 1.0, 0.0)
        verts11, verts11_Rs = healpix_verts_rots(self.hl_target, 1.0, 1.0)
        verts01, verts01_Rs = healpix_verts_rots(self.hl_target, 0.0, 1.0)
        vertsmm, vertsmm_Rs = healpix_verts_rots(self.hl_target, 0.5, 0.5)
        self.hpy_verts = [
            verts00.to(torch.float32),
            verts10.to(torch.float32),
            verts11.to(torch.float32),
            verts01.to(torch.float32),
            vertsmm.to(torch.float32),
        ]
        self.hpy_verts_Rs_target = [
            verts00_Rs.to(torch.float32),
            verts10_Rs.to(torch.float32),
            verts11_Rs.to(torch.float32),
            verts01_Rs.to(torch.float32),
            vertsmm_Rs.to(torch.float32),
        ]

        self.verts_local = []
        verts = torch.stack([verts10, verts11, verts01, vertsmm])
        temp = ref - torch.stack(locs_to_cell_coords_ctrs(verts00_Rs, verts.transpose(0, 1)))
        self.verts_local.append(temp.flatten(1, 2))

        verts = torch.stack([verts00, verts11, verts01, vertsmm])
        temp = ref - torch.stack(locs_to_cell_coords_ctrs(verts10_Rs, verts.transpose(0, 1)))
        self.verts_local.append(temp.flatten(1, 2))

        verts = torch.stack([verts00, verts10, verts01, vertsmm])
        temp = ref - torch.stack(locs_to_cell_coords_ctrs(verts11_Rs, verts.transpose(0, 1)))
        self.verts_local.append(temp.flatten(1, 2))

        verts = torch.stack([verts00, verts11, verts10, vertsmm])
        temp = ref - torch.stack(locs_to_cell_coords_ctrs(verts01_Rs, verts.transpose(0, 1)))
        self.verts_local.append(temp.flatten(1, 2))

        verts = torch.stack([verts00, verts10, verts11, verts01])
        temp = ref - torch.stack(locs_to_cell_coords_ctrs(vertsmm_Rs, verts.transpose(0, 1)))
        self.verts_local.append(temp.flatten(1, 2))

        self.hpy_verts_local_target = torch.stack(self.verts_local).transpose(0, 1)

        # add local coords wrt to center of neighboring cells
        # (since the neighbors are used in the prediction)
        num_healpix_cells = 12 * 4**self.hl_target
        with warnings.catch_warnings(action="ignore"):
            temp = hp.neighbours(
                np.arange(num_healpix_cells), 2**self.hl_target, order="nested"
            ).transpose()
        # fix missing nbors with references to self
        for i, row in enumerate(temp):
            temp[i][row == -1] = i
        self.hpy_nctrs_target = (
            vertsmm[temp.flatten()]
            .reshape((num_healpix_cells, 8, 3))
            .transpose(1, 0)
            .to(torch.float32)
        )

        worker_info = torch.utils.data.get_worker_info()
        div_factor = (worker_info.id + 1) if worker_info is not None else 1
        self.rng = np.random.default_rng(int(time.time() / div_factor))

        self.size_time_embedding = 6

    def get_size_time_embedding(self) -> int:
        """Get size of time embedding"""
        return self.size_time_embedding

    def reset(self) -> None:
        worker_info = torch.utils.data.get_worker_info()
        div_factor = (worker_info.id + 1) if worker_info is not None else 1
        self.rng = np.random.default_rng(int(time.time() / div_factor))

    def batchify_source(
        self,
        stream_info: dict,
        coords: np.array,
        geoinfos: np.array,
        source: np.array,
        times: np.array,
        time_win: tuple,
        normalizer,  # dataset
    ):
        init_loggers()
        token_size = stream_info["token_size"]
        is_diagnostic = stream_info.get("diagnostic", False)
        tokenize_spacetime = stream_info.get("tokenize_spacetime", False)

        tokenize_window = partial(
            tokenize_window_spacetime if tokenize_spacetime else tokenize_window_space,
            time_win=time_win,
            token_size=token_size,
            hl=self.hl_source,
            hpy_verts_Rs=self.hpy_verts_Rs_source[-1],
            n_coords=normalizer.normalize_coords,
            n_geoinfos=normalizer.normalize_geoinfos,
            n_data=normalizer.normalize_source_channels,
            enc_time=encode_times_source,
        )

        self.token_size = token_size

        # return empty if there is no data or we are in diagnostic mode
        if is_diagnostic or source.shape[1] == 0 or len(source) < 2:
            source_tokens_cells = torch.tensor([])
            source_tokens_lens = torch.zeros([self.num_healpix_cells_source], dtype=torch.int32)
            source_centroids = torch.tensor([])
            return (source_tokens_cells, source_tokens_lens, source_centroids)

        # tokenize all data first
        tokenized_data = tokenize_window(
            0,
            coords,
            geoinfos,
            source,
            times,
        )

        tokenized_data = [
            torch.stack(c) if len(c) > 0 else torch.tensor([]) for c in tokenized_data
        ]

        # Use the masker to get source tokens and the selection mask for the target
        source_tokens_cells = self.masker.mask_source(tokenized_data)

        source_tokens_lens = torch.tensor([len(s) for s in source_tokens_cells], dtype=torch.int32)

        if source_tokens_lens.sum() > 0:
            source_means = [
                (
                    self.hpy_verts[-1][i].unsqueeze(0).repeat(len(s), 1)
                    if len(s) > 0
                    else torch.tensor([])
                )
                for i, s in enumerate(source_tokens_cells)
            ]
            source_means_lens = [len(s) for s in source_means]
            # merge and split to vectorize computations
            source_means = torch.cat(source_means)
            source_centroids = torch.cat(
                [source_means.to(torch.float32), r3tos2(source_means).to(torch.float32)], -1
            )
            source_centroids = torch.split(source_centroids, source_means_lens)
        else:
            source_centroids = torch.tensor([])

        return (source_tokens_cells, source_tokens_lens, source_centroids)

    def batchify_target(
        self,
        stream_info: dict,
        sampling_rate_target: float,
        coords: torch.tensor,
        geoinfos: torch.tensor,
        source: torch.tensor,
        times: np.array,
        time_win: tuple,
        normalizer,  # dataset
    ):
        token_size = stream_info["token_size"]
        tokenize_spacetime = stream_info.get("tokenize_spacetime", False)

        target_tokens, target_coords = torch.tensor([]), torch.tensor([])
        target_tokens_lens = torch.zeros([self.num_healpix_cells_target], dtype=torch.int32)

        # target is empty
        if len(self.masker.perm_sel) == 0:
            return (target_tokens, target_coords, torch.tensor([]), torch.tensor([]))

        # identity function
        def id(arg):
            return arg

        # set tokenization function, no normalization of coords
        tokenize_window = partial(
            tokenize_window_spacetime if tokenize_spacetime else tokenize_window_space,
            time_win=time_win,
            token_size=token_size,
            hl=self.hl_source,
            hpy_verts_Rs=self.hpy_verts_Rs_source[-1],
            n_coords=id,
            n_geoinfos=normalizer.normalize_geoinfos,
            n_data=normalizer.normalize_target_channels,
            enc_time=encode_times_target,
            pad_tokens=False,
            local_coords=False,
        )

        # tokenize
        target_tokens_cells = tokenize_window(
            0,
            coords,
            geoinfos,
            source,
            times,
        )

        target_tokens = self.masker.mask_target(target_tokens_cells, coords, geoinfos, source)

        target_tokens_lens = [len(t) for t in target_tokens]
        if torch.tensor(target_tokens_lens).sum() == 0:
            return (torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]))

        tt_lin = torch.cat(target_tokens)
        tt_lens = target_tokens_lens
        # TODO: can we avoid setting the offsets here manually?
        # TODO: ideally we would not have recover it; but using tokenize_window seems necessary for
        #       consistency -> split tokenize_window in two parts with the cat only happening in the
        #       second
        offset = 6
        # offset of 1 : stream_id
        target_times = torch.split(tt_lin[..., 1:offset], tt_lens)
        target_coords = torch.split(tt_lin[..., offset : offset + coords.shape[-1]], tt_lens)
        offset += coords.shape[-1]
        target_geoinfos = torch.split(tt_lin[..., offset : offset + geoinfos.shape[-1]], tt_lens)
        offset += geoinfos.shape[-1]
        target_tokens = torch.split(tt_lin[..., offset:], tt_lens)

        offset = 6
        target_coords_raw = torch.split(tt_lin[:, offset : offset + coords.shape[-1]], tt_lens)
        # recover absolute time from relatives in encoded ones
        # TODO: avoid recover; see TODO above
        deltas_sec = arc_alpha(tt_lin[..., 1], tt_lin[..., 2]) / (2.0 * np.pi) * (12 * 3600)
        deltas_sec = deltas_sec.numpy().astype("timedelta64[s]")
        target_times_raw = np.split(time_win[0] + deltas_sec, np.cumsum(tt_lens)[:-1])

        # compute encoding of target coordinates used in prediction network
        if torch.tensor(target_tokens_lens).sum() > 0:
            target_coords = get_target_coords_local_ffast(
                self.hl_target,
                target_coords,
                target_geoinfos,
                target_times,
                self.hpy_verts_Rs_target,
                self.hpy_verts_local_target,
                self.hpy_nctrs_target,
            )
            target_coords.requires_grad = False
            target_coords = list(target_coords.split(target_tokens_lens))

        return (target_tokens, target_coords, target_coords_raw, target_times_raw)
