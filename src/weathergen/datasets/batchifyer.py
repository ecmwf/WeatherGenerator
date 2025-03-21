# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import warnings
from functools import partial

import astropy_healpix as hp
import numpy as np
import pandas as pd
import torch
from astropy_healpix.healpy import ang2pix

from weathergen.datasets.utils import (
    get_target_coords_local_ffast,
    healpix_verts_rots,
    locs_to_cell_coords_ctrs,
    r3tos2,
    s2tor3,
)


def encode_times_source(times, time_win) -> torch.tensor:
    # assemble tensor as fed to the network, combining geoinfo and data
    fp32 = torch.float32
    dt = pd.to_datetime(times)
    dt_win = pd.to_datetime(time_win)
    dt_delta = dt - dt_win[0]
    time_tensor = torch.cat(
        (
            torch.tensor(dt.year, dtype=fp32).unsqueeze(1),
            torch.tensor(dt.dayofyear, dtype=fp32).unsqueeze(1),
            torch.tensor(dt.hour * 60 + dt.minute, dtype=fp32).unsqueeze(1),
            torch.tensor(dt_delta.seconds, dtype=fp32).unsqueeze(1),
            torch.tensor(dt_delta.seconds, dtype=fp32).unsqueeze(1),
        ),
        1,
    )

    # normalize
    time_tensor[..., 0] /= 2100.0
    time_tensor[..., 1] = time_tensor[..., 1] / 365.0
    time_tensor[..., 2] = time_tensor[..., 2] / 1440.0
    time_tensor[..., 3] = np.sin(time_tensor[..., 3] / (12.0 * 3600.0) * 2.0 * np.pi)
    time_tensor[..., 4] = np.cos(time_tensor[..., 4] / (12.0 * 3600.0) * 2.0 * np.pi)

    return time_tensor


def encode_times_target(times, time_win) -> torch.tensor:
    dt = pd.to_datetime(times)
    dt_win = pd.to_datetime(time_win)
    # for target only provide local time
    dt_delta = torch.tensor((dt - dt_win[0]).seconds, dtype=torch.float32).unsqueeze(1)
    time_tensor = torch.cat(
        (
            dt_delta,
            dt_delta,
            dt_delta,
            dt_delta,
            dt_delta,
        ),
        1,
    )

    # normalize
    time_tensor[..., 0] = np.sin(time_tensor[..., 0] / (12.0 * 3600.0) * 2.0 * np.pi)
    time_tensor[..., 1] = np.cos(time_tensor[..., 1] / (12.0 * 3600.0) * 2.0 * np.pi)
    time_tensor[..., 2] = np.sin(time_tensor[..., 2] / (12.0 * 3600.0) * 2.0 * np.pi)
    time_tensor[..., 3] = np.cos(time_tensor[..., 3] / (12.0 * 3600.0) * 2.0 * np.pi)
    time_tensor[..., 4] = np.sin(time_tensor[..., 4] / (12.0 * 3600.0) * 2.0 * np.pi)

    return time_tensor


def tokenize_window_space(
    stream_id,
    coords,
    geoinfos,
    source,
    times,
    tokens_cells,
    time_win,
    token_size,
    hl,
    hpy_verts_Rs,
    rng,
    normalizer,
    mr,
):
    """Process one window into tokens"""

    # len(source)==1 would require special case handling that is not worth the effort
    if len(source) < 2:
        return tokens_cells

    thetas = ((90.0 - coords[:, 0]) / 180.0) * np.pi
    phis = ((coords[:, 1] + 180.0) / 360.0) * 2.0 * np.pi
    posr3 = s2tor3(thetas, phis)
    hpy_idxs = ang2pix(2**hl, thetas, phis, nest=True)

    hpy_idxs_ord = torch.argsort(torch.from_numpy(hpy_idxs), stable=True)
    splits = np.flatnonzero(np.diff(hpy_idxs[hpy_idxs_ord]))
    cells_idxs = np.concatenate(
        [hpy_idxs[hpy_idxs_ord][splits], np.array([hpy_idxs[hpy_idxs_ord[-1]]])]
    )
    hpy_idxs_ord_split = np.split(hpy_idxs_ord, splits + 1)

    for i, c in enumerate(cells_idxs):
        # use sorting based on theta
        thetas_sorted = torch.argsort(thetas[hpy_idxs_ord_split[i]], stable=True)
        posr3_cell = posr3[hpy_idxs_ord_split[i]][thetas_sorted]
        R = hpy_verts_Rs[c]
        coords_cell = r3tos2(torch.matmul(R, posr3_cell.transpose(1, 0)).transpose(1, 0)).to(
            torch.float32
        )

        source_cell = torch.cat(
            (
                torch.full([coords_cell.shape[0], 1], stream_id, dtype=torch.float32),
                # use np.take to retain array and collapse it if index is 1-element
                encode_times_source(
                    np.take(np.take(times, hpy_idxs_ord_split[i]), thetas_sorted), time_win
                ),
                normalizer.normalize_coords(coords_cell),
                normalizer.normalize_geoinfos(geoinfos[hpy_idxs_ord_split[i]][thetas_sorted]),
                normalizer.normalize_source_channels(source[hpy_idxs_ord_split[i]][thetas_sorted]),
            ),
            1,
        )

        # split into tokens for current cells and pad last one to have full size
        pad = (
            token_size - (len(source_cell) % token_size) if len(source_cell) % token_size > 0 else 0
        )

        source_cell = torch.nn.functional.pad(
            source_cell, (0, 0, 0, pad), mode="constant", value=0.0
        )
        source_cell = source_cell.reshape((len(source_cell) // token_size, token_size, -1))

        # apply masking (discarding) of tokens
        if mr > 0.0:
            idx_sel = rng.permutation(len(source_cell))[
                : max(1, int((1.0 - mr) * len(source_cell)))
            ]
            source_cell = source_cell[idx_sel]

        tokens_cells[c] += [source_cell]

    return tokens_cells


#############################################
def tokenize_window_spacetime(
    stream_id,
    coords,
    geoinfos,
    source,
    times,
    tokens_cells,
    time_win,
    token_size,
    hl,
    hpy_verts_Rs,
    rng,
    normalizer,
    mr,
):
    t_unique = np.unique(times)
    for _, t in enumerate(t_unique):
        mask = t == times
        tokens_cells = tokenize_window_space(
            coords[mask],
            geoinfos[mask],
            source[mask],
            None,
            tokens_cells,
            time_win,
            token_size,
            hl,
            hpy_verts_Rs,
            rng,
            normalizer,
            mr,
        )

    return tokens_cells


####################################################################################################
class Batchifyer:
    def __init__(self, hl):
        ref = torch.tensor([1.0, 0.0, 0.0])

        self.hl_source = hl
        self.hl_target = hl

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

        self.rng = np.random.default_rng()

    ##############################################
    def batchify_source(
        self,
        stream_info,
        masking_rate,
        masking_rate_sampling,
        rng,
        coords,
        geoinfos,
        source,
        times,
        time_win,
        normalizer,
    ):
        si = stream_info
        token_size = si["token_size"]
        is_diagnostic = si["diagnostic"] if "diagnostic" in stream_info else False
        tokenize_spacetime = (
            si["tokenize_spacetime"] if "tokenize_spacetime" in stream_info else False
        )

        if masking_rate > 0.0:
            # adjust if there's a per-stream masking rate
            masking_rate = si["masking_rate"] if "masking_rate" in si else masking_rate
            # mask either patches or entire stream
            if masking_rate_sampling:
                # masking_rate = self.rng.uniform( low=0., high=masking_rate)
                masking_rate = np.clip(
                    np.abs(self.rng.normal(loc=0.0, scale=1.0 / np.pi)), 0.0, 1.0
                )
            else:
                masking_rate = 1.0 if self.rng.uniform() < masking_rate else 0.0

        tokenize_window = partial(
            tokenize_window_spacetime if tokenize_spacetime else tokenize_window_space,
            time_win=time_win,
            token_size=token_size,
            hl=self.hl_source,
            hpy_verts_Rs=self.hpy_verts_Rs_source[-1],
            normalizer=normalizer,
        )

        if is_diagnostic or source.shape[1] == 0 or len(source) < 2 or masking_rate == 1.0:
            source_tokens_cells = torch.tensor([])
            source_centroids = torch.tensor([])
            source_tokens_lens = torch.zeros([self.num_healpix_cells_source], dtype=torch.int32)

        else:
            source_tokens_cells = [[] for _ in range(self.num_healpix_cells_source)]
            # TODO: properly set stream_id; don't forget to normalize
            source_tokens_cells = tokenize_window(
                0,
                coords,
                geoinfos,
                source,
                times,
                source_tokens_cells,
                rng=self.rng,
                mr=masking_rate,
            )

            source_tokens_cells = [
                torch.cat(c) if len(c) > 0 else torch.tensor([]) for c in source_tokens_cells
            ]
            source_tokens_lens = torch.tensor(
                [len(s) for s in source_tokens_cells], dtype=torch.int32
            )

            if source_tokens_lens.sum() > 0:
                source_means = [
                    self.hpy_verts[-1][i].unsqueeze(0).repeat(len(s), 1)
                    if len(s) > 0
                    else torch.tensor([])
                    for i, s in enumerate(source_tokens_cells)
                ]
                source_means_lens = [len(s) for s in source_means]
                # merge and split to vectorize computations
                source_means = torch.cat(source_means)
                # TODO: precompute also source_means_r3 and then just cat
                source_centroids = torch.cat(
                    [source_means.to(torch.float32), r3tos2(source_means).to(torch.float32)], -1
                )
                source_centroids = torch.split(source_centroids, source_means_lens)
            else:
                source_centroids = torch.tensor([])

        return (source_tokens_cells, source_tokens_lens, source_centroids)

    ##############################################
    def batchify_target(
        self,
        stream_info,
        sampling_rate_target,
        rng,
        coords,
        geoinfos,
        source,
        times,
        time_win,
        normalizer,
    ):
        if len(source) < 2:
            target_tokens, target_coords = torch.tensor([]), torch.tensor([])
            target_tokens_lens = torch.zeros([self.num_healpix_cells_target], dtype=torch.int32)

        else:
            thetas = ((90.0 - coords[:, 0]) / 180.0) * np.pi
            phis = ((coords[:, 1] + 180.0) / 360.0) * 2.0 * np.pi
            hpy_idxs = ang2pix(2**self.hl_target, thetas, phis, nest=True)
            hpy_idxs_ord = np.argsort(hpy_idxs)

            # extract per cell data
            splits = np.flatnonzero(np.diff(hpy_idxs[hpy_idxs_ord]))
            cells_idxs = np.concatenate(
                [hpy_idxs[hpy_idxs_ord][splits], np.array([hpy_idxs[hpy_idxs_ord[-1]]])]
            )
            hpy_idxs_ord_split = np.split(hpy_idxs_ord, splits + 1)

            times_enc = encode_times_target(times, time_win)

            target_tokens = [torch.tensor([]) for _ in range(self.num_healpix_cells_target)]
            target_coords = [torch.tensor([]) for _ in range(self.num_healpix_cells_target)]
            target_geoinfos = [torch.tensor([]) for _ in range(self.num_healpix_cells_target)]
            target_times = [torch.tensor([]) for _ in range(self.num_healpix_cells_target)]
            target_times_enc = [torch.tensor([]) for _ in range(self.num_healpix_cells_target)]
            for i, c in enumerate(cells_idxs):
                t = normalizer.normalize_target_channels(source[hpy_idxs_ord_split[i]])
                t = t[self.rng.permutation(len(t))][: int(len(t) * sampling_rate_target)]
                target_tokens[c] = t
                target_coords[c] = coords[hpy_idxs_ord_split[i]]
                target_geoinfos[c] = normalizer.normalize_geoinfos(geoinfos[hpy_idxs_ord_split[i]])
                target_times[c] = times[hpy_idxs_ord_split[i]]
                target_times_enc[c] = times_enc[hpy_idxs_ord_split[i]]

            target_tokens_lens = torch.tensor([len(s) for s in target_tokens], dtype=torch.int32)

            # if target_coords_local and target_tokens_lens.sum()>0 :
            if target_tokens_lens.sum() > 0:
                target_coords = get_target_coords_local_ffast(
                    self.hl_target,
                    target_coords,
                    target_geoinfos,
                    target_times_enc,
                    self.hpy_verts_Rs_target,
                    self.hpy_verts_local_target,
                    self.hpy_nctrs_target,
                )
                target_coords.requires_grad = False
                target_coords = list(target_coords.split(target_tokens_lens.tolist()))

        return (target_tokens, target_coords, target_times)
