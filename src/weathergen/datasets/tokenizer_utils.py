import numpy as np
import pandas as pd
import torch
from astropy_healpix.healpy import ang2pix

from weathergen.datasets.utils import (
    r3tos2,
    s2tor3,
)


def arc_alpha(sin_alpha, cos_alpha):
    """Invert cosine/sine for alpha in [0,2pi] using both functions"""
    t = torch.arccos(cos_alpha)
    mask = sin_alpha < 0.0
    t[mask] = (2.0 * np.pi) - t[mask]
    return t


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


def hpy_cell_splits(coords, hl):
    """Compute healpix cell id for each coordinate on given level hl"""
    thetas = ((90.0 - coords[:, 0]) / 180.0) * np.pi
    phis = ((coords[:, 1] + 180.0) / 360.0) * 2.0 * np.pi
    hpy_idxs = ang2pix(2**hl, thetas, phis, nest=True)
    posr3 = s2tor3(thetas, phis)

    hpy_idxs_ord = np.argsort(hpy_idxs, stable=True)

    # extract per cell data
    splits = np.flatnonzero(np.diff(hpy_idxs[hpy_idxs_ord]))
    hpy_idxs_ord_split = np.split(hpy_idxs_ord, splits + 1)

    return (hpy_idxs_ord_split, thetas, phis, posr3)


def hpy_splits(coords, hl, token_size, pad_tokens):
    """???"""

    (hpy_idxs_ord_split, thetas, phis, posr3) = hpy_cell_splits(coords, hl)

    # if token_size is exceeed split based on latitude
    # TODO: split by hierarchically traversing healpix scheme
    thetas_sorted = [torch.argsort(thetas[idxs], stable=True) for idxs in hpy_idxs_ord_split]
    # remainder for padding to token size
    if pad_tokens:
        rem = [
            token_size - (len(idxs) % token_size if len(idxs) % token_size != 0 else token_size)
            for idxs in hpy_idxs_ord_split
        ]
    else:
        rem = np.zeros(len(hpy_idxs_ord_split), dtype=np.int32)

    # helper variables to split according to cells
    # pad to token size *and* offset by +1 to account for the index 0 that is added for the padding
    idxs_ord = [
        torch.split(
            torch.cat((torch.from_numpy(np.take(idxs, ts) + 1), torch.zeros(r, dtype=torch.int32))),
            token_size,
        )
        for idxs, ts, r in zip(hpy_idxs_ord_split, thetas_sorted, rem, strict=True)
    ]
    # extract length and flatten nested list
    idxs_ord_lens = [len(a) for aa in idxs_ord for a in aa]
    idxs_ord = torch.cat([idxs for iidxs in idxs_ord for idxs in iidxs])

    return idxs_ord, idxs_ord_lens, posr3


def tokenize_window_space(
    stream_id,
    coords,
    geoinfos,
    source,
    times,
    time_win,
    token_size,
    hl,
    hpy_verts_Rs,
    n_coords,
    n_geoinfos,
    n_data,
    enc_time,
    pad_tokens=True,
):
    """Process one window into tokens"""

    # len(source)==1 would require special case handling that is not worth the effort
    if len(source) < 2:
        return

    idxs_ord, idxs_ord_lens, posr3 = hpy_splits(coords, hl, token_size, pad_tokens)

    times_enc = enc_time(times, time_win)
    times_enc_padded = torch.cat([torch.zeros_like(times_enc[0]).unsqueeze(0), times_enc])
    geoinfos_padded = torch.cat([torch.zeros_like(geoinfos[0]).unsqueeze(0), n_geoinfos(geoinfos)])
    source_padded = torch.cat([torch.zeros_like(source[0]).unsqueeze(0), n_data(source)])

    # pad with zero at the beggining for token size padding
    posr3 = torch.cat([torch.zeros_like(posr3[0]).unsqueeze(0), posr3])
    # reorder based on cells and split
    posr3 = torch.split(posr3[idxs_ord], idxs_ord_lens)
    # convert to local coordinates
    # TODO: how to vectorize it so that there's no list comprhension (and the Rs are not duplicated)
    # TODO: avoid that padded lists are rotated, which means potentially a lot of zeros
    coords_local = torch.cat(
        [
            n_coords(r3tos2(torch.matmul(R, p.transpose(1, 0)).transpose(1, 0)).to(torch.float32))
            for R, p in zip(hpy_verts_Rs, posr3, strict=True)
        ]
    )

    # reorder based on cells (except for coords_local) and then cat along
    # (time,coords,geoinfos,source) dimension and then split based on cells
    tokens_cells = torch.split(
        torch.cat(
            (
                torch.full([len(idxs_ord), 1], stream_id, dtype=torch.float32),
                times_enc_padded[idxs_ord],
                coords_local,
                geoinfos_padded[idxs_ord],
                source_padded[idxs_ord],
            ),
            1,
        ),
        idxs_ord_lens,
    )

    tokens_cells = [[t] for t in tokens_cells]

    return tokens_cells


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
    n_coords,
    n_geoinfos,
    n_data,
    enc_time,
    pad_tokens=True,
):
    """Tokenize respecting an intrinsic time step in the data, i.e. each time step is tokenized
    separately
    """

    num_healpix_cells = 12 * 4**hl
    tokens_cells = [[] for _ in range(num_healpix_cells)]

    t_unique = np.unique(times)
    for _, t in enumerate(t_unique):
        mask = t == times
        tokens_cells_cur = tokenize_window_space(
            stream_id,
            coords[mask],
            geoinfos[mask],
            source[mask],
            times[mask],
            time_win,
            token_size,
            hl,
            hpy_verts_Rs,
            n_coords,
            n_geoinfos,
            n_data,
            pad_tokens,
        )

        tokens_cells = [t + tc for t, tc in zip(tokens_cells, tokens_cells_cur, strict=True)]

    return tokens_cells
