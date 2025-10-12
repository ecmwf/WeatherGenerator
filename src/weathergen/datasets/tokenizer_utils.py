from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from astropy_healpix.healpy import ang2pix
from torch import Tensor

from weathergen.datasets.utils import (
    r3tos2,
    s2tor3,
)

CoordNormalizer = Callable[[torch.Tensor], torch.Tensor]

# on some clusters our numpy version is pinned to be 1.x.x where the np.argsort does not
# the stable=True argument
numpy_argsort_args = {"stable": True} if int(np.__version__.split(".")[0]) >= 2 else {}


def arc_alpha(sin_alpha, cos_alpha):
    """Maps a point on the unit circle (np.array or torch.tensor), defined by its (cosine, sine)
    coordinates to its spherical coordinate in [0,2pi)
    """
    t = torch.arccos(cos_alpha)
    mask = sin_alpha < 0.0
    t[mask] = (2.0 * np.pi) - t[mask]
    return t


def encode_times_source(times, time_win) -> torch.tensor:
    """Encode times in the format used for source

    Return:
        len(times) x 5
    """
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
    """Encode times in the format used for target (relative time in window)

    Return:
        len(times) x 5
    """
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

    # We add + 0.5 as in ERA5 very often we otherwise get 0 as the first time and to prevent too
    # many zeros in the input, where we cannot learn anything we add an offset
    return time_tensor + 0.5


def hpy_cell_splits(coords: torch.tensor, hl: int):
    """Compute healpix cell id for each coordinate on given level hl

    Returns
      hpy_idxs_ord_split : list of per cell indices into thetas,phis,posr3
      thetas : thetas in rad
      phis : phis in rad
      posr3 : (thetas,phis) as position in R3
    """
    thetas = ((90.0 - coords[:, 0]) / 180.0) * np.pi
    phis = ((coords[:, 1] + 180.0) / 360.0) * 2.0 * np.pi
    # healpix cells for all points
    hpy_idxs = ang2pix(2**hl, thetas, phis, nest=True)
    posr3 = s2tor3(thetas, phis)

    # extract information to split according to cells by first sorting and then finding split idxs
    hpy_idxs_ord = np.argsort(hpy_idxs, **numpy_argsort_args)
    splits = np.flatnonzero(np.diff(hpy_idxs[hpy_idxs_ord]))

    # extract per cell data
    hpy_idxs_ord_temp = np.split(hpy_idxs_ord, splits + 1)
    hpy_idxs_ord_split = [np.array([], dtype=np.int64) for _ in range(12 * 4**hl)]
    # TODO: split smarter (with a augmented splits list?) so that this loop is not needed
    for b, x in zip(np.unique(np.unique(hpy_idxs[hpy_idxs_ord])), hpy_idxs_ord_temp, strict=True):
        hpy_idxs_ord_split[b] = x

    return (hpy_idxs_ord_split, thetas, phis, posr3)


def hpy_splits(
    coords: torch.Tensor, hl: int, token_size: int, pad_tokens: bool
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Compute healpix cell for each data point and splitting information per cell;
       when the token_size is exceeded then splitting based on lat is used;
       tokens can be padded

    Return :
        idxs_ord : flat list of indices (to data points) per healpix cell
        idxs_ord_lens : lens of lists per cell
        (so that data[idxs_ord].split( idxs_ord_lens) provides per cell data)
        posr3 : R^3 positions of coords
    """

    # list of data points per healpix cell
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
    idxs_ord_lens = [[len(a) for a in aa] for aa in idxs_ord]
    idxs_ord = [torch.cat([idxs for idxs in iidxs]) for iidxs in idxs_ord]

    return idxs_ord, idxs_ord_lens, posr3


def tokenize_window_space(
    stream_id: float,
    coords: torch.tensor,
    geoinfos,
    source,
    times,
    time_win,
    token_size,
    hl,
    hpy_verts_rots,
    n_coords: CoordNormalizer,
    n_geoinfos,
    n_data,
    enc_time,
    pad_tokens=True,
    local_coords=True,
):
    """Process one window into tokens"""

    # len(source)==1 would require special case handling that is not worth the effort
    if len(source) < 2:
        return

    # idx_ord_lens is length is number of tokens per healpix cell
    idxs_ord, idxs_ord_lens, posr3 = hpy_splits(coords, hl, token_size, pad_tokens)

    # pad with zero at the beggining for token size padding
    times_enc = enc_time(times, time_win)
    times_enc_padded = torch.cat([torch.zeros_like(times_enc[0]).unsqueeze(0), times_enc])
    geoinfos_padded = torch.cat([torch.zeros_like(geoinfos[0]).unsqueeze(0), n_geoinfos(geoinfos)])
    source_padded = torch.cat([torch.zeros_like(source[0]).unsqueeze(0), n_data(source)])

    # convert to local coordinates
    # TODO: avoid that padded lists are rotated, which means potentially a lot of zeros
    if local_coords:
        coords_local = _coords_local(posr3, hpy_verts_rots, idxs_ord, n_coords)
    else:
        coords_local = torch.cat([torch.zeros_like(coords[0]).unsqueeze(0), coords])
        coords_local = [coords_local[idxs] for idxs in idxs_ord]

    # reorder based on cells (except for coords_local) and then cat along
    # (time,coords,geoinfos,source) dimension and then split based on cells
    tokens_cells = [
        (
            list(
                torch.split(
                    torch.cat(
                        (
                            torch.full([len(idxs), 1], stream_id, dtype=torch.float32),
                            times_enc_padded[idxs],
                            coords_local[i],
                            geoinfos_padded[idxs],
                            source_padded[idxs],
                        ),
                        1,
                    ),
                    idxs_lens,
                )
            )
            if idxs_lens[0] > 0
            else []
        )
        for i, (idxs, idxs_lens) in enumerate(zip(idxs_ord, idxs_ord_lens, strict=True))
    ]

    return tokens_cells


def tokenize_window_spacetime(
    stream_id,
    coords,
    geoinfos,
    source,
    times,
    time_win,
    token_size,
    hl,
    hpy_verts_rots,
    n_coords,
    n_geoinfos,
    n_data,
    enc_time,
    pad_tokens=True,
    local_coords=True,
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
            hpy_verts_rots,
            n_coords,
            n_geoinfos,
            n_data,
            enc_time,
            pad_tokens,
            local_coords,
        )

        tokens_cells = [t + tc for t, tc in zip(tokens_cells, tokens_cells_cur, strict=True)]

    return tokens_cells


def _coords_local(
    posr3: Tensor, hpy_verts_rots: Tensor, idxs_ord: list[Tensor], n_coords: CoordNormalizer
) -> list[Tensor]:
    """Compute simple local coordinates for a set of 3D positions on the unit sphere."""
    fp32 = torch.float32
    posr3 = torch.cat([torch.zeros_like(posr3[0]).unsqueeze(0), posr3])  # prepend zero

    idxs_ords_lens_l = [len(idxs) for idxs in idxs_ord]
    # int32 should be enough
    idxs_ords_lens = torch.tensor(idxs_ords_lens_l, dtype=torch.int32)
    # concat all indices
    idxs_ords_c = torch.cat(idxs_ord)
    # Copy the rotation matrices for each healpix cell
    # num_points x 3 x 3
    rots = torch.repeat_interleave(hpy_verts_rots, idxs_ords_lens, dim=0)
    # BMM only works for b x n x m and b x m x 1
    # adding a dummy dimension to posr3
    # numpoints x 3 x 1
    posr3_sel = posr3[idxs_ords_c].unsqueeze(-1)
    # print(f"rots = {rots} type(rots) = {type(rots)}", flush=True)
    # print(f"posr3_sel = {posr3_sel} type(posr3_sel) = {type(posr3_sel)}", flush=True)
    vec_rot = torch.bmm(rots, posr3_sel.to(dtype=rots.dtype))
    vec_rot = vec_rot.squeeze(-1)
    vec_scaled = n_coords(r3tos2(vec_rot).to(fp32))
    # split back to ragged list
    # num_points x 2
    coords_local = torch.split(vec_scaled, idxs_ords_lens_l, dim=0)
    return list(coords_local)
