import numpy as np
import pandas as pd
import torch
from astropy_healpix.healpy import ang2pix
from torch import Tensor

from weathergen.common.io import IOReaderData
from weathergen.datasets.utils import (
    locs_to_cell_coords_ctrs,
    locs_to_ctr_coords,
    r3tos2,
    s2tor3,
)

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


def theta_phi_to_standard_coords(coords):
    thetas = ((90.0 - coords[:, 0]) / 180.0) * np.pi
    phis = ((coords[:, 1] + 180.0) / 360.0) * 2.0 * np.pi

    return thetas, phis


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

    # We add + 0.5 as for datasets with regular time steps we otherwise very often get 0 as the
    # first time and to prevent too many zeros in the input
    return time_tensor + 0.5


def hpy_cell_splits(coords: torch.tensor, hl: int):
    """Compute healpix cell id for each coordinate on given level hl

    Returns
      hpy_idxs_ord_split : list of per cell indices into thetas,phis,posr3
      thetas : thetas in rad
      phis : phis in rad
    """
    thetas, phis = theta_phi_to_standard_coords(coords)
    # healpix cells for all points
    hpy_idxs = ang2pix(2**hl, thetas, phis, nest=True)

    # extract information to split according to cells by first sorting and then finding split idxs
    hpy_idxs_ord = np.argsort(hpy_idxs, **numpy_argsort_args)
    splits = np.flatnonzero(np.diff(hpy_idxs[hpy_idxs_ord]))

    # extract per cell data
    hpy_idxs_ord_temp = np.split(hpy_idxs_ord, splits + 1)
    hpy_idxs_ord_split = [np.array([], dtype=np.int64) for _ in range(12 * 4**hl)]
    # TODO: split smarter (with a augmented splits list?) so that this loop is not needed
    for b, x in zip(np.unique(np.unique(hpy_idxs[hpy_idxs_ord])), hpy_idxs_ord_temp, strict=True):
        hpy_idxs_ord_split[b] = x

    return (hpy_idxs_ord_split, thetas, phis)


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
    """

    # list of data points per healpix cell
    (hpy_idxs_ord_split, thetas, phis) = hpy_cell_splits(coords, hl)

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
    offset = 1 if pad_tokens else 0
    int32 = torch.int32
    idxs_ord = [
        list(
            torch.split(
                torch.cat(
                    (torch.from_numpy(np.take(idxs, ts) + offset), torch.zeros(r, dtype=int32))
                ),
                token_size,
            )
        )
        if len(idxs) > 0
        else []
        for idxs, ts, r in zip(hpy_idxs_ord_split, thetas_sorted, rem, strict=True)
    ]

    # extract length and flatten nested list
    idxs_ord_lens = [[len(a) for a in aa] for aa in idxs_ord]

    return idxs_ord, idxs_ord_lens


def tokenize_space(
    rdata,
    token_size,
    hl,
    pad_tokens=True,
):
    """Process one window into tokens"""

    # idx_ord_lens is length is number of tokens per healpix cell
    idxs_ord, idxs_ord_lens = hpy_splits(rdata.coords, hl, token_size, pad_tokens)

    return idxs_ord, idxs_ord_lens


def tokenize_spacetime(
    rdata,
    token_size,
    hl,
    pad_tokens=True,
):
    """Tokenize respecting an intrinsic time step in the data, i.e. each time step is tokenized
    separately
    """

    num_healpix_cells = 12 * 4**hl
    idxs_cells = [[] for _ in range(num_healpix_cells)]
    idxs_cells_lens = [[] for _ in range(num_healpix_cells)]

    t_unique = np.unique(rdata.datetimes)
    for _, t in enumerate(t_unique):
        # data for current time step
        mask = t == rdata.datetimes
        rdata_cur = IOReaderData(
            rdata.coords[mask], rdata.geoinfos[mask], rdata.data[mask], rdata.datetimes[mask]
        )
        idxs_cur, idxs_cur_lens = tokenize_space(rdata_cur, token_size, hl, pad_tokens)

        # collect data for all time steps
        idxs_cells = [t + tc for t, tc in zip(idxs_cells, idxs_cur, strict=True)]
        idxs_cells_lens = [t + tc_l for t, tc_l in zip(idxs_cells_lens, idxs_cur_lens, strict=True)]

    return idxs_cells, idxs_cells_lens


def tokenize_apply_mask_source(
    idxs_cells,
    idxs_cells_lens,
    mask_tokens,
    mask_channels,
    stream_id,
    rdata,
    time_win,
    hpy_verts_rots,
    enc_time,
):
    """
    Apply masking to the data.

    Conceptually, the data is a matrix with the rows corresponding to data points / tokens and
    the cols the channels. Thereby mask_tokens acts on the rows, grouped according to the tokens as
    specified in idxs_cells and mask_channels acts on the columns.

    """

    # convert to token level, forgetting about cells
    idxs_tokens = [i for t in idxs_cells for i in t]
    idxs_lens = [i for t in idxs_cells_lens for i in t]

    # apply spatial masking on a per token level
    if mask_tokens is not None:
        # filter tokens using mask to obtain flat per data point index list
        idxs_data = [t for t, m in zip(idxs_tokens, mask_tokens, strict=True) if m]

        if len(idxs_data) == 0:
            tokens_cells = [torch.tensor([])]
            tokens_per_cell = torch.zeros(len(idxs_cells_lens), dtype=torch.int32)
            return tokens_cells, tokens_per_cell

        idxs_data = torch.cat(idxs_data)
        # filter list of token lens using mask and obtain flat list for splitting
        idxs_data_lens = torch.tensor([t for t, m in zip(idxs_lens, mask_tokens, strict=True) if m])

        # pad with zero at the begining of the conceptual 2D data tensor:
        # idxs_cells -> idxs_tokens -> idxs_data has been prepared so
        # that the zero-index is used to add the padding to the tokens to ensure fixed size
        times_enc = enc_time(rdata.datetimes, time_win)
        zeros_like = torch.zeros_like
        datetimes_enc_padded = torch.cat([zeros_like(times_enc[0]).unsqueeze(0), times_enc])
        geoinfos_padded = torch.cat([zeros_like(rdata.geoinfos[0]).unsqueeze(0), rdata.geoinfos])
        coords_padded = torch.cat([zeros_like(rdata.coords[0]).unsqueeze(0), rdata.coords])
        data_padded = torch.cat([zeros_like(rdata.data[0]).unsqueeze(0), rdata.data])

        # apply mask
        datetimes = datetimes_enc_padded[idxs_data]
        geoinfos = geoinfos_padded[idxs_data]
        coords = coords_padded[idxs_data]
        data = data_padded[idxs_data]

    if mask_channels is not None:
        assert False, "to be implemented"
        # data = data_padded[ : channel_mask]

    # local coords
    num_tokens_per_cell = [len(idxs) for idxs in idxs_cells_lens]
    mask_tokens_per_cell = torch.split(torch.from_numpy(mask_tokens), num_tokens_per_cell)
    tokens_per_cell = torch.tensor([t.sum() for t in mask_tokens_per_cell])
    masked_points_per_cell = torch.tensor(
        [
            torch.tensor([len(t) for t, m in zip(tt, mm, strict=False) if m]).sum()
            for tt, mm in zip(idxs_cells, mask_tokens_per_cell, strict=False)
        ]
    ).to(dtype=torch.int32)
    coords_local = get_source_coords_local(coords, hpy_verts_rots, masked_points_per_cell)

    # create tensor that contains all data
    stream_ids = torch.full([len(datetimes), 1], stream_id, dtype=torch.float32)
    tokens = torch.cat((stream_ids, datetimes, coords_local, geoinfos, data), 1)

    # split up tensor into tokens
    # TODO: idxs_data_lens is currently only defined when mask_tokens is not None
    idxs_data_lens = idxs_data_lens.tolist()
    tokens_cells = torch.split(tokens, idxs_data_lens)

    return tokens_cells, tokens_per_cell


def tokenize_apply_mask_target(
    hl,
    idxs_cells,
    idxs_cells_lens,
    mask_tokens,
    mask_channels,
    rdata,
    time_win,
    hpy_verts_rots,
    hpy_verts_local,
    hpy_nctrs,
    enc_time,
):
    """
    Apply masking to the data.

    Conceptually, the data is a matrix with the rows corresponding to data points / tokens and
    the cols the channels. Thereby mask_tokens acts on the rows, grouped according to the tokens as
    specified in idxs_cells and mask_channels acts on the columns.

    """

    # convert to token level, forgetting about cells
    idxs_tokens = [i for t in idxs_cells for i in t]
    idxs_lens = [i for t in idxs_cells_lens for i in t]

    # apply spatial masking on a per token level
    if mask_tokens is not None:
        # filter tokens using mask to obtain flat per data point index list
        idxs_data = [t for t, m in zip(idxs_tokens, mask_tokens, strict=True) if m]

        if len(idxs_data) == 0:
            do = torch.zeros([0, rdata.data.shape[-1]])
            coords = torch.zeros([0, rdata.coords.shape[-1]])
            dt = np.array([], dtype=np.datetime64)
            masked_points_per_cell = torch.zeros(len(idxs_cells_lens), dtype=torch.int32)
            # data, datetimes, coords, coords_local, masked_points_per_cell
            return do, dt, coords, coords, masked_points_per_cell

        idxs_data = torch.cat(idxs_data)

        # apply mask
        datetimes = rdata.datetimes[idxs_data]
        datetimes_enc = enc_time(datetimes, time_win)
        geoinfos = rdata.geoinfos[idxs_data]
        coords = rdata.coords[idxs_data]
        data = rdata.data[idxs_data]

    if mask_channels is not None:
        assert False, "to be implemented"
        # data = data_padded[ : channel_mask]

    num_tokens_per_cell = [len(idxs) for idxs in idxs_cells_lens]
    mask_tokens_per_cell = torch.split(torch.from_numpy(mask_tokens), num_tokens_per_cell)
    masked_points_per_cell = torch.tensor(
        [
            torch.tensor([len(t) for t, m in zip(tt, mm, strict=False) if m]).sum()
            for tt, mm in zip(idxs_cells, mask_tokens_per_cell, strict=False)
        ]
    ).to(dtype=torch.int32)

    # compute encoding of target coordinates used in prediction network
    if torch.tensor(idxs_lens).sum() > 0:
        coords_local = get_target_coords_local(
            hl,
            masked_points_per_cell,
            coords,
            geoinfos,
            datetimes_enc,
            hpy_verts_rots,
            hpy_verts_local,
            hpy_nctrs,
        )
        coords_local.requires_grad = False
    else:
        coords_local = torch.tensor([])

    return data, datetimes, coords, coords_local, masked_points_per_cell


def get_source_coords_local(
    coords: Tensor,
    hpy_verts_rots: Tensor,
    masked_points_per_cell,
) -> list[Tensor]:
    """Compute simple local coordinates for a set of 3D positions on the unit sphere."""

    # remove padding from coords
    posr3 = s2tor3(*theta_phi_to_standard_coords(coords))
    posr3[0, 0] = 0.0
    posr3[0, 1] = 0.0
    posr3[0, 2] = 0.0

    rots = torch.repeat_interleave(hpy_verts_rots, masked_points_per_cell, dim=0)
    # BMM only works for b x n x m and b x m x 1
    # adding a dummy dimension to posr3
    vec_rot = torch.bmm(rots, posr3.unsqueeze(-1)).squeeze(-1)
    vec_scaled = r3tos2(vec_rot).to(torch.float32)

    # TODO: vec_scaled are small -> should they be normalized/rescaled?

    return vec_scaled


def get_target_coords_local(
    hlc,
    masked_points_per_cell,
    coords,
    target_geoinfos,
    target_times,
    verts_rots,
    verts_local,
    nctrs,
):
    """Generate local coordinates for target coords w.r.t healpix cell vertices and
    and for healpix cell vertices themselves
    """

    # target_coords_lens = [len(t) for t in target_coords]
    # tcs, target_coords = tcs_optimized(target_coords)
    target_coords = s2tor3(*theta_phi_to_standard_coords(coords))
    tcs = torch.split(target_coords, masked_points_per_cell.tolist())

    if target_coords.shape[0] == 0:
        return torch.tensor([])
    # target_geoinfos = torch.cat(target_geoinfos)
    # target_times = torch.cat(target_times)

    verts00_rots, verts10_rots, verts11_rots, verts01_rots, vertsmm_rots = verts_rots

    a = torch.zeros(
        [
            *target_coords.shape[:-1],
            1 + target_geoinfos.shape[1] + target_times.shape[1] + 5 * (3 * 5) + 3 * 8,
        ]
    )
    # TODO: properly set stream_id, implicitly zero at the moment
    geoinfo_offset = 1
    a[..., geoinfo_offset : geoinfo_offset + target_times.shape[1]] = target_times
    geoinfo_offset += target_times.shape[1]
    a[..., geoinfo_offset : geoinfo_offset + target_geoinfos.shape[1]] = target_geoinfos
    geoinfo_offset += target_geoinfos.shape[1]

    ref = torch.tensor([1.0, 0.0, 0.0])

    tcs_lens = torch.tensor([tt.shape[0] for tt in tcs], dtype=torch.int32)
    tcs_lens_mask = tcs_lens > 0
    tcs_lens = tcs_lens[tcs_lens_mask]

    vls = torch.cat(
        [
            vl.repeat([tt, 1, 1])
            for tt, vl in zip(tcs_lens, verts_local[tcs_lens_mask], strict=False)
        ],
        0,
    )
    vls = vls.transpose(0, 1)

    zi = 0
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - locs_to_cell_coords_ctrs(
        verts00_rots, tcs
    )

    zi = 3
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + vls.shape[-1])] = vls[0]

    zi = 15
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - locs_to_cell_coords_ctrs(
        verts10_rots, tcs
    )

    zi = 18
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + vls.shape[-1])] = vls[1]

    zi = 30
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - locs_to_cell_coords_ctrs(
        verts11_rots, tcs
    )

    zi = 33
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + vls.shape[-1])] = vls[2]

    zi = 45
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - locs_to_cell_coords_ctrs(
        verts01_rots, tcs
    )

    zi = 48
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + vls.shape[-1])] = vls[3]

    zi = 60
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - locs_to_cell_coords_ctrs(
        vertsmm_rots, tcs
    )

    zi = 63
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + vls.shape[-1])] = vls[4]

    tcs_ctrs = torch.cat([ref - torch.cat(locs_to_ctr_coords(c, tcs)) for c in nctrs], -1)
    zi = 75
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + (3 * 8))] = tcs_ctrs

    # remaining geoinfos (zenith angle etc)
    zi = 99
    a[..., (geoinfo_offset + zi) :] = target_coords[..., (geoinfo_offset + 2) :]

    return a
