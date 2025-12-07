# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import astropy_healpix as hp
import numpy as np
import torch
from astropy_healpix.healpy import ang2pix

from weathergen.datasets.stream_data import StreamData


####################################################################################################
def arc_alpha(sin_alpha, cos_alpha):
    """Invert cosine/sine for alpha in [0,2pi] using both functions"""
    t = torch.arccos(cos_alpha)
    mask = sin_alpha < 0.0
    t[mask] = (2.0 * np.pi) - t[mask]
    return t


####################################################################################################
def vecs_to_rots(vecs):
    """
    Convert vectors to rotations that align with (1,0,0) ie coordinate origin in geophysical
    spherical coordinates. A variant of Rodrigues formula is used
    """

    rots = torch.zeros((vecs.shape[0], 3, 3), dtype=torch.float64)
    c1 = vecs[:, 0]
    c2 = vecs[:, 1]
    c3 = vecs[:, 2]
    s = torch.square(c2) + torch.square(c3)
    rots[:, 0, 0] = c1
    rots[:, 0, 1] = c2
    rots[:, 0, 2] = c3
    rots[:, 1, 0] = -c2
    rots[:, 1, 1] = (c1 * torch.square(c2) + torch.square(c3)) / s
    rots[:, 1, 2] = (-1.0 + c1) * c2 * c3 / s
    rots[:, 2, 0] = -c3
    rots[:, 2, 1] = (-1.0 + c1) * c2 * c3 / s
    rots[:, 2, 2] = (torch.square(c2) + c1 * torch.square(c3)) / s

    return rots


####################################################################################################
def s2tor3(lats, lons):
    """
    Convert from spherical to Cartesion R^3 coordinates

    Note: mathematics convention with lats in [0,pi] and lons in [0,2pi] is used
          (which is not problematic for lons but for lats care is required)
    """
    sin_lats = torch.sin(lats)
    cos_lats = torch.cos(lats)

    # Calculate the x, y, and z coordinates using vectorized operations.
    x = sin_lats * torch.cos(lons)
    y = sin_lats * torch.sin(lons)
    z = cos_lats

    # Stack the x, y, and z tensors along the last dimension.
    return torch.stack([x, y, z], dim=-1)


####################################################################################################
def r3tos2(pos: torch.Tensor) -> torch.Tensor:
    """
    Convert from spherical to Cartesion R^3 coordinates

    This optimized version is faster and more numerically stable by:
    1. Unbinding the input tensor to get x, y, and z components directly.
    2. Using torch.hypot for a more efficient and stable calculation of
       the xy-plane norm.
    3. Stacking the final latitude and longitude tensors along the last
       dimension, which avoids an expensive permute operation.

    Args:
        pos (torch.Tensor): A tensor of Cartesian coordinates with shape `(..., 3)`.

    Returns:
        torch.Tensor: .
    """
    # Unbind the last dimension to get x, y, and z tensors.
    x, y, z = torch.unbind(pos, dim=-1)

    # Use torch.hypot(x, y)
    xy_norm = torch.sqrt(x**2 + y**2)

    # Calculate latitudes and longitudes using atan2.
    # The output is directly a tensor with the same batch dimensions as the input.
    lats = torch.atan2(z, xy_norm)
    lons = torch.atan2(y, x)

    # Stack the results along the final dimension to get a `(..., 2)` tensor.
    return torch.stack([lats, lons], dim=-1)


####################################################################################################
def locs_to_cell_coords(hl: int, locs: list, dx=0.5, dy=0.5) -> list:
    """
    Map a list of locations per cell to spherical local coordinates centered
    at the healpix cell center
    """

    assert locs[13].shape[-1] == 3 if len(locs[13]) > 0 else True

    # centroids of healpix cells
    num_healpix_cells = 12 * 4**hl
    assert len(locs) == num_healpix_cells

    lons, lats = hp.healpix_to_lonlat(
        np.arange(0, num_healpix_cells), 2**hl, dx=dx, dy=dy, order="nested"
    )
    healpix_centers = s2tor3(
        torch.from_numpy(np.pi / 2.0 - lats.value), torch.from_numpy(lons.value)
    )
    healpix_centers_rots = vecs_to_rots(healpix_centers)

    ## express each centroid in local coordinates w.r.t to healpix center
    #  by rotating center to origin
    local_locs = [
        torch.matmul(R, s.transpose(-1, -2)).transpose(-2, -1) if len(s) > 0 else torch.tensor([])
        for i, (R, s) in enumerate(zip(healpix_centers_rots, locs, strict=False))
    ]

    return local_locs


####################################################################################################
def locs_to_ctr_coords(ctrs_r3, locs: list[torch.Tensor]) -> list:
    """
    Map a list of locations per cell to spherical local coordinates centered
    at the healpix cell center
    """

    ctrs_rots = vecs_to_rots(ctrs_r3).to(torch.float32)

    ## express each centroid in local coordinates w.r.t to healpix center
    #  by rotating center to origin

    # Concatenate all points into single tensor
    all_points = torch.cat(locs, dim=0)

    lengths = torch.tensor([len(s) for s in locs], device=all_points.device)
    batch_indices = torch.repeat_interleave(
        torch.arange(len(locs), device=all_points.device), lengths
    )

    point_rotations = ctrs_rots[batch_indices]

    # Single vectorized batch matrix multiplication
    rotated_points = torch.bmm(point_rotations, all_points.unsqueeze(-1)).squeeze(-1)

    # Split back using tensor operations
    local_locs = torch.split(rotated_points, lengths.tolist())

    return list(local_locs)


####################################################################################################
def healpix_verts(hl: int, dx=0.5, dy=0.5):
    """
    healpix cell center
    """

    # centroids of healpix cells
    num_healpix_cells = 12 * 4**hl
    lons, lats = hp.healpix_to_lonlat(
        np.arange(0, num_healpix_cells), 2**hl, dx=dx, dy=dy, order="nested"
    )
    verts = s2tor3(torch.from_numpy(np.pi / 2.0 - lats.value), torch.from_numpy(lons.value))

    return verts


####################################################################################################
def healpix_verts_rots(hl: int, dx=0.5, dy=0.5):
    """
    healpix cell center
    """

    # centroids of healpix cells
    num_healpix_cells = 12 * 4**hl
    lons, lats = hp.healpix_to_lonlat(
        np.arange(0, num_healpix_cells), 2**hl, dx=dx, dy=dy, order="nested"
    )
    verts = s2tor3(torch.from_numpy(np.pi / 2.0 - lats.value), torch.from_numpy(lons.value))
    verts_rot3 = vecs_to_rots(verts)

    return verts, verts_rot3


####################################################################################################
def locs_to_cell_coords_ctrs(
    healpix_centers_rots: torch.Tensor, locs: list[torch.Tensor]
) -> torch.Tensor:
    """
    Map a list of locations per cell to spherical local coordinates centered
    at the healpix cell center
    """

    ## express each centroid in local coordinates w.r.t to healpix center
    #  by rotating center to origin

    # Concatenate all non-empty locations
    all_points = torch.cat(locs, dim=0)
    lengths = torch.tensor([len(s) for s in locs], device=all_points.device)

    # Efficiently create batch indices using torch.repeat_interleave
    batch_indices = torch.repeat_interleave(
        torch.arange(len(locs), device=all_points.device), lengths
    )

    # Select rotation matrices for each point
    rotations_selected = healpix_centers_rots[batch_indices]

    # Vectorized matrix multiplication
    local_locs = torch.bmm(rotations_selected, all_points.unsqueeze(-1)).squeeze(-1)

    return local_locs


####################################################################################################
def coords_to_hpyidxs(hl, thetas, phis):
    thetas = ((90.0 - thetas) / 180.0) * np.pi
    phis = ((180.0 + phis) / 360.0) * 2.0 * np.pi
    hpyidxs = ang2pix(2**hl, thetas, phis, nest=True)

    return hpyidxs


####################################################################################################
def add_local_vert_coords(hl, a, verts, tcs, zi, dx, dy, geoinfo_offset):
    ref = torch.tensor([1.0, 0.0, 0.0])
    aa = locs_to_cell_coords(hl, verts.unsqueeze(1), dx, dy)
    aa = ref - torch.cat(
        [
            aaa.repeat([*tt.shape[:-1], 1]) if len(tt) > 0 else torch.tensor([])
            for tt, aaa in zip(tcs, aa, strict=False)
        ]
    )
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = aa
    return a


####################################################################################################
def add_local_vert_coords_ctrs2(verts_local, tcs_lens, a, zi, geoinfo_offset):
    ref = torch.tensor([1.0, 0.0, 0.0])
    aa = ref - torch.cat(
        [
            aaa.unsqueeze(0).repeat([*tcs_lens, 1, 1]) if len(tt) > 0 else torch.tensor([])
            for tt, aaa in zip(tcs_lens, verts_local, strict=False)
        ],
        0,
    )
    aa = aa.flatten(1, 2)
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + aa.shape[-1])] = aa
    return a


def compute_offsets_scatter_embed(batch: StreamData, num_steps_input: int) -> StreamData:
    """
    Compute auxiliary information for scatter operation that changes from stream-centric to
    cell-centric computations

    Parameters
    ----------
    batch : str
        batch of stream data information for which offsets have to be computed

    Returns
    -------
    StreamData
        stream data with offsets added as members
    """

    # collect source_tokens_lens for all stream datas
    source_tokens_lens = [
        torch.stack(
            [
                torch.stack(
                    [
                        s.source_tokens_lens[i]
                        if (len(s.source_tokens_lens) > 0) and (len(s.source_tokens_lens[i]) > 0)
                        else torch.tensor([])
                        for s in stl_b
                    ]
                )
                for stl_b in batch
            ]
        )
        for i in range(num_steps_input)
    ]

    # precompute index sets for scatter operation after embed
    offsets_base = [s.sum(1).sum(0).cumsum(0) for s in source_tokens_lens]
    offsets = [torch.cat([torch.zeros(1, dtype=torch.int32), o[:-1]]) for o in offsets_base]
    offsets_pe = [torch.zeros_like(o) for o in offsets]

    if torch.cat(offsets_base).shape[0] == 0:
        return batch

    for i_s in range(num_steps_input):
        for ib, sb in enumerate(batch):  # batch items
            for itype, s in enumerate(sb):  # streams, i.e. here we have StreamData object
                if not s.source_empty():
                    s.source_idxs_embed[i_s] = torch.cat(
                        [
                            torch.arange(offset, offset + token_len, dtype=torch.int64)
                            for offset, token_len in zip(
                                offsets[i_s], source_tokens_lens[i_s][ib, itype], strict=False
                            )
                        ]
                    )
                    s.source_idxs_embed_pe[i_s] = torch.cat(
                        [
                            torch.arange(offset, offset + token_len, dtype=torch.int32)
                            for offset, token_len in zip(
                                offsets_pe[i_s], source_tokens_lens[i_s][ib][itype], strict=False
                            )
                        ]
                    )

                # advance offsets
                offsets[i_s] += source_tokens_lens[i_s][ib][itype]
                offsets_pe[i_s] += source_tokens_lens[i_s][ib][itype]

    return batch


def compute_idxs_predict(forecast_dt: int, batch: StreamData, streams: list[dict]) -> list:
    """
    Compute auxiliary information for prediction

    Parameters
    ----------
    forecast_dt : str
        number of forecast steps
    batch :
        StreamData information for current batch

    Returns
    -------
    tuple[list,list]
        - lens for each item for varlen flash attention
    """

    target_coords_lens = [[s.target_coords_lens for s in sb] for sb in batch]

    # target coords idxs
    tcs_lens_merged = {}
    pad = torch.zeros(1, dtype=torch.int32)
    for ii in range(len(batch[0])):
        # generate len lists for varlen attention (per batch list for local, per-cell attention and
        # global
        tcs_lens_merged[streams[ii]["name"]] = [
            torch.cat(
                [
                    pad,
                    torch.cat(
                        [
                            target_coords_lens[i_b][ii][fstep]
                            for i_b in range(len(target_coords_lens))
                        ]
                    ),
                ]
            ).to(torch.int32)
            for fstep in range(forecast_dt + 1)
        ]

    return tcs_lens_merged


def compute_source_cell_lens(
    batch: list[list[StreamData]], num_steps_input: int
) -> list[torch.tensor]:
    """
    Compute auxiliary information for varlen attention for local assimilation

    Parameters
    ----------
    batch :
        StreamData information for current batch for each batch item and each stream

    Returns
    -------
    torch.tensor
        Offsets for varlen attention
    """

    # precompute for processing in the model (with varlen flash attention)
    source_cell_lens_raw = [
        torch.stack(
            [
                torch.stack(
                    [
                        s.source_tokens_lens[i]
                        if (len(s.source_tokens_lens) > 0) and (len(s.source_tokens_lens[i]) > 0)
                        else torch.tensor([])
                        for s in stl_b
                    ]
                )
                for stl_b in batch
            ]
        )
        for i in range(num_steps_input)
    ]

    source_cell_lens = [torch.sum(c, 1).flatten().to(torch.int32) for c in source_cell_lens_raw]
    source_cell_lens = [torch.cat([torch.zeros(1, dtype=torch.int32), c]) for c in source_cell_lens]

    return source_cell_lens
