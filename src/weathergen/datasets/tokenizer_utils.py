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


def get_random_rot():
    """Generate a simple random rotation matrix for quick testing."""
    # Generate random Euler angles
    alpha = torch.rand(1).item() * 2 * np.pi  # rotation around z
    beta = torch.rand(1).item() * np.pi       # rotation around y
    gamma = torch.rand(1).item() * 2 * np.pi  # rotation around x
    
    #print(f"[DEBUG] Random Euler angles: α={alpha:.4f}, β={beta:.4f}, γ={gamma:.4f}")
    
    # Rotation matrices
    Rz = torch.tensor([
        [np.cos(alpha), -np.sin(alpha), 0],
        [np.sin(alpha),  np.cos(alpha), 0],
        [0,              0,              1]
    ], dtype=torch.float32)
    
    Ry = torch.tensor([
        [ np.cos(beta), 0, np.sin(beta)],
        [ 0,            1, 0           ],
        [-np.sin(beta), 0, np.cos(beta)]
    ], dtype=torch.float32)
    
    Rx = torch.tensor([
        [1, 0,             0            ],
        [0, np.cos(gamma), -np.sin(gamma)],
        [0, np.sin(gamma),  np.cos(gamma)]
    ], dtype=torch.float32)
    
    # Combined rotation: Rz @ Ry @ Rx
    R = Rz @ Ry @ Rx
    
    # Verify it's a valid rotation (determinant = 1, orthogonal)
    det = torch.det(R).item()
    #print(f"[DEBUG] Rotation matrix determinant: {det:.6f} (should be ~1.0)")
    #print(f"[DEBUG] Rotation matrix:\n{R}")
    
    return R


def hpy_cell_splits(coords: torch.tensor, hl: int, R_fixed: torch.Tensor | None = None):
    """Compute healpix cell id for each coordinate on given level hl

    Args:
        coords: [N, 2] tensor of (lat, lon) in degrees
        hl: HEALPix level
        R_fixed: Optional fixed rotation matrix to use. If None, generates random rotation.

    Returns
      hpy_idxs_ord_split : list of per cell indices into thetas,phis,posr3
      thetas : thetas in rad
      phis : phis in rad
      posr3 : (thetas,phis) as position in R3
      R : rotation matrix used (either R_fixed or newly generated)
    """
    #print(f"\n[DEBUG] ========== hpy_cell_splits ==========")
    #print(f"[DEBUG] Input: {len(coords)} coords, healpix level={hl}")
    #print(f"[DEBUG] Using fixed rotation: {R_fixed is not None}")
    
    # Convert lat/lon to spherical
    thetas = ((90.0 - coords[:, 0]) / 180.0) * np.pi
    phis = ((coords[:, 1] + 180.0) / 360.0) * 2.0 * np.pi
    
    # Convert to 3D Cartesian
    posr3 = s2tor3(thetas, phis)  # [N, 3]
    
    # Use provided rotation or generate new one
    if R_fixed is not None:
        R_random = R_fixed
        #print(f"[DEBUG] Using provided rotation matrix")
    else:
        R_random = get_random_rot()  # [3, 3]
        #print(f"[DEBUG] Generated new rotation matrix")
    
    posr3_rotated = posr3 @ R_random.T  # [N, 3] @ [3, 3]^T = [N, 3]
    
    print(f"[DEBUG] Rotated 3D norms (should be ~1.0): {torch.norm(posr3_rotated[:3], dim=1)}")
    
    # Convert rotated 3D back to spherical
    thetas_rot = torch.acos(torch.clamp(posr3_rotated[:, 2], -1.0, 1.0))
    phis_rot = torch.atan2(posr3_rotated[:, 1], posr3_rotated[:, 0])
    phis_rot = torch.where(phis_rot < 0, phis_rot + 2.0 * np.pi, phis_rot)
    
    # Assign healpix cells using ROTATED coordinates
    hpy_idxs = ang2pix(2**hl, thetas_rot, phis_rot, nest=True)   
    
    #print(f"[DEBUG] Number of unique cells: {len(np.unique(hpy_idxs))} / {12 * 4**hl} total")

    # extract information to split according to cells
    hpy_idxs_ord = np.argsort(hpy_idxs, **numpy_argsort_args)
    splits = np.flatnonzero(np.diff(hpy_idxs[hpy_idxs_ord]))

    hpy_idxs_ord_temp = np.split(hpy_idxs_ord, splits + 1)
    hpy_idxs_ord_split = [np.array([], dtype=np.int64) for _ in range(12 * 4**hl)]
    
    for b, x in zip(np.unique(hpy_idxs[hpy_idxs_ord]), hpy_idxs_ord_temp, strict=True):
        hpy_idxs_ord_split[b] = x
    
    #print(f"[DEBUG] ========================================\n")

    return (hpy_idxs_ord_split, thetas, phis, posr3, R_random)


def hpy_splits(
    coords: torch.Tensor, hl: int, token_size: int, pad_tokens: bool, R_fixed: torch.Tensor | None = None
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Compute healpix cell for each data point and splitting information per cell

    Args:
        R_fixed: Optional fixed rotation matrix. If None, generates random rotation.

    Returns:
        idxs_ord : flat list of indices per healpix cell
        idxs_ord_lens : lens of lists per cell
        posr3 : R^3 positions of coords
        R : rotation matrix used
    """

    # list of data points per healpix cell
    (hpy_idxs_ord_split, thetas, phis, posr3, R) = hpy_cell_splits(coords, hl, R_fixed=R_fixed)

    # if token_size is exceeded split based on latitude
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

    return idxs_ord, idxs_ord_lens, posr3, R


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
    enc_time,
    pad_tokens=True,
    local_coords=True,
    R_fixed: torch.Tensor | None = None,  # ← NEW PARAMETER
):
    """Process one window into tokens
    
    Args:
        R_fixed: Optional fixed rotation matrix. If None, generates random rotation.
                Returns the rotation used for consistency with target tokenization.
    """

    # len(source)==1 would require special case handling
    if len(source) < 2:
        return None, None  # ← Return None for both tokens and rotation

    # idx_ord_lens length is number of tokens per healpix cell
    idxs_ord, idxs_ord_lens, posr3, R = hpy_splits(coords, hl, token_size, pad_tokens, R_fixed=R_fixed)

    # pad with zero at the beginning for token size padding
    times_enc = enc_time(times, time_win)
    times_enc_padded = torch.cat([torch.zeros_like(times_enc[0]).unsqueeze(0), times_enc])
    geoinfos_padded = torch.cat([torch.zeros_like(geoinfos[0]).unsqueeze(0), geoinfos])
    source_padded = torch.cat([torch.zeros_like(source[0]).unsqueeze(0), source])

    # convert to local coordinates
    if local_coords:
        coords_local = _coords_local(posr3, hpy_verts_rots, idxs_ord, n_coords)
    else:
        coords_local = torch.cat([torch.zeros_like(coords[0]).unsqueeze(0), coords])
        coords_local = [coords_local[idxs] for idxs in idxs_ord]

    # reorder based on cells and cat along (time,coords,geoinfos,source) dimension
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

    return tokens_cells, R  # ← Return rotation matrix


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
    enc_time,
    pad_tokens=True,
    local_coords=True,
    R_fixed: torch.Tensor | None = None,  # ← ADD THIS PARAMETER
):
    """Tokenize respecting an intrinsic time step in the data, i.e. each time step is tokenized
    separately
    
    Args:
        R_fixed: Optional fixed rotation matrix. If provided, use same rotation for all timesteps.
                If None, generate new rotation for first timestep and reuse for subsequent ones.
    
    Returns:
        tokens_cells: List of token lists per cell
        R: Rotation matrix used (for consistency with source/target)
    """

    num_healpix_cells = 12 * 4**hl
    tokens_cells = [[] for _ in range(num_healpix_cells)]
    
    # Track rotation across timesteps
    R_used = R_fixed  # Start with provided rotation (or None)

    t_unique = np.unique(times)
    for i, t in enumerate(t_unique):
        mask = t == times
        
        # First timestep: generate or use provided rotation
        # Subsequent timesteps: reuse the rotation from first timestep
        result = tokenize_window_space(
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
            enc_time,
            pad_tokens,
            local_coords,
            R_fixed=R_used,  # ← Pass rotation (None for first iter if not provided)
        )
        
        # Unpack the result
        if result[0] is None:  # Handle empty case
            continue
            
        tokens_cells_cur, R_cur = result
        
        # Store rotation from first timestep
        if i == 0:
            R_used = R_cur

        tokens_cells = [t + tc for t, tc in zip(tokens_cells, tokens_cells_cur, strict=True)]

    return tokens_cells, R_used  # ← Return rotation matrix

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
    vec_rot = torch.bmm(rots, posr3_sel)
    vec_rot = vec_rot.squeeze(-1)
    vec_scaled = n_coords(r3tos2(vec_rot).to(fp32))
    # split back to ragged list
    # num_points x 2
    coords_local = torch.split(vec_scaled, idxs_ords_lens_l, dim=0)
    return list(coords_local)
