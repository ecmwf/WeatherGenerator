# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import dataclasses
import hashlib
import logging
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

_logger = logging.getLogger(__name__)

stat_loss_fcts = ["stats", "kernel_crps"]  # Names of loss functions that need std computed


def gaussian(x, mu=0.0, std_dev=1.0):
    # unnormalized Gaussian where maximum is one
    return torch.exp(-0.5 * (x - mu) * (x - mu) / (std_dev * std_dev))


def normalized_gaussian(x, mu=0.0, std_dev=1.0):
    return (1 / (std_dev * np.sqrt(2.0 * np.pi))) * torch.exp(
        -0.5 * (x - mu) * (x - mu) / (std_dev * std_dev)
    )


def erf(x, mu=0.0, std_dev=1.0):
    c1 = torch.sqrt(torch.tensor(0.5 * np.pi))
    c2 = torch.sqrt(1.0 / torch.tensor(std_dev * std_dev))
    c3 = torch.sqrt(torch.tensor(2.0))
    val = c1 * (1.0 / c2 - std_dev * torch.special.erf((mu - x) / (c3 * std_dev)))
    return val


def gaussian_crps(target, ens, mu, stddev):
    # see Eq. A2 in S. Rasp and S. Lerch. Neural networks for postprocessing ensemble weather
    # forecasts. Monthly Weather Review, 146(11):3885 – 3900, 2018.
    c1 = np.sqrt(1.0 / np.pi)
    t1 = 2.0 * erf((target - mu) / stddev) - 1.0
    t2 = 2.0 * normalized_gaussian((target - mu) / stddev)
    val = stddev * ((target - mu) / stddev * t1 + t2 - c1)
    return torch.mean(val)  # + torch.mean( torch.sqrt( stddev) )


def stats(target, ens, mu, stddev):
    diff = gaussian(target, mu, stddev) - 1.0
    return torch.mean(diff * diff) + torch.mean(torch.sqrt(stddev))


def stats_normalized(target, ens, mu, stddev):
    a = normalized_gaussian(target, mu, stddev)
    max = 1 / (np.sqrt(2 * np.pi) * stddev)
    d = a - max
    return torch.mean(d * d) + torch.mean(torch.sqrt(stddev))


def stats_normalized_erf(target, ens, mu, stddev):
    delta = -torch.abs(target - mu)
    d = 0.5 + torch.special.erf(delta / (np.sqrt(2.0) * stddev))
    return torch.mean(d * d)  # + torch.mean( torch.sqrt( stddev) )


def mse_ens(target, ens, mu, stddev):
    mse_loss = torch.nn.functional.mse_loss
    return torch.stack([mse_loss(target, mem) for mem in ens], 0).mean()


def kernel_crps(
    targets,
    preds,
    weights_channels: torch.Tensor | None,
    weights_points: torch.Tensor | None,
    fair=True,
):
    """
    Compute kernel CRPS

    Params:
    target : shape ( num_data_points , num_channels )
    pred : shape ( ens_dim , num_data_points , num_channels)
    weights_channels : shape = (num_channels,)
    weights_points : shape = (num_data_points)

    Returns:
    loss: scalar - overall weighted CRPS
    loss_chs: [C] - per-channel CRPS (location-weighted, not channel-weighted)
    """

    ens_size = preds.shape[0]
    assert ens_size > 1, "Ensemble size has to be greater than 1 for kernel CRPS."
    assert len(preds.shape) == 3, "if data has batch dimension, remove unsqueeze() below"

    # replace NaN by 0
    mask_nan = ~torch.isnan(targets)
    targets = torch.where(mask_nan, targets, 0)
    preds = torch.where(mask_nan, preds, 0)

    # permute to enable/simply broadcasting and contractions below
    preds = preds.permute([2, 1, 0]).unsqueeze(0).to(torch.float32)
    targets = targets.permute([1, 0]).unsqueeze(0).to(torch.float32)

    mae = torch.mean(torch.abs(targets[..., None] - preds), dim=-1)

    ens_n = -1.0 / (ens_size * (ens_size - 1)) if fair else -1.0 / (ens_size**2)
    abs = torch.abs
    ens_var = torch.zeros(size=preds.shape[:-1], device=preds.device)
    # loop to reduce memory usage
    for i in range(ens_size):
        ens_var += torch.sum(ens_n * abs(preds[..., i].unsqueeze(-1) - preds[..., i + 1 :]), dim=-1)

    kcrps_locs_chs = mae + ens_var

    # apply point weighting
    if weights_points is not None:
        kcrps_locs_chs = kcrps_locs_chs * weights_points
    # apply channel weighting
    kcrps_chs = torch.mean(torch.mean(kcrps_locs_chs, 0), -1)
    if weights_channels is not None:
        kcrps_chs = kcrps_chs * weights_channels

    return torch.mean(kcrps_chs), kcrps_chs


def lp_loss(
    target: torch.Tensor,
    pred: torch.Tensor,
    p_norm: int,
    with_p_root: bool = False,
    with_mean: bool = True,
    weights_channels: torch.Tensor | None = None,
    weights_points: torch.Tensor | None = None,
):
    """
    This function computes the Lp-norm for any arbitrary integer p < inf.
    By default, the Lp-norm is normalized by the number of samples (i.e. with_mean=True).
    * For example: p=1 corresponds to MAE; p=2 corresponds to MSE.
    The samples are weighted by location if weights_points is not None.
    The norm can optionally be normalised by the pth root.
    * For example: p=2 and with_p_root=True corresponds to RMSE.
    The mean across all channels can optionally be weighted by channel weights.

    The function implements:

    loss = Mean_{channels}  ( weight_channels *
                                ( Mean_{data_pts}(|(target - pred)|**p * weights_points)
                                ) ** (1/p)
                            )

    Geometrically,

        ------------------------     -
        |                      |    |  |
        |                      |    |  |
        |                      |    |  |
        |     target - pred    | x  |wp|
        |                      |    |  |
        |                      |    |  |
        |                      |    |  |
        ------------------------     -
                    x
        ------------------------
        |          wc          |
        ------------------------

    where wp = weights_points and wc = weights_channels and "x" denotes row/col-wise multiplication.

    The computations are:
    1. weight the rows of |(target - pred)|**p by wp = weights_points (if given)
    2. take the mean over the row
    3. weight the collapsed cols by wc = weights_channels (if given)
    4. take the mean over the channel-weighted cols

    Params:
        target : tensor of shape ( num_data_points , num_channels )
        pred : tensor of shape ( ens_dim , num_data_points , num_channels)
        p_norm : integer defining the p the type of the norm
        with_mean : boolean defining whether the norm is summed or averaged
        with_p_root : boolean defining whether the p-th root of the norm is returned
        weights_channels (optional): tensor of shape = (num_channels,)
        weights_points (optional): tensor of shape = (num_data_points)

    Return:
        loss : (weighted) scalar loss (e.g. for gradient computation)
        loss_chs : losses per channel (if given with location weighting but no channel weighting)
    """

    assert type(p_norm) is int, "Only integer p supported for p-norm loss"

    mask_nan = ~torch.isnan(target)
    pred = pred[0] if pred.shape[0] == 0 else pred.mean(0)

    diff_p = torch.pow(
        torch.abs(torch.where(mask_nan, target, 0) - torch.where(mask_nan, pred, 0)), p_norm
    )
    if weights_points is not None:
        diff_p = (diff_p.transpose(1, 0) * weights_points).transpose(1, 0)
    loss_chs = diff_p.mean(0) if with_mean else diff_p.sum(0)
    loss_chs = torch.pow(loss_chs, 1.0 / p_norm) if with_p_root else loss_chs
    loss = torch.mean(loss_chs * weights_channels if weights_channels is not None else loss_chs)

    return loss, loss_chs


def mse(
    target: torch.Tensor,
    pred: torch.Tensor,
    weights_channels: torch.Tensor | None,
    weights_points: torch.Tensor | None,
):
    """
    Computes the mean squared error (mse).
    See lp_loss function above for a detailed explanation of arguments.
    """
    return lp_loss(
        target=target,
        pred=pred,
        p_norm=2,
        with_p_root=False,
        with_mean=True,
        weights_channels=weights_channels,
        weights_points=weights_points,
    )


def rss(
    target: torch.Tensor,
    pred: torch.Tensor,
    weights_channels: torch.Tensor | None,
    weights_points: torch.Tensor | None,
):
    """
    Computes the residual sum of squares (rss).
    See lp_loss function above for a detailed explanation of arguments.
    """
    return lp_loss(
        target=target,
        pred=pred,
        p_norm=2,
        with_p_root=False,
        with_mean=False,
        weights_channels=weights_channels,
        weights_points=weights_points,
    )


def rmse(
    target: torch.Tensor,
    pred: torch.Tensor,
    weights_channels: torch.Tensor | None,
    weights_points: torch.Tensor | None,
):
    """
    Computes the root mean squared error (rmse).
    See lp_loss function above for a detailed explanation of arguments.
    """
    return lp_loss(
        target=target,
        pred=pred,
        p_norm=2,
        with_p_root=True,
        with_mean=True,
        weights_channels=weights_channels,
        weights_points=weights_points,
    )


def mae(
    target: torch.Tensor,
    pred: torch.Tensor,
    weights_channels: torch.Tensor | None,
    weights_points: torch.Tensor | None,
):
    """
    Computes the mean absolute error (mae).
    See lp_loss function above for a detailed explanation of arguments.
    """
    return lp_loss(
        target=target,
        pred=pred,
        p_norm=1,
        with_p_root=False,
        with_mean=True,
        weights_channels=weights_channels,
        weights_points=weights_points,
    )


def cosine_latitude(target_coords, min_value=1e-3, max_value=1.0):
    latitudes_radian = target_coords[:, 0] * np.pi / 180
    return (max_value - min_value) * torch.cos(latitudes_radian) + min_value


def gamma_decay(num_forecast_steps, gamma):
    fsteps = np.arange(num_forecast_steps)
    weights = gamma**fsteps
    return weights * (len(fsteps) / np.sum(weights))


def student_teacher_softmax(student_patches, teacher_patches, student_temp):
    """
    Cross-entropy between softmax outputs of the teacher and student networks.
    student_patches: (B, N, D) tensor
    teacher_patches: (B, N, D) tensor
    student_temp: float
    """
    loss = torch.sum(
        teacher_patches * F.log_softmax(student_patches / student_temp, dim=-1), dim=-1
    )
    loss = torch.mean(loss, dim=-1)
    return -loss.mean()


def softmax(t, s, temp):
    return torch.sum(t * F.log_softmax(s / temp, dim=-1), dim=-1)


def masked_student_teacher_patch_softmax(
    student_patches_masked,
    teacher_patches_masked,
    student_masks,
    teacher_masks,
    student_temp,
    n_masked_patches=None,
    masks_weight=None,
):
    """
    Cross-entropy between softmax outputs of the teacher and student networks.
    student_patches_masked,
    teacher_patches_masked,
    student_masks_flat,
    student_temp,
    n_masked_patches=None,
    masks_weight=None,
    """
    mask = torch.logical_and(teacher_masks, torch.logical_not(student_masks))
    loss = softmax(teacher_patches_masked[mask], student_patches_masked[mask], student_temp)
    if masks_weight is None:
        masks_weight = (
            (1 / student_masks.sum(-1).clamp(min=1.0))
            .unsqueeze(-1)
            .expand_as(student_masks)  # [student_masks_flat]
        )
    loss = loss * masks_weight[mask]
    return -loss.sum() / student_masks.shape[0]


def student_teacher_global_softmax(student_outputs, teacher_output, student_temp):
    """
    This comment is outdated TODO fix. Leaving it for now so we remember the context

    This assumes that student_outputs : list[Tensor[2*batch_size, num_class_tokens, channel_size])
                 and  teacher_outputs : Tensor[2*batch_size, num_class_tokens, channel_size]
    The 2* is because there is two global views and they are concatenated in the batch dim
    in DINOv2 as far as I can tell.
    """
    total_loss = 0
    for s in student_outputs:
        lsm = F.log_softmax(s / student_temp, dim=-1)
        loss = torch.sum(teacher_output * lsm, dim=-1)
        total_loss -= loss.mean()
    return total_loss


##############################################
# Haar wavelet losses
##############################################


def haar_2d(field: torch.Tensor):
    """
    One-level 2D Haar wavelet decomposition over the LAST TWO dims.
    field: (..., H, W), H and W must be even.
    Returns LL, LH, HL, HH each of shape (..., H//2, W//2).

    What the four subbands mean physically:
    LL: average of average. The smooth, low-frequency content (large-scale spatial mean).
    LH: average of difference. Contrast between left and right columns (east-west gradients).
    HL: difference of average. Contrast between top and bottom rows (north-south gradients).
    HH: difference of difference. Point features and diagonal / checkerboard-like variation.
    """
    c = 2**-0.5  # 1/sqrt(2) - orthonormal Haar scaling

    # 1D Haar along rows (second-to-last dim)
    L = (field[..., 0::2, :] + field[..., 1::2, :]) * c
    H = (field[..., 0::2, :] - field[..., 1::2, :]) * c

    # 1D Haar along columns (last dim)
    LL = (L[..., 0::2] + L[..., 1::2]) * c
    LH = (L[..., 0::2] - L[..., 1::2]) * c
    HL = (H[..., 0::2] + H[..., 1::2]) * c
    HH = (H[..., 0::2] - H[..., 1::2]) * c

    return LL, LH, HL, HH


def _warn_once(owner, key: str, msg: str):
    """Log `msg` only the first time `key` is seen on `owner` (a function object)."""
    if owner is not None and not getattr(owner, key, False):
        setattr(owner, key, True)
        _logger.warning(msg)


# template_path -> (template_grid (ny*nx, 2) [lat, lon], (ny, nx))
_haar_templates: dict[str, tuple[NDArray, tuple[int, int]]] = {}


def _load_haar_template(owner, prefix, template_path, stream_name, num_channels, dev):
    """
    Load and cache (per process, by path) the 2D lat/lon template output grid.

    Supports a NetCDF file with 2D `latitude`/`longitude` and dims `y`/`x`, or an `.npz`
    file with keys `lat`, `lon`, `ny`, `nx`.

    Returns (template_grid, (ny, nx)) or (None, zero_loss) if no template is available.
    """
    hit = _haar_templates.get(template_path)
    if hit is not None:
        return hit

    zero_loss = (
        torch.tensor(0.0, device=dev, requires_grad=True),
        torch.zeros(num_channels, device=dev),
    )

    if not template_path:
        _warn_once(
            owner,
            f"_{prefix}_no_template_reported_{stream_name}",
            f"[{owner.__name__}] stream={stream_name} "
            f"template_path is empty. Setting loss to zero.",
        )
        return None, zero_loss

    try:
        if template_path.endswith(".npz"):
            _z = np.load(template_path)
            olat = _z["lat"]
            olon = _z["lon"]
            ny = int(_z["ny"])
            nx = int(_z["nx"])
        else:
            import xarray as xr

            template = xr.open_dataset(template_path)
            olat = template.latitude.values.flatten()
            olon = template.longitude.values.flatten()
            ny = len(template.y.values)
            nx = len(template.x.values)
            template.close()

        template_grid = np.stack([olat, olon], axis=1)
        _haar_templates[template_path] = (template_grid, (ny, nx))

        _logger.info(
            f"[{owner.__name__}] stream={stream_name} "
            f"template loaded: grid=({ny}x{nx})  n_output_points={ny * nx}"
        )
        return template_grid, (ny, nx)

    except Exception as _e:
        _warn_once(
            owner,
            f"_{prefix}_template_error_reported_{stream_name}",
            f"[{owner.__name__}] stream={stream_name} "
            f"failed to load template '{template_path}': {_e}. Setting loss to zero.",
        )
        return None, zero_loss


#: Max number of regrid-geometry entries cached per process (LRU). One entry is used per
#: (template, target point set, regrid method, device), so a fixed-grid stream needs one entry
#: per regrid method it uses. Entries live on the loss device and hold O(ny*nx) index/weight
#: tensors (~6 MB "nearest", ~50 MB "linear" on a 949x739 MEPS template). Set to 0 to disable
#: the point-set cache (template loading and template spacing are always cached).
HAAR_REGRID_CACHE_SIZE = 8

# id(template_grid) -> (template_grid, template spacing). The template arrays themselves are
# cached in _haar_templates (see _load_haar_template), so their id is stable; the array is
# kept in the value so the id can never be recycled while the entry exists.
_haar_template_dx_cache: dict[int, tuple[NDArray, float]] = {}
# point-set key -> _RegridGeometry, LRU-ordered
_haar_regrid_cache: "OrderedDict[tuple, _RegridGeometry]" = OrderedDict()


@dataclasses.dataclass
class _RegridGeometry:
    """Data-independent part of the regridding for one (template, point set, device)."""

    isort_t: torch.Tensor  # (ny*nx,) long: nearest data point of every template cell
    valid_cell: torch.Tensor  # (ny*nx,) bool: template cell has a data point nearby
    lin_w: torch.Tensor | None = None  # (ny*nx, N) sparse barycentric weights ("linear")
    lin_inside: torch.Tensor | None = None  # (ny*nx, 1) bool: cell inside the triangulation
    lin_error: str | None = None  # why "linear" geometry could not be built


def _haar_template_spacing(template_grid: NDArray) -> float:
    """Median nearest-neighbour spacing of the template grid (depends on the template only)."""
    from scipy.spatial import cKDTree as _cKDTree

    hit = _haar_template_dx_cache.get(id(template_grid))
    if hit is not None and hit[0] is template_grid:
        return hit[1]

    _tree_tmpl = _cKDTree(template_grid)
    _tmpl_nn, _ = _tree_tmpl.query(template_grid, k=2)
    _tmpl_dx = float(np.median(_tmpl_nn[:, 1]))
    _haar_template_dx_cache[id(template_grid)] = (template_grid, _tmpl_dx)
    return _tmpl_dx


def _haar_regrid_geometry(ipoints, template_grid, linear: bool, dev) -> _RegridGeometry:
    """
    Build (or fetch from cache) the regrid geometry of the scattered points `ipoints`
    (N, 2) [lat, lon] onto `template_grid` (ny*nx, 2).

    The result depends only on the template and on the target coordinates, not on the data
    values, so it is reused whenever the same point set comes back (e.g. a fixed anemoi grid
    without target subsampling). The cache key hashes the coordinates bytes, so a different
    point set (random subsampling, different crop) is simply a cache miss.
    """
    from scipy.spatial import cKDTree as _cKDTree

    key = None
    if HAAR_REGRID_CACHE_SIZE > 0:
        _pts = np.ascontiguousarray(ipoints)
        key = (
            id(template_grid),
            _pts.shape,
            str(_pts.dtype),
            hashlib.blake2b(_pts.tobytes(), digest_size=16).digest(),
            linear,
            str(dev),
        )
        hit = _haar_regrid_cache.get(key)
        if hit is not None:
            _haar_regrid_cache.move_to_end(key)
            return hit

    # nearest index gather (always computed: it is the default and the fallback).
    # A single KD-tree over the data points serves the nearest-neighbour gather, the
    # distance to the nearest data point and the data spacing. This is the same tree and
    # query that scipy's NearestNDInterpolator performs internally.
    _tree_data = _cKDTree(ipoints)
    _data_nn, _isort = _tree_data.query(template_grid, k=1)
    isort_t = torch.from_numpy(_isort.astype(int)).long().to(dev)

    # --- domain crop: mask template cells with no nearby data point ---
    # template_grid is the FULL (uncropped) grid. If the data covers only a sub-box,
    # the nearest-neighbour gather extrapolates every outside cell to the nearest edge
    # point, smearing it across the sphere. Reject cells whose nearest actual data point
    # is far away. The radius is scaled by the coarser of {template spacing, data spacing}
    # so a legitimately coarse stream (ERA5) on a fine template (MEPS) fills its natural
    # footprint while genuine out-of-domain cells are still rejected.
    _tmpl_dx = _haar_template_spacing(template_grid)
    if ipoints.shape[0] >= 2:
        _self_nn, _ = _tree_data.query(ipoints, k=2)
        _data_dx = float(np.median(_self_nn[:, 1]))
    else:
        _data_dx = _tmpl_dx
    _max_dist = max(max(_tmpl_dx, _data_dx) * 2.0, 1e-6)
    valid_cell = torch.from_numpy(_data_nn <= _max_dist).to(dev)  # (ny*nx,)

    geom = _RegridGeometry(isort_t=isort_t, valid_cell=valid_cell)

    # --- optional linear (barycentric) regridding weights ---
    if linear and ipoints.shape[0] >= 3:
        try:
            from scipy.spatial import Delaunay as _Delaunay

            _tri = _Delaunay(ipoints)
            _simplex = _tri.find_simplex(template_grid)
            _inside = _simplex >= 0
            if _inside.any():
                _sx = _simplex[_inside]
                _T = _tri.transform[_sx, :2]
                _r = template_grid[_inside] - _tri.transform[_sx, 2]
                _bary2 = np.einsum("mij,mj->mi", _T, _r)
                _bary = np.concatenate([_bary2, 1.0 - _bary2.sum(axis=1, keepdims=True)], axis=1)
                _verts = _tri.simplices[_sx]
                _rows = np.repeat(np.flatnonzero(_inside), 3)
                _cols = _verts.reshape(-1)
                _wts = _bary.reshape(-1).astype(np.float32)
                geom.lin_w = torch.sparse_coo_tensor(
                    torch.from_numpy(np.stack([_rows, _cols])).long(),
                    torch.from_numpy(_wts),
                    size=(template_grid.shape[0], ipoints.shape[0]),
                    device=dev,
                    check_invariants=False,
                ).coalesce()
                geom.lin_inside = torch.from_numpy(_inside).to(dev).unsqueeze(-1)
        except Exception as _e_lin:
            geom.lin_error = str(_e_lin)

    if key is not None:
        _haar_regrid_cache[key] = geom
        while len(_haar_regrid_cache) > HAAR_REGRID_CACHE_SIZE:
            _haar_regrid_cache.popitem(last=False)
    return geom


def clear_haar_regrid_cache():
    """Drop all cached templates and regrid geometry (e.g. after changing a template file)."""
    _haar_templates.clear()
    _haar_template_dx_cache.clear()
    _haar_regrid_cache.clear()


def _reshape_regrid_to_grid(
    target,
    pred_mean,
    target_coords_raw,
    template_grid,
    ny,
    nx,
    num_channels,
    dev,
    regrid_method="nearest",
    stream_name="",
    _report_owner=None,
):
    """
    Regrid a scattered (points, channels) target/pred pair onto the fixed 2D
    template grid, returning flat (ny*nx, C) tensors plus a per-cell validity mask.

    Used by global_haar_wavelet_reshape_varweighted (domain-crop fix and the
    nearest/linear regridding option).

    The geometry (nearest indices, validity mask, linear weights) is cached per
    template and target point set, see _haar_regrid_geometry.

    Returns
    -------
    t_flat, p_flat : (ny*nx, C) float tensors gathered onto the template grid.
    valid_cell     : (ny*nx,) bool; False for template cells with no nearby data
                     point (out-of-domain under a regional crop). On those cells
                     p_flat is set equal to t_flat.detach() so every Haar
                     coefficient of (target - pred) vanishes and the cell is
                     loss-neutral.
    empty          : bool; True if NO cell is valid (template/domain mismatch),
                     in which case the caller should return a zero loss.

    regrid_method:
      "nearest" : block replication via nearest-neighbour index gather
      "linear"  : barycentric (bilinear-on-a-triangulation) resampling, applied
                  as a differentiable sparse weight matrix to both target and
                  pred. Cells outside the source convex hull fall back to nearest.
    """
    owner = _report_owner  # object to hang one-shot warning flags on (a function)

    ilat = target_coords_raw[:, 0].cpu().numpy()
    ilon = target_coords_raw[:, 1].cpu().numpy()
    ipoints = np.concatenate([ilat[:, None], ilon[:, None]], axis=1)

    if regrid_method not in ("nearest", "linear"):
        _warn_once(
            owner,
            f"_regrid_bad_method_{stream_name}",
            f"[reshape_regrid] stream={stream_name} unknown "
            f"regrid_method={regrid_method!r}; using nearest.",
        )

    geom = _haar_regrid_geometry(ipoints, template_grid, regrid_method == "linear", dev)
    valid_cell = geom.valid_cell

    # nearest gather
    t_flat = target.float()[geom.isort_t]
    p_flat = pred_mean.float()[geom.isort_t]

    _valid_c = valid_cell.unsqueeze(-1)
    p_flat = torch.where(_valid_c, p_flat, t_flat.detach())

    # --- optional linear (barycentric) regridding ---
    if geom.lin_w is not None:
        _t_lin = torch.sparse.mm(geom.lin_w, target.float())
        _p_lin = torch.sparse.mm(geom.lin_w, pred_mean.float())
        t_flat = torch.where(geom.lin_inside, _t_lin, t_flat)
        p_flat = torch.where(geom.lin_inside, _p_lin, p_flat)
        p_flat = torch.where(_valid_c, p_flat, t_flat.detach())
    elif geom.lin_error is not None:
        _warn_once(
            owner,
            f"_regrid_linear_failed_{stream_name}",
            f"[reshape_regrid] stream={stream_name} linear regrid "
            f"failed ({geom.lin_error}); falling back to nearest.",
        )

    empty = not bool(valid_cell.any())
    return t_flat, p_flat, valid_cell, empty


def global_haar_wavelet_reshape_varweighted(
    target: torch.Tensor,
    pred: torch.Tensor,
    target_coords_raw: torch.Tensor,
    weights_channels: torch.Tensor | None,
    weights_points: torch.Tensor | None,
    template_path: str = "",
    detail_weight: float = 2.0,
    num_levels: int = 3,
    var_weight_epsilon: float = 1e-3,
    stream_name: str = "",
    regrid_method: str = "nearest",
    level_start: int = 0,
):
    """
    Global 2D Haar wavelet MSE with inverse-variance spatial weighting.

    The scattered target/prediction points are regridded onto a fixed 2D template grid,
    decomposed with a multi-level 2D Haar transform, and at each Haar level the detail
    subband MSE is weighted by the inverse of the local target variance at that spatial
    scale. Regions where the target is smooth (low local variance, e.g. sea for 2m
    temperature) receive higher weight, penalising prediction errors there more strongly.
    Regions where the target is rough (high local variance, e.g. complex terrain) receive
    lower weight.

    The local variance at level lvl and position (i,j) is estimated directly
    from the Haar detail coefficients of the target at that level:
        local_var[i,j] = t_LH[i,j]**2 + t_HL[i,j]**2 + t_HH[i,j]**2
    This is exact for orthonormal Haar and requires no extra computation.
    The weight is then:
        weight[i,j] = 1.0 / (local_var[i,j] + var_weight_epsilon)
    Weights are normalised by their sum so the total contribution of each
    level is scale-invariant.

    Args:
        target              : (num_points, num_channels)
        pred                : (ens_size, num_points, num_channels)
        target_coords_raw   : (num_points, 2) -> geographic (lat, lon) degrees
        weights_channels    : (num_channels,) or None
        weights_points      : unused
        template_path       : path to NetCDF (or .npz) template with 2D lat/lon
        detail_weight       : relative weight of LH/HL/HH vs each other (currently unused)
        num_levels          : number of Haar decomposition levels
        var_weight_epsilon  : floor for local variance to avoid division by
                              zero and to control the maximum upweighting of
                              smooth regions. Larger = less aggressive
                              weighting. Default 1e-3 (normalised units).
        stream_name         : used for log messages
        regrid_method       : "nearest" or "linear" (see _reshape_regrid_to_grid)
        level_start         : first Haar level that contributes to the loss

    Returns:
        loss     : scalar
        loss_chs : (num_channels,) per-channel loss
    """
    _self = global_haar_wavelet_reshape_varweighted

    num_points, num_channels = target.shape
    dev = target.device
    pred_mean = pred.mean(0)

    # --- load and cache template output grid ---
    template_grid, ny_nx = _load_haar_template(
        _self, "vw", template_path, stream_name, num_channels, dev
    )
    if template_grid is None:
        return ny_nx  # zero loss

    ny, nx = ny_nx

    # --- regrid scattered points onto template grid (shared helper) ---
    t_flat, p_flat, valid_cell, _empty = _reshape_regrid_to_grid(
        target,
        pred_mean,
        target_coords_raw,
        template_grid,
        ny,
        nx,
        num_channels,
        dev,
        regrid_method=regrid_method,
        stream_name=stream_name,
        _report_owner=_self,
    )
    if _empty:
        _warn_once(
            _self,
            f"_vw_empty_domain_reported_{stream_name}",
            f"[global_haar_varweighted] stream={stream_name} no template cell "
            f"near any data point; check template_path vs domain. Loss=0.",
        )
        return (
            torch.tensor(0.0, device=dev, requires_grad=True),
            torch.zeros(num_channels, device=dev),
        )

    t_grid_raw = t_flat.view(ny, nx, num_channels)
    p_grid_raw = p_flat.view(ny, nx, num_channels)

    # --- pad to next multiple of 2^num_levels ---
    factor = 2**num_levels
    ny_p = int(math.ceil(ny / factor)) * factor
    nx_p = int(math.ceil(nx / factor)) * factor

    valid_grid_raw = valid_cell.view(ny, nx)  # (ny, nx) bool
    if ny_p > ny or nx_p > nx:
        t_grid = torch.zeros(ny_p, nx_p, num_channels, device=dev)
        p_grid = torch.zeros(ny_p, nx_p, num_channels, device=dev)
        t_grid[:ny, :nx, :] = t_grid_raw
        p_grid[:ny, :nx, :] = p_grid_raw
        # padded region is invalid too (target==pred==0 there, so it is already
        # loss-neutral, but exclude it from the field_var normaliser)
        valid_grid = torch.zeros(ny_p, nx_p, dtype=torch.bool, device=dev)
        valid_grid[:ny, :nx] = valid_grid_raw
    else:
        t_grid = t_grid_raw
        p_grid = p_grid_raw
        valid_grid = valid_grid_raw

    # --- multi-level Haar per channel ---
    loss_chs = torch.zeros(num_channels, device=dev)

    for c in range(num_channels):
        t_field = t_grid[:, :, c]
        p_field = p_grid[:, :, c]
        level_loss = torch.tensor(0.0, device=dev)

        for lvl in range(num_levels):
            t_LL, t_LH, t_HL, t_HH = haar_2d(t_field)
            p_LL, p_LH, p_HL, p_HH = haar_2d(p_field)

            # local variance of target at this level estimated from its own
            # detail coefficients — exact for orthonormal Haar, free to compute
            # shape: (ny/2^(lvl+1), nx/2^(lvl+1))
            local_var = (
                t_LH**2 + t_HL**2 + t_HH**2
            ).detach()  # detach: weight is a constant, not part of gradient

            # inverse variance weight - smooth regions (low var) get high weight
            inv_var_weight = 1.0 / (local_var + var_weight_epsilon)

            # all cells are active
            _active_mask = torch.ones_like(t_LH, dtype=torch.bool)
            _n_active = _active_mask.sum().clamp(min=1)

            if _n_active > 0:
                w = inv_var_weight[_active_mask]
                w_sum = w.sum().clamp(min=1e-8)

                lh_loss = (((t_LH - p_LH) ** 2)[_active_mask] * w).sum() / w_sum
                hl_loss = (((t_HL - p_HL) ** 2)[_active_mask] * w).sum() / w_sum
                hh_loss = (((t_HH - p_HH) ** 2)[_active_mask] * w).sum() / w_sum

                # only count levels >= level_start (fine fabricated levels are
                # still traversed via the cascade, just not penalised). Default
                # level_start=0 counts every level, unchanged.
                if lvl >= level_start:
                    level_loss = level_loss + (lh_loss + hl_loss + hh_loss) / 3.0

            t_field = t_LL
            p_field = p_LL

        # normalise by the target field variance over IN-DOMAIN cells only.
        # Including the masked out-of-domain cells (which hold smeared edge
        # values) would bias this normaliser; restrict to valid_grid. Falls back
        # to the full-grid variance if for some reason nothing is valid.
        _tvals = t_grid[:, :, c]
        if bool(valid_grid.any()):
            field_var = _tvals[valid_grid].var().clamp(min=1e-8)
        else:
            field_var = _tvals.var().clamp(min=1e-8)
        _n_active_lvl = max(num_levels - level_start, 1)
        loss_chs[c] = level_loss / (_n_active_lvl * field_var)

    if weights_channels is not None:
        loss = torch.mean(loss_chs * weights_channels.to(dev))
    else:
        loss = torch.mean(loss_chs)

    return loss, loss_chs
