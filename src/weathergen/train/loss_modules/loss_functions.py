# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import torch

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


def mse(target, ens, mu, *kwargs):
    return torch.nn.functional.mse_loss(target, mu)


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


def mse_channel_location_weighted(
    target: torch.Tensor,
    pred: torch.Tensor,
    weights_channels: torch.Tensor | None,
    weights_points: torch.Tensor | None,
):
    """
    Compute weighted MSE loss for one window or step

    The function implements:

    loss = Mean_{channels}( weight_channels * Mean_{data_pts}( (target - pred) * weights_points ))

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
    1. weight the rows of (target - pred) by wp = weights_points
    2. take the mean over the row
    3. weight the collapsed cols by wc = weights_channels
    4. take the mean over the channel-weighted cols

    Params:
        target : shape ( num_data_points , num_channels )
        target : shape ( ens_dim , num_data_points , num_channels)
        weights_channels : shape = (num_channels,)
        weights_points : shape = (num_data_points)

    Return:
        loss : weight loss for gradient computation
        loss_chs : losses per channel with location weighting but no channel weighting
    """

    mask_nan = ~torch.isnan(target)
    pred = pred[0] if pred.shape[0] == 0 else pred.mean(0)

    diff2 = torch.square(torch.where(mask_nan, target, 0) - torch.where(mask_nan, pred, 0))
    if weights_points is not None:
        diff2 = (diff2.transpose(1, 0) * weights_points).transpose(1, 0)
    loss_chs = diff2.mean(0)
    loss = torch.mean(loss_chs * weights_channels if weights_channels is not None else loss_chs)

    return loss, loss_chs


def mae(
    target: torch.Tensor,
    pred: torch.Tensor,
    weights_channels: torch.Tensor | None,
    weights_points: torch.Tensor | None,
):
    """
    Compute weighted MAE loss for one window or step

    The function implements:

    loss = Mean_{channels}( weight_channels * Mean_{data_pts}( (target - pred) * weights_points ))

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
    1. weight the rows of (target - pred) by wp = weights_points
    2. take the mean over the row
    3. weight the collapsed cols by wc = weights_channels
    4. take the mean over the channel-weighted cols

    Params:
        target : shape ( num_data_points , num_channels )
        target : shape ( ens_dim , num_data_points , num_channels)
        weights_channels : shape = (num_channels,)
        weights_points : shape = (num_data_points)

    Return:
        loss : weight loss for gradient computation
        loss_chs : losses per channel with location weighting but no channel weighting
    """

    mask_nan = ~torch.isnan(target)
    pred = pred[0] if pred.shape[0] == 0 else pred.mean(0)

    diff2 = torch.abs(torch.where(mask_nan, target, 0) - torch.where(mask_nan, pred, 0))
    if weights_points is not None:
        diff2 = (diff2.transpose(1, 0) * weights_points).transpose(1, 0)
    loss_chs = diff2.mean(0)
    loss = torch.mean(loss_chs * weights_channels if weights_channels is not None else loss_chs)

    return loss, loss_chs


def cosine_latitude(stream_data, forecast_offset, fstep, min_value=1e-3, max_value=1.0):
    latitudes_radian = stream_data.target_coords_raw[forecast_offset + fstep][:, 0] * np.pi / 180
    return (max_value - min_value) * np.cos(latitudes_radian) + min_value


def gamma_decay(forecast_steps, gamma):
    fsteps = np.arange(forecast_steps)
    weights = gamma**fsteps
    return weights * (len(fsteps) / np.sum(weights))
