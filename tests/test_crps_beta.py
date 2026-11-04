# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Spread-scaled kernel CRPS: E|X-y| - beta/2 E|X-X'| (beta = 1 is the fair CRPS)."""

from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from weathergen.train.crps_beta import append_record, beta_from_r, last_beta, mse_ratio
from weathergen.train.loss_modules.loss_functions import kernel_crps
from weathergen.train.loss_modules.loss_module_physical import LossPhysical
from weathergen.train.utils import TRAIN, VAL

STREAM = "ERA5"
CHANNELS = ["a", "b", "c"]


def _ens(ens_size=4, num_points=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    target = torch.randn(num_points, len(CHANNELS), generator=g)
    preds = target + 0.5 * torch.randn(ens_size, num_points, len(CHANNELS), generator=g)
    return target, preds


def test_beta_one_is_default():
    target, preds = _ens()
    loss, chs = kernel_crps(target, preds, None, None)
    loss_b, chs_b = kernel_crps(target, preds, None, None, beta=1.0)
    torch.testing.assert_close(loss, loss_b)
    torch.testing.assert_close(chs, chs_b)


def test_beta_scales_only_the_spread_term():
    target, preds = _ens()
    mae = (preds - target).abs().mean((0, 1))
    pairs = (preds[:, None] - preds[None, :]).abs().sum((0, 1)).mean(0)  # sum over i != j
    m = preds.shape[0]
    for beta in (0.5, 1.0, 1.1):
        _, chs = kernel_crps(target, preds, None, None, beta=beta)
        torch.testing.assert_close(chs, mae - beta * pairs / (2 * m * (m - 1)))


def test_loss_physical_passes_loss_extra_args():
    target, preds = _ens()
    num_points = target.shape[0]
    meta = {STREAM: SimpleNamespace(global_params={"idx": 0})}
    preds_ns = SimpleNamespace(physical=[{}, {STREAM: [preds]}])
    targets_ns = SimpleNamespace(
        physical=[
            {},
            {
                STREAM: {
                    "target": [target],
                    "target_coords": [torch.zeros(num_points, 2)],
                    "target_times": [np.zeros(num_points, dtype="datetime64[ns]")],
                    "target_metda_data": [meta],
                    "is_spoof": [False],
                }
            },
        ],
        output_idxs=[1],
    )
    output_info = [SimpleNamespace(global_params={"correspondence": 0, "loss": ["kernel_crps"]})]
    metadata = (None, output_info, None, None)
    cf = OmegaConf.create(
        {
            "streams": {
                STREAM: {
                    "train_target_channels": CHANNELS,
                    "val_target_channels": CHANNELS,
                    "target_channel_weights": [1.0, 1.0, 1.0],
                    "loss_weight": 1.0,
                }
            }
        }
    )
    mode_cfg = OmegaConf.create({"forecast": {}})
    losses = OmegaConf.create({"kernel_crps": {"loss_extra_args": {"beta": 1.1}}})
    out = LossPhysical(cf, mode_cfg, TRAIN, "cpu", **losses).compute_loss(
        preds_ns, targets_ns, metadata
    )
    torch.testing.assert_close(out.loss, kernel_crps(target, preds, None, None, beta=1.1)[0])
    out1 = LossPhysical(cf, mode_cfg, TRAIN, "cpu", kernel_crps={}).compute_loss(
        preds_ns, targets_ns, metadata
    )
    assert out.loss < out1.loss
    # validation reports the proper CRPS
    out_val = LossPhysical(cf, mode_cfg, VAL, "cpu", **losses).compute_loss(
        preds_ns, targets_ns, metadata
    )
    torch.testing.assert_close(out_val.loss, kernel_crps(target, preds, None, None)[0])


# --- beta set from R = MSE_train / MSE_val


def test_beta_from_r():
    # note v7, table of the calibrating beta
    for r, beta in ((1.0, 1.0), (0.81, 1.051), (0.64, 1.104), (0.49, 1.159)):
        assert abs(beta_from_r(r, beta_max=2.0) - beta) < 1e-3
    # kappa(beta)^2 R = 1
    beta = beta_from_r(0.7, beta_max=2.0)
    assert abs(beta**2 / (2 - beta**2) * 0.7 - 1.0) < 1e-12
    assert beta_from_r(0.1) == 1.2
    assert beta_from_r(1.3) == 1.0


def test_mse_ratio():
    val = {
        "LossPhysical.ERA5.1.mse.a": [2.0, 2.0],
        "LossPhysical.ERA5.1.mse.b": [4.0],
        "LossPhysical.ERA5.1.kernel_crps.a": [1.0],
        "LossPhysical.ERA5.2.mse.a": [float("nan")],
        "LossPhysical.loss_avg": [1.0],
    }
    train = {
        "LossPhysical.ERA5.1.mse.a": [1.0, float("nan")],
        "LossPhysical.ERA5.1.mse.b": [3.0, 5.0],
        "LossPhysical.ERA5.1.kernel_crps.a": [0.1],
        "LossPhysical.ERA5.2.mse.a": [1.0],
    }
    r, n = mse_ratio(train, val)
    assert n == 2
    assert abs(r - (0.5 + 1.0) / 2) < 1e-12
    assert mse_ratio({}, val)[1] == 0


def test_set_loss_extra_args():
    cf = OmegaConf.create({"streams": {}})
    mod = LossPhysical(cf, OmegaConf.create({"forecast": {}}), TRAIN, "cpu", kernel_crps={})
    assert mod.set_loss_extra_args("kernel_crps", beta=1.1)
    assert not mod.set_loss_extra_args("mse", beta=1.1)
    target, preds = _ens()
    torch.testing.assert_close(
        mod.loss_fcts[0][0](target, preds, None, None)[0],
        kernel_crps(target, preds, None, None, beta=1.1)[0],
    )


def test_last_beta(tmp_path):
    path = tmp_path / "crps_beta.jsonl"
    assert last_beta(path) is None
    append_record(path, {"mini_epoch": -1, "R": 0.9, "beta": 1.03})
    append_record(path, {"mini_epoch": 0, "R": 0.8, "beta": 1.05})
    assert last_beta(path) == 1.05
