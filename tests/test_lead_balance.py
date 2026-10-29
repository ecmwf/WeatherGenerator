# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for the lead-time balancing of the physical training loss."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from weathergen.train.loss_modules.lead_balance import LeadBalance, get_lead_balance
from weathergen.train.loss_modules.loss_functions import kernel_crps
from weathergen.train.loss_modules.loss_module_physical import LossPhysical
from weathergen.train.utils import TRAIN, VAL

REPO = Path(__file__).parent.parent
STREAM = "ERA5"
CHANNELS = ["z_500", "t_850", "q_700"]
# growth of the loss relative to the first step, per channel, for steps 1-3
GROWTH = {"z_500": [1.0, 4.0, 16.0], "t_850": [1.0, 2.0, 4.0], "q_700": [1.0, 1.5, 2.0]}


def test_weights():
    lb = LeadBalance({STREAM: GROWTH}, power=1.0, first_step=1)
    assert torch.equal(lb.weights(STREAM, CHANNELS, 1, "cpu"), torch.ones(3))
    torch.testing.assert_close(
        lb.weights(STREAM, CHANNELS, 3, "cpu"), torch.tensor([1 / 16, 1 / 4, 1 / 2])
    )
    # channels in the order requested, extra table channels ignored
    torch.testing.assert_close(
        lb.weights(STREAM, ["q_700", "z_500"], 2, "cpu"), torch.tensor([1 / 1.5, 1 / 4])
    )


def test_weights_power_and_first_step():
    lb = LeadBalance({STREAM: GROWTH}, power=0.5, first_step=2)
    torch.testing.assert_close(
        lb.weights(STREAM, CHANNELS, 4, "cpu"), torch.tensor([1 / 4, 1 / 2, 2**-0.5])
    )
    lb0 = LeadBalance({STREAM: GROWTH}, power=0.0)
    assert torch.equal(lb0.weights(STREAM, CHANNELS, 3, "cpu"), torch.ones(3))


@pytest.mark.parametrize(
    "growth, power, first_step",
    [
        ({}, 1.0, 1),
        ({STREAM: {"z_500": [1.0, 0.0]}}, 1.0, 1),
        ({STREAM: {"z_500": [1.0, float("nan")]}}, 1.0, 1),
        ({STREAM: {"z_500": [1.0, 2.0]}}, -1.0, 1),
        ({STREAM: {"z_500": [1.0, 2.0]}}, 1.0, -1),
    ],
)
def test_invalid_table(growth, power, first_step):
    with pytest.raises(AssertionError):
        LeadBalance(growth, power=power, first_step=first_step)


@pytest.mark.parametrize(
    "stream, channels, step",
    [
        ("other", CHANNELS, 1),  # stream not in the table
        (STREAM, [*CHANNELS, "2t"], 1),  # channel not in the table
        (STREAM, CHANNELS, 0),  # before the first step
        (STREAM, CHANNELS, 4),  # after the last step
    ],
)
def test_missing_entries(stream, channels, step):
    lb = LeadBalance({STREAM: GROWTH})
    with pytest.raises(AssertionError):
        lb.weights(stream, channels, step, "cpu")


def test_get_lead_balance_from_config():
    assert get_lead_balance(OmegaConf.create({})) is None
    assert get_lead_balance(None) is None
    cfg = OmegaConf.create({"lead_balance": {"power": 1, "growth": {STREAM: GROWTH}}})
    lb = get_lead_balance(cfg)
    torch.testing.assert_close(
        lb.weights(STREAM, CHANNELS, 2, "cpu"), torch.tensor([1 / 4, 1 / 2, 1 / 1.5])
    )


# --- the physical loss with lead balancing


def _loss_inputs(num_steps=3, num_points=50, seed=0):
    """Predictions and targets of one sample with a single source-target correspondence."""
    rng = np.random.default_rng(seed)
    preds = SimpleNamespace(physical=[{}])
    targets = SimpleNamespace(physical=[{}], output_idxs=list(range(1, num_steps + 1)))
    meta = {STREAM: SimpleNamespace(global_params={"idx": 0})}
    for step in range(1, num_steps + 1):
        target = torch.tensor(rng.normal(size=(num_points, len(CHANNELS))), dtype=torch.float32)
        # errors grow with the step, at a different rate per channel
        err = torch.tensor(rng.normal(size=(num_points, len(CHANNELS))), dtype=torch.float32)
        pred = target + err * torch.tensor([step**2, step, 1.0]) * 0.1
        preds.physical.append({STREAM: [pred.unsqueeze(0)]})
        targets.physical.append(
            {
                STREAM: {
                    "target": [target],
                    "target_coords": [torch.zeros(num_points, 2)],
                    "target_times": [np.zeros(num_points, dtype="datetime64[ns]")],
                    "target_metda_data": [meta],
                    "is_spoof": [False],
                }
            }
        )
    output_info = [SimpleNamespace(global_params={"correspondence": 0, "loss": ["mse"]})]
    return preds, targets, (None, output_info, None, None)


def _loss_module(stage, lead_balance=None, channel_weights=(1.0, 1.0, 1.0)):
    cf = OmegaConf.create(
        {
            "streams": {
                STREAM: {
                    "train_target_channels": CHANNELS,
                    "val_target_channels": CHANNELS,
                    "target_channel_weights": list(channel_weights),
                    "loss_weight": 1.0,
                }
            }
        }
    )
    forecast = {} if lead_balance is None else {"lead_balance": lead_balance}
    mode_cfg = OmegaConf.create({"forecast": forecast})
    return LossPhysical(cf, mode_cfg, stage, "cpu", mse={"weight": 1.0})


def _expected_loss(preds, targets, weights_fn):
    """Mean over steps of Mean_channels(weight * mse) for the inputs of _loss_inputs."""
    losses = []
    for step in targets.output_idxs:
        pred = preds.physical[step][STREAM][0][0]
        target = targets.physical[step][STREAM]["target"][0]
        mse_chs = ((pred - target) ** 2).mean(0)
        losses.append((mse_chs * weights_fn(step)).mean())
    return torch.stack(losses).mean()


def test_loss_without_lead_balance_unchanged():
    preds, targets, metadata = _loss_inputs()
    w = torch.tensor([10.0, 5.0, 1.0])
    out = _loss_module(TRAIN, channel_weights=w.tolist()).compute_loss(preds, targets, metadata)
    torch.testing.assert_close(out.loss, _expected_loss(preds, targets, lambda s: w))


def test_loss_with_lead_balance():
    preds, targets, metadata = _loss_inputs()
    w = torch.tensor([10.0, 5.0, 1.0])
    balance = {"power": 1.0, "growth": {STREAM: GROWTH}}
    module = _loss_module(TRAIN, lead_balance=balance, channel_weights=w.tolist())
    out = module.compute_loss(preds, targets, metadata)

    growth = torch.tensor([GROWTH[ch] for ch in CHANNELS]).T  # [step, channel]
    expected = _expected_loss(preds, targets, lambda s: w / growth[s - 1])
    torch.testing.assert_close(out.loss, expected)

    # the logged per-channel losses stay unweighted
    out_ref = _loss_module(TRAIN, channel_weights=w.tolist()).compute_loss(preds, targets, metadata)
    for ch in CHANNELS:
        for step in targets.output_idxs:
            v = out.losses_all[STREAM]["mse"][ch][str(step)]
            v_ref = out_ref.losses_all[STREAM]["mse"][ch][str(step)]
            assert torch.equal(v, v_ref)


def test_loss_lead_balance_gradient():
    """Balancing scales the gradient of each step and channel by its weight."""
    preds, targets, metadata = _loss_inputs(num_steps=2)
    for p in preds.physical[1:]:
        p[STREAM][0].requires_grad_(True)
    balance = {"power": 1.0, "growth": {STREAM: GROWTH}}
    out = _loss_module(TRAIN, lead_balance=balance).compute_loss(preds, targets, metadata)
    out.loss.backward()
    out_ref = _loss_module(TRAIN).compute_loss(preds, targets, metadata)
    grads = [p[STREAM][0].grad.clone() for p in preds.physical[1:]]
    for p in preds.physical[1:]:
        p[STREAM][0].grad = None
    out_ref.loss.backward()
    for step, g in enumerate(grads, start=1):
        g_ref = preds.physical[step][STREAM][0].grad
        scale = 1.0 / torch.tensor([GROWTH[ch][step - 1] for ch in CHANNELS])
        torch.testing.assert_close(g, g_ref * scale)


def test_validation_is_not_balanced():
    preds, targets, metadata = _loss_inputs()
    balance = {"power": 1.0, "growth": {STREAM: GROWTH}}
    out = _loss_module(VAL, lead_balance=balance).compute_loss(preds, targets, metadata)
    out_ref = _loss_module(VAL).compute_loss(preds, targets, metadata)
    assert torch.equal(out.loss, out_ref.loss)


def test_kernel_crps_logs_unweighted_channels():
    rng = np.random.default_rng(0)
    target = torch.tensor(rng.normal(size=(40, 3)), dtype=torch.float32)
    preds = torch.tensor(rng.normal(size=(4, 40, 3)), dtype=torch.float32)
    w = torch.tensor([10.0, 5.0, 1.0])
    loss_w, chs_w = kernel_crps(target, preds, w, None)
    loss, chs = kernel_crps(target, preds, None, None)
    assert torch.equal(chs_w, chs)
    torch.testing.assert_close(loss_w, (chs * w).mean())
    torch.testing.assert_close(loss, chs.mean())


# --- the growth table script


def test_compute_lead_balance_script(tmp_path):
    metrics = tmp_path / "run_train_metrics.json"
    rows = []
    for i in range(30):
        row = {"stage": "train"}
        for ch, g in GROWTH.items():
            for step, v in enumerate(g, start=1):
                # warm-up entries (skipped) are far off; later entries scatter around the growth
                row[f"LossPhysical.{STREAM}.mse.{ch}.{step}"] = (
                    100.0 if i < 10 else 0.01 * v * (1.0 + 0.1 * (-1) ** i)
                )
        rows.append(row)
    rows.append({"stage": "val", f"LossPhysical.{STREAM}.mse.z_500.1": 1.0})
    metrics.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "overlay.yml"
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "compute_lead_balance.py"), str(metrics), str(out)],
        check=True,
        capture_output=True,
    )
    cfg = OmegaConf.load(out)
    lb_cfg = cfg.training_config.forecast.lead_balance
    assert lb_cfg.first_step == 1 and lb_cfg.power == 1.0
    for ch, g in GROWTH.items():
        np.testing.assert_allclose(list(lb_cfg.growth[STREAM][ch]), g, rtol=1e-3)
    assert get_lead_balance(cfg.training_config.forecast) is not None


def test_committed_table_covers_era5_targets():
    """The table made from B' (xxlm80yw) covers the ERA5 targets of the long-rollout runs."""
    cfg = OmegaConf.load(REPO / "config" / "config_lead_balance_xxlm80yw.yml")
    lb = get_lead_balance(cfg.training_config.forecast)
    streams_dir = "gjl6a0kb_reproduction_n320_new_channels_forecast_iasi_channels_mnt_cw"
    era5 = OmegaConf.load(REPO / "config" / "streams" / streams_dir / "era5.yml").ERA5
    channels = list(era5.target)
    for step in range(1, 41):
        w = lb.weights(STREAM, channels, step, "cpu")
        assert w.shape == (len(channels),) and torch.isfinite(w).all() and (w > 0).all()
    assert torch.equal(lb.weights(STREAM, channels, 1, "cpu"), torch.ones(len(channels)))
