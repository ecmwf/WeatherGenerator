# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Spatio-temporal JEPA masking: masks shared across streams, per-sample target time offsets and
the per-sample JEPA loss region (temporal="auto")."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from weathergen.datasets.masking import Masker
from weathergen.train.loss_modules.loss_module_ssl import jepa_loss

HL = 3  # 768 cells, fast
NUM_CELLS = 12 * 4**HL


def _stream(name):
    return {
        "name": name,
        "train_source_channels": ["a"],
        "train_target_channels": ["a"],
    }


def _mode_cfg(source_shared=True, time_offset=None, time_offset_probs=None):
    target = {"masking_strategy": "forecast", "num_samples": 1, "masking_strategy_config": {}}
    if time_offset is not None:
        target["time_offset"] = time_offset
    if time_offset_probs is not None:
        target["time_offset_probs"] = time_offset_probs
    return OmegaConf.create(
        {
            "losses": {
                "student-teacher": {
                    "loss_fcts": {"JEPA": {"target_source_correspondence": {0: {0: "independent"}}}}
                }
            },
            "model_input": {
                "masked": {
                    "masking_strategy": "healpix",
                    "num_samples": 1,
                    "masking_strategy_config": {
                        "rate": 0.5,
                        "hl_mask": 2,
                        "shared_across_streams": source_shared,
                    },
                }
            },
            "target_input": {"target": target},
        }
    )


def _masker(mode_cfg, streams=("ERA5", "OBS"), seed=0):
    streams_cfg = OmegaConf.create({s: _stream(s) for s in streams})
    masker = Masker(HL, "train", streams_cfg, mode_cfg)
    masker.reset_rng(np.random.default_rng(seed))
    return masker


def _build_item(masker, streams=("ERA5", "OBS")):
    masker.start_item()
    return {
        s: masker.build_samples_for_stream(["student_teacher"], NUM_CELLS, _stream(s))
        for s in streams
    }


@pytest.mark.parametrize("shared", [True, False])
def test_source_masks_shared_across_streams(shared):
    masker = _masker(_mode_cfg(source_shared=shared))
    same = []
    for _ in range(5):
        item = _build_item(masker)
        m_era5, m_obs = item["ERA5"][1].masks[0], item["OBS"][1].masks[0]
        assert 0 < int(m_era5.sum()) < NUM_CELLS
        same.append(torch.equal(m_era5, m_obs))
    assert all(same) if shared else not any(same)


def test_shared_masks_differ_between_items():
    masker = _masker(_mode_cfg(source_shared=True))
    m0 = _build_item(masker)["ERA5"][1].masks[0]
    m1 = _build_item(masker)["ERA5"][1].masks[0]
    assert not torch.equal(m0, m1)


def test_time_offset_int_and_unset():
    item = _build_item(_masker(_mode_cfg(time_offset=1)))
    assert item["ERA5"][0].metadata[0].params["time_offset"] == 1
    item = _build_item(_masker(_mode_cfg()))
    assert "time_offset" not in item["ERA5"][0].metadata[0].params


def test_time_offset_choices_shared_across_streams_and_distributed():
    masker = _masker(_mode_cfg(time_offset=[0, 1], time_offset_probs=[0.25, 0.75]))
    draws = []
    for _ in range(400):
        item = _build_item(masker)
        offsets = {s: item[s][0].metadata[0].params["time_offset"] for s in item}
        assert len(set(offsets.values())) == 1, "time offset differs between streams"
        draws.append(offsets["ERA5"])
    assert set(draws) == {0, 1}
    assert 0.65 < np.mean(draws) < 0.85


def test_time_offset_probs_validated():
    masker = _masker(_mode_cfg(time_offset=[0, 1], time_offset_probs=[0.5, 0.6]))
    with pytest.raises(AssertionError):
        _build_item(masker)


def _loss_inputs(num_samples=4, cells=16, dim=8):
    torch.manual_seed(0)
    student = torch.randn(num_samples, cells, dim)
    teacher = torch.randn(num_samples, cells, dim)
    student_masks = torch.rand(num_samples, 1, cells) < 0.5
    teacher_masks = torch.ones(num_samples, 1, cells, dtype=torch.bool)
    return student, student_masks, teacher, teacher_masks


def _manual_loss(student, teacher, mask):
    per_token = (student - teacher).abs().mean(-1)
    return torch.stack([per_token[i][mask[i]].mean() for i in range(mask.shape[0])]).mean()


def test_jepa_loss_auto_matches_manual_regions():
    student, s_masks, teacher, t_masks = _loss_inputs()
    offsets = torch.tensor([0, 1, 0, 1])
    loss = jepa_loss(
        student, s_masks, teacher, t_masks, teacher_time_offsets=offsets, temporal="auto"
    )
    spatial = t_masks.squeeze(1) & ~s_masks.squeeze(1)
    region = torch.where((offsets > 0).reshape(-1, 1), t_masks.squeeze(1), spatial)
    torch.testing.assert_close(loss, _manual_loss(student, teacher, region))


@pytest.mark.parametrize("temporal", [True, False])
def test_jepa_loss_auto_reduces_to_fixed_mode(temporal):
    student, s_masks, teacher, t_masks = _loss_inputs()
    offsets = torch.full((4,), 1 if temporal else 0)
    auto = jepa_loss(
        student, s_masks, teacher, t_masks, teacher_time_offsets=offsets, temporal="auto"
    )
    fixed = jepa_loss(student, s_masks, teacher, t_masks, temporal=temporal)
    torch.testing.assert_close(auto, fixed)


def _sampler_stub(teacher_time_offset, use_target_time_offsets=True):
    from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler

    sampler = object.__new__(MultiStreamDataSampler)
    sampler.teacher_time_offset = teacher_time_offset
    sampler.use_target_time_offsets = use_target_time_offsets
    return sampler


@pytest.mark.parametrize(
    "global_offset, target_cfg, use, expected",
    [
        (1, {}, True, [1]),  # global teacher_time_offset (temporal JEPA as before)
        (1, {"time_offset": 0}, True, [0]),  # per-strategy override
        (0, {"time_offset": [2]}, True, [2]),
        (1, {"time_offset": 1}, False, [0]),  # not student_teacher: always 0
    ],
)
def test_sampler_target_time_offsets(global_offset, target_cfg, use, expected):
    cfg = _mode_cfg()
    cfg.target_input.target.update(target_cfg)
    item = _build_item(_masker(cfg))
    sampler = _sampler_stub(global_offset, use)
    assert sampler._get_target_time_offsets(item) == expected
