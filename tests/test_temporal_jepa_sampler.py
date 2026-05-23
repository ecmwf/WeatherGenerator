"""Tests for temporal JEPA sampler index bounds."""

from types import SimpleNamespace

import numpy as np

from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler


def test_teacher_time_offset_reduces_base_permutation_range():
    sampler = MultiStreamDataSampler.__new__(MultiStreamDataSampler)
    sampler.index_range = SimpleNamespace(start=0, end=16)
    sampler.output_offset = 0
    sampler.time_step = np.timedelta64(0, "s")
    sampler.step_timedelta = np.timedelta64(6, "h")
    sampler.teacher_time_offset = 1

    perms = sampler._calc_baseperms(fsm=0)

    assert perms.tolist() == list(range(15))


def test_teacher_time_offset_combines_with_forecast_horizon():
    sampler = MultiStreamDataSampler.__new__(MultiStreamDataSampler)
    sampler.index_range = SimpleNamespace(start=0, end=20)
    sampler.output_offset = 1
    sampler.time_step = np.timedelta64(6, "h")
    sampler.step_timedelta = np.timedelta64(6, "h")
    sampler.teacher_time_offset = 2

    perms = sampler._calc_baseperms(fsm=3)

    assert perms.tolist() == list(range(14))
