import pytest

import numpy as np

from weathergen.common.timing import Timer, Timing, _get_timer


@pytest.fixture
def root_timer():
    lvl_2_timer = Timer("root.foo.baz", {}, [])
    substep1 = Timer("root.foo", {lvl_2_timer.name: lvl_2_timer}, [])
    substep2 = Timer("root.bar", {}, [])

    root = Timer("root", {substep1.name: substep1, substep2.name: substep2}, [])

    lvl_2_timer._parent = substep1
    substep1._parent = root
    substep2._parent = root

    return root


@pytest.fixture
def lvl_2_timer(root_timer) -> Timer:
    return _get_timer(root_timer, "foo", "baz")


@pytest.fixture
def substep1(root_timer) -> Timer:
    return _get_timer(root_timer, "foo")


@pytest.fixture
def substep2(root_timer) -> Timer:
    return _get_timer(root_timer, "bar")


def test_get_timer_missing_create(root_timer: Timer):
    timer = _get_timer(root_timer, "foo", "bar", "baz", create=True)

    assert timer.name == "root.foo.bar.baz"


def test_get_timer_missing_no_create(root_timer: Timer):
    with pytest.raises(ValueError):
        _ = _get_timer(root_timer, "foo", "bar")


def test_get_timer(root_timer: Timer, lvl_2_timer: Timer):
    labels = lvl_2_timer._get_labels()
    timer = _get_timer(root_timer, *labels)

    assert timer == lvl_2_timer


def test_get_timer_empty(root_timer: Timer):
    timer = _get_timer(root_timer)

    assert timer == root_timer


def test_record_first(lvl_2_timer: Timer):
    n = len(lvl_2_timer.records)
    lvl_2_timer.record()

    assert len(lvl_2_timer.records) == n


def test_record_following(lvl_2_timer: Timer):
    lvl_2_timer.record()
    n = len(lvl_2_timer.records)

    lvl_2_timer.record()

    assert len(lvl_2_timer.records) == n + 1


def test_record_substeps(substep1: Timer, substep2: Timer):
    substep1.record()  # first recording
    substep2.record()  # end recording of first substep, begin recording 2. substep

    assert len(substep1.records) == 1 and len(substep2.records) == 0

def test_record_substeps_via_parent(root_timer: Timer, substep1: Timer, substep2: Timer):
    root_timer.record("foo")
    root_timer.record("bar")
    
    assert len(substep1.records) == 1 and len(substep2.records) == 0


def test_get_result_name(lvl_2_timer: Timer):
    result = lvl_2_timer.get_result()

    assert lvl_2_timer.name in result.keys()


def test_get_result_emtpy(lvl_2_timer: Timer):
    result = lvl_2_timer.get_result()
    expected = Timing(np.nan, np.nan, np.nan, np.nan, 0)

    timing = result[lvl_2_timer.name]
    assert timing == expected


def test_get_result_values(lvl_2_timer: Timer):
    records = [np.timedelta64(1, "ns"), np.timedelta64(1, "ns"), np.timedelta64(1, "ns")]
    lvl_2_timer.records = records

    timings = lvl_2_timer.get_result()[lvl_2_timer.name]

    assert (
        timings.mean == 1.0
        and timings.std == 0.0
        and timings.max == 1
        and timings.min == 1
        and timings.n == 3
    )


def test_reset_records(substep1: Timer):
    substep1.record()

    _ = substep1.reset()

    assert len(substep1.records) == 0


def test_reset_substeps(root_timer: Timer):
    _ = root_timer.reset()

    assert len(root_timer.substeps) == 0


def test_reset_get_all_timings(root_timer: Timer):
    results = root_timer.reset()
    expected_names = ["root", "root.foo", "root.foo.baz"]

    assert all(name in results.keys() for name in expected_names)