import numpy as np

from weathergen.readers_extra.data_reader_anemoi_operan import latest_available_analysis_index


def test_materialized_analysis_selects_latest_snapshot_record() -> None:
    valid_times = np.array(["2026-08-05T00:00", "2026-08-05T06:00"], dtype="datetime64[h]")

    selected, availability_times = latest_available_analysis_index(
        valid_times,
        np.datetime64("2026-08-05T11:00"),
        "dataset",
        available_until=np.datetime64("2026-08-05T11:00"),
    )

    assert selected == 1
    assert np.array_equal(availability_times, valid_times)


def test_materialized_analysis_respects_snapshot_boundary() -> None:
    valid_times = np.array(
        ["2026-08-05T00:00", "2026-08-05T06:00", "2026-08-05T12:00"], dtype="datetime64[h]"
    )

    selected, _ = latest_available_analysis_index(
        valid_times,
        np.datetime64("2026-08-05T17:00"),
        "dataset",
        available_until=np.datetime64("2026-08-05T11:00"),
    )

    assert selected == 1


def test_nominal_mapping_preserves_strict_window_boundary() -> None:
    valid_times = np.array(["2026-08-05T00:00", "2026-08-05T06:00"], dtype="datetime64[h]")

    selected, availability_times = latest_available_analysis_index(
        valid_times,
        np.datetime64("2026-08-05T11:00"),
        "nominal_time_mapping",
        {"0": 5, "6": 11},
    )

    assert selected == 0
    assert np.array_equal(
        availability_times,
        np.array(["2026-08-05T05:00", "2026-08-05T11:00"], dtype="datetime64[h]"),
    )
