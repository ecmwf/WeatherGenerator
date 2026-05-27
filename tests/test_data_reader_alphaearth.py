from pathlib import Path

import numpy as np
import zarr

from weathergen.datasets.data_reader_alphaearth import DataReaderAlphaEarthGeoinfo


def _create_alphaearth_zarr(path: Path) -> None:
    root = zarr.open(path, mode="w")
    dates = np.array(["2020-01-01", "2021-01-01"], dtype="datetime64[ns]")
    metadata = np.array(
        [
            (10.0, 20.0, 0.0, 0.0, 1.0, 1.0, 4326),
            (-30.0, 170.0, 0.0, 0.0, 1.0, 1.0, 4326),
        ],
        dtype=[
            ("lat", "f4"),
            ("lon", "f4"),
            ("bbox_west", "f4"),
            ("bbox_south", "f4"),
            ("bbox_east", "f4"),
            ("bbox_north", "f4"),
            ("crs_code", "i4"),
        ],
    )
    data = np.zeros((2, 2, 3, 3, 3), dtype=np.int8)
    data[0, 0, :, 1, 1] = [1, 2, 3]
    data[0, 1, :, 1, 1] = [4, 5, 6]
    data[1, 1, :, 1, 1] = [7, 8, 9]

    root.create_array("dates", data=dates)
    root.create_array("metadata", data=metadata)
    root.create_array("station_id", data=np.array(["station_0", "station_1"]))
    root.create_array("data", data=data)


def test_alphaearth_geoinfo_reader_matches_station_and_date(tmp_path: Path) -> None:
    alphaearth_path = tmp_path / "alphaearth.zarr"
    _create_alphaearth_zarr(alphaearth_path)

    reader = DataReaderAlphaEarthGeoinfo(
        alphaearth_path,
        {
            "patch_mode": "center",
            "max_distance_deg": 0.1,
            "stats_sample_size": 0,
        },
    )

    coords = np.array([[10.02, 20.01], [10.02, 20.01], [0.0, 0.0]], dtype=np.float32)
    datetimes = np.array(
        ["2020-02-01", "2021-02-01", "2021-02-01"], dtype="datetime64[ns]"
    )

    features = reader.get(coords, datetimes)

    np.testing.assert_array_equal(features[0], np.array([1, 2, 3], dtype=np.float32))
    np.testing.assert_array_equal(features[1], np.array([4, 5, 6], dtype=np.float32))
    np.testing.assert_array_equal(features[2], np.zeros(3, dtype=np.float32))
