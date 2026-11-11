# (C) Copyright 2026 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from weathergen.common.io import ItemKey, OutputItem, ZarrIO


def test_write_zarr_rejects_existing_output_item(tmp_path):
    key = ItemKey(sample=0, forecast_step=0, stream="ERA5")
    item = OutputItem(key=key, forecast_offset=0)  # type: ignore[arg-type]

    with ZarrIO(tmp_path / "output.zarr", read_only=False) as writer:
        writer.write_zarr(item)

        with pytest.raises(ValueError, match=r"0/ERA5/0.*refusing to overwrite"):
            writer.write_zarr(item)
