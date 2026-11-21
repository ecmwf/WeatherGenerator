# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the configurable layout of written zarr arrays (common.io.ZarrWriteConfig)."""

import numpy as np
import pytest
import zarr
from omegaconf import OmegaConf

from weathergen.common.config import _DEFAULT_CONFIG_PTH
from weathergen.common.io import ZarrIO, ZarrWriteConfig


def test_default_config_matches_code_defaults():
    """The documented default_config.yml section keeps the previous layout."""
    cf = OmegaConf.load(_DEFAULT_CONFIG_PTH)
    assert ZarrWriteConfig.from_config(cf) == ZarrWriteConfig()
    assert ZarrWriteConfig.from_config(OmegaConf.create({})) == ZarrWriteConfig()


def test_from_config_overrides_only_given_keys():
    cf = OmegaConf.create({"zarr_write": {"chunk_nsamples": 100, "shard_nsamples": 300}})
    layout = ZarrWriteConfig.from_config(cf)

    assert (layout.chunk_nsamples, layout.shard_nsamples) == (100, 300)
    assert layout.scale_factor == ZarrWriteConfig().scale_factor


def test_shards_must_be_a_multiple_of_chunks():
    with pytest.raises(AssertionError, match="multiple"):
        ZarrWriteConfig(chunk_nsamples=100, shard_nsamples=250)


@pytest.mark.parametrize("sharding", [True, False])
def test_written_arrays_follow_the_layout(tmp_path, sharding):
    layout = ZarrWriteConfig(
        sharding=sharding, chunk_nsamples=100, shard_nsamples=300, scale_factor=2
    )
    with ZarrIO(tmp_path / "out.zarr", read_only=False, write_config=layout) as zio:
        assert zio.data_root is not None
        group = zio.data_root.create_group("item")
        zio._create_dataset(group, "data", np.zeros((1000, 8), dtype=np.float32))
        array = group["data"]

        # chunks: chunk_nsamples x (size // scale_factor); shards: (scale_factor + 1) chunks
        assert array.chunks == (100, 4)
        assert array.shards == ((300, 12) if sharding else None)


def test_async_concurrency_is_set_only_while_writing():
    before = zarr.config.get("async.concurrency")

    with ZarrWriteConfig(async_concurrency=before + 7).zarr_options():
        assert zarr.config.get("async.concurrency") == before + 7

    assert zarr.config.get("async.concurrency") == before
    with ZarrWriteConfig().zarr_options():
        assert zarr.config.get("async.concurrency") == before
