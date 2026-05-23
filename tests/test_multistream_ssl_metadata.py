"""Tests for SSL metadata merging across multiple streams."""

import torch

from weathergen.datasets.batch import ModelBatch, SampleMetaData
from weathergen.datasets.stream_data import StreamData
from weathergen.train.utils import extract_batch_metadata, merge_sample_metadata


def _stream_data(source_lens: list[int]) -> StreamData:
    stream_data = StreamData(idx=0, input_steps=1, output_steps=1, healpix_cells=len(source_lens))
    stream_data.source_tokens_lens[0] = torch.tensor(source_lens, dtype=torch.int32)
    return stream_data


def _metadata(mask: list[bool], losses=None, relationship="independent") -> SampleMetaData:
    return SampleMetaData(
        params={},
        mask=torch.tensor(mask, dtype=torch.bool),
        global_params={
            "idx": 0,
            "correspondence": [0],
            "loss": ["JEPA"] if losses is None else losses,
            "relationship": relationship,
        },
    )


def test_extract_batch_metadata_merges_sparse_stream_masks():
    streams = [{"name": "empty_first"}, {"name": "active_second"}]
    batch = ModelBatch(
        streams,
        num_source_samples=1,
        num_target_samples=1,
        output_offset=0,
        output_steps=1,
    )

    batch.add_source_stream(
        0,
        0,
        "empty_first",
        _stream_data([0, 0, 0, 0]),
        _metadata([False, False, False, False]),
    )
    batch.add_source_stream(
        0,
        0,
        "active_second",
        _stream_data([0, 1, 0, 1]),
        _metadata([True, True, True, False]),
    )
    batch.add_target_stream(
        0,
        [0],
        "empty_first",
        _stream_data([0, 0, 0, 0]),
        _metadata([False, False, False, False]),
    )
    batch.add_target_stream(
        0,
        [0],
        "active_second",
        _stream_data([0, 0, 1, 1]),
        _metadata([False, True, True, True]),
    )

    _, source_metadata, _, target_metadata = extract_batch_metadata(batch)

    assert torch.equal(
        source_metadata[0].mask,
        torch.tensor([False, True, False, False]),
    )
    assert torch.equal(
        target_metadata[0].mask,
        torch.tensor([False, False, True, True]),
    )


def test_merge_sample_metadata_combines_losses_and_relationships():
    streams = [{"name": "identity_stream"}, {"name": "jepa_stream"}]
    batch = ModelBatch(
        streams,
        num_source_samples=1,
        num_target_samples=1,
        output_offset=0,
        output_steps=1,
    )
    batch.add_source_stream(
        0,
        0,
        "identity_stream",
        _stream_data([1, 0]),
        _metadata([True, False], losses=["DINO"], relationship="identity"),
    )
    batch.add_source_stream(
        0,
        0,
        "jepa_stream",
        _stream_data([0, 1]),
        _metadata([False, True], losses=["JEPA"], relationship="independent"),
    )

    merged_metadata = merge_sample_metadata(batch.source_samples.get_samples()[0])

    assert torch.equal(merged_metadata.mask, torch.tensor([True, True]))
    assert merged_metadata.global_params["loss"] == ["DINO", "JEPA"]
    assert merged_metadata.global_params["relationship"] == "independent"
