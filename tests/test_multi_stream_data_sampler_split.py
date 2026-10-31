import pytest

from weathergen.datasets.multi_stream_data_sampler import samples_per_rank, worker_sample_range


@pytest.mark.parametrize(
    "samples, world_size, batch_size, expected",
    [
        (4096, 8, 1, 512),  # typical training setup, unchanged
        (80, 8, 1, 10),  # was floored to 8 per rank by num_workers=8
        (64, 4, 1, 16),
        (59, 4, 1, 14),  # was floored to 8 per rank (32 instead of 56 samples)
        (8, 4, 1, 2),  # was 0 -> empty validation and NCCL deadlock
        (10, 2, 4, 4),  # whole batches only: 10 // (2 * 4) = 1 batch per rank
        (3, 4, 1, 1),  # fewer samples than ranks: at least one batch per rank
    ],
)
def test_samples_per_rank(samples, world_size, batch_size, expected):
    assert samples_per_rank(samples, world_size, batch_size) == expected


@pytest.mark.parametrize("rank_len, batch_size", [(10, 1), (16, 1), (3, 1), (12, 4), (0, 1)])
@pytest.mark.parametrize("num_workers", [1, 4, 8])
def test_worker_sample_range_covers_rank_slice(rank_len, batch_size, num_workers):
    rank_start = 100
    ranges = [
        worker_sample_range(rank_start, rank_len, batch_size, wid, num_workers)
        for wid in range(num_workers)
    ]

    # contiguous, disjoint, and covering exactly the rank's slice
    assert ranges[0][0] == rank_start
    assert ranges[-1][1] == rank_start + rank_len
    for (_, end), (start, _) in zip(ranges[:-1], ranges[1:], strict=True):
        assert end == start

    # whole batches only, balanced to within one batch
    sizes = [end - start for start, end in ranges]
    assert all(size % batch_size == 0 for size in sizes)
    assert max(sizes) - min(sizes) <= batch_size
