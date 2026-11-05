# Zarr reader benchmark

Times `DataReaderAnemoi._get` per window on different Zarr encodings of the ERA5/AIFS
datasets (Zarr 3, sharding, time range) on JUPITER. One record per dataset
file in `experiments/`; each gives copy-paste commands for its prerequisites, the checkout of
the pinned commits, and the run.

- `benchmark_anemoi_reader.py` — the benchmark: times `_get` over a stream's windows and
  writes one JSONL line per window; `--nsys` re-launches it under nsys with NVTX ranges.
  Slurm wrapper: `hpc/jupiter/benchmark_anemoi_reader.sh` in WeatherGenerator-private.
- `restore_zarr_attrs.py` — restores the metadata a Zarr 2 → 3 conversion dropped, from the
  Zarr 2 source, and checks that the result matches the data (see the known issue below).
- `stage_v2_time_slice.py` — copies a Zarr 2 store with only a range of time steps of its
  `data` chunks (all metadata unchanged), e.g. a one-month Zarr 2 baseline.
- `with_squashfs.sh` — mounts a squashfs image with `squashfuse_ll` for one command (one
  mount per task); used only by the dropped squashfs variant of ZRB-E002.
- `summarize_benchmark.py` — one row per benchmark run (time per window, node throughput)
  and a check that runs on different encodings read the same data.

## Repositories

- WeatherGenerator: branch [`exp/anemoi-reader-zarr-benchmark`](https://github.com/florianscheidl/WeatherGenerator/tree/exp/anemoi-reader-zarr-benchmark)
- WeatherGenerator-private: branch [`exp/anemoi-reader-zarr-benchmark`](https://gitlab.jsc.fz-juelich.de/esde/WeatherGenerator-private/-/tree/exp/anemoi-reader-zarr-benchmark?ref_type=heads)
- weathergen-configs: branch [`zarr-experiment-configs`](https://github.com/florianscheidl/weathergen-configs/tree/zarr-experiment-configs)

Commits are pinned per record.

## Known issue: `restore_zarr_attrs.py` broke time-sliced stores

Up to commit `65dfdf2f`, `restore_zarr_attrs.py` copied every Zarr 2 array missing from the
Zarr 3 store and merged the Zarr 2 root attributes, without checking them against the Zarr 3
data. On a store holding only a time slice of the source, it wrote the source's full `dates`
array and `start_date`/`end_date`. anemoi then reports the full date range over a short `data`
array, and the reader requests time steps past its end (see ZRB-E001).

Fixed in [`5a4f785d`](https://github.com/florianscheidl/WeatherGenerator/commit/5a4f785d7067674f34535596bf4d3fc36fa68e02)
(refuses a store whose `data` shape differs from the source's, checks the result against the
data, `--check-only` for existing stores) and
[`acfe78b6`](https://github.com/florianscheidl/WeatherGenerator/commit/acfe78b60d662449acd3f6e9f16b7d07ed1c8139)
(`--start-date` restores a time slice correctly, after checking its first and last time steps
against the source data). Re-check every store restored with the old version.

## Experiments

New IDs are `ZRB-E<nnn>`; a repeat or failed attempt of the same question keeps the number
and appends `a`, `b`, ...

| ID | Dataset | Outcome | Record |
| --- | --- | --- | --- |
| ZRB-E001 | N320 v2, sharded, March 2022 | Blocked: store's time metadata covers 1979–2024, data only March 2022; fix ready, needs the store owner | [`experiments/ZRB-E001-n320-sharded-month3-2022.md`](experiments/ZRB-E001-n320-sharded-month3-2022.md) |
| ZRB-E001a | same, store not fixed | Planned: training range given in the store's wrong dates (January 1979) | [`experiments/ZRB-E001a-workaround-1979-range.md`](experiments/ZRB-E001a-workaround-1979-range.md) |
| ZRB-E002 | N320 March 2022: Zarr 2 vs Zarr 3 sharded | Zarr 3 with 24-step shards ~1.2× slower per window than Zarr 2 (login and compute, 1 and 4 tasks, cold and warm), 1/3 the inodes. Squashfs variant dropped (1.4–3× slower than Zarr 2): [`experiments/ZRB-E002-squashfs-dropped.md`](experiments/ZRB-E002-squashfs-dropped.md) | [`experiments/ZRB-E002-n320-v2-v3-march-2022.md`](experiments/ZRB-E002-n320-v2-v3-march-2022.md) |
