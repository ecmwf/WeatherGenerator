# ZRB-E001a — N320 v2, sharded, March 2022: run before the store fix

- Date: 2026-10-30
- Status: done 2026-11-04 (Slurm run under nsys; an unprofiled repeat is optional)
- Question: same as [ZRB-E001](ZRB-E001-n320-sharded-month3-2022.md), run on the store as it
  is, without its prerequisite (the time-metadata fix).
- Code: WeatherGenerator [`acfe78b6`](https://github.com/florianscheidl/WeatherGenerator/commit/acfe78b60d662449acd3f6e9f16b7d07ed1c8139)
  (newer than ZRB-E001's `65dfdf2f`; only `restore_zarr_attrs.py` differs, the benchmark is
  unchanged), WeatherGenerator-private [`7fb48aba`](https://gitlab.jsc.fz-juelich.de/esde/WeatherGenerator-private/-/commit/7fb48aba885b2df117b1a9d0387e5c2103b8fec0),
  weathergen-configs [`ab7a0c53`](https://github.com/florianscheidl/weathergen-configs/commit/ab7a0c535f8dceeb1631dcbf606f137ad161358f)
- Config, dataset: as ZRB-E001.

## Workaround

The store's `dates` start at 1979-01-01T00 (see ZRB-E001, Result), and the reader numbers time
steps from there. So it dates the 744 stored steps (March 2022) as 1979-01-01T00 …
1979-01-31T23, and the training range must be given in those dates:
`training_config.start_date=1979-01-01T00:00`, `training_config.end_date=1979-01-31T18:00`
(the last 6 h window still ends inside the data). Only the window timestamps are wrong; the
bytes read are those a run on the fixed store with the matching 2022 range would read, so
`--checksum` values should agree with it.

## Reproducibility

Run on JUPITER; paste each block as a whole. `WG_ROOT` is the directory holding the three
checkouts side by side.

**1. Checkout.** Check out the three repositories at the recorded commits (detached HEAD) and
sync the env. `git switch` refuses if a checkout has uncommitted changes.

```bash
WG_ROOT=/e/project1/weatherai/scheidl2
(
set -e
cd "$WG_ROOT/WeatherGenerator"
git fetch \
https://github.com/florianscheidl/WeatherGenerator.git \
exp/anemoi-reader-zarr-benchmark
git switch --detach acfe78b60d662449acd3f6e9f16b7d07ed1c8139
./scripts/actions.sh sync
cd "$WG_ROOT/WeatherGenerator-private"
git fetch \
git@gitlab.jsc.fz-juelich.de:esde/WeatherGenerator-private.git \
exp/anemoi-reader-zarr-benchmark
git switch --detach 7fb48aba885b2df117b1a9d0387e5c2103b8fec0
cd "$WG_ROOT/weathergen-configs"
git fetch \
https://github.com/florianscheidl/weathergen-configs.git \
zarr-experiment-configs
git switch --detach ab7a0c535f8dceeb1631dcbf606f137ad161358f
echo "Checkouts ready"
)
```

**2a. Run through Slurm** (`booster`, 1 node, 1 task, 72 CPUs). The job runs from the
checkout, so don't switch commits until it finishes.

```bash
WG_ROOT=/e/project1/weatherai/scheidl2
DS=/e/scratch/hclimrep/kreshpa1/zarr3_datasets
DS=$DS/aifs-ea-an-oper-0001-mars-n320-1979-2024-1h-v2-with-era51
DS=${DS}_sharded_month3_2022.zarr/
cd "$WG_ROOT/WeatherGenerator"
WEATHERGEN_NSYS_PROFILING=1 sbatch \
../WeatherGenerator-private/hpc/jupiter/benchmark_anemoi_reader.sh \
--config ../weathergen-configs/forecast/config_era5_lowres_dataonly.yml \
--stream ERA5 --stage train --order shuffled --seed 0 \
--num-windows 100 --warmup 2 --checksum \
--filename "$DS" \
--options training_config.start_date=1979-01-01T00:00 \
training_config.end_date=1979-01-31T18:00
```

**2b. Run directly**, on a compute node (`--nsys` needs nsys on PATH):

```bash
WG_ROOT=/e/project1/weatherai/scheidl2
DS=/e/scratch/hclimrep/kreshpa1/zarr3_datasets
DS=$DS/aifs-ea-an-oper-0001-mars-n320-1979-2024-1h-v2-with-era51
DS=${DS}_sharded_month3_2022.zarr/
cd "$WG_ROOT/WeatherGenerator"
ml Stages/2025 Nsight-Systems/2024.7.1
uv run python scripts/zarr_investigation/benchmark_anemoi_reader.py \
--config ../weathergen-configs/forecast/config_era5_lowres_dataonly.yml \
--stream ERA5 --stage train --order shuffled --seed 0 \
--num-windows 100 --warmup 2 --checksum \
--filename "$DS" \
--options training_config.start_date=1979-01-01T00:00 \
training_config.end_date=1979-01-31T18:00 \
--nsys
```

Check that the summary line reports `num_empty: 0`.

- Job ID(s) / output name(s): login-node check run (2b, see Result):
  `aifs-ea-an-oper-0001-mars-n320-1979-2024-1h-v2-with-era51_sharded_month3_2022_shards24_20261104T134936.jsonl`;
  Slurm run (2a): job 2683819,
  `aifs-ea-an-oper-0001-mars-n320-1979-2024-1h-v2-with-era51_sharded_month3_2022_shards24_20261104T140224_2683819.jsonl`
  and the matching `.nsys-rep` in `profiling/zarr_experiment_results/profiler_traces/`

## Result

**Slurm run** (2a, job 2683819, `jpbo-030-44`, 1 task, 72 CPUs, under nsys). Both runs read
the same 100 windows (seed 0) and their checksums agree on all 100.

| Run | Node | Median ms | p90 ms | p99 ms | Max ms | Decoded MiB/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Slurm, under nsys | `jpbo-030-44` | 3021 | 3167 | 3538 | 3669 | 466.5 |
| Login node (below) | `jpbl-s01-01` | 3110 | 3476 | 8032 | 13765 | 426.5 |

- One reader takes about 3.0 s per 6-step window (about 1.4 GiB decoded), i.e. roughly
  470 MiB/s, with a tight distribution: the slowest window is 1.2× the median.
- No warmup effect (warmup windows 2981 and 3025 ms) and no drift (median of the first and
  second half 3001 and 3033 ms).
- Windows that cross a 24-step shard boundary are not slower: median 2855 ms (24 windows)
  against 3044 ms within one shard.
- Caveats: the run is under nsys (overhead not measured);
  it started about 4 minutes after the login-node run had read the same data, so a
  storage-side cache can't be ruled out, and these need not be cold-read times.
  `zarr_threading_max_workers` is unset (zarr default), `zarr_async_concurrency` 10.

**Login-node check run** (2026-11-04, `jpbl-s01-01`). Step 2b run on a login node, not a
compute node, and without `--nsys` (the pasted command ended at the line before it). It
checks correctness; its timings are not results.

- `num_empty: 0` over all 100 windows: the 1979 workaround range reads the March 2022 data.
- Per-window `--checksum` hashes are recorded. They do not depend on the node, so they are a
  reference for the Slurm run and for a later run on the fixed store (ZRB-E001).
- Timings, 98 timed windows: median 3110 ms, p90 3476 ms, p99 8032 ms, max 13765 ms;
  426.5 MiB/s decoded (about 1.4 GiB per window). Not comparable to a `booster` node: other
  users' processes compete for the CPU (chunk decompression runs on it) and memory
  bandwidth; the login node's cores, per-user limits and filesystem client may differ from
  a compute node's; and its page cache may hold parts of the store from earlier reads.
- The long tail is contention, not data: it comes from two windows close together in time
  (positions 79 and 81, idx 5 at 13765 ms and idx 515 at 7855 ms), and the same windows are
  not among the slowest in the Slurm run (max 3669 ms). Outside them the login node was
  only a little slower (median +3 %, p90 +10 %).
- Wall time 505 s against 322 s timed: the rest is reader init, the 2 warmup windows and the
  checksum (about 1.5 GiB hashed per window), all outside the timed `_get`.
