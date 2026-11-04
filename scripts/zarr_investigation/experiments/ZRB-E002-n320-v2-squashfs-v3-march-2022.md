# ZRB-E002 — N320, March 2022: Zarr 2 vs Zarr 2 in squashfs vs Zarr 3 sharded

- Date: 2026-11-03
- Status: login-node part done; squashfs dropped for datasets (see Conclusion); compute
  runs pending
- Question: for the reader as training uses it, how do the same data compare as a Zarr 2
  store on GPFS, the same Zarr 2 store packed into a squashfs image mounted with
  `squashfuse_ll`, and the sharded Zarr 3 store? Read time per window, node throughput,
  `squashfuse_ll` CPU, cold vs re-read, 1 vs 4 tasks per node; plus build cost and inode
  count of each.
- Code: WeatherGenerator [`9e257596`](https://github.com/florianscheidl/WeatherGenerator/commit/9e257596e51242c7d5c5bab27d11bed1c093d7b8), WeatherGenerator-private
  [`7fb48aba`](https://gitlab.jsc.fz-juelich.de/esde/WeatherGenerator-private/-/commit/7fb48aba885b2df117b1a9d0387e5c2103b8fec0)
  (private config only), weathergen-configs
  [`ab7a0c53`](https://github.com/florianscheidl/weathergen-configs/commit/ab7a0c535f8dceeb1631dcbf606f137ad161358f)
- Config: `forecast/config_era5_lowres_dataonly.yml`, stream `ERA5` (as ZRB-E001)

## Datasets

All three hold March 2022 of the N320 ERA5 dataset, and all three live on `/e/scratch`, so
the filesystem is the same.

| Variant | Path | Built by |
| --- | --- | --- |
| `v2` | `$WORK/stage/n320-v2-2022-03.zarr` | `stage_v2_time_slice.py` from the Zarr 2 source in `/e/data1/slmet/ml_training/` |
| `sq` | `$WORK/n320-v2-2022-03.sqfs`, mounted per task | `mksquashfs` of `$WORK/stage` |
| `v3` | ZRB-E001's sharded store in `/e/scratch/hclimrep/kreshpa1/zarr3_datasets/` | its owner |

- `v2` is a sparse copy of the source: all metadata (full `dates`, attributes, statistics
  arrays; the source has no `.zmetadata`) and only the `data` chunk files of steps 378336–379127 (March 2022 plus a
  day each side; 378360 is 2022-03-01T00, see ZRB-E001). The chunk files are copied
  unchanged, so `v2` and `sq` return the source's bytes; the copy checks the first and last
  step. Same training dates as for the source.
- `sq` is built with `-noD -noF` (no squashfs compression of data and fragments: the Zarr
  chunks are already compressed) and `-b 1M`. `-noX` only leaves xattrs uncompressed; it
  doesn't drop them, so `mksquashfs` warns `Unrecognised xattr prefix system.posix_acl_*`
  for every directory carrying scratch's ACLs (harmless; `-no-xattrs` would silence it).
- `v3` still has the wrong time metadata (ZRB-E001), so it runs with the ZRB-E001a
  workaround: the same 744 steps given as January 1979. Window index `k` covers the same
  data in all three, and `--checksum` checks it. If the store has been fixed by then, use the
  2022 dates for `v3` too (`$D22 $E22` instead of `$D79 $E79` in blocks 4 and 5).

## Design

The copy, the image build and a short check run (10 windows per variant: mount wrapper,
date workaround, checksums) run on a login node, with reduced parallelism; their times are
only indicative (shared node). Timings come from one allocation of 2 nodes: `A` runs 1-task,
`B` runs 4-task, so neither benchmark node has read the files before (per-node GPFS cache).
The storage servers' cache can still be warm from the copy and earlier runs; nothing in user
space can clear it.

A window is 6 h long and reads 6 hourly steps, i.e. 6 chunks (the check run decodes
~6 × 234 MiB per window); windows start every hour, so each step falls into 6 windows.

- `s6` = `--stride 6`: every step once (~122 windows of 6 steps). Pass `p1` is the first read
  of the data on that node, `p2` re-reads it straight away.
- `s1` = all windows, shuffled (~740): the training pattern; each step is read 6 times, so
  cache hits are part of the result.

| Node | Tasks | Runs per variant |
| --- | --- | --- |
| A | 1 | `s6-p1`, `s6-p2` |
| B | 4 | `s6-p1`, `s1` |

`sq` mounts once per task and run, so a pass never inherits the previous mount's page cache;
only the GPFS cache of the image file stays.

What we expect: anemoi chunks are a full time step each (113 variables × 542080 points,
~245 MB as float32 before compression), so reads are large and few. The venv spike's gain
(30.7 s → 8.4 s cold Python start) came from many small files on GPFS, which these reads
don't have. So `sq` probably can't beat `v2` here and the FUSE path may cost throughput and
CPU; the question is how much. Its inode win (one per image) is certain. `v3`'s result
depends on its shard and codec layout, which the summary prints.

Not covered: DataLoader workers inside a task, many nodes reading one image, appending to a
dataset, a TB-scale build (extrapolate from block 3).

## Reproducibility

Run on JUPITER. Each block is self-contained; paste it as a whole. Lines are flush-left and
short on purpose.

**1. Checkout** (login node). WeatherGenerator gets its own worktree,
`$WG_ROOT/WeatherGenerator-zrb`, at the recorded commit, with its own env: every block runs
from there, so the main checkout can move on to other work (on 2026-11-04 it did, and runs
from it used the wrong code). The other two repositories are switched to their commits
(detached HEAD; `git switch` refuses if a checkout has uncommitted changes). The module loads
come first: a `git` left on `PATH` without its module fails with
`libiconv.so.2: cannot open shared object file`. If the worktree already exists (created on
2026-11-04), `git worktree add` fails and stops the block: drop that line and the
`actions.sh sync` after it.

```bash
module --force purge
module load Stages/2025 GCC/13.3.0
module load git/2.45.1 Python/3.12.3
WG_ROOT=/e/project1/weatherai/scheidl2
(
set -e
cd "$WG_ROOT/WeatherGenerator"
git fetch \
https://github.com/florianscheidl/WeatherGenerator.git \
exp/anemoi-reader-zarr-benchmark
P=9e257596e51242c7d5c5bab27d11bed1c093d7b8
git worktree add --detach "$WG_ROOT/WeatherGenerator-zrb" $P
cd "$WG_ROOT/WeatherGenerator-zrb"
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

**2. Set-up files and dry run** (login node). Writes `vars.sh` (paths), `env.sh` (vars,
modules, venv, `cd` into the checkout) and `bench.sh` (one benchmark run) into the work
directory, records tool versions, and prints how much the copy will take (expect ~800 chunk
files; check the GiB against the scratch quota: the copy and the image each need it once).

```bash
WORK=/e/scratch/weatherai/scheidl2/zrb-e002
mkdir -p $WORK && cd $WORK
cat > vars.sh <<'EOF'
WG_ROOT=/e/project1/weatherai/scheidl2
WORK=/e/scratch/weatherai/scheidl2/zrb-e002
N=aifs-ea-an-oper-0001-mars-n320-1979-2024-1h-v2
SRC=/e/data1/slmet/ml_training/$N-with-era51.zarr
V3=/e/scratch/hclimrep/kreshpa1/zarr3_datasets
V3=$V3/$N-with-era51_sharded_month3_2022.zarr
NAME=n320-v2-2022-03.zarr
V2=$WORK/stage/$NAME
IMG=$WORK/n320-v2-2022-03.sqfs
ZI=scripts/zarr_investigation
OUT=profiling/zarr_experiment_results
OUT=$OUT/anemoi_reader_benchmark/ZRB-E002
CFG=../weathergen-configs/forecast
CFG=$CFG/config_era5_lowres_dataonly.yml
D22=training_config.start_date=2022-03-01T00:00
E22=training_config.end_date=2022-03-31T18:00
D79=training_config.start_date=1979-01-01T00:00
E79=training_config.end_date=1979-01-31T18:00
EOF
cat > env.sh <<'EOF'
. /e/scratch/weatherai/scheidl2/zrb-e002/vars.sh
module --force purge
module load Stages/2025 GCC/13.3.0
module load git/2.45.1 Python/3.12.3
cd $WG_ROOT/WeatherGenerator-zrb
source .venv/bin/activate
export SRUN_CPUS_PER_TASK=72
nodes() { scontrol show hostnames $SLURM_JOB_NODELIST; }
A=$(nodes 2>/dev/null | sed -n 1p)
B=$(nodes 2>/dev/null | sed -n 2p)
EOF
cat > bench.sh <<'EOF'
#!/bin/bash
# bench.sh RUN DATASET START END [benchmark args]
. /e/scratch/weatherai/scheidl2/zrb-e002/vars.sh
run=$1 ds=$2 start=$3 end=$4
shift 4
exec python -u $ZI/benchmark_anemoi_reader.py \
--config $CFG --stream ERA5 --stage train \
--order shuffled --seed 0 --warmup 2 --checksum \
--filename "$ds" --output $OUT/$run.jsonl "$@" \
--options $start $end
EOF
chmod +x bench.sh
. ./env.sh
rpm -q squashfs-tools squashfuse fuse-libs
git rev-parse HEAD
python -u $ZI/stage_v2_time_slice.py $SRC $V2 \
--first-step 378336 --last-step 379127 --dry-run
df -h $WORK
```

**3. Copy and image build** (login node; 8 copy threads and 8 `mksquashfs` processors to
spare the shared node). Prints the copy and build times (indicative only), the sizes and the
inode counts, then lists the store through the mount wrapper. If the dry run showed several
hundred GiB, run this block in an allocation instead (prefix the two commands with
`srun -N1 -n1 -c72`).

```bash
. /e/scratch/weatherai/scheidl2/zrb-e002/env.sh
hostname
t0=$SECONDS
python -u $ZI/stage_v2_time_slice.py $SRC $V2 \
--first-step 378336 --last-step 379127 --workers 8 \
2>&1 | tee -a $WORK/copy.log
echo "copy: $((SECONDS - t0)) s" | tee -a $WORK/copy.log
t0=$SECONDS
mksquashfs $WORK/stage $IMG -noD -noF -noX -b 1M \
-processors 8 -no-progress 2>&1 | tee -a $WORK/build.log
echo "build: $((SECONDS - t0)) s" | tee -a $WORK/build.log
(
du -s --apparent-size -B1M $V2 $IMG $V3
echo "inodes v2: $(find $V2 | wc -l)"
echo "inodes v3: $(find $V3 | wc -l)"
) 2>&1 | tee $WORK/sizes.log
$ZI/with_squashfs.sh $IMG ls -la @MOUNT@/$NAME
```

Expected: `copy.log` ends with two `identical` lines; the last command prints a
`with_squashfs[0]: mounted ...` line, the store's `.zattrs`, `.zgroup`, `data`, `dates`, the
statistics arrays, and a `cpu_seconds=` line.

Run it in `tmux`/`screen`: the copy and the build take a while, and a closed shell kills
`mksquashfs` before it writes the superblock (last), leaving an image of full size that
doesn't mount. To repeat the block, first `rm -rf $WORK/stage $IMG` (after `. env.sh`); the
copy refuses an existing target and `mksquashfs` won't overwrite a non-squashfs file.

**4. Check run** (login node; a few minutes). The first 10 windows of the `s6` order on each
variant, `sq` through the mount wrapper. Checks the wrapper, the `v3` date workaround and
that the three variants return the same data; the times are not results.

```bash
. /e/scratch/weatherai/scheidl2/zrb-e002/env.sh
mkdir -p $OUT
CK="--stride 6 --num-windows 10 --warmup 1"
(
set -x
$WORK/bench.sh check-v2 $V2 $D22 $E22 $CK
$ZI/with_squashfs.sh $IMG $WORK/bench.sh check-sq \
@MOUNT@/$NAME $D22 $E22 $CK
$WORK/bench.sh check-v3 $V3 $D79 $E79 $CK
) 2>&1 | tee $WORK/check.log
cd $OUT
python $WG_ROOT/WeatherGenerator-zrb/$ZI/summarize_benchmark.py \
check-v2.jsonl check-sq.jsonl check-v3.jsonl \
| tee $WORK/check-summary.txt
findmnt -rn -t fuse.squashfuse_ll -o TARGET \
| grep '^/tmp/zrb-' || echo "no zrb mounts left"
```

Expected: all three rows with `empty 0` and checksums `10/10`; `with_squashfs` lines in
`check.log` (mount time, `cpu_seconds=`); `no zrb mounts left`. If `v3` shows empty windows
or mismatches, the workaround dates are off; if `squashfuse_ll -f` fails, report the error.
To repeat block 4, delete its outputs first (the benchmark refuses to overwrite):

```bash
. /e/scratch/weatherai/scheidl2/zrb-e002/env.sh
rm -v $OUT/check-v2.jsonl $OUT/check-sq.jsonl \
$OUT/check-v3.jsonl
```

**5. Runs** (compute nodes; ~1 h). `v2` and `v3` only: squashfs was dropped after the
login-node diagnosis (see Conclusion). 1 task on `A`, then 4 tasks on `B`. First the
allocation:

```bash
salloc -A weatherai -p booster -N 2 -t 03:00:00
```

Then, in the shell `salloc` opens:

```bash
. /e/scratch/weatherai/scheidl2/zrb-e002/env.sh
echo "A=$A B=$B"
mkdir -p $OUT
r1() { srun -N1 -n1 -c72 -w $A "$@"; }
r4() { srun -N1 -n4 -c72 -w $B "$@"; }
git log -1 --format='%h %s'
B6="$WORK/bench.sh"
(
set -x
for p in 1 2; do
r1 $B6 v2-1t-s6-p$p $V2 $D22 $E22 --stride 6
done
for p in 1 2; do
r1 $B6 v3-1t-s6-p$p $V3 $D79 $E79 --stride 6
done
r4 $B6 v2-4t-s6-p1 $V2 $D22 $E22 --stride 6
r4 $B6 v2-4t-s1 $V2 $D22 $E22
r4 $B6 v3-4t-s6-p1 $V3 $D79 $E79 --stride 6
r4 $B6 v3-4t-s1 $V3 $D79 $E79
) 2>&1 | tee -a $WORK/runs.log
```

The first line must print `9e257596 Updated benchmark`.

**6. Summary** (login node or the allocation).

```bash
. /e/scratch/weatherai/scheidl2/zrb-e002/env.sh
cd $OUT
python $WG_ROOT/WeatherGenerator-zrb/$ZI/summarize_benchmark.py \
v2-1t-s6-p1.jsonl v2-1t-s6-p2.jsonl \
v3-1t-s6-p1.jsonl v3-1t-s6-p2.jsonl \
v2-4t-s6-p1.rank*.jsonl v3-4t-s6-p1.rank*.jsonl \
v2-4t-s1.rank*.jsonl v3-4t-s1.rank*.jsonl \
| tee $WORK/summary.txt
```

Every run's checksum column should read `n/n`; a mismatch means the variants don't hold the
same data and the timings aren't comparable. Bring back `copy.log`, `build.log`,
`sizes.log`, `check-summary.txt`, `summary.txt` (all in `$WORK`), the login node's name
(block 3) and the job ID.

- Job ID(s): —

## Result

### Blocks 2–3: data, copy, build (2026-11-03, login node `jpbl-s01-01`)

- Tools: `squashfs-tools-4.4-10.git1.el9`, `squashfuse-0.1.104-1.el9`, `fuse-libs-2.9.9-17.el9`.
- Source `data`: shape `[403248, 113, 1, 542080]`, chunks `[1, 113, 1, 542080]` (one time
  step per chunk), `<f4`, blosc lz4 clevel 5, byte shuffle. 792 chunks = 91.8 GiB, ~119 MiB
  per chunk on disk, ~234 MiB decoded.
- Copy (`--workers 8`): 178 metadata files and 792 chunk files, 91.8 GiB in 21 s
  (4505 MiB/s); first and last step (2022-02-28T00, 2022-04-01T23) identical to the source.
  A first attempt had already read the same source on this node, so the source was probably
  partly cached.
- Build (`-processors 8`): 13 s, from the just-written copy. 1024 inodes, 99 duplicate files
  (small metadata), data uncompressed (100 %).
- Mount on the login node: 18 ms.

| Variant | Size (MiB) | Inodes |
| --- | --- | --- |
| `v2` | 94017 | 1023 |
| `sq` | 94016 | 1 |
| `v3` | 88332 | 342 |

`v3` holds 744 steps (March only), `v2`/`sq` 792 (a day of margin each side):
94017 × 744 / 792 = 88318 MiB, so the encodings are the same size per step.

The first attempt at block 3 left an image of full size without a superblock (`mksquashfs`
stopped before writing it; cause unknown, likely a closed shell), and its rerun overwrote
the logs; hence the `tee -a` and the clean-up note in block 3.

### Block 4: check run (2026-11-03, login node `jpbl-s01-01`)

10 windows of the `s6` order per variant (1 warmup, 9 timed), 1 task. Shared login node:
indicative only. `v2` and the image were written on this node in block 3 (GPFS cache
possibly warm), `v3` was not.

| Run | Median ms | p90 ms | Decoded MiB/s | Empty | Checksums |
| --- | --- | --- | --- | --- | --- |
| `check-v2` | 2458 | 2501 | 580 | 0 | 10/10 |
| `check-sq` | 7584 | 8937 | 177 | 0 | 10/10 |
| `check-v3` | 3124 | 3239 | 457 | 0 | 10/10 |

- All three return identical data; the `v3` 1979-date workaround works; no mounts left.
- `v3`: shards of 24 steps (one day) `[24, 113, 1, 542080]`, inner chunks as `v2`, same
  blosc lz4 clevel 5 + shuffle, `crc32c` shard index at the end.
- `sq` is ~3× slower per window than `v2`: ~90 MiB/s read from the image (decoded
  177 MiB/s at ~2:1 compression); diagnosed below.

### Squashfs diagnosis (2026-11-04, login node `jpbl-s01-01`)

Commands: `notes/2026-11-03-zrb-e002-squashfs-slow.md` and
`notes/2026-11-04-zrb-e002-rerun-d6-d7.md` in Florian's local workspace (D1–D7, R0–R3). R2
and R3 ran from a dedicated worktree at `9e257596` (`$WG_ROOT/WeatherGenerator-zrb`): the
first D6/D7 attempt ran from the main checkout, which had moved to another commit, and was
discarded. Squashfuse 0.6.0 is the copy inside Apptainer's `libexec` (FUSE 3,
multithreaded), run directly for the test; JUPITER's own is 0.1.104 (FUSE 2).

Raw reads with `dd` (6 chunk files, ~710 MiB):

| Path | Cache | One after another | 6 at once |
| --- | --- | --- | --- |
| GPFS files | cold | 1.05–1.20 s | 0.21–0.22 s |
| sq 128 KiB, 0.1.104 | cold | 0.47 s | 0.28 s* |
| sq 1 MiB, 0.1.104 | cold | 0.42 s | 0.45 s* |
| sq 128 KiB, 0.6.0 | cold | 0.55 s | 0.21 s* |
| sq 1 MiB, 0.6.0 | cold | 0.48 s | 0.09 s* |
| sq 1 MiB, 0.1.104 | warm | 0.24 s | 0.74 s |
| sq 128 KiB, 0.1.104 | warm | 0.25 s | 0.25 s |

\* optimistic: in the image the "at once" files follow the "one after another" files, so
GPFS had already prefetched part of them.

Check runs (10 windows of the `s6` order, 1 task; medians):

| Runs (seed) | Zarr concurrency | `v2` | `sq` |
| --- | --- | --- | --- |
| 0.1.104, 1 MiB (0, 1) | 10 | 2458, 2646 ms | 7584, 7248 ms |
| 0.1.104, 128 KiB (1) | 10 | 2646 ms | 6645 ms |
| 0.6.0, 1 MiB (4) | 10 | 2486 ms | 3553 ms (1.43×) |
| 0.6.0, 128 KiB (4) | 10 | 2486 ms | 4215 ms (1.70×) |
| 0.1.104, 128 KiB (5) | 1 | 3867 ms | 3790 ms (p90 7254) |

Daemon CPU stayed small (0.4–8.4 s per run of ~40–70 s).

- The image itself reads fast: cold, one file after another, faster than separate GPFS
  files (one large file streams better).
- The gap comes from concurrent reads. A window's 6 chunks are read concurrently (the reader
  takes `ds[start:end]`); `v2` gains ~1.55× from that (3867 → 2486 ms), squashfuse 0.1.104
  serialises them (and 1 MiB blocks thrash under concurrency when warm), 0.6.0 recovers only
  part. With Zarr reading one chunk at a time, `sq` and `v2` are equal.
- Likely reason for the remaining gap (not measured): one large read of a GPFS file is
  spread over many storage servers by the GPFS client; through FUSE it arrives as a stream
  of small requests served by a few daemon threads.

## Conclusion on squashfs for datasets

Dropped for these datasets (decided 2026-11-04 against the agreed bar of ~1.2× of `v2`):
with JUPITER's squashfuse 0.1.104 reads are 2.5–3× slower than `v2`; even squashfuse 0.6.0
(not provided on JUPITER, only inside Apptainer) is 1.4× slower. With chunks of ~119 MiB
there are no small-file costs to save, so squashfs could at best tie `v2`, and its only gain
over Zarr 3 sharding is the inode count (1 vs ~17k for N320 with one-day shards), at the
price of mount handling in every job and a squashfuse we would have to supply. The venv
images are unaffected: there the many small files are what squashfs saves.
