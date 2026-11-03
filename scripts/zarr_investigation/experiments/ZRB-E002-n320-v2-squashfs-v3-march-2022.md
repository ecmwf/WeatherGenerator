# ZRB-E002 — N320, March 2022: Zarr 2 vs Zarr 2 in squashfs vs Zarr 3 sharded

- Date: 2026-11-03
- Status: planned
- Question: for the reader as training uses it, how do the same data compare as a Zarr 2
  store on GPFS, the same Zarr 2 store packed into a squashfs image mounted with
  `squashfuse_ll`, and the sharded Zarr 3 store? Read time per window, node throughput,
  `squashfuse_ll` CPU, cold vs re-read, 1 vs 4 tasks per node; plus build cost and inode
  count of each.
- Code: WeatherGenerator `<pin after commit>`, WeatherGenerator-private
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

- `v2` is a sparse copy of the source: all metadata (full `dates`, attributes,
  `.zmetadata`) and only the `data` chunk files of steps 378336–379127 (March 2022 plus a
  day each side; 378360 is 2022-03-01T00, see ZRB-E001). The chunk files are copied
  unchanged, so `v2` and `sq` return the source's bytes; the copy checks the first and last
  step. Same training dates as for the source.
- `sq` is built with `-noD -noF` (no squashfs compression: the Zarr chunks are already
  compressed) and `-b 1M`.
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

The stream reads every 6th hourly step (`frequency: 06:00:00`), and the training windows are
6 h long with a 1 h step, so each 6-hourly step falls into 6 windows.

- `s6` = `--stride 6`: every 6-hourly step once (~122 windows). Pass `p1` is the first read
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

**1. Checkout** (login node). Check out the three repositories at the recorded commits
(detached HEAD) and sync the env. `git switch` refuses if a checkout has uncommitted changes.

```bash
WG_ROOT=/e/project1/weatherai/scheidl2
(
set -e
cd "$WG_ROOT/WeatherGenerator"
git fetch \
https://github.com/florianscheidl/WeatherGenerator.git \
exp/anemoi-reader-zarr-benchmark
git switch --detach <pin after commit>
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
module load Stages/2025 GCC/13.3.0 Python/3.12.3
cd $WG_ROOT/WeatherGenerator
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
2>&1 | tee $WORK/copy.log
echo "copy: $((SECONDS - t0)) s" | tee -a $WORK/copy.log
t0=$SECONDS
mksquashfs $WORK/stage $IMG -noD -noF -noX -b 1M \
-processors 8 -no-progress 2>&1 | tee $WORK/build.log
echo "build: $((SECONDS - t0)) s" | tee -a $WORK/build.log
(
du -s --apparent-size -B1M $V2 $IMG $V3
echo "inodes v2: $(find $V2 | wc -l)"
echo "inodes v3: $(find $V3 | wc -l)"
) 2>&1 | tee $WORK/sizes.log
$ZI/with_squashfs.sh $IMG ls -la @MOUNT@/$NAME
```

Expected: `copy.log` ends with two `identical` lines; the last command prints a
`with_squashfs[0]: mounted ...` line, the store's `.zattrs`, `.zmetadata`, `data`, `dates`,
... and a `cpu_seconds=` line.

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
python $WG_ROOT/WeatherGenerator/$ZI/summarize_benchmark.py \
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

**5. Runs** (compute nodes; ~1–2 h). 1 task on `A`, then 4 tasks on `B`. First the
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
SQ="$ZI/with_squashfs.sh $IMG"
SQDS=@MOUNT@/$NAME
B6="$WORK/bench.sh"
(
set -x
for p in 1 2; do
r1 $B6 v2-1t-s6-p$p $V2 $D22 $E22 --stride 6
done
for p in 1 2; do
r1 $SQ $B6 sq-1t-s6-p$p $SQDS $D22 $E22 --stride 6
done
for p in 1 2; do
r1 $B6 v3-1t-s6-p$p $V3 $D79 $E79 --stride 6
done
r4 $B6 v2-4t-s6-p1 $V2 $D22 $E22 --stride 6
r4 $B6 v2-4t-s1 $V2 $D22 $E22
r4 $SQ $B6 sq-4t-s6-p1 $SQDS $D22 $E22 --stride 6
r4 $SQ $B6 sq-4t-s1 $SQDS $D22 $E22
r4 $B6 v3-4t-s6-p1 $V3 $D79 $E79 --stride 6
r4 $B6 v3-4t-s1 $V3 $D79 $E79
) 2>&1 | tee $WORK/runs.log
srun -N2 -n2 -w $A,$B bash -c \
'hostname; findmnt -rn -t fuse.squashfuse_ll \
-o TARGET | grep "^/tmp/zrb-"'
```

The last command should print only the two host names (no leftover mounts).

**6. Summary** (login node or the allocation).

```bash
. /e/scratch/weatherai/scheidl2/zrb-e002/env.sh
cd $OUT
python $WG_ROOT/WeatherGenerator/$ZI/summarize_benchmark.py \
v2-1t-s6-p1.jsonl v2-1t-s6-p2.jsonl \
sq-1t-s6-p1.jsonl sq-1t-s6-p2.jsonl \
v3-1t-s6-p1.jsonl v3-1t-s6-p2.jsonl \
v2-4t-s6-p1.rank*.jsonl sq-4t-s6-p1.rank*.jsonl \
v3-4t-s6-p1.rank*.jsonl v2-4t-s1.rank*.jsonl \
sq-4t-s1.rank*.jsonl v3-4t-s1.rank*.jsonl \
| tee $WORK/summary.txt
grep -h 'with_squashfs' $WORK/runs.log \
| grep -v 'cleared' | tee -a $WORK/summary.txt
```

Every run's checksum column should read `n/n`; a mismatch means the variants don't hold the
same data and the timings aren't comparable. Bring back `copy.log`, `build.log`,
`sizes.log`, `check-summary.txt`, `summary.txt` (all in `$WORK`), the login node's name
(block 3) and the job ID.

- Job ID(s): —

## Result

—
