# ZRB-E003 — N320, 2022: Zarr 3 shard sizes 6, 24, 96, 216 vs Zarr 2

- Date: 2026-11-05
- Status: planned; block 3 failed on 2026-11-05 on permissions. The four 2022 stores are
  `drwxrwS---` with (default) ACL `group::rw-`, so group members can't enter their
  directories. Fix, run by the stores' owner, then rerun block 3:

  ```bash
  Z3=/e/scratch/hclimrep/kreshpa1/zarr3_datasets
  N=aifs-ea-an-oper-0001-mars-n320-1979-2024-1h-v2-with-era51
  for s in 6 24 96 216; do
  v=$Z3/${N}_2022-01_2022-12_$s.zarr
  chmod -R g+rX $v
  find $v -type d -exec setfacl -d -m g::rwx {} +
  ls -ld $v $v/data
  done
  ```

  Expected: `drwxrws---` (lower-case `s`) for every store and its `data`.
- Question: how does the shard size of a Zarr 3 store change the reader's time per window
  and node throughput, against the Zarr 2 baseline? Also: does Zarr's concurrent chunk
  reading (`async.concurrency`) matter differently per shard size, and are windows that
  span two shards slower?
- Code: WeatherGenerator [`9e257596`](https://github.com/florianscheidl/WeatherGenerator/commit/9e257596e51242c7d5c5bab27d11bed1c093d7b8)
  (as ZRB-E002; the worktree `WeatherGenerator-zrb` from ZRB-E002 is reused),
  WeatherGenerator-private
  [`7fb48aba`](https://gitlab.jsc.fz-juelich.de/esde/WeatherGenerator-private/-/commit/7fb48aba885b2df117b1a9d0387e5c2103b8fec0)
  (private config only), weathergen-configs
  [`ab7a0c53`](https://github.com/florianscheidl/weathergen-configs/commit/ab7a0c535f8dceeb1631dcbf606f137ad161358f)
- Config: `forecast/config_era5_lowres_dataonly.yml`, stream `ERA5` (as ZRB-E001/E002)
- Builds on [ZRB-E002](ZRB-E002-n320-v2-v3-march-2022.md): Zarr 3 with 24-step shards was
  ~1.2× slower per window than Zarr 2; re-reads changed ≤3%.

## Datasets

The four Zarr 3 stores hold all of 2022 (8760 steps), converted by kreshpa1 from the Zarr 2
source; they differ only in the shard size (steps per shard file). Inner chunks are expected
to be one time step each with the source's blosc lz4 settings; block 4 prints the layout.

| Variant | Path | Shard file | Shard files per year |
| --- | --- | --- | --- |
| `v2` | `$WORK2/stage/n320-v2-2022-03.zarr` (ZRB-E002's copy, March 2022) | one step, ~119 MiB | 8760 |
| `v3-6` | `$Z3/${N}_2022-01_2022-12_6.zarr` | 6 steps, ~0.7 GiB | 1460 |
| `v3-24` | `$Z3/${N}_2022-01_2022-12_24.zarr` | 24 steps, ~2.8 GiB | 365 |
| `v3-96` | `$Z3/${N}_2022-01_2022-12_96.zarr` | 96 steps, ~11 GiB | 92 |
| `v3-216` | `$Z3/${N}_2022-01_2022-12_216.zarr` | 216 steps, ~25 GiB | 41 |

`N=aifs-ea-an-oper-0001-mars-n320-1979-2024-1h-v2-with-era51`,
`Z3=/e/scratch/hclimrep/kreshpa1/zarr3_datasets`,
`WORK2=/e/scratch/weatherai/scheidl2/zrb-e002`.

**Time range: March 2022 for every variant.** The reader selects steps by date
(`training_config.start_date`/`end_date`), not by what the store holds, so no variant needs a
cut to a year: the full 1979–2024 Zarr 2 source and the 2022 Zarr 3 stores give the same
windows for the same dates (the `--checksum` comparison relies on this). What matters for
the baseline is the filesystem: the Zarr 2 source lives on `/e/data1`, the Zarr 3 stores on
`/e/scratch`, so reading the source directly would mix the encoding with the filesystem.
ZRB-E002's sparse copy of the source on `/e/scratch` holds only Feb 28–Apr 1 2022, so the
range is March 2022 (as in ZRB-E002); copying all of 2022 would take ~1 TiB of scratch for no
gain:

- A window's read is the same whatever the range: 6 steps from one or two shard files.
  The range only sets how many windows exist; E002 showed caching across windows changes
  ≤3%, and a year at `s1` (8760 windows) would take hours per run.
- One difference to a full year: March touches few shards of the large sizes (4 files for
  `v3-216`), so the 4 tasks of a node read from the same few files. In training over many
  years they would be spread over many files.
- Results are directly comparable with ZRB-E002 (same `v2` copy, same windows); `v3-24`
  should repeat E002's `v3` (~3.0 s per window on compute nodes).

Windows spanning two shard files, out of the 739 March windows (stores start at
2022-01-01T00, March at step 1416; a window is 6 hourly steps):

| Shard steps | Shards touched by March | Windows spanning two shards |
| --- | --- | --- |
| 6 | 124 | 615 (83%) |
| 24 | 31 | 150 (20%) |
| 96 | 9 | 40 (5%) |
| 216 | 4 | 15 (2%) |

## Design

As ZRB-E002: a window is 6 h long and reads 6 hourly steps (6 chunks, ~1.4 GiB decoded),
requested as one slice, so Zarr reads its chunks concurrently (`async.concurrency`, default
10). One batch job of 3 nodes runs everything one run at a time; each node reads each variant
once before anything else touches that variant on that node (per-node GPFS cache). The
storage servers' cache can still be warm from earlier runs; nothing in user space clears it.

- `s6` = `--stride 6`: every 6th window, ~123 windows, every step once.
- `s1` = all ~740 windows, shuffled: the training pattern, each step read 6 times.
- `c1` = `ZARR_ASYNC__CONCURRENCY=1`: a window's chunks read one after another (E002: `v2`
  1.55× slower on the login node). With large shards all chunks of a window come from one
  file, so the effect may differ by shard size.

| Node | Tasks | Runs per variant | Measures |
| --- | --- | --- | --- |
| A | 1 | `1t-s6` | cold read, one reader |
| C | 1 | `1t-s6-c1` | cold read, chunks read serially |
| B | 4 | `4t-s6`, then `4t-s1` | node throughput; training pattern |

E002 found no difference between a first read and an immediate re-read, so the re-read pass
(`p2`) is dropped. 5 variants × 4 runs = 20 runs, ~2.5–3 h.

Not covered: DataLoader workers inside a task, ranges longer than a month, writing/appending.

## Reproducibility

Run on JUPITER. Each block is self-contained; paste it as a whole. Lines are flush-left and
short on purpose.

**1. Checkout** (login node). Reuses ZRB-E002's worktree `$WG_ROOT/WeatherGenerator-zrb` at
`9e257596`; switches the private and configs checkouts to the recorded commits (detached
HEAD; `git switch` refuses if a checkout has uncommitted changes). If the worktree is gone,
create it with ZRB-E002 block 1.

```bash
module --force purge
module load Stages/2025 GCC/13.3.0
module load git/2.45.1 Python/3.12.3
WG_ROOT=/e/project1/weatherai/scheidl2
(
set -e
cd "$WG_ROOT/WeatherGenerator-zrb"
git log -1 --format='%h %s'
cd "$WG_ROOT/WeatherGenerator-private"
git switch --detach 7fb48aba885b2df117b1a9d0387e5c2103b8fec0
cd "$WG_ROOT/weathergen-configs"
git switch --detach ab7a0c535f8dceeb1631dcbf606f137ad161358f
echo "Checkouts ready"
)
```

Expected: `9e257596 Updated benchmark`, then `Checkouts ready`.

**2. Set-up files** (login node). Writes `vars.sh`, `env.sh` (vars, modules, venv, `cd` into
the worktree) and `bench.sh` (one benchmark run) into the work directory and lists the
variants.

```bash
WORK=/e/scratch/weatherai/scheidl2/zrb-e003
mkdir -p $WORK && cd $WORK
cat > vars.sh <<'EOF'
WG_ROOT=/e/project1/weatherai/scheidl2
WORK=/e/scratch/weatherai/scheidl2/zrb-e003
N=aifs-ea-an-oper-0001-mars-n320-1979-2024-1h-v2
N=$N-with-era51
SRC=/e/data1/slmet/ml_training/$N.zarr
Z3=/e/scratch/hclimrep/kreshpa1/zarr3_datasets
V2=/e/scratch/weatherai/scheidl2/zrb-e002/stage
V2=$V2/n320-v2-2022-03.zarr
V3_6=$Z3/${N}_2022-01_2022-12_6.zarr
V3_24=$Z3/${N}_2022-01_2022-12_24.zarr
V3_96=$Z3/${N}_2022-01_2022-12_96.zarr
V3_216=$Z3/${N}_2022-01_2022-12_216.zarr
ZI=scripts/zarr_investigation
OUT=profiling/zarr_experiment_results
OUT=$OUT/anemoi_reader_benchmark/ZRB-E003
CFG=../weathergen-configs/forecast
CFG=$CFG/config_era5_lowres_dataonly.yml
D=training_config.start_date=2022-03-01T00:00
E=training_config.end_date=2022-03-31T18:00
EOF
cat > env.sh <<'EOF'
. /e/scratch/weatherai/scheidl2/zrb-e003/vars.sh
module --force purge
module load Stages/2025 GCC/13.3.0
module load git/2.45.1 Python/3.12.3
cd $WG_ROOT/WeatherGenerator-zrb
source .venv/bin/activate
export SRUN_CPUS_PER_TASK=72
nodes() { scontrol show hostnames $SLURM_JOB_NODELIST; }
A=$(nodes 2>/dev/null | sed -n 1p)
B=$(nodes 2>/dev/null | sed -n 2p)
C=$(nodes 2>/dev/null | sed -n 3p)
EOF
cat > bench.sh <<'EOF'
#!/bin/bash
# bench.sh RUN DATASET [benchmark args]
. /e/scratch/weatherai/scheidl2/zrb-e003/vars.sh
run=$1 ds=$2
shift 2
exec python -u $ZI/benchmark_anemoi_reader.py \
--config $CFG --stream ERA5 --stage train \
--order shuffled --seed 0 --warmup 2 --checksum \
--filename "$ds" --output $OUT/$run.jsonl "$@" \
--options $D $E
EOF
chmod +x bench.sh
. ./env.sh
git log -1 --format='%h %s'
ls -d $V2 $V3_6 $V3_24 $V3_96 $V3_216
```

Expected: `9e257596 Updated benchmark` and the five paths without errors.

**3. Metadata and sizes** (login node; a few minutes). `restore_zarr_attrs.py --check-only`
per store: its `dates`, start/end date and frequency must match its time axis, variables,
statistics and coordinates must match the data, and its first and last step must equal the
source's at 2022-01-01T00 and 2022-12-31T23. Then size and inode count per store.

```bash
. /e/scratch/weatherai/scheidl2/zrb-e003/env.sh
for v in $V3_6 $V3_24 $V3_96 $V3_216; do
echo "== $v"
python $ZI/restore_zarr_attrs.py $SRC $v \
--check-only --start-date 2022-01-01T00:00
done 2>&1 | tee $WORK/metadata.log
(
for v in $V3_6 $V3_24 $V3_96 $V3_216; do
du -s --apparent-size -B1M $v
echo "inodes: $(find $v | wc -l)"
done
) 2>&1 | tee $WORK/sizes.log
```

Expected: every store ends with `metadata matches the data`. If one fails on its dates, stop:
the March range would select different steps in it.

**4. Check run** (login node; ~10 min). The first 10 windows of the `s6` order on each
variant: all variants must return the same data (checksums against `v2`); the summary also
prints each store's layout. Times are not results.

```bash
. /e/scratch/weatherai/scheidl2/zrb-e003/env.sh
mkdir -p $OUT
CK="--stride 6 --num-windows 10 --warmup 1"
(
set -x
$WORK/bench.sh check-v2 $V2 $CK
$WORK/bench.sh check-v3-6 $V3_6 $CK
$WORK/bench.sh check-v3-24 $V3_24 $CK
$WORK/bench.sh check-v3-96 $V3_96 $CK
$WORK/bench.sh check-v3-216 $V3_216 $CK
) 2>&1 | tee $WORK/check.log
cd $OUT
python $WG_ROOT/WeatherGenerator-zrb/$ZI/summarize_benchmark.py \
check-v2.jsonl check-v3-6.jsonl check-v3-24.jsonl \
check-v3-96.jsonl check-v3-216.jsonl \
| tee $WORK/check-summary.txt
```

Expected: five rows with `empty 0` and checksums `10/10`; layout lines with chunks
`[1, 113, 1, 542080]`, shards `[<steps>, 113, 1, 542080]` and the same blosc settings. To
repeat block 4, delete its outputs first (the benchmark refuses to overwrite):

```bash
. /e/scratch/weatherai/scheidl2/zrb-e003/env.sh
rm -v $OUT/check-*.jsonl
```

**5. Runs** (compute nodes; ~2.5–3 h; one batch job). Writes `runs.sbatch` into the work
directory and submits it. 3 nodes: per variant, 1 task on `A` (`1t-s6`), 1 task on `C` with
serial chunk reads (`1t-s6-c1`), 4 tasks on `B` (`4t-s6`, then `4t-s1`), one run after
another; then the summary into `$WORK/summary.txt`. Log: `$WORK/runs-<job ID>.out`. It stops
before the first run if the worktree is not at the recorded commit or the nodes are not three
distinct ones; a failed run does not stop the later ones, and the log ends with each run's
exit code. Don't move the worktree until the job has finished.

```bash
W=/e/scratch/weatherai/scheidl2/zrb-e003
cat > $W/runs.sbatch <<'EOF'
#!/bin/bash
#SBATCH --account=weatherai
#SBATCH --partition=booster
#SBATCH --nodes=3
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --job-name=zrb-e003
#SBATCH --output=/e/scratch/weatherai/scheidl2/zrb-e003/runs-%j.out
# Lmod is not set up in batch shells (as benchmark_anemoi_reader.sh)
. "$HOME/.bashrc"
. "$LMOD_ROOT/lmod/init/bash"
. /e/scratch/weatherai/scheidl2/zrb-e003/env.sh
P=9e257596e51242c7d5c5bab27d11bed1c093d7b8
echo "job $SLURM_JOB_ID: A=$A B=$B C=$C"
git log -1 --format='%h %s'
if [ "$(git rev-parse HEAD)" != "$P" ]; then
echo "worktree not at $P, stopping"
exit 1
fi
if [ -z "$C" ] || [ "$A" = "$B" ] || [ "$B" = "$C" ] \
|| [ "$A" = "$C" ]; then
echo "need three distinct nodes, stopping"
exit 1
fi
mkdir -p $OUT
codes=()
# run NTASKS NODE RUN DATASET [benchmark args]
run() {
local n=$1 node=$2
shift 2
echo "== $1: $n task(s) on $node, $(date -Is)"
srun -N1 -n$n -c72 -w $node $WORK/bench.sh "$@"
codes+=("$? $1")
}
VARIANTS="v2=$V2 v3-6=$V3_6 v3-24=$V3_24"
VARIANTS="$VARIANTS v3-96=$V3_96 v3-216=$V3_216"
for x in $VARIANTS; do
run 1 $A ${x%%=*}-1t-s6 ${x#*=} --stride 6
done
export ZARR_ASYNC__CONCURRENCY=1
for x in $VARIANTS; do
run 1 $C ${x%%=*}-1t-s6-c1 ${x#*=} --stride 6
done
unset ZARR_ASYNC__CONCURRENCY
for x in $VARIANTS; do
run 4 $B ${x%%=*}-4t-s6 ${x#*=} --stride 6
run 4 $B ${x%%=*}-4t-s1 ${x#*=}
done
echo "== exit codes (0 = ok), $(date -Is)"
printf '%s\n' "${codes[@]}"
cd $OUT
F=
for m in 1t-s6 1t-s6-c1 4t-s6 4t-s1; do
for x in $VARIANTS; do
r=${x%%=*}-$m
case $m in 4t-*) f="$r.rank*.jsonl" ;; *) f="$r.jsonl" ;; esac
F="$F $(ls $f 2>/dev/null)"
done
done
python $WG_ROOT/WeatherGenerator-zrb/$ZI/summarize_benchmark.py $F \
| tee $WORK/summary.txt
EOF
sbatch $W/runs.sbatch
```

Once it has finished, check the (latest) log: the second line must print `9e257596 Updated benchmark`,
`A`, `B` and `C` must differ, every exit code must be 0, and the `c1` runs' headers must show
concurrency 1:

```bash
. /e/scratch/weatherai/scheidl2/zrb-e003/env.sh
LOG=$(ls -t $WORK/runs-*.out | head -1)
echo $LOG
sed -n 1,2p $LOG
sed -n '/^== exit codes/,$p' $LOG
cd $OUT
for f in *-1t-s6-c1.jsonl v2-1t-s6.jsonl; do
echo "$f: $(head -1 $f | grep -o '"zarr_async_concurrency": [0-9]*')"
done
```

**6. Summary** (login node). The job writes it; to produce it again, e.g. after repeating
a run. Every run's checksums are compared with the first run that read the same window,
i.e. `v2`.

```bash
. /e/scratch/weatherai/scheidl2/zrb-e003/env.sh
VARIANTS="v2 v3-6 v3-24 v3-96 v3-216"
cd $OUT
F=
for m in 1t-s6 1t-s6-c1 4t-s6 4t-s1; do
for x in $VARIANTS; do
case $m in 4t-*) f="$x-$m.rank*.jsonl" ;; *) f="$x-$m.jsonl" ;; esac
F="$F $(ls $f 2>/dev/null)"
done
done
python $WG_ROOT/WeatherGenerator-zrb/$ZI/summarize_benchmark.py $F \
| tee $WORK/summary.txt
```

**7. Bring back** `metadata.log`, `sizes.log`, `check-summary.txt`, `summary.txt` and
`runs-<job ID>.out` (all in `$WORK`), plus the JSONL outputs for the shard-boundary analysis
(each window records its dataset steps):

```bash
. /e/scratch/weatherai/scheidl2/zrb-e003/env.sh
tar czf $WORK/zrb-e003-jsonl.tgz -C $OUT .
ls -l $WORK/zrb-e003-jsonl.tgz
```

- Job ID(s):

## Result

Pending.
