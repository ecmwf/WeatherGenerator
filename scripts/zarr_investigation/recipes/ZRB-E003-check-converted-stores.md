# ZRB-E003: integration check of the converted Zarr 3 stores

Login node, ~30 min including set-up. Reads each converted store the
way training does (WeatherGenerator's anemoi reader, ZRB benchmark)
and checks it returns exactly the Zarr 2 source's data:

- metadata: `restore_zarr_attrs.py --check-only` (dates, start/end
  date and frequency against the time axis, variables, statistics,
  coordinates; first and last step equal to the source);
- short benchmark: 10 windows per store, same windows as on the
  Zarr 2 source; `--checksum` hashes every window, the summary
  compares them with the source's and prints each store's layout
  (inner chunks, shards, codecs).

Everything goes into one work directory `CHK` (fresh clones of the
two public repos at pinned commits, a venv, the outputs); your own
checkouts are only read (WeatherGenerator-private, for the platform
config). Code: WeatherGenerator `124987a0`, weathergen-configs
`ab7a0c53` (as ZRB-E002).

Each block is self-contained; paste it as a whole. First replace
`<project>/<user>` everywhere in this file with your project and
user, e.g. `hclimrep/kreshpa1` (it appears at the top of every block),
and check `PRIV` in block 1.

## 1. Set-up (once, ~5 min)

Fill in the first lines of `vars.sh`: `CHK` (work directory, e.g. on
scratch; ~1 GB for the venv), `PRIV` (your WeatherGenerator-private
checkout), `STORES` (`<steps per shard>=<path>`, one per line). `D`/`E`
are the training range: March 2022, valid if the stores' `dates` are
correct (see block 2).

```bash
CHK=/e/scratch/<project>/<user>/zrb-e003-check
mkdir -p $CHK/out && cd $CHK
cat > vars.sh <<'EOF'
CHK=/e/scratch/<project>/<user>/zrb-e003-check
PRIV=/e/project1/<project>/<user>/WeatherGenerator-private
STORES="
6=/path/to/store-shards-6.zarr
24=/path/to/store-shards-24.zarr
96=/path/to/store-shards-96.zarr
384=/path/to/store-shards-384.zarr
"
N=aifs-ea-an-oper-0001-mars-n320-1979-2024-1h-v2
SRC=/e/data1/slmet/ml_training/$N-with-era51.zarr
START=2022-03-01T00:00
D=training_config.start_date=2022-03-01T00:00
E=training_config.end_date=2022-03-31T18:00
WG=$CHK/WeatherGenerator
ZI=$WG/scripts/zarr_investigation
CFG=$CHK/weathergen-configs/forecast
CFG=$CFG/config_era5_lowres_dataonly.yml
EOF
cat > env.sh <<'EOF'
. "$(dirname "${BASH_SOURCE[0]}")/vars.sh"
module --force purge
module load Stages/2025 GCC/13.3.0
module load git/2.45.1 Python/3.12.3
export WEATHERGEN_PRIVATE_REPO_PATH=$PRIV
source $WG/.venv/bin/activate
cd $WG
# bench NAME DATASET [benchmark args]: one run, output out/NAME.jsonl
bench() {
local name=$1 ds=$2
shift 2
python -u $ZI/benchmark_anemoi_reader.py \
--config $CFG --stream ERA5 --stage train \
--order shuffled --seed 0 --warmup 1 --checksum \
--filename "$ds" --output $CHK/out/$name.jsonl "$@" \
--options $D $E
}
EOF
. ./vars.sh
module --force purge
module load Stages/2025 GCC/13.3.0
module load git/2.45.1 Python/3.12.3
(
set -e
G=https://github.com/florianscheidl
git init -q $WG && cd $WG
git fetch -q --depth 1 $G/WeatherGenerator.git \
124987a0d871a0fadc87ead21eac71359617d73e
git checkout -q FETCH_HEAD
./scripts/actions.sh sync
git init -q $CHK/weathergen-configs
cd $CHK/weathergen-configs
git fetch -q --depth 1 $G/weathergen-configs.git \
ab7a0c535f8dceeb1631dcbf606f137ad161358f
git checkout -q FETCH_HEAD
)
. $CHK/env.sh
git log -1 --format='%h %s'
ls $PRIV/hpc/platform-env.py $CFG
for s in $STORES; do ls -d ${s#*=}; done
```

Expected: `124987a0 Remove check_v3_store.py`, then the platform
script, the config and every store path listed without errors.

## 2. Metadata check (~1 min)

```bash
. /e/scratch/<project>/<user>/zrb-e003-check/env.sh
for s in $STORES; do
echo "== ${s%%=*}: ${s#*=}"
python $ZI/restore_zarr_attrs.py $SRC ${s#*=} \
--check-only --start-date $START
done 2>&1 | tee $CHK/metadata.log
```

Expected for every store: it compares the first and last step with
the source and ends with `metadata matches the data`. If a store
fails with dates spanning 1979–2024 (its `dates` were not sliced),
the windows in block 3 will not match either: fix the store first
(`restore_zarr_attrs.py ... --start-date $START` without
`--check-only`, by its owner).

## 3. Short benchmark (~2 min per store, plus the source)

The same 10 windows (every 6th window, shuffled, seed 0) on the Zarr 2
source and on each store. Times on a login node are not results.

```bash
. /e/scratch/<project>/<user>/zrb-e003-check/env.sh
K="--stride 6 --num-windows 10"
(
set -x
bench v2 $SRC $K
for s in $STORES; do
bench s${s%%=*} ${s#*=} $K
done
) 2>&1 | tee $CHK/bench.log
cd $CHK/out
F=
for r in v2 $(for s in $STORES; do echo s${s%%=*}; done); do
if [ -f $r.jsonl ]; then F="$F $r.jsonl"
else echo "no output for $r: see bench.log"; fi
done
python $ZI/summarize_benchmark.py $F | tee $CHK/summary.txt
```

Expected: one row per run with `empty 0` and checksums `10/10`, and
in the layout lines below the table chunks `[1, 113, 1, 542080]`,
shards `[<steps>, 113, 1, 542080]` and the same blosc lz4 settings
for every store. A run that crashed is listed as `no output` (or
`incomplete` if it stopped midway); its error is in `bench.log`. If
`v2` itself failed, the checksums are compared with the first store
instead and mean nothing. To repeat a run, delete its `out/<name>.jsonl`
first (the benchmark refuses to overwrite).

## 4. Optional: read every time step (~6 min per store)

Every 6th window covers each time step exactly once (~123 windows,
6 steps each), so this reads the whole month of each store and
compares it with the source. Run it in `tmux`/`screen`.

```bash
. /e/scratch/<project>/<user>/zrb-e003-check/env.sh
(
set -x
bench full-v2 $SRC --stride 6
for s in $STORES; do
bench full-s${s%%=*} ${s#*=} --stride 6
done
) 2>&1 | tee $CHK/bench-full.log
cd $CHK/out
F=
for r in v2 $(for s in $STORES; do echo s${s%%=*}; done); do
if [ -f full-$r.jsonl ]; then F="$F full-$r.jsonl"
else echo "no output for full-$r: see bench-full.log"; fi
done
python $ZI/summarize_benchmark.py $F | tee $CHK/summary-full.txt
```

Expected: `empty 0` and checksums `n/n` in every row.

## 5. Send back

`$CHK/metadata.log` and `$CHK/summary.txt` (and `summary-full.txt`
if block 4 ran), plus how the stores were converted (tool and
version).
