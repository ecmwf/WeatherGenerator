# packages/ — UV Workspace Packages

Reference for the UV workspace packages under `packages/`: shared config/I-O utilities, the evaluation pipeline, MLFlow integration, extra data readers, and the (non-workspace) dashboard. Part of the agent docs rooted at [`AGENT-README.md`](../AGENT-README.md). Related: [`config/DOCS-Config.md`](../config/DOCS-Config.md) (config file semantics) and [`src/weathergen/datasets/DOCS-Datasets.md`](../src/weathergen/datasets/DOCS-Datasets.md) (reader interface).

## Workspace mechanics

The root `pyproject.toml` (`[tool.uv.workspace]`) declares four members:

```
packages/common         → weathergen-common
packages/evaluate       → weathergen-evaluate
packages/metrics        → weathergen-metrics
packages/readers_extra  → weathergen-readers-extra
```

`packages/dashboard` is **deliberately not a member** (it breaks Streamlit deployment) and `packages/science` is a script directory, not a package (no `pyproject.toml`).

Key facts:

- **Install**: `uv sync --all-packages --extra cpu|gpu` (wrapped by `./scripts/actions.sh sync`, which picks `cpu` on macOS, `gpu` on Linux). All workspace members are installed editable into the single root venv.
- **Independent versioning**: each package has its own `pyproject.toml` with its own version (`weathergen-evaluate` is at `1.0`, the others at `0.1.0`).
- **Type checking is per package**: `./scripts/actions.sh type-check` runs `uv sync --project packages/<pkg> --no-install-workspace` and then `pyrefly check packages/<pkg>` for common, metrics, and evaluate individually (rebuilding deps each time so implicit imports fail), then checks root `src/` with the full workspace. Each package carries its own `[tool.pyrefly]` config.
- **Dependency direction** (from the package `pyproject.toml` files):
  - `common` → depends only on third-party libs (xarray, zarr, omegaconf, ...). Everything else may depend on it.
  - `metrics` → `common` + `mlflow-skinny`.
  - `evaluate` → `common`, `metrics` + plotting/verification stack (cartopy, xskillscore, earthkit, eccodes, ...).
  - `readers_extra` → `common` declared, but it **also implicitly imports the root package** (`weathergen.datasets.data_reader_base`, `weathergen.train.utils`). This is a known circular dependency, documented as a TODO in `packages/readers_extra/pyproject.toml`; it only works because everything shares one venv.
  - Root `weathergen` → `common`, `evaluate`, `readers_extra` (and, transitively, `metrics`).
- All packages use the `src/weathergen/<name>/` layout, so they merge into the single `weathergen` namespace at import time.

## packages/common — config system and I/O

The most important package for agents: **all config handling lives here**.

### `weathergen/common/config.py`

Full path: `packages/common/src/weathergen/common/config.py`. `Config` is an alias for `omegaconf.DictConfig`.

**Config assembly — `load_merge_configs(private_home, from_run_id, mini_epoch, base, *overwrites)`** is the single entry point that builds a run config. Merge precedence (ascending, later wins):

1. Base config: `config/default_config.yml` by default; a `Path`/`Config` if given; or, when `from_run_id` is set, the saved config of that run via `load_run_config()` (which reads `models/<run_id>/model_<run_id>[_latest|_chkpt<N>].json`).
2. Private config.
3. `*overwrites` in order — each may be a `Path` (possibly several paths joined with `:`, as passed by slurm), a `dict`, or a `DictConfig`.

Notes on behavior:
- Any overwrite with a `streams_directory` key gets its streams loaded via `load_streams()`, and its presence **nulls out the base config's inherited `streams`** so they are replaced, not merged.
- `merge_configs(base, update)` is a thin wrapper over `OmegaConf.merge` and is the sanctioned way to combine configs (do not do dict updates).
- `from_cli_arglist()` turns `a.b=c` CLI items into an overwrite config.

**Private config — `_load_private_conf()`** resolves the platform-specific config (data paths, `path_shared_working_dir`, MLFlow settings) in this order:
1. Explicit `private_home` path argument.
2. `WEATHERGEN_PRIVATE_CONF` environment variable.
3. Running `<private-repo>/hpc/platform-env.py hpc-config` as a subprocess (auto-detects the HPC).

The `secrets` section of the private config is **always deleted** before merging, so secrets never end up in saved run configs.

**Custom OmegaConf resolvers**: `${timedelta:...}` (`parse_timedelta` — ints/floats are seconds, strings via `pandas.to_timedelta`, result is `np.timedelta64[ms]`) and `${datetime:...}` (`str_to_datetime64`), registered at import time. Time keys in `training_config`/`validation_config`/`test_config` (`start_date`, `end_date`, `time_window_step`, `time_window_len`, `forecast.time_step`) are rewritten by `_sanitize_time_keys()`: the raw string is kept in a backup key `_<key>` and `<key>` becomes an interpolation, so accessing `cfg.training_config.start_date` yields a real `np.datetime64`. `_strip_interpolation()` reverses this for serialization (used by `save()` and `format_cf()`); keys starting with `_` are hidden.

**Backward compatibility**: `_apply_fixes()` is the central hook applied when loading old run configs — `_check_time_interpolation` (unwraps stored resolver strings), `_check_datasets` (collects legacy `data_path_*` keys into `data_paths`), `_check_streams` (converts list-style `streams` to a dict keyed by stream name). Add new compat fixes here, and expect them to be removed eventually.

**Streams**: `load_streams(streams_directory)` globs `[!.#]*.yml` recursively, loads each file's top-level mapping as `{stream_name: stream_config}`, injects `name` into each config, rejects duplicate names, and patches `frequency` with the timedelta resolver. It also consults a hardcoded `streams_history` rename map (e.g. `streams_anemoi` → `era5_1deg`) for old directory names.

**Other utilities**: `get_run_id()` (random 8-char id, letter first), `set_run_id()` (new / assigned / reused id logic), `get_path_run()` / `get_path_model()` / `get_path_results()` / `get_model_results()` (all rooted at the private config's `path_shared_working_dir`: `results/<run_id>`, `models/<run_id>`, validation zarr names `validation_chkpt<NNNNN>_rank<NNNN>.<ext>`), `save()` / `load_run_config()` (JSON round-trip of the config into the model dir), and `validate_forecast_policy_and_steps()` (enforces the `forecast.offset` / `num_steps` / `policy` rules).

### Other modules

- `packages/common/src/weathergen/common/io.py` — the Zarr inference-output format shared between training/inference (writer) and evaluate (reader). Key pieces: `ZarrIO` / `ZipZarrIO` context managers (obtain via `zarrio_reader()` / `zarrio_writer()`, which dispatch on extension), `StoreType` (valid store extensions), `ItemKey` (`sample/stream/forecast_step` addressing), `OutputDataset` / `OutputItem` (source/target/prediction + coords, `as_xarray()`), `OutputBatchData` (extracts per-item data from model batch output), `TimeRange`, `IOReaderData`.
- `packages/common/src/weathergen/common/paths.py` — `_REPO_ROOT` (computed from the file location) and `get_wg_private_path()` (`WEATHERGEN_PRIVATE_REPO_PATH` env var, else sibling `../WeatherGenerator-private`).
- `packages/common/src/weathergen/common/platform_env.py` — `get_platform_env()` dynamically imports `<private-repo>/hpc/platform-env.py` as a module (cached); provides `get_hpc()`, `get_hpc_user()`, `get_hpc_user_org()`, `get_hpc_config()`, `get_hpc_certificate()`.
- `packages/common/src/weathergen/common/logger.py` — `init_loggers()` (singleton, resets handlers, JSON-configurable, colored relative-path formatter). Call it at every entry point.
- `__init__.py` contains only a placeholder `common_function()` and a TODO list of things to move here.

## packages/evaluate — evaluation pipeline

Reads Zarr inference output (the `common/io.py` format) and produces scores, plots, and exports. See `packages/evaluate/README.md` for motivation ("fast evaluation": no GRIB/netCDF conversion, developer diagnostics).

**Entry points** (root `pyproject.toml`):
- `uv run evaluate --config config/evaluate/eval_config.yml [--push-metrics] [--options a.b=c ...] [--run-ids id1 id2 ...]` → `weathergen.evaluate.run_evaluation:evaluate` in `packages/evaluate/src/weathergen/evaluate/run_evaluation.py`. Flow: load eval config → per run × stream, `_process_stream()` builds a reader, loads data, plots (`plotting/plot_orchestration.py`) and/or scores (`scores/score_orchestration.py`), caches computed scores as JSON → summary plots → optional MLFlow upload of scores (`--push-metrics`, via `weathergen-metrics`).
- `uv run export --run-id <id> --stream <name> ...` → `weathergen.evaluate.export.export_inference:export` — converts inference Zarr to external formats. `export/parser_factory.py` supports `netcdf`, `quaver`, and `verif` output parsers, with optional regridding (`export/verif_interpolator.py`).
- The package's own `pyproject.toml` additionally names the evaluation script `evaluation`; the root alias `evaluate` is the one normally on PATH.

**Reader types** (`get_reader()` in `run_evaluation.py`, selected by the per-run `type` key): `zarr` (default, `io/wegen_reader.py:WeatherGenZarrReader`), `csv` (`io/csv_reader.py`), `json` (cached scores), `merge` / `jsonmerge` (`io/merge_reader.py`, multi-rank zarr merging).

**Layout** under `packages/evaluate/src/weathergen/evaluate/`: `io/` (readers + parallel data loading in `io/data/`), `scores/` (metric computation, `scores/score.py`, PSD), `plotting/` (maps, line/bar/quantile plots, score cards, timeseries), `export/` (format conversion), `utils/` (regions, climatology, derived channels), `example_extras/` (tropical-cyclone and power-spectra case studies).

**Config**: templates live in `config/evaluate/` (`eval_config.yml`, `eval_config_default.yml`, plus `config_zarr2cf.yaml` / `config_zarr2verif.yaml` for export). The reference doc `docs/evaluate_config_reference.md` is accurate and current (run types, metrics, regions, score caching all match the code) — except it points to the template as `config/eval_config.yml`; the real path is `config/evaluate/eval_config.yml`.

## packages/metrics — MLFlow integration

Single module: `packages/metrics/src/weathergen/metrics/mlflow_utils.py` (dep: `mlflow-skinny`). Tracks runs on Databricks-hosted MLFlow (`tracking_uri = "databricks"`, registry `databricks-uc`, experiment `/Shared/weathergen-dev/core-model/defaultExperiment`).

- `setup_mlflow(private_config)` — sets `DATABRICKS_HOST`/`DATABRICKS_TOKEN` from the private config's `mlflow.tracking_uri` and `secrets.mlflow_token` (or requires the env vars if `private_config` is None) and returns an `MlflowClient`. This is one of the few places that reads the private `secrets` section directly.
- `MlFlowUpload` — experiment/run tag conventions (run_id, stage, hpc, uploader org via `common/platform_env.py`).
- `get_or_create_mlflow_parent_run()` — one parent MLFlow run per training `run_id`; training/eval phases attach as nested runs.
- `log_metrics()` — batch-uploads training metric dicts (keys prefixed `weathergen.` carry step/timestamp and are stripped).
- `log_scores()` — uploads evaluation scores as `score.<region>.<metric>.<stream>.<channel>` vs `forecast_step` (used by `evaluate --push-metrics`).

## packages/readers_extra — additional data readers

Extra dataset readers beyond the two built-ins (`obs`, `anemoi`) that live in the main package. Contents of `packages/readers_extra/src/weathergen/readers_extra/`:

| stream config `type` | class | module |
|---|---|---|
| `fesom` | `DataReaderFesom` | `data_reader_fesom.py` |
| `iconesm` | `DataReaderIconEsm` | `data_reader_icon_esm.py` |
| `iconart` | `DataReaderIconArt` | `data_reader_iconart.py` |
| `cams` | `DataReaderCams` | `data_reader_cams.py` |
| `grep` | `DataReaderGREP` | `data_reader_grep.py` |
| `mesh` | `DataReaderMesh` | `data_reader_mesh.py` |
| `anemoi_operan` | `DataReaderAnemoiOperan` | `data_reader_anemoi_operan.py` |

**Discovery**: `registry.py:get_extra_reader(stream_type)` is a simple `match` statement with lazy imports (no sanity checks — a broken reader fails at import time when first requested). The main pipeline hooks in at `src/weathergen/datasets/multi_stream_data_sampler.py:_init_stream_datasets()`: a stream's `type` is matched against the built-ins first, then `get_extra_reader()`; `None` raises "Unsupported stream type". **To add a reader**: implement it here subclassing the base classes from `src/weathergen/datasets/data_reader_base.py` (most extras subclass `DataReaderTimestep`; `DataReaderAnemoiOperan` subclasses `DataReaderAnemoi`), add a `case` to `registry.py`, and reference the new `type` in a stream config. See `src/weathergen/datasets/DOCS-Datasets.md` for the reader interface.

Note the readers import from the root package (`weathergen.datasets.*`, `weathergen.train.utils`) — the known circular dependency mentioned above.

## packages/dashboard — Streamlit dashboard (not installed)

Internal experiment-tracking dashboard (`dashboard.py` plus pages: `exp_tracker.py`, `atmo_training.py`, `atmo_eval.py`, `data_overview.py`, `data_sources.py`, `eng_overview.py`; helpers in `weathergen/dashboard/`). It is **not a workspace member** (deployment issues) and is a self-contained uv project with its own `uv.lock`, depending on `weathergen-common` and `weathergen-metrics` via relative `path` sources. Run from `packages/dashboard/` with `uv run --env-file=.env streamlit run dashboard.py` (see its README for deployment docs).

## packages/science

Not a package — a single standalone script, `packages/science/compute_spatial_autocorrelation.py` (PEP-723 inline deps, run with `uv run`), which estimates per-variable spatial correlation lengths in a dataset and emits suggested `masking_override` / `hl_mask` YAML snippets for stream configs.
