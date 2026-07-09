# DOCS-Streams: Stream Configuration

Stream configs define the data sources (reanalyses, observations, ocean/climate model output) a
WeatherGenerator run trains on — one YAML block per stream, one directory per stream *set*.

Part of the agent docs rooted at [`AGENT-README.md`](../../AGENT-README.md). For the general
config system (merge order, private config, CLI overrides) see
[`config/DOCS-Config.md`](../DOCS-Config.md).

## How streams are loaded

- `streams_directory` in the main config selects a stream set
  (`config/default_config.yml:97` → `"./config/streams/era5_1deg/"`).
- `load_streams()` in `packages/common/src/weathergen/common/config.py` recursively globs
  `[!.#]*.yml` in that directory. Every **top-level YAML key** in every file becomes one stream;
  the key is the stream name (injected as `stream_config.name`) and must be unique across the
  whole directory. An empty (fully commented-out) file is skipped with a warning.
- `_load_streams_in_config()` runs on the base config and on every `--config` overwrite. If any
  overwrite sets `streams_directory`, the inherited `streams` are cleared and replaced (see
  `load_config()` in the same file) — stream sets replace each other, they do not merge.
- `load_streams()` also maintains a rename history for old directory names
  (`streams_anemoi` → `era5_1deg`, `streams_mixed` → `era5_nppatms_synop`,
  `streams_ocean` → `fesom`, `streams_icon` → `icon`,
  `streams_mixed_experimental` → `cerra_seviri`), so configs from old runs still resolve.
- At runtime `MultiStreamDataSampler._init_stream_datasets()`
  (`src/weathergen/datasets/multi_stream_data_sampler.py`) instantiates one reader per entry in
  `filenames` for each stream.

## Available stream sets

| Directory | Contents |
|---|---|
| `era5_1deg/` | ERA5 reanalysis at o96 (~1 deg), single `anemoi` stream; the default set |
| `era5_1deg_forecasting/` | Same ERA5 zarr, tuned for forecasting (per-channel `channel_weights`, `max_num_targets: 20000`, larger embed dims) |
| `era5_nppatms_synop/` | ERA5 + NPP ATMS satellite radiances + SYNOP surface obs (`anemoi` + 2x `obs`) |
| `era5_decoding_synop/` | ERA5 + SYNOP for decoding experiments |
| `era5_synop_finetuning/` | ERA5 as `forcing: True` + SYNOP as `diagnostic: True` (fine-tune a SYNOP decoder) |
| `era5_iasi_finetuning/` | ERA5 + Metop-B IASI radiances fine-tuning |
| `era5_georing_avhrr/` | ERA5 in/out pair + geostationary ring (SEVIRI-like) + AVHRR; `era5_out.yml` is a second diagnostic ERA5 stream with `masking_override` |
| `operan_georing_avhrr_synop_lowres/` | Operational analysis (`anemoi_operan`) + geo-ring + AVHRR + SYNOP + ERA5, low-res |
| `cerra_seviri/` | CERRA regional reanalysis (`anemoi`) + SEVIRI obs |
| `cams/` | CAMS atmospheric composition (EAC4 + analysis), `cams` reader |
| `igra/` | IGRA radiosonde archive (`obs` reader, custom `base_datetime`) |
| `fesom/` | FESOM ocean model nodes/elements + IFS atmosphere (`fesom` reader) |
| `eerie_native/`, `eerie_gridded/`, `eerie_downscaling/` | EERIE IFS-FESOM output on native/gridded meshes (`mesh` reader; downscaling uses gridded source → native target) |
| `icon_esm_historical_day/`, `icon_esm_historical_mon/` | ICON-ESM historical CMIP output, daily/monthly (`iconesm` reader, one stream per CMIP table) |
| `iconart_6h/` | ICON-ART composition run (`iconart` reader) |

`config/evaluate/` and `config/profiling/` have their own configs; they reuse these stream sets.

## Stream YAML schema

A minimal gridded stream (`era5_1deg/era5.yml`, abridged):

```yaml
ERA5 :                      # stream name (must be unique in the set)
  type : anemoi             # reader selection, see "Reader types"
  filenames : ['aifs-ea-an-oper-0001-mars-o96-1979-2024-1h-v3-with-era51.zarr']
  stream_id : 0
  source_exclude : ['z', 'w_10', ...]   # drop channels from model input
  target_exclude : ['z', 'w_10', ...]   # drop channels from prediction targets
  geoinfo_channels : ['z', 'lsm', 'slor', 'sdor', 'insolation', ...]
  loss_weight : 1.
  location_weight : cosine_latitude
  token_size : 8
  tokenize_spacetime : True
  max_num_targets: -1
  frequency : 06:00:00
  embed :
    net : transformer
    num_tokens : 1
    num_heads : 8
    dim_embed : 256
    num_blocks : 2
  embed_target_coords :
    net : linear
    dim_embed : 256
  target_readout :
    num_layers : 2
    num_heads : 4
  pred_head :
    ens_size : 1
    num_layers : 1
```

An observation stream differs mainly in reader type and channel defaults
(`era5_nppatms_synop/synop.yml`: `type: obs`, no explicit channel lists → all zarr columns), and
an unstructured-mesh stream adds explicit channel lists and sampling controls
(`fesom/fesom.yml`: `type: fesom`, `filenames: ['ocean_node']` resolving to a zarr directory,
`source: null` → all, `target: ["sst", "sss", "ssh", "temp_", "salt_"]`;
`eerie_native/eerie_ocean_node.yml`: `type: mesh`, absolute path in `filenames`,
`sampling_mode: 'global_sparse'`, `sample_points: 65536`, `patch_size_deg: null`).

### Identity and data location

| Key | Meaning (where consumed) |
|---|---|
| `type` | Reader selection. Required. (`multi_stream_data_sampler.py:_init_stream_datasets`) |
| `filenames` | List of dataset names/paths; one reader instance per entry. Each entry is used as-is if it exists as a path, otherwise it is resolved against every entry of the top-level `data_paths` list (which comes from the **private config**; legacy keys `data_path_anemoi`, `data_path_obs`, ... are collected into `data_paths` by `_check_datasets` in `common/config.py`). Missing data raises `FileNotFoundError`. |
| `stream_id` | Numeric ID written as the first feature of every token (`tokenizer_utils.py:299`). Distinguishes streams for the model; not used for file lookup. |
| `anemoi_config` | (anemoi reader only) Full `anemoi.open_dataset` dict; if set, `filenames` is ignored (`data_reader_anemoi.py:62`). |
| `target_file` | (fesom/mesh readers) Separate target dataset path — source and target can come from different files (`data_reader_fesom.py:51`, `data_reader_mesh.py:49`). |

### Channel selection

| Key | Meaning |
|---|---|
| `source` / `target` | Explicit include lists for input/target channels. `null`/absent → all available channels; `[]` → none (stream becomes forcing/diagnostic implicitly, see below). |
| `source_exclude` / `target_exclude` | Channels to drop (applied after the include list; `data_reader_anemoi.py:select_channels`, `data_reader_obs.py`). |
| `geoinfo_channels` | Static/auxiliary channels (orography, land-sea mask, insolation, ...) appended to every token as geo-info features, normalized separately. |
| `channel_weights` | Per-channel loss weight map `{channel: weight}`; unspecified channels get 1.0. Validated against target channels (`data_reader_base.py:490-504`), consumed via `target_channel_weights` in `train/loss_modules/loss_module_physical.py`. |
| `coords_channels` | (obs reader) Names of the two coordinate columns, default `["lat", "lon"]` (`data_reader_obs.py:77`). |
| `base_datetime` | (obs reader) Epoch for the zarr time axis, default `1970-01-01T00:00:00`; e.g. IGRA uses 1750 (`data_reader_obs.py:39`). |
| `variables`, `pressure_levels` | (cams reader) Column names and level list (`data_reader_cams.py:69,90`). |
| `channels`, `plev`, `lev`, `depth` | (iconesm reader) Variable list and vertical-level selections per coordinate type (`data_reader_icon_esm.py:67-73`). |
| `attributes` | (iconart reader) Names of lon/lat/grid attributes in the dataset (`data_reader_iconart.py:86-88`). |

### Time and subsampling

| Key | Meaning |
|---|---|
| `frequency` | (anemoi) Temporal subsampling of the dataset, `HH:MM:SS` (parsed to timedelta by the config system, passed to `anemoi_datasets.open_dataset`; `data_reader_anemoi.py:84`). |
| `subsampling_rate` | Deprecated for anemoi — only triggers a warning telling you to use `frequency` (`data_reader_anemoi.py:87`). |
| `max_num_targets` | Cap on the number of target points per sample (random subset); `-1` = no cap (`multi_stream_data_sampler.py:72`). |
| `shuffle_source` / `shuffle_target` | Shuffle points before (sub)sampling, default `False` (`multi_stream_data_sampler.py:68,73`). |
| `sampling_mode`, `sample_points`, `patch_size_deg` | (mesh reader) Spatial sampling: `"patch"` (default), `"global_sparse"` (random `sample_points` per step, default 4096), `"regular"` (`data_reader_mesh.py:56-77,349-359`). |
| `nominal_time_mapping` | (anemoi_operan reader) Mapping to nominal analysis times (`data_reader_anemoi_operan.py:118`). |

### Stream role flags

| Key | Meaning |
|---|---|
| `diagnostic` | `True` → stream is decoded/evaluated but **not fed into the encoder** (no embedding network is built, `model/engines.py:54`; source mask forced empty, `datasets/masking.py:424`). Also implied by empty source channels (`utils/utils.py:is_stream_diagnostic`). |
| `forcing` | `True` → stream is model **input only**, no predictions/decoder/loss (`model/model.py:404`; target mask empty, `masking.py:370`). Also implied by empty target channels (`utils/utils.py:is_stream_forcing`). |

(One file, `era5_decoding_synop/synop.yml`, uses `is_diagnostic` — that key is not
read anywhere; the consumed key is `diagnostic`.)

### Loss

| Key | Meaning |
|---|---|
| `loss_weight` | Scalar weight of this stream in the total loss, default 1.0 (`train/loss_modules/loss_module_physical.py:140`). |
| `location_weight` | Name of a per-location weighting function in `train/loss_modules/loss_functions.py`, e.g. `cosine_latitude` (`loss_module_physical.py:174-180`). |

### Tokenization

| Key | Meaning |
|---|---|
| `token_size` | Max points per token within a HEALPix cell (`tokenizer_masking.py:65`, `tokenizer_utils.py`). Gridded streams use small values (8–16), dense obs use 64+. |
| `tokenize_spacetime` | `True` → tokenize jointly over space and time (`tokenize_spacetime`); `False`/absent → per-timestep spatial tokenization (`tokenize_space`) (`tokenizer_masking.py:62`). |

### Masking

Masking is configured **globally** per training mode in the main config
(`training_config.model_input` / `target_input` sections with `masking_strategy`,
`masking_strategy_config: {rate, rate_sampling, hl_mask, ...}`; strategies: `"random"`,
`"healpix"`, `"cropping_healpix"`, `"forecast"` (alias `"causal"`) — see
`src/weathergen/datasets/masking.py`).
Per-stream, only one key matters:

| Key | Meaning |
|---|---|
| `masking_override` | Per-stream override of the global masking config, flat per section (`model_input` / `target_input`); can replace `masking_strategy` and deep-merge `masking_strategy_config` fields, and set `randomly_drop_as_source_rate` (`masking.py:merge_masking_config`, `build_effective_masking_cfgs`). Example in `era5_georing_avhrr/era5_out.yml` (forces `target_input.masking_strategy_config.rate: 1.0`). |

**Note:** `masking_rate` and `masking_rate_none` still appear in many stream YAMLs but are
**dead keys** — nothing in `src/` or `packages/` reads them anymore. Do not add them to new
streams; use the global masking sections plus `masking_override`.

### Model components (per-stream networks)

| Key | Meaning |
|---|---|
| `embed` | Source-token embedding network (`model/engines.py:EmbeddingEngine`, ~line 54). `net`: `transformer` or `linear`. For `transformer`: `num_tokens`, `num_heads`, `dim_embed`, `num_blocks`. Skipped for diagnostic streams. |
| `embed_target_coords` | Embedding of target coordinates for the decoder (`model/model.py:411,430-446`). `net`: `linear` or `mlp`; `dim_embed` sets the decoder width. |
| `target_readout` | Cross-attention readout / TargetPredictionEngine (`model/model.py:412-418`, `model/engines.py:732`). Keys: `num_layers`, `num_heads`, optional `mlp_hidden_factor` (default 2), `dim_head_proj`, `softcap`. Some shared engines take `num_heads` from the *first* stream in the config (`model/blocks.py:204`). The commented-out `sampling_rate` seen in many YAMLs is not consumed anywhere. |
| `pred_head` | Final prediction head (`model/model.py:477-490`, `EnsPredictionHead`): `num_layers`, `ens_size` (ensemble members for probabilistic output), optional `final_activation` (default `Identity`). |
| `pred_spatial_shared` | Name of another stream whose coord-embedding and target-readout this stream reuses (own `pred_head` is still built); must reference an existing, different stream (`model/model.py:498-517`). |

The reader also writes derived keys back into the stream config at runtime
(`train_source_channels`, `val_target_channels`, `target_channel_weights`, `data_paths`, `name`)
— you will see these in saved run configs; never set them by hand.

## Reader types

Resolution: `MultiStreamDataSampler._init_stream_datasets()` matches `stream_info["type"]` —
`"obs"` and `"anemoi"` are built in (`src/weathergen/datasets/`); anything else is looked up via
`get_extra_reader()` in `packages/readers_extra/src/weathergen/readers_extra/registry.py`
(lazy imports). Unknown types raise `ValueError`.

| `type` | Class (file) | For |
|---|---|---|
| `anemoi` | `DataReaderAnemoi` (`src/weathergen/datasets/data_reader_anemoi.py`) | Gridded zarr via anemoi-datasets (ERA5, CERRA, ...) |
| `obs` | `DataReaderObs` (`src/weathergen/datasets/data_reader_obs.py`) | Point observations in zarr (SYNOP, IASI, ATMS, IGRA, SEVIRI-ring) |
| `fesom` | `DataReaderFesom` (`packages/readers_extra/.../data_reader_fesom.py`) | FESOM ocean node/element zarr groups |
| `mesh` | `DataReaderMesh` (`.../data_reader_mesh.py`) | Unstructured meshes from parquet (EERIE), patch/sparse/regular sampling |
| `iconesm` | `DataReaderIconEsm` (`.../data_reader_icon_esm.py`) | ICON-ESM CMIP output (kerchunk JSON indexes) |
| `iconart` | `DataReaderIconArt` (`.../data_reader_iconart.py`) | ICON-ART composition zarr |
| `cams` | `DataReaderCams` (`.../data_reader_cams.py`) | CAMS composition zarr |
| `anemoi_operan` | `DataReaderAnemoiOperan` (`.../data_reader_anemoi_operan.py`) | Operational analysis with nominal-time mapping |
| `grep` | `DataReaderGREP` (`.../data_reader_grep.py`) | GREP ensemble reanalysis |

All readers subclass `DataReaderBase` (`src/weathergen/datasets/data_reader_base.py`), which
handles time windows, normalization statistics, and channel-weight validation. A stream whose
time range does not overlap the training window is skipped with a warning (`init_empty`).

## Adding a new stream

1. **Create a directory** `config/streams/<my_set>/` (or extend an existing one) and add
   `<my_stream>.yml`. Every YAML file in the directory gets loaded — copy only the streams you
   want.
2. **Minimum keys**: top-level stream name, `type`, `filenames`, `token_size`, `loss_weight`,
   `embed`, `embed_target_coords`, `target_readout`, `pred_head`. Add `stream_id` (unique per
   set), channel selections, and `tokenize_spacetime` as appropriate. Start from
   `era5_1deg/era5.yml` (gridded) or `era5_nppatms_synop/synop.yml` (obs).
3. **Point the run at it**: set `streams_directory: "./config/streams/<my_set>/"` in your
   experiment config (an overwrite passed via `--config` fully replaces the inherited stream
   set), or override on the CLI.
4. **Data paths**: keep `filenames` as bare dataset names; the machine-specific `data_paths`
   list from the private config (`WeatherGenerator-private`, selected via
   `WEATHERGEN_PRIVATE_CONF` / `platform-env.py`) is prepended at load time. Absolute paths in
   `filenames` also work (used by the EERIE set) but tie the config to one machine. The data
   file must exist on the target machine or startup fails with `FileNotFoundError`.
5. If no existing reader fits, add one under `packages/readers_extra/.../` subclassing
   `DataReaderBase` and register its `type` string in `registry.py:get_extra_reader`.

## Masking / tokenizer interactions worth knowing

- `TokenizerMasking` (`src/weathergen/datasets/tokenizer_masking.py`) tokenizes each stream once
  per sample (`get_tokens_windows`, using `token_size` + `tokenize_spacetime`), then applies
  cell-level masks from `Masker` (`src/weathergen/datasets/masking.py`); cell masks are expanded
  to token-level masks in `cell_to_token_mask`.
- Masks are generated at HEALPix-cell granularity. With `masking_strategy: "healpix"` and
  `hl_mask: <level>` the mask is drawn at a coarser HEALPix level and broadcast to all child
  cells of the data-level grid (`healpix_level: 5` in `default_config.yml` → 12,288 cells).
- Forcing streams get an empty *target* mask, diagnostic streams an empty *source* mask
  (`masking.py:369-431`) — so the `diagnostic`/`forcing` flags and empty channel lists directly
  shape masking behavior.
- `rate` is a **keep/sampling rate** inside `masking_strategy_config`; with
  `rate_sampling: True` it is resampled per call from a clipped normal around the configured
  value (`masking.py:_get_sampling_rate`).
- Multiple source/target mask strategies can be combined per training mode; source-target
  correspondence is validated (`one-to-one`, `equal-split-all`; `masking.py:64-78`). Per-stream
  deviations always go through `masking_override`, never per-stream rate keys.
