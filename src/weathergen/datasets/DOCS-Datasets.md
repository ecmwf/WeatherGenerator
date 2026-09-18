# DOCS-Datasets: The Data Pipeline (`src/weathergen/datasets/`)

One-line purpose: turns heterogeneous weather data (gridded reanalysis, point observations, ocean/atmosphere model output) into masked, tokenized, HEALPix-organized `ModelBatch` objects for training and inference.

Part of the agent docs rooted at `AGENT-README.md`. Stream configuration itself is documented in `config/streams/DOCS-Streams.md`.

Data flow in one sentence:

```
stream configs -> DataReader* (per stream, per file) -> ReaderData/IOReaderData (flat point arrays)
   -> TokenizerMasking (HEALPix cell tokens + masks from Masker)
   -> StreamData (one stream, one sample) -> Sample -> BatchSamples -> ModelBatch
   -> DataLoader (batch_size=None) -> pin_memory -> to_device -> model
```

---

## 1. File-by-file overview

### `data_reader_base.py` (~820 lines)
Reader abstractions and time indexing. `TimeWindowHandler` (~line 67) maps an integer time index (`TIndex`) to a `[start, end)` datetime window given `start_date`, `end_date`, `time_window_len`, `time_window_step`. `ReaderData` (~line 148) is the flat return type of readers: `coords [N,2] (lat,lon deg)`, `geoinfos [N,G]`, `data [N,C]`, `datetimes [N]`, plus `shuffle()` / `remove_nan_coords_and_geoinfos()` helpers. `DataReaderBase` (~line 296) is the abstract base (channel selection, normalization). `DataReaderTimestep` (~line 684) specializes it for datasets with a fixed period; `get_dataset_indexes_timestep()` (~line 761) converts a window index into dataset row indices.

### `data_reader_anemoi.py` (~340 lines)
`DataReaderAnemoi` (~line 35), reader for gridded zarr datasets opened via `anemoi.datasets.open_dataset`. Subclass of `DataReaderTimestep`. Handles channel selection from `source`/`target`/`*_exclude`/`geoinfo_channels` config keys, normalization statistics from the anemoi dataset, `frequency` subsetting, and lat/lon caching. Can be driven by an inline `anemoi_config` dict in the stream config instead of `filenames`.

### `data_reader_obs.py` (~300 lines)
`DataReaderObs` (~line 29), reader for scattered observations stored in a zarr with a `data` table (columns named in `colnames` attr), a `dates` array, and a precomputed hourly row index `idx_<base>_1`. Selects `obsvalue*` columns via substring matching, builds per-window start/end row indices in `_setup_sample_index()`, and enforces the `[t_start, t_end)` convention with a datetime mask in `_get()`. Normalization stats come from zarr attrs (`means`, `vars`).

### `multi_stream_data_sampler.py` (~840 lines)
The central `torch.utils.data.IterableDataset`: `MultiStreamDataSampler` (~line 94). Combines all streams, applies masking/tokenization, and yields fully-assembled `ModelBatch` objects. See section 2.

### `batch.py` (~500 lines)
Batch containers. `SampleMetaData` (~line 19): masking params + mask tensor per stream. `Sample` (~line 29): dict `stream_name -> StreamData` plus meta info; one "view" of the world. `BatchSamples` (~line 160): list of `Sample`s plus stacked `tokens_lens`. `ModelBatch` (~line 278): `source_samples` and `target_samples` (`BatchSamples` each) with `source2target_matching_idxs` / `target2source_matching_idxs` linking views (1-to-1 for MTM/forecast, many-to-one for student-teacher). All levels implement `pin_memory()`, `to_device()`, and `*_empty()` / `*_nan()` validity checks.

### `stream_data.py` (~500 lines)
`StreamData` (~line 49): all tensors the model ingests for one stream in one sample — per input step `source_tokens_cells` (stacked tokens per HEALPix cell), `source_tokens_lens`; per output step `target_tokens`, `target_coords` (the ~105-dim local coordinate encoding), `target_coords_lens`, `idxs_inv` (inverse permutation to restore input order, kept only outside TRAIN). `spoof()` (~line 462) fabricates a minimal 2-point dataset for empty windows — a workaround for pytorch#158719 so no tensor is ever truly empty; spoofed steps are flagged via `source_is_spoof`/`target_is_spoof`.

### `masking.py` (~790 lines)
`Masker` (~line 86) generates boolean *keep* masks over HEALPix cells (True = keep). `MaskData` (~line 19) is a list of masks + `SampleMetaData`. `Masker.build_samples_for_stream()` (~line 333) builds all target and source masks for one stream from the mode config (`model_input` / `target_input` sections) plus the per-stream `masking_override`, and returns `(target_masks, source_masks, source_target_mapping)`. Also parses `target_source_correspondence` from the loss config (~line 243). See section 4.

### `tokenizer.py` (~140 lines)
`Tokenizer` (~line 22), base class holding precomputed HEALPix geometry: cell vertex positions and rotation matrices (`healpix_verts_rots`) at 5 anchor points per cell (4 corners + center), local vertex coordinates, and neighbor-center coordinates used in target coordinate encoding. Also defines `size_time_embedding = 6`.

### `tokenizer_masking.py` (~220 lines)
`TokenizerMasking` (~line 42), the tokenizer actually instantiated by the sampler. Wraps a `Masker` and glues masks to tokenization: `get_tokens_windows()` tokenizes each time window once (amortized across views), `cell_to_token_mask()` expands cell-level masks to token-level, and `get_source()` / `get_target_coords()` / `get_target_values()` produce the masked, encoded tensors that go into `StreamData`.

### `tokenizer_utils.py` (~520 lines)
Free functions doing the heavy lifting: `tokenize_space()` / `tokenize_spacetime()` (~lines 176/191) bucket points into HEALPix cells (via `ang2pix`) and split them into fixed-size tokens (padding with index 0, which points to a prepended zero row); `tokenize_apply_mask_source()` (~line 223) and `tokenize_apply_mask_target()` (~line 310) apply token masks and assemble the final token tensors `[stream_id | time_enc(5) | local_coords | geoinfos | data]`; `encode_times_source()` / `encode_times_target()` (~lines 27/63) build the 5-dim time encodings; `get_target_coords_local()` (~line 421) builds the large per-point target coordinate encoding (`geoinfo + 5*(3*5) + 3*8 (+ stream_id, times)` — matches `MultiStreamDataSampler.get_targets_coords_size()`).

### `utils.py` (~290 lines)
Spherical geometry helpers used by the tokenizer: `s2tor3`/`r3tos2` (sphere <-> R^3), `vecs_to_rots` (Rodrigues rotation aligning a cell center with the origin), `healpix_verts_rots`, `locs_to_cell_coords_ctrs`, `locs_to_ctr_coords`. Also `get_tokens_lens()` (~line 260), which stacks `source_tokens_lens` across (steps, samples, streams) for `ModelBatch.tokens_lens`. (`utils_test.py` holds its unit tests.)

### `memory_pinning.py` (42 lines)
`Pinnable` runtime-checkable protocol (anything with `pin_memory()`) and a recursive `pin_object()` helper for tensors/lists/dicts/`IOReaderData`. See section 5.

Note on `ReaderData` vs `IOReaderData`: `weathergen.common.io.IOReaderData` is a structurally identical twin of `ReaderData` living in `packages/common` (which must not depend on the core model). `collect_datasources()` in the sampler combines reader outputs via `IOReaderData.combine()`, so downstream code (tokenizer, `StreamData`) sees `IOReaderData`.

---

## 2. `MultiStreamDataSampler` in detail

Constructed in `train/trainer.py` (one instance per stage: `TRAIN` with `training_config`, `VAL` with `validation_config`) as `MultiStreamDataSampler(cf, mode_cfg, stage)`. It is an `IterableDataset`; the `DataLoader` is created with `batch_size=None` — no collate function, the sampler yields complete `ModelBatch` objects itself.

### Stream setup (`_init_stream_datasets`, ~line 220)
For every entry in `cf.streams`, resolves the reader class from `stream_info["type"]`:
- `"obs"` -> `DataReaderObs`
- `"anemoi"` -> `DataReaderAnemoi`
- anything else -> `weathergen.readers_extra.registry.get_extra_reader(type)`; `ValueError` if unknown.

One reader is created *per filename* in `stream_info["filenames"]` (paths resolved against `cf.data_paths`), so a stream is a `_Stream(info, readers: list)` (~line 88). Multiple readers per stream are concatenated point-wise at load time by `collect_datasources()` (~line 52), which also applies normalization, `shuffle_source`/`shuffle_target`, and `max_num_targets` subsetting. The resolved source/target channels are written back into `stream_info` (`<stage>_source_channels` etc.) so validation reuses the training channel selection.

### Sampling windows and mini-epochs
Time is discretized by `TimeWindowHandler` using `mode_cfg.start_date/end_date/time_window_len/time_window_step`; a sample is identified by a window index `idx`. `reset()` (~line 282) reseeds the RNG and builds a permutation of valid indices (`_calc_baseperms` excludes indices that cannot accommodate `max_input_steps` history or the forecast horizon), optionally tiling it when `data_loading.repeat_data_in_mini_epoch` is set. It also draws per-batch forecast step counts according to `forecast.policy` (`None`, `"fixed"`/`"sequential"`, `"random"`/`"sequential_random"`); `forecast.num_steps` may be a list indexed by mini-epoch. `check_samples()` (~line 156) clamps `samples_per_mini_epoch` to what the date range supports. `advance()` bumps `mini_epoch` on the template object between mini-epochs.

### Distributed sharding (`worker_workset`, ~line 808)
`len(self)` is the per-rank sample count: `((samples_per_mini_epoch // world_size) // batch_size) * batch_size`. Each DDP rank owns `[rank*len, (rank+1)*len)` and each DataLoader worker takes an equal slice of that. Worker processes are bit-wise copies of the template, so the RNG seed is re-derived per (DDP rank, worker id, mini_epoch) inside `worker_workset()` — if you touch seeding, keep it unique across all three.

### Batch assembly (`_get_batch`, ~line 652)
For a window index `idx` and forecast step count:
1. `_get_source_target_masks()` builds per-stream source/target HEALPix masks via the `Masker`.
2. The `training_mode` string selects what gets built: `"masking"` -> sources get `network_input` + `target_coords`, targets get `target_values`; `"student_teacher"` / `"latent_loss"` -> both sides get `network_input` (targets are teacher views). Modes are substring-matched and combinable.
3. Per stream: `_get_data_windows()` loads raw data for all input steps (`idx - num_steps_input + 1 .. idx`) and all output steps (`idx + offset .. idx + offset + num_forecast_steps`, stepped by `forecast.time_step`), spoofing empty windows; `get_tokens_windows()` tokenizes each window once; then `_build_stream_data()` produces one `StreamData` per source mask and per target mask, wired into the `ModelBatch` via `add_source_stream` / `add_target_stream` with the `source_to_target` mapping.
4. `_preprocess_model_batch()` stacks `tokens_lens`.

`__iter__` (~line 761) then loops over its index slice, drawing `idx` from the permutation, and *skips invalid batches* (all-empty sources, all-NaN, or empty targets in masking mode) by advancing to the next index — so emitted batch count is exact but data indices may drift past the nominal slice.

### What a batch looks like
One `ModelBatch` = one sample (per-rank batch semantics live in the model, which consumes `tokens_lens` shaped `[steps, samples, streams]`):
- `batch.source_samples: BatchSamples` — `num_source_samples` `Sample`s (e.g. student views); each `Sample.streams_data[name]` is a `StreamData` with `source_tokens_cells[step]` = tensor `[num_tokens_total, token_size, channels]` stacked per HEALPix cell and `source_tokens_lens[step]` = `[num_cells]` int32.
- `batch.target_samples: BatchSamples` — target/teacher views with `target_tokens[fstep]`, `target_coords[fstep]`, `target_coords_lens[fstep]`, `idxs_inv[fstep]`.
- `source2target_matching_idxs` / `target2source_matching_idxs` connect views for the losses.

---

## 3. Reader architecture: adding a new reader

Hierarchy:

```
DataReaderBase (data_reader_base.py ~296)          # abstract
├── DataReaderObs (data_reader_obs.py ~29)         # scattered observations, zarr
└── DataReaderTimestep (data_reader_base.py ~684)  # fixed-period datasets
    ├── DataReaderAnemoi (data_reader_anemoi.py ~35)   # anemoi-datasets zarr, gridded
    │   └── DataReaderAnemoiOperan (readers_extra)
    ├── DataReaderFesom     (readers_extra)  # FESOM ocean, type: "fesom"
    ├── DataReaderIconEsm   (readers_extra)  # type: "iconesm"
    ├── DataReaderIconArt   (readers_extra)  # type: "iconart"
    ├── DataReaderCams      (readers_extra)  # type: "cams"
    ├── DataReaderGREP      (readers_extra)  # type: "grep"
    └── DataReaderMesh      (readers_extra)  # type: "mesh"
```

Extra readers live in `packages/readers_extra/src/weathergen/readers_extra/` (a separate uv workspace package), **not** in this directory. They are resolved by name in `registry.py` (`get_extra_reader(stream_type)`) with lazy imports; `MultiStreamDataSampler._init_stream_datasets` falls through to this registry for any `type:` it does not recognize. There is no entry-point magic: to register a reader you add a `case` to the `match` in `packages/readers_extra/src/weathergen/readers_extra/registry.py`.

To implement a reader an agent must:
1. Subclass `DataReaderBase` (or `DataReaderTimestep` for fixed-frequency data — it provides `_get_dataset_idxs()` for window-index -> row-index translation).
2. Match the constructor signature used by the sampler: `__init__(self, filename: Path, tw_handler: TimeWindowHandler, stream_info: dict, stage: Stage)` (passed as kwargs plus `filename`).
3. Set the abstract attributes: `source_channels`, `target_channels`, `geoinfo_channels` (names), `source_idx`, `target_idx`, `geoinfo_idx` (column indices), `target_channel_weights`, plus normalization stats `mean`, `stdev`, `mean_geoinfo`, `stdev_geoinfo`. Call `init_empty()` and return early if the dataset does not overlap the requested time range.
4. Implement `length()` and `_get(idx, channels_idx) -> ReaderData`. `_get` must return flat per-point arrays with lat in [-90, 90], lon in [-180, 180], and all datetimes inside `[window.start, window.end)` — validate with `check_reader_data(rdata, dtr)`. Return `ReaderData.empty(...)` for windows without data (the sampler spoofs them).
5. Register the class in `registry.py` and pick a `type:` name for stream configs; put data-specific options in the stream YAML (see `config/streams/DOCS-Streams.md`).

Base-class services you get for free: `get_source`/`get_target` (channel-sliced `_get`), normalize/denormalize for source/target/geoinfo channels, `normalize_coords`, `parse_target_channel_weights()` (reads `channel_weights` from the stream config).

---

## 4. Tokenization and masking

### Raw data -> HEALPix tokens
All data — gridded or scattered — is treated as a point cloud. `tokenize_space()` assigns each point to a HEALPix cell at `cf.healpix_level` (default 5 -> 12,288 cells, nested ordering) via `ang2pix`, sorts within cells by latitude, and chunks each cell's points into tokens of `stream_info["token_size"]` points, zero-padded to full size (padding indices point at a prepended all-zero row, so no masking logic is needed for pads). `tokenize_spacetime()` (enabled by `tokenize_spacetime: true` in the stream config) does this per unique timestamp so a token never mixes times. The result is `idxs_cells` (per cell: list of index tensors into the flat point array) + `idxs_cells_lens`, computed once per window in `TokenizerMasking.get_tokens_windows()` and shared across all views.

A source token row is `[stream_id, time_enc(5), local_coords(2, cell-relative via rotation), geoinfos, data]` (`tokenize_apply_mask_source`). Target coordinates are encoded per point relative to the 5 cell anchor vertices and the 8 neighbor cell centers (`get_target_coords_local`), giving the `geoinfo + 5*(3*5) + 3*8 + 6` layout hard-coded in `MultiStreamDataSampler.get_targets_coords_size()` (~line 369) — if you change the coordinate encoding, update both.

### Masking
Masks are boolean *keep* arrays over the 12·4^L HEALPix cells, generated by `Masker` per stream and per view, then expanded to token level by `TokenizerMasking.cell_to_token_mask()`. Strategies (`Masker._generate_cell_mask`, ~line 514):
- `"random"` — iid per-cell keep with rate `rate` (optionally noisy via `rate_sampling: true`).
- `"healpix"` — mask whole parent cells at a coarser level `masking_strategy_config.hl_mask`; all children of a kept parent are kept (block masking).
- `"cropping_healpix"` — spatially contiguous crops (for JEPA/DINO-style views); `masking_strategy_config.method` is `"geodesic_disk"` (default, circular), `"disk"` (frontier growth), or `"random_walk"`.
- `"forecast"` / `"causal"` — keep everything (mask of ones); temporal separation does the work. Supports `diffusion_rn` in the strategy config to sample a noise level into the mask params.

Source masks can be tied to their target mask via a relationship (`Masker._get_mask`, ~line 447): `complement` (default for `random` — masked-token modeling), `identity`, `subset`, `disjoint`, `independent` (default for forecasting). The relationship and the source->target wiring come from `target_source_correspondence` inside the loss config, parsed by `parse_src_target_correspondence()`.

Where it is configured: the mode config (`training_config` / `validation_config`) defines `model_input` and `target_input` sections — each a dict of named strategies with `masking_strategy`, `masking_strategy_config`, `num_samples`, `num_steps_input`. Streams can override strategy fields (not structure) via `masking_override` in their YAML, merged in `Masker.merge_masking_config()` (~line 139); `randomly_drop_as_source_rate` drops an entire stream from the source with the given probability (training only). Diagnostic streams get empty source masks, forcing streams get empty target masks (`is_stream_diagnostic` / `is_stream_forcing`). Per-stream options are documented in `config/streams/DOCS-Streams.md`.

---

## 5. Memory pinning and performance notes

- **Pinning path.** `ModelBatch.pin_memory()` -> `BatchSamples` -> `Sample` -> `StreamData.pin_memory()` pins every tensor to page-locked host memory so the subsequent `to_device(..., non_blocking=True)` transfers are async. It is triggered two ways: the DataLoader's `pin_memory=cf.data_loading.memory_pinning` flag (PyTorch calls `pin_memory()` on the yielded `ModelBatch` because it exposes the method), and an explicit `batch.pin_memory()` in the trainer for the prefetched next batch (`train/trainer.py` ~line 454). `memory_pinning.py`'s `Pinnable`/`pin_object` is a generic recursive helper (handles `IOReaderData`, whose tensors may be numpy); the batch classes currently implement pinning directly.
- **Tokenization is amortized.** Windows are tokenized once per stream (`get_tokens_windows`) and reused for every source/target view; keep it that way when adding views.
- **Empty-batch resilience costs time.** Empty windows are spoofed (2-point dummy data) rather than skipped per-stream; fully empty/NaN batches are skipped in `__iter__` by loading the next index — a data-sparse period can silently multiply I/O.
- **Training-stage stripping.** In TRAIN, `StreamData.add_source` drops `source_raw` and `add_target_values` drops `idxs_inv` to reduce host memory / IPC volume; don't rely on these fields during training.
- **Worker cost.** Each DataLoader worker holds its own copy of every reader (zarr handles are cheap, but anemoi metadata is not free); `persistent_workers` is available via `cf.data_loading.persistent_workers`.
- **NaN handling.** Coordinates/geoinfos with NaN are dropped in `collect_datasources`; NaN *data* values in source tokens are replaced by `mask_value = 0.0` in `StreamData.add_source` (post-normalization zero = mean).
