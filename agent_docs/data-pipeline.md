# Data pipeline — from stream configs to ModelBatch

Summary: stream YAMLs under `config/streams/` define what data a run reads. Per stream,
`MultiStreamDataSampler` opens one reader per file (built-in `obs`/`anemoi` types or a
`packages/readers_extra` type via its registry), windows the data in time, tokenizes it
onto the healpix grid, applies masks per training mode, and yields fully assembled
`ModelBatch` objects — the DataLoader does no collation (`batch_size=None`).

## Streams (`config/streams/`, loaded in `common/config.py`)

- `run_train.py` loads `cf.streams = config.load_streams(cf.streams_directory)`; an
  overwrite config with `streams_directory` set replaces inherited streams entirely
  (`load_merge_configs`, config.py:441).
- Each stream entry carries `type`, `filenames`, and channel/normalization info that
  readers consume.

## Readers (`multi_stream_data_sampler.py:_init_stream_datasets`)

- Reader class by `stream_info.type`: `obs` → `DataReaderObs`, `anemoi` →
  `DataReaderAnemoi`, anything else → `readers_extra/registry.py:get_extra_reader`
  (lazy import; unknown type raises ValueError, a broken reader fails at first use).
- `filenames` resolve against `cf.data_paths` unless the path exists as given; a stream
  may have several files → several readers.
- The reader defines `source_channels`, `target_channels`, `target_channel_weights`;
  these are written back into `stream_info` (in-place config mutation).
- To add a reader: `agent_docs/recipes/add-data-reader.md`.

## Batch assembly (`_get_batch` → `__iter__`)

1. Masks per `training_mode` via `_get_source_target_masks` /
   `datasets/masking.py:Masker` (strategies e.g. `random`, healpix-cell and
   `geodesic_disk` variants; per-target overrides via `masking_strategy_config`).
   `source_to_target` maps each source (student) view to its target.
2. `_get_data_windows`: input/output time windows around the sample index;
   `num_forecast_steps` is drawn per batch and constant within it (`reset()`).
3. `tokenizer.py:Tokenizer.get_tokens_windows` tokenizes each window (healpix cells).
4. `_build_stream_data` assembles per-(stream, mask) `StreamData`; what goes into
   sources vs targets depends on `training_mode`: `"masking"` → inputs +
   `target_coords` as sources, `target_values` as targets; `"student_teacher"` /
   `"latent_loss"` → `network_input` on both sides.
5. `ModelBatch` (`datasets/batch.py`) collects source/target streams;
   `_preprocess_model_batch` computes `tokens_lens` for varlen attention.
6. `__iter__`: per-worker range from `worker_workset`; empty or NaN batches are
   skipped with a warning (targets-empty only invalidates `"masking"` mode).

## Consumption

- `Trainer` moves the batch (`to_device`, optional `pin_memory` via
  `datasets/memory_pinning.py`) and calls `model(model_params,
  batch.get_source_samples())`; target-aux calculators get
  `batch.get_target_samples(idxs)`. See `agent_docs/training-step.md`.
- Validation output is denormalized via `dataset.denormalize_target_channels` unless
  `output.normalized_samples` is set.

## Coupling & invariants

- `stream_info.type` string couples stream YAMLs ↔ the sampler match ↔ the
  readers_extra registry. New type = registry case + stream config.
- `num_steps_input` must be constant across all `model_input` entries (assert in
  `_get_batch`).
- `stream_info` is mutated at runtime (`data_paths`, per-stage channels, weights) —
  stream configs are not read-only records.
- Batch validity rules differ by `training_mode` (`__iter__`); changing modes → check
  the skip conditions.
- `advance()` must be called after each (mini-)epoch so validation subsets rotate
  (`trainer.py` does this for both datasets).
