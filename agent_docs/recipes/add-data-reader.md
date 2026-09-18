# Recipe: add a data reader for a new source

Use when: a new dataset type needs to be read into training (anything that isn't
already covered by the built-in `obs`/`anemoi` readers).

1. Implement `packages/readers_extra/src/weathergen/readers_extra/data_reader_<name>.py`.
   Subclass the appropriate base from `src/weathergen/datasets/data_reader_base.py`
   (existing readers there and in `readers_extra/` are the templates).
2. Expose `source_channels`, `target_channels`, `target_channel_weights` — the sampler
   copies these into the stream config at init
   (`multi_stream_data_sampler.py:_init_stream_datasets`).
3. Register the type: add a `case "<type>":` with a lazy import in
   `packages/readers_extra/src/weathergen/readers_extra/registry.py:get_extra_reader`.
4. Add a stream config under `config/streams/<...>` with `type: <type>` and
   `filenames` (resolved against `cf.data_paths` unless absolute/existing).
5. Verify: `scripts/actions.sh unit-test`, then a small run with a config using the
   stream (GPU: `scripts/actions.sh integration-test`).

Pitfalls:

- Imports in the registry are lazy and unchecked — a broken reader import fails at
  runtime, not at startup.
- Zarr stores are directories; file-existence checks account for that, don't "fix"
  them to require regular files.
- Context: `agent_docs/data-pipeline.md`.
