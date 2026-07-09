# config/ — YAML run configurations

- `default_config.yml` — base config; every run starts from it. Other `config_*.yml` are overwrites layered on top (ascending order). Loader: `packages/common/src/weathergen/common/config.py`.
- `streams/<name>/` — input data-stream definitions (which datasets a run reads).
- `evaluate/` — evaluation configs (reference: `docs/evaluate_config_reference.md`); `profiling/` — profiling configs.
- New options belong in `default_config.yml` with a sensible default, so partial overwrite configs stay valid.
