# weathergen-common — shared foundation

Imported as `weathergen.common` by src/ and the other packages; keep its dependencies minimal.

- `config.py` — the Config API used everywhere (OmegaConf-based): loads `config/default_config.yml`, layers overwrite configs on top in ascending order.
- `paths.py` — canonical repo/run paths; `io.py` — shared I/O helpers; `logger.py` — logging setup; `platform_env.py` — per-HPC platform detection.
