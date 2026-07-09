# weathergen-common — shared foundation

Imported as `weathergen.common` by src/ and the other packages; keep its dependencies minimal.

- `config.py` — the Config API used everywhere (OmegaConf-based): loads `config/default_config.yml`, layers overwrite configs on top in ascending order. Full picture: `agent_docs/config-system.md` — read before changing merge/loading logic; update after.
- `paths.py` — canonical repo/run paths; `io.py` — shared I/O helpers; `logger.py` — logging setup; `platform_env.py` — per-HPC platform detection.
