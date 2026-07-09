# packages/ — uv workspace libraries

- Workspace members: `common`, `evaluate`, `metrics`, `readers_extra` (see `[tool.uv.workspace]` in root pyproject.toml). `dashboard` and `science` are NOT members; `dashboard` is excluded deliberately — adding it breaks the streamlit deployment.
- Each package: own `pyproject.toml`, code under `src/weathergen/<name>/`, imported as `weathergen.<name>`, own AGENTS.md with detail.
- Packages must be self-contained: type-check (`scripts/actions.sh type-check`) builds each package's deps separately to catch implicit cross-package imports.
