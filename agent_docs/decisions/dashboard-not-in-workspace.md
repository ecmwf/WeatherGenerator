# Decision: packages/dashboard is not a uv workspace member

Context: all other `packages/*` libraries are members of the uv workspace
(`[tool.uv.workspace]` in the root `pyproject.toml`) and share the root lockfile.

Decision: `packages/dashboard` is deliberately excluded and keeps its own
`pyproject.toml`, `uv.lock`, and `.venv`.

Why: including it in the workspace causes issues when deploying the streamlit
dashboard (see the comment in the root `pyproject.toml` workspace section).

Consequences: run and manage it from its own directory
(`uv run --env-file=.env streamlit run dashboard.py`); do not add it to the workspace
members, and do not "clean up" its standalone lockfile.
