# tests/ — unit tests

- Run: `uv run --extra cpu pytest tests/` (`--extra cpu` on machines without GPU, `--extra gpu` otherwise).
- CI's `unit-test` target runs `pytest src/` — inline `*_test.py` files there; keep both green.
- Integration tests live in `integration_tests/`, need a GPU: `scripts/actions.sh integration-test*`.
- pytest options in root `pyproject.toml` `[tool.pytest.ini_options]` (log_cli enabled).
