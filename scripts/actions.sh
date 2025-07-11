#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"

case "$1" in
  sync)
    (
      cd "$SCRIPT_DIR" || exit 1
      uv sync
    )
    ;;
  lint)
    (
      cd "$SCRIPT_DIR" || exit 1
      uv run --no-project --with "ruff==0.9.7" ruff format --target-version py312 && \
      uv run --no-project --with "ruff==0.9.7" ruff check --fix
    )
    ;;
  unit-test)
    (
      cd "$SCRIPT_DIR" || exit 1
      uv run pytest src/
    )
    ;;
  integration-test)
    (
      cd "$SCRIPT_DIR" || exit 1
      uv run pytest ./integration_tests/small1_test.py --verbose
    )
    ;;
  *)
    echo "Usage: $0 {sync|lint|unit-test|integration-test}"
    exit 1
    ;;
esac
