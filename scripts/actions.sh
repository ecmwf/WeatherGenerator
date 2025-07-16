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
  create-links)
    (
      cd "$SCRIPT_DIR" || exit 1
      # This script creates symbolic links to the shared working directories.
      # 1. Get the path of the private config of the cluster
      # 2. Read the yaml and extract the path of the shared conf
      # This uses the yq command. It is a python package so uvx (bundled with uv) will donwload and create the right venv
      export working_dir=$(cat $(../WeatherGenerator-private/hpc/platform-env.py hpc-config) | uvx yq .path_shared_working_dir)
      # Remove quotes
      export working_dir=$(echo "$working_dir" | sed 's/[\"\x27]//g')
      # If the working directory does not exist, exit with an error
      if [ ! -d "$working_dir" ]; then
        echo "Working directory $working_dir does not exist. Please check the configuration."
        exit 1
      fi
      # Ensure the working directory ends with a slash
      if [[ "$working_dir" != */ ]]; then
        working_dir="$working_dir/"
      fi
      echo "Working directory: $working_dir"
      # Create all the links
      for d in "logs" "models" "output" "plots" "results"
      do
        # If the link already exists, do nothing
        # If a file with the same name exists, skip it
        if [ -e "$d" ]; then
          echo "'$d' already exists, skipping. The results in $d will not be linked to the shared working directory."
          continue
        fi
        echo "$d -> $working_dir$d"
        ln -s "$working_dir$d" "$d"
      done
    )
    ;;
  *)
    echo "Usage: $0 {sync|lint|unit-test|integration-test|create-links}"
    exit 1
    ;;
esac
