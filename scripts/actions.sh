#!/bin/bash

# TODO: this is the root weathergenerator directory, rename the variable.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"

case "$1" in
  sync)
    (
      cd "$SCRIPT_DIR" || exit 1
      # If we are running on a mac, use the cpu extra
      if [[ "$(uname)" == "Darwin" ]]; then
        uv sync --all-packages --extra cpu
        exit 0
      fi
      # Otherwise, use the gpu extra
      uv sync --all-packages --extra gpu
    )
    ;;
  lint)
    (
      cd "$SCRIPT_DIR" || exit 1
      uv run --no-project --with "ruff==0.12.2" ruff format --target-version py312 \
        src/ scripts/ packages/ \
        && \
      uv run --no-project --with "ruff==0.12.2" \
        ruff check --target-version py312 \
        --fix  \
        src/ scripts/ packages/
    )
    ;;
  lint-check)
    (
      cd "$SCRIPT_DIR" || exit 1
      uv run --no-project --with "ruff==0.12.2" ruff format --target-version py312 \
        -n \
        src/ scripts/ packages/ \
        && \
      uv run --no-project --with "ruff==0.12.2" \
       ruff check  --target-version py312  \
       src/ scripts/ packages/ \
        && \
      uv run --no-project --with "pylint==4.0.3" \
       pylint src/ packages/
    )
    ;;
  type-check)
    (
      # The dependencies are rebuilt for each package to ensure that they do not rely on implicit imports.
      cd "$SCRIPT_DIR" || exit 1

      # weathergen-common
      uv sync --project packages/common --no-install-workspace
      uv pip list
      uv run --project packages/common --frozen pyrefly check packages/common
      # Fail for errors on weathergen-common:
      if [ $? -ne 0 ]; then
        echo "Type checking failed for weathergen-common."
        exit 1
      fi

      # weathergen-metrics
      uv sync --project packages/metrics --no-install-workspace
      uv pip list
      uv run --project packages/metrics --frozen pyrefly check packages/metrics
      # Fail for errors on weathergen-metrics:
      if [ $? -ne 0 ]; then
        echo "Type checking failed for weathergen-metrics."
        exit 1
      fi

      # weathergen-evaluate
      uv sync --project packages/evaluate --no-install-workspace --package weathergen-evaluate 
      uv pip list
      uv run --project packages/evaluate --frozen pyrefly check packages/evaluate

      # weathergen (root)
      # Install the whole workspace. It also needs the extra cpu option for the right version of pytorch.
      uv sync --all-packages --extra cpu --no-install-workspace
      uv pip list
      uv run --all-packages pyrefly check src
      echo "Type checking completed."
    )
    ;;
  unit-test)
    (
      cd "$SCRIPT_DIR" || exit 1
      uv sync --extra cpu 
      uv run --extra cpu pytest src/
    )
    ;;
  toml-check)
    (
      cd "$SCRIPT_DIR" || exit 1
      uv run --no-project python scripts/check_tomls.py
    )
    ;;
  integration-test-single)
    (
      cd "$SCRIPT_DIR" || exit 1
      uv sync --offline --all-packages --extra gpu
      uv run --offline pytest ./integration_tests/small1_test.py --verbose -s
    )
    ;;
    integration-test)
    (
      cd "$SCRIPT_DIR" || exit 1
      uv sync --offline --all-packages --extra gpu
      uv run --offline pytest ./integration_tests/small_multi_stream_test.py --verbose -s
    );;
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
  create-jupyter-kernel)
    (
      cd "$SCRIPT_DIR" || exit 1
      uv sync --all-packages
      uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=weathergen_kernel --display-name "Python (WeatherGenerator)" 
      echo "Jupyter kernel created. You can now use it in Jupyter Notebook or JupyterLab."
      echo "To use this kernel, select 'Python (WeatherGenerator)' from the kernel options in Jupyter Notebook or JupyterLab."
      echo "If you want to remove the kernel later, you can run:"
      echo "jupyter kernelspec uninstall weathergen_kernel"
    )
    ;;
  jupytext-sync)
    (
      cd "$SCRIPT_DIR" || exit 1
      # Run on any python or jupyter notebook files in the WeatherGenerator-private/notebooks directory
      uv run jupytext --set-formats ipynb,py:percent --sync  ../WeatherGenerator-private/notebooks/*.ipynb ../WeatherGenerator-private/notebooks/*.py
      echo "Jupytext sync completed."
    )
    ;;
  *)
    # Automatically extract all options from the case statement
    options=$(grep -oP '^\s*\K[\w-]+(?=\))' "$0" | tr '\n' '|' | sed 's/|$//')
    echo "Usage: $0 {$options}"
    exit 1
    ;;
esac
