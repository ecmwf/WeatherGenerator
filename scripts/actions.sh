#!/bin/bash

# TODO: this is the root weathergenerator directory, rename the variable.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
PRIVATE_REPO_PATH="${WEATHERGEN_PRIVATE_REPO_PATH:-$SCRIPT_DIR/../WeatherGenerator-private}"

# uv_link_mode from WeatherGenerator-private's hpc/<hpc>/config/paths.yml, if set.
hpc_uv_link_mode() {
  local platform_env="$PRIVATE_REPO_PATH/hpc/platform-env.py"
  [[ -x "$platform_env" ]] && "$platform_env" uv-link-mode 2>/dev/null
}

# Git hooks that re-sync the venv when dependencies change, so that `uv run`
# doesn't sync with symlinks. Existing hooks are left alone.
install_sync_hooks() {
  local hooks_dir hook file
  hooks_dir=$(git rev-parse --git-path hooks 2>/dev/null) || return
  mkdir -p "$hooks_dir"
  for hook in post-merge post-rewrite post-checkout; do
    file="$hooks_dir/$hook"
    if [[ -e "$file" ]] && ! grep -q weathergen-sync-venv "$file"; then
      echo "Not installing the $hook hook: $file exists." >&2
      continue
    fi
    hook_stub "$hook" > "$file"
    chmod +x "$file"
  done
}

# Hook that runs scripts/git-hooks/sync-venv.sh, if the checkout has it.
hook_stub() {
  cat <<EOF
#!/bin/sh
# weathergen-sync-venv: installed by scripts/actions.sh sync
top=\$(git rev-parse --show-toplevel) || exit 0
f="\$top/scripts/git-hooks/sync-venv.sh"
[ -x "\$f" ] && exec "\$f" $1 "\$@"
exit 0
EOF
}

# Venv packages installed as symlinks.
symlinked_packages() {
  local metadata dist_info
  for metadata in .venv/lib/python3*/site-packages/*.dist-info/METADATA; do
    [[ -L "$metadata" ]] || continue
    dist_info=$(basename "$(dirname "$metadata")")
    echo "${dist_info%%-*}"
  done
}

case "$1" in
  sync)
    # Extra arguments go to uv sync.
    shift
    (
      cd "$SCRIPT_DIR" || exit 1
      # Link mode: --link-mode argument > UV_LINK_MODE > HPC config > pyproject.toml.
      link_mode=""
      if [[ "$*" == *--link-mode* ]]; then
        :
      elif [[ -n "${UV_LINK_MODE:-}" ]]; then
        link_mode="$UV_LINK_MODE"
        echo "uv link mode from UV_LINK_MODE in your environment: $link_mode"
      else
        link_mode=$(hpc_uv_link_mode)
        if [[ -n "$link_mode" ]]; then
          echo "uv link mode from the HPC config: $link_mode"
          export UV_LINK_MODE="$link_mode"
          install_sync_hooks
        fi
      fi
      # Reinstall symlinked packages in the chosen link mode.
      relink=()
      if [[ -n "$link_mode" && "$link_mode" != symlink ]]; then
        packages=()
        while read -r package; do
          packages+=("$package")
        done < <(symlinked_packages)
        if (( ${#packages[@]} )); then
          echo "Relinking ${#packages[@]} symlinked packages as $link_mode"
          for package in "${packages[@]}"; do
            relink+=(--reinstall-package "$package")
          done
        fi
      fi
      # Creates a virtual environment without checking the integrity of the cache.
      # If we are running on a mac, use the cpu extra
      if [[ "$(uname)" == "Darwin" ]]; then
        uv sync --all-packages --extra cpu "${relink[@]}" "$@"
        exit 0
      fi
      # Otherwise, use the gpu extra
      uv sync --all-packages --extra gpu "${relink[@]}" "$@"
    )
    ;;
  sync-safe)
    (
      # Creates a virtual environment, checking the integrity of the cache
      # and copying the files.
      # This is slower (+ 60 seconds to the sync) but it is prevents issues with
      # corrupted cache. These issues happen when a shared filesystem such as LUSTRE
      # is used along with symlinks and a SCRATCH deletion policy: some files from
      # cached packages may get deleted because they seem to not be touched enough.
      cd "$SCRIPT_DIR" || exit 1
      # --refresh --reinstall : LUSTRE may clean up some pieces of the cache.
      # This ensures basic integrity but it is too slow (adds 60 seconds to the sync) to do it on every sync. So we only do it on mac, where the cache is more likely to get corrupted.
      # --link-mode=copy overrides the pyproject.toml setting (symlink) to fully
      # detach installed files from the cache, so SCRATCH cleanups can't corrupt the venv.
      # If we are running on a mac, use the cpu extra
      if [[ "$(uname)" == "Darwin" ]]; then
        uv sync --all-packages --extra cpu --refresh --reinstall --link-mode=copy
        exit 0
      fi
      # Otherwise, use the gpu extra
      uv sync --all-packages --extra gpu --refresh --reinstall --link-mode=copy
    )
    ;;
  lint)
    (
      cd "$SCRIPT_DIR" || exit 1
      uv run --no-project --with "ruff==0.12.2" \
        ruff format --target-version py312 src/ scripts/ packages/ \
      && \
      uv run --no-project --with "ruff==0.12.2" \
        ruff check --target-version py312 --fix  src/ scripts/ packages/
    )
    ;;
  lint-check)
    (
      # Warning: make sure that the ruff versions are aligned
      # with pyproject.toml
      cd "$SCRIPT_DIR" || exit 1
      uv run --no-project --with "ruff==0.12.2" \
      ruff format --target-version py312 --check src/ scripts/ packages/ \
      && \
      uv run --no-project --with "ruff==0.12.2" \
        ruff check  --target-version py312 src/ scripts/ packages/ \
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
      uv run --project packages/common --frozen pyrefly check packages/common
      # Fail for errors on weathergen-common:
      if [ $? -ne 0 ]; then
        echo "Type checking failed for weathergen-common."
        exit 1
      fi

      # weathergen-metrics
      uv sync --project packages/metrics --no-install-workspace
      uv run --project packages/metrics --frozen pyrefly check packages/metrics
      # Fail for errors on weathergen-metrics:
      if [ $? -ne 0 ]; then
        echo "Type checking failed for weathergen-metrics."
        exit 1
      fi

      # weathergen-evaluate
      uv sync --project packages/evaluate --no-install-workspace --package weathergen-evaluate
      uv run --project packages/evaluate --frozen pyrefly check packages/evaluate

      # weathergen (root)
      # Install the whole workspace. It also needs the extra cpu option for the right version of pytorch.
      uv sync --all-packages --extra cpu --no-install-workspace
      uv run --all-packages pyrefly check src
      echo "Type checking completed."
    )
    ;;
  unit-test)
    (
      cd "$SCRIPT_DIR" || exit 1
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
  integration-test-jepa)
    (
      cd "$SCRIPT_DIR" || exit 1
      uv sync --offline --all-packages --extra gpu
      uv run --offline pytest ./integration_tests/jepa1_test.py --verbose -s
    )
    ;;
    integration-test)
    (
      cd "$SCRIPT_DIR" || exit 1
      uv sync --offline --all-packages --extra gpu
      uv run --offline pytest ./integration_tests/small_multi_stream_test.py --verbose -s
    );;
    integration-test-export)
    (
      cd "$SCRIPT_DIR" || exit 1
      uv sync --offline --all-packages --extra gpu
      uv run --offline pytest ./integration_tests/export_test.py --verbose -s
    );;
    integration-test-all)
    (
      cd "$SCRIPT_DIR" || exit 1
      uv sync --offline --all-packages --extra gpu
      uv run --offline pytest ./integration_tests/small1_test.py --verbose -s
      uv run --offline pytest ./integration_tests/small_multi_stream_test.py --verbose -s
      uv run --offline pytest ./integration_tests/jepa1_test.py --verbose -s
      uv run --offline pytest ./integration_tests/export_test.py --verbose -s
    );;
  create-links)
    (
      cd "$SCRIPT_DIR" || exit 1
      # This script creates symbolic links to the shared working directories.
      # 1. Get the path of the private config: WEATHERGEN_PRIVATE_CONF if set
      #    (e.g. in CI), otherwise inferred from the hostname by platform-env.py
      # 2. Read the yaml and extract the path of the shared conf
      # This uses the yq command. It is a python package so uvx (bundled with uv) will donwload and create the right venv
      # The 'yq' command is used in a separate virtual environment, because the cache
      # around that tool can get corrupted.
      # See https://github.com/ecmwf/WeatherGenerator/issues/2298
      private_conf="${WEATHERGEN_PRIVATE_CONF:-$("$PRIVATE_REPO_PATH"/hpc/platform-env.py hpc-config)}"
      echo "Private config: $private_conf"
      export working_dir=$(cat "$private_conf" |
        UV_CACHE_DIR="$(mktemp -d)" VIRTUAL_ENV=""  uvx yq .path_shared_working_dir)
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
      uv run jupytext --set-formats ipynb,py:percent --sync  "$PRIVATE_REPO_PATH"/notebooks/*.ipynb "$PRIVATE_REPO_PATH"/notebooks/*.py
      echo "Jupytext sync completed."
    )
    ;;
  *)
    (
      # Automatically extract all options from the case statement
      options=$(sed -nE 's/^[[:space:]]*([a-zA-Z][a-zA-Z0-9_-]*)\).*/\1/p' "$0" | tr '\n' '|' | sed 's/|$//')
      echo "Usage: $0 {$options}"
      exit 1
    )
    ;;
esac
