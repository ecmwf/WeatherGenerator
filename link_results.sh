#!/usr/bin/env bash
# Create incrementally-numbered symlinks in results/xq6/ that point to the
# validation zip files inside all results/xq6XXXXX source directories.
#
# Usage:
#   bash link_results.sh            # actually create the symlinks
#   bash link_results.sh --dry-run  # only print what would be done
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" || "${1:-}" == "-n" ]]; then
    DRY_RUN=1
    echo "== DRY RUN: no changes will be made ==" >&2
fi

#results/xq6
BASE="results"
RUNID="xq6"
TARGET="${BASE}/${RUNID}"
CHKPT="validation_chkpt00000"

if [[ "${DRY_RUN}" -eq 0 ]]; then
    mkdir -p "${TARGET}"
fi

# Global counter for the target rank numbering.
dest=0

# Discover all source run directories (e.g. results/xq650000 ... results/xq650015),
# sorted numerically, while skipping the aggregated target directory itself.
for src_dir in $(ls -d "${BASE}/${RUNID}"[0-9]*/ 2>/dev/null | sort); do
    src_dir="${src_dir%/}"          # strip trailing slash
    src_name="$(basename "${src_dir}")"

    if [[ "${src_name}" == "${RUNID}" ]]; then
        continue                    # never link the target into itself
    fi
    if [[ ! -d "${src_dir}" ]]; then
        echo "Skipping missing directory: ${src_dir}" >&2
        continue
    fi

    # Iterate source ranks in order (0,1,2,3,...).
    for src_file in $(ls "${src_dir}/${CHKPT}_rank"*.zip 2>/dev/null | sort); do
        dest_name=$(printf "%s_rank%04d.zip" "${CHKPT}" "${dest}")
        dest_path="${TARGET}/${dest_name}"

        # Relative link target as seen from inside ${TARGET} (results/xq6/),
        # hence the leading ../ to step out of xq6/ back into results/.
        rel_src="../${src_name}/$(basename "${src_file}")"

        if [[ -e "${dest_path}" || -L "${dest_path}" ]]; then
            echo "Exists, skipping: ${dest_path}" >&2
        else
            if [[ "${DRY_RUN}" -eq 1 ]]; then
                echo "[dry-run] ln -s ${rel_src} ${dest_path}"
            else
                ln -s "${rel_src}" "${dest_path}"
                echo "ln -s ${rel_src} ${dest_path}"
            fi
        fi

        dest=$((dest + 1))
    done
done
