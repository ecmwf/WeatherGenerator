# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Merge individual PDF plot files into combined documents per plot subdirectory.

Only PDFs whose filenames reference run_ids from the current config are merged,
so different evaluations can coexist in the same base directory.
"""

import logging
from itertools import groupby
from pathlib import Path

from pypdf import PdfWriter

from weathergen.evaluate.plotting.plot_utils import PlotSubdir

_logger = logging.getLogger(__name__)


def _get_known_subdirs() -> list[str]:
    """Return the default list of plot subdirectory names to scan."""
    return [subdir.value for subdir in PlotSubdir]


def _merge_pdfs(pdf_files: list[Path], output_path: Path) -> Path | None:
    """Merge *pdf_files* into *output_path*.  Returns path on success, else None."""
    if not pdf_files:
        return None
    writer = PdfWriter()
    n = 0
    for f in pdf_files:
        try:
            writer.append(str(f))
            n += 1
        except Exception as e:
            _logger.warning(f"Skipping {f.name}: {e}")
    if n == 0:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as fh:
        writer.write(fh)
    writer.close()
    _logger.info(f"Merged {n} PDF(s) → {output_path}")
    return output_path


def merge_pdf_subdirectories(
    base_dir: Path,
    run_ids: list[str],
    subdirs: list[str] | None = None,
) -> list[Path]:
    """Merge PDFs in each plot subdirectory, filtered by run_ids.

    Scans the given subdirectories (e.g. ``line_plots/``, ``ratio_plots/``, …),
    each of which contains a ``<metric>/<region>/`` nested structure, and produces:

    1. One merged PDF per leaf directory (metric/region).
    2. One merged PDF per plot-type aggregating all metrics/regions.

    Parameters
    ----------
    base_dir : Path
        The run_ids-prefixed output directory (e.g. ``plots/runA_runB/``).
    run_ids : list[str]
        Run identifiers from the evaluation config.
    subdirs : list[str] | None
        Subdirectory names to scan, typically the plot types enabled in the
        evaluation config. Missing subdirectories are skipped. Defaults to
        all known plot dirs if not provided.
    """
    if not run_ids:
        _logger.warning("No run_ids provided — skipping PDF merge.")
        return []

    subdirs = subdirs or _get_known_subdirs()
    out_name = "merged_" + "_".join(["plots"] + sorted(run_ids)) + ".pdf"
    merged: list[Path] = []

    for subdir_name in subdirs:
        subdir = base_dir / subdir_name
        if not subdir.is_dir():
            continue

        # Find all matching PDFs recursively, sorted so entries sharing the same
        # parent directory are contiguous (required for groupby below).
        pdfs = sorted(
            (
                f
                for f in subdir.rglob("*.pdf")
                if not f.stem.startswith("merged_") and any(rid in f.stem for rid in run_ids)
            ),
            key=lambda f: (f.parent, f.name),
        )
        if not pdfs:
            continue

        # Merge per leaf directory (metric/region)
        leaf_dirs = set()
        for leaf_dir, group in groupby(pdfs, key=lambda f: f.parent):
            leaf_dirs.add(leaf_dir)
            leaf_pdfs = list(group)
            if len(leaf_pdfs) >= 2:
                r = _merge_pdfs(leaf_pdfs, leaf_dir / out_name)
                if r:
                    merged.append(r)

        # Top-level merge for this plot type, aggregating all metrics/regions.
        # Skip when every PDF already lives directly under `subdir` (no nested
        # metric/region structure), since that would just redo the leaf merge above.
        if len(pdfs) >= 2 and leaf_dirs != {subdir}:
            r = _merge_pdfs(pdfs, subdir / out_name)
            if r:
                merged.append(r)

    if merged:
        _logger.info(f"Created {len(merged)} merged PDF(s) under {base_dir}")
    return merged
