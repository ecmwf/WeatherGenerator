# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Utilities for merging multiple PDF plot files into combined documents.

When evaluation is configured with ``image_format: pdf``, individual PDF files
are generated per variable/metric/region.  This module provides helpers to merge
them into fewer documents for easier browsing.

Only PDFs whose filenames reference the run_ids from the current evaluation
config are included — so different evaluations can coexist in the same directory
without contaminating each other's merged output.

Typical usage after summary plotting::

    from weathergen.evaluate.plotting.pdf_merge import merge_pdf_subdirectories

    merge_pdf_subdirectories(summary_dir, run_ids=["run_A", "run_B"])
"""

import logging
from collections import defaultdict
from pathlib import Path

from pypdf import PdfWriter

_logger = logging.getLogger(__name__)

_KNOWN_SUBDIRS = [
    "line_plots",
    "ratio_plots",
    "psd_plots",
    "score_cards",
    "bar_plots",
    "qq_plots",
]


def _file_belongs_to_runs(pdf_file: Path, run_ids: list[str]) -> bool:
    """Check whether a PDF filename references *any* of the given run_ids.

    The convention is that plot filenames embed the run_ids that contributed
    to the plot (e.g. ``rmse_global_runA_runB_ERA5_2t.pdf``).  A file belongs
    to the current evaluation if at least one of the config's run_ids appears
    in its filename.
    """
    stem = pdf_file.stem
    return any(run_id in stem for run_id in run_ids)


def _build_merged_filename(subdir_name: str, run_ids: list[str]) -> str:
    """Build the merged PDF filename from the subdirectory name and run_ids.

    Examples
    --------
    >>> _build_merged_filename("line_plots", ["runA", "runB"])
    'merged_line_plots_runA_runB.pdf'
    """
    parts = ["merged", subdir_name] + sorted(run_ids)
    return "_".join(parts) + ".pdf"


# ---------------------------------------------------------------------------
# Core merge helper
# ---------------------------------------------------------------------------


def merge_pdfs(pdf_files: list[Path], output_path: Path) -> Path | None:
    """Merge a list of PDF files into a single output PDF.

    Parameters
    ----------
    pdf_files : list[Path]
        Ordered list of input PDF file paths.
    output_path : Path
        Destination path for the merged PDF.

    Returns
    -------
    Path | None
        The output path if merging succeeded, None if no valid pages were found.
    """
    if not pdf_files:
        return None

    writer = PdfWriter()
    pages_added = 0
    for pdf_file in pdf_files:
        try:
            writer.append(str(pdf_file))
            pages_added += 1
        except Exception as e:
            _logger.warning(f"Skipping {pdf_file.name}: {e}")

    if pages_added == 0:
        _logger.warning(f"No valid PDFs to merge into {output_path.name}")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        writer.write(f)
    writer.close()

    _logger.info(f"Merged {pages_added} PDF(s) → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Per-directory merge (filtered by run_ids)
# ---------------------------------------------------------------------------


def merge_pdfs_in_directory(
    directory: Path,
    run_ids: list[str],
    output_name: str | None = None,
) -> Path | None:
    """Merge PDF files belonging to given run_ids in a single directory.

    Only files whose filename contains at least one of the ``run_ids`` are
    included.  Previously-merged files (prefixed with ``merged_``) are skipped.

    Parameters
    ----------
    directory : Path
        Directory to scan for ``*.pdf`` files (non-recursive).
    run_ids : list[str]
        Run identifiers from the evaluation config.
    output_name : str | None
        Filename for the merged output. If None, built automatically from
        the directory name and run_ids.

    Returns
    -------
    Path | None
        Path to the merged PDF, or None if no matching PDFs found.
    """
    if output_name is None:
        output_name = _build_merged_filename(directory.name, run_ids)

    all_pdfs = list(directory.glob("*.pdf"))
    # Filter: only files that belong to these runs, and skip previous merges
    pdf_files = sorted(
        f for f in all_pdfs if not f.stem.startswith("merged_") and _file_belongs_to_runs(f, run_ids)
    )

    if not pdf_files:
        _logger.debug(f"No PDFs matching run_ids {run_ids} in {directory}")
        return None

    output_path = directory / output_name
    return merge_pdfs(pdf_files, output_path)


# ---------------------------------------------------------------------------
# Group by plot type (metric_region prefix) within a directory
# ---------------------------------------------------------------------------


def merge_pdfs_by_plot_type(
    directory: Path,
    run_ids: list[str],
    separator: str = "_",
    prefix_depth: int = 2,
) -> list[Path]:
    """Group PDF files by plot type prefix and merge each group.

    Filenames follow the convention ``<metric>_<region>_<run_ids>_<stream>_<channel>.pdf``.
    The first ``prefix_depth`` parts (before run_ids) identify the *plot type*.
    Files are first filtered to only those belonging to the given run_ids,
    then grouped by their metric/region prefix.

    For example, with ``run_ids=["runA", "runB"]``::

        rmse_global_runA_runB_ERA5_2t.pdf  → group "rmse_global"
        rmse_global_runA_runB_ERA5_10u.pdf → group "rmse_global"
        bias_global_runA_runB_ERA5_2t.pdf  → group "bias_global"

    Each group is merged into e.g. ``merged_rmse_global_runA_runB.pdf``.

    Parameters
    ----------
    directory : Path
        Directory containing PDF plot files.
    run_ids : list[str]
        Run identifiers to filter on.
    separator : str
        Character used to split filenames into parts.
    prefix_depth : int
        Number of filename parts (from the left) that define the plot type.

    Returns
    -------
    list[Path]
        Paths to all successfully created merged PDFs.
    """
    all_pdfs = list(directory.glob("*.pdf"))
    pdf_files = [
        f for f in all_pdfs if not f.stem.startswith("merged_") and _file_belongs_to_runs(f, run_ids)
    ]
    if not pdf_files:
        return []

    # Group by prefix (metric_region)
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in pdf_files:
        parts = f.stem.split(separator)
        prefix = separator.join(parts[:prefix_depth]) if len(parts) >= prefix_depth else f.stem
        groups[prefix].append(f)

    merged_paths = []
    run_id_tag = "_".join(sorted(run_ids))
    for prefix, files in sorted(groups.items()):
        if len(files) < 2:
            continue  # no point merging a single file
        files = sorted(files)
        output_path = directory / f"merged_{prefix}_{run_id_tag}.pdf"
        result = merge_pdfs(files, output_path)
        if result:
            merged_paths.append(result)

    return merged_paths


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def merge_pdf_subdirectories(
    base_dir: Path,
    run_ids: list[str],
    subdirs: list[str] | None = None,
) -> list[Path]:
    """Merge PDFs in each plot subdirectory, filtered by run_ids.

    This is the main entry point called after ``plot_summary``.  It scans
    known subdirectories (line_plots, ratio_plots, etc.) and produces:

    1. One merged PDF per subdirectory containing all plots for the given
       run_ids (e.g. ``merged_line_plots_runA_runB.pdf``).
    2. Finer-grained merged PDFs grouped by plot type / metric
       (e.g. ``merged_rmse_global_runA_runB.pdf``).

    Parameters
    ----------
    base_dir : Path
        The summary/output directory that contains plot subdirectories.
    run_ids : list[str]
        Run identifiers from the evaluation config.  Only PDFs whose filenames
        reference these run_ids will be included.
    subdirs : list[str] | None
        Explicit list of subdirectory names to scan.  If None, uses the
        default set of known evaluation plot directories.

    Returns
    -------
    list[Path]
        Paths of all successfully created merged PDFs.
    """
    if subdirs is None:
        subdirs = _KNOWN_SUBDIRS

    if not run_ids:
        _logger.warning("No run_ids provided — skipping PDF merge.")
        return []

    merged_paths = []

    for subdir_name in subdirs:
        subdir = base_dir / subdir_name
        if not subdir.is_dir():
            continue

        # Full merge of all matching PDFs in this subdirectory
        result = merge_pdfs_in_directory(subdir, run_ids=run_ids)
        if result:
            merged_paths.append(result)

        # Finer-grained: group by plot type (metric_region prefix)
        prefix_results = merge_pdfs_by_plot_type(subdir, run_ids=run_ids)
        merged_paths.extend(prefix_results)

    if merged_paths:
        _logger.info(
            f"Created {len(merged_paths)} merged PDF(s) under {base_dir} "
            f"for run_ids={sorted(run_ids)}"
        )
    else:
        _logger.debug(f"No PDFs to merge under {base_dir} for run_ids={sorted(run_ids)}")

    return merged_paths
