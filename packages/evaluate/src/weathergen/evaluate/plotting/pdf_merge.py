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

Typical usage after summary plotting::

    from weathergen.evaluate.plotting.pdf_merge import merge_pdf_subdirectories

    merge_pdf_subdirectories(summary_dir)
"""

import logging
from collections import defaultdict
from pathlib import Path

from pypdf import PdfWriter

_logger = logging.getLogger(__name__)


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


def merge_pdfs_in_directory(
    directory: Path,
    output_name: str = "merged.pdf",
    sort: bool = True,
) -> Path | None:
    """Find all PDF files in a directory and merge them into one.

    Parameters
    ----------
    directory : Path
        Directory to scan for ``*.pdf`` files (non-recursive).
    output_name : str
        Filename for the merged output (written into the same directory).
    sort : bool
        Whether to sort PDF files alphabetically before merging.

    Returns
    -------
    Path | None
        Path to the merged PDF, or None if no PDFs found.
    """
    pdf_files = list(directory.glob("*.pdf"))
    # Exclude any previously-merged files to allow re-runs
    pdf_files = [f for f in pdf_files if f.name != output_name]

    if not pdf_files:
        _logger.debug(f"No PDFs found in {directory}")
        return None

    if sort:
        pdf_files = sorted(pdf_files)

    output_path = directory / output_name
    return merge_pdfs(pdf_files, output_path)


def merge_pdfs_by_prefix(
    directory: Path,
    output_dir: Path | None = None,
    separator: str = "_",
    prefix_depth: int = 2,
    sort: bool = True,
) -> list[Path]:
    """Group PDF files by filename prefix and merge each group.

    Filenames are split on ``separator`` and the first ``prefix_depth`` parts
    form the group key.  For example, with default settings:
        ``rmse_global_2t.pdf`` and ``rmse_global_10u.pdf``
    both have prefix ``rmse_global`` and would be merged together.

    Parameters
    ----------
    directory : Path
        Directory containing PDF plot files.
    output_dir : Path | None
        Where to write merged files.  Defaults to ``directory``.
    separator : str
        Character used to split filenames into parts.
    prefix_depth : int
        Number of filename parts (from the left) that define the group.
    sort : bool
        Sort files within each group alphabetically.

    Returns
    -------
    list[Path]
        Paths to all successfully created merged PDFs.
    """
    if output_dir is None:
        output_dir = directory

    pdf_files = [f for f in directory.glob("*.pdf") if not f.stem.startswith("merged_")]
    if not pdf_files:
        return []

    # Group by prefix
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in pdf_files:
        parts = f.stem.split(separator)
        prefix = separator.join(parts[:prefix_depth]) if len(parts) >= prefix_depth else f.stem
        groups[prefix].append(f)

    merged_paths = []
    for prefix, files in groups.items():
        if len(files) < 2:
            continue  # no point merging a single file
        if sort:
            files = sorted(files)
        output_path = output_dir / f"merged_{prefix}.pdf"
        result = merge_pdfs(files, output_path)
        if result:
            merged_paths.append(result)

    return merged_paths


def merge_pdf_subdirectories(
    base_dir: Path,
    subdirs: list[str] | None = None,
    output_name: str = "merged.pdf",
) -> list[Path]:
    """Merge PDFs in each plot subdirectory under a base directory.

    This is the main entry point called after ``plot_summary``.  It scans
    known subdirectories (line_plots, ratio_plots, etc.) and produces one
    merged PDF per subdirectory.

    Parameters
    ----------
    base_dir : Path
        The summary/output directory that contains plot subdirectories.
    subdirs : list[str] | None
        Explicit list of subdirectory names to scan.  If None, uses the
        default set of known evaluation plot directories.
    output_name : str
        Name of the merged file in each subdirectory.

    Returns
    -------
    list[Path]
        Paths of all successfully created merged PDFs.
    """
    if subdirs is None:
        subdirs = [
            "line_plots",
            "ratio_plots",
            "psd_plots",
            "score_cards",
            "bar_plots",
            "qq_plots",
        ]

    merged_paths = []
    for subdir_name in subdirs:
        subdir = base_dir / subdir_name
        if subdir.is_dir():
            result = merge_pdfs_in_directory(subdir, output_name=output_name)
            if result:
                merged_paths.append(result)

    # Also try merging by metric/region prefix within each subdir for
    # finer-grained booklets (e.g. "merged_rmse_global.pdf")
    for subdir_name in subdirs:
        subdir = base_dir / subdir_name
        if subdir.is_dir():
            prefix_results = merge_pdfs_by_prefix(subdir, output_dir=subdir)
            merged_paths.extend(prefix_results)

    if merged_paths:
        _logger.info(f"Created {len(merged_paths)} merged PDF(s) under {base_dir}")
    else:
        _logger.debug(f"No PDFs to merge under {base_dir}")

    return merged_paths
