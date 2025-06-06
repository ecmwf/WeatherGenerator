#!/usr/bin/env python3
# (C) Copyright [Year] [Your Organization/Project contributors]
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, [Your Organization] does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
# (Note: Added a placeholder license header. Please replace or remove.)

"""
Radklim Dataset Reference Generator using Kerchunk.

Generates a single consolidated JSON reference for all Radklim NetCDF datasets
via Kerchunk, without saving individual per-file references.

This script processes NetCDF files typically organized in year-named
subdirectories, creates in-memory Kerchunk references for each, combines them
into a single reference, and saves it as a JSON file. Options are provided
for dropping variables, specifying dimensions for concatenation and identity,
and skipping optional steps like metadata consolidation or time verification.

Usage:
    python3 generate_radklim_reference.py <input_dir> <output_dir> [options]

Example:
    python3 generate_radklim_reference.py /data/radklim/input /data/radklim/output \\
        --output-name my_combined_ref.json \\
        --drop-variables crs rotated_pole \\
        --concat-dims time \\
        --identical-dims y x \\
        --no-consolidate \\
        --skip-verify

Arguments:
    input_dir       Directory with year-named subdirectories containing
                    NetCDF (.nc) files.
    output_dir      Directory where the combined reference JSON will be saved.

Optional arguments:
    --output-name       Filename for combined reference
                        (default: radklim_combined_reference.json).
    --drop-variables    Space-separated list of variables to drop from each
                        reference (default: crs).
    --concat-dims       Space-separated list of dimensions to concatenate over
                        (default: time).
    --identical-dims    Space-separated list of dimensions identical across all
                        files (default: y x).
    --no-consolidate    Skip Zarr metadata consolidation of the final reference.
    --skip-verify       Skip the time consistency check between original files
                        and the combined reference.
"""

import argparse
import json
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import fsspec
import numpy as np
import xarray as xr
import zarr
from kerchunk.combine import MultiZarrToZarr
from kerchunk.hdf import SingleHdf5ToZarr


def find_year_directories(parent_dir: Path) -> List[Path]:
    """
    Find subdirectories named like years (e.g., 2001, 2020) under a given directory.

    Parameters
    ----------
    parent_dir : Path
        The directory to search within.

    Returns
    -------
    List[Path]
        A sorted list of Path objects representing valid year directories.

    Raises
    ------
    FileNotFoundError
        If `parent_dir` does not exist or is not a directory.
    """
    if not parent_dir.exists() or not parent_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {parent_dir}")
    dirs: List[Path] = []
    for sub in parent_dir.iterdir():
        if sub.is_dir() and sub.name.isdigit():
            try:
                year = int(sub.name)
                if 1900 <= year <= 2100:  
                    dirs.append(sub)
            except ValueError:
                continue  
    return sorted(dirs)


def collect_netcdf_files(year_dirs: List[Path]) -> List[Path]:
    """
    Find all NetCDF (.nc) files inside a list of year directories.

    Files are sorted alphabetically by their full path.

    Parameters
    ----------
    year_dirs : List[Path]
        A list of directories to search for .nc files.

    Returns
    -------
    List[Path]
        A sorted list of Path objects for all found NetCDF files.
    """
    files: List[Path] = []
    for ydir in year_dirs:
        for fp_str in sorted(glob(str(ydir / "*.nc"))):
            files.append(Path(fp_str))
    return sorted(files)


def extract_time_values(filepath: Path) -> np.ndarray:
    """
    Load the 'time' coordinate values from a NetCDF file.

    Parameters
    ----------
    filepath : Path
        Path to the NetCDF file.

    Returns
    -------
    np.ndarray
        A NumPy array containing the time values.

    Raises
    ------
    RuntimeError
        If the file cannot be opened or 'time' coordinate cannot be extracted.
    """
    try:
        with xr.open_dataset(str(filepath), chunks={"time": 1000}) as ds: # type: ignore
            if "time" not in ds.coords and "time" not in ds.data_vars:
                raise KeyError(f"'time' coordinate or variable not found in {filepath}")
            return ds.time.values # type: ignore
    except Exception as e:
        raise RuntimeError(f"Failed to extract time from {filepath}: {e}") from e


def generate_kerchunk_references(
    nc_files: List[Path], drop_vars: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Create in-memory Kerchunk references for each NetCDF file.

    Optionally removes specified variables from the references. This implementation
    does not save individual JSONs but returns a list of reference dictionaries.

    Parameters
    ----------
    nc_files : List[Path]
        A list of paths to NetCDF files.
    drop_vars : Optional[List[str]], optional
        A list of variable names to drop from the references.
        Defaults to ["crs"] if None.

    Returns
    -------
    List[Dict[str, Any]]
        A list of Kerchunk reference dictionaries.
    """
    if drop_vars is None:
        current_drop_vars: List[str] = ["crs"]
    else:
        current_drop_vars = drop_vars

    refs: List[Dict[str, Any]] = []
    for nc_path in nc_files:
        with fsspec.open(str(nc_path), "rb") as fobj:
            translator = SingleHdf5ToZarr(fobj, str(nc_path))
            ref = translator.translate()

        if current_drop_vars and "refs" in ref:
            for var_to_drop in current_drop_vars:
                ref["refs"].pop(f"{var_to_drop}/.zarray", None)
                ref["refs"].pop(f"{var_to_drop}/.zattrs", None)
            if "zarr_consolidated" in ref and isinstance(ref["zarr_consolidated"], dict) :
                zmeta = ref["zarr_consolidated"].get("metadata", {})
                if isinstance(zmeta, dict):
                    for var_to_drop in current_drop_vars:
                        zmeta.pop(var_to_drop, None) # Remove var entry itself if present

        refs.append(ref)
    return refs


def combine_kerchunk_references(
    refs: List[Dict[str, Any]],
    concat_dims: List[str],
    identical_dims: List[str],
) -> Dict[str, Any]:
    """
    Merge multiple Kerchunk references into one consolidated reference.

    Parameters
    ----------
    refs : List[Dict[str, Any]]
        A list of individual Kerchunk reference dictionaries.
    concat_dims : List[str]
        A list of dimension names along which to concatenate datasets.
    identical_dims : List[str]
        A list of dimension names that are identical across all datasets.

    Returns
    -------
    Dict[str, Any]
        A single, combined Kerchunk reference dictionary.

    Raises
    ------
    RuntimeError
        If the combination process fails.
    """
    try:
        mzz = MultiZarrToZarr(
            refs,
            concat_dims=concat_dims,
            identical_dims=identical_dims,
        )
        return mzz.translate()
    except Exception as e:
        raise RuntimeError(f"Failed to combine Kerchunk references: {e}") from e


def consolidate_metadata_on_file(ref_path: Path) -> None:
    """
    Perform Zarr metadata consolidation on a combined reference JSON file.

    This function reads the reference JSON, consolidates metadata in memory
    using a reference filesystem, and then writes the updated reference
    (potentially with a new .zmetadata key/entry) back to the original file path.

    Parameters
    ----------
    ref_path : Path
        Path to the combined Kerchunk reference JSON file.
    """
    # Load the existing reference file content
    try:
        with open(ref_path, 'r') as f:
            ref_content = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Reference file {ref_path} not found for consolidation.")
        return
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {ref_path} for consolidation.")
        return


    fs = fsspec.filesystem("reference", fo=ref_content)
    store = fs.get_mapper("")
    try:
        zarr.consolidate_metadata(store)

        with open(ref_path, 'w') as f:
            json.dump(ref_content, f, indent=2, default=str)

    except Exception as e:
        print(f"Warning: Metadata consolidation failed for {ref_path}: {e}")
        pass


# This method is experimental, can be totally ignored
def verify_time_consistency(
    original_files: List[Path], combined_ref_path: Path
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if the combined reference has the same time values as the original files.

    This function is primarily for debugging and verification.

    Parameters
    ----------
    original_files : List[Path]
        List of paths to the original NetCDF files.
    combined_ref_path : Path
        Path to the combined Kerchunk reference JSON file.

    Returns
    -------
    Tuple[bool, Dict[str, Any]]
        A tuple containing:
        - bool: True if time values match, False otherwise.
        - Dict[str, Any]: A dictionary with detailed verification information.
    """
    orig_times_list: List[np.datetime64] = []
    file_time_counts: Dict[str, int] = {}
    for fp in original_files:
        try:
            vals = extract_time_values(fp) # Returns np.ndarray
            orig_times_list.extend(list(vals)) # type: ignore
            file_time_counts[fp.name] = len(vals)
        except RuntimeError as e:
            print(f"Warning: Could not extract time for verification from {fp}: {e}")

    orig_times_sorted = np.array(sorted(orig_times_list), dtype='datetime64[ns]')

    try:
        with open(combined_ref_path, "r") as f:
            ref_dict = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Combined reference file {combined_ref_path} not found for verification.")
        return False, {"error": "Combined reference file not found."}
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {combined_ref_path} for verification.")
        return False, {"error": "Could not decode combined reference JSON."}


    fs_comb = fsspec.filesystem("reference", fo=ref_dict)
    store_comb = fs_comb.get_mapper("")
    try:
        zarr.consolidate_metadata(store_comb)
    except Exception as e:
        print(f"Note: Minor issue during pre-verification consolidation of {combined_ref_path}: {e}")
        pass

    try:
        with xr.open_dataset(
            store_comb, engine="zarr", consolidated=True, chunks={"time": 1000} # type: ignore
        ) as ds_combined:
            combined_times_data = ds_combined.time.load().data # type: ignore
            combined_times_sorted = np.array(sorted(combined_times_data), dtype='datetime64[ns]')
    except Exception as e:
        print(f"Error opening or reading combined reference dataset for verification: {e}")
        return False, {"error": f"Failed to read combined dataset: {e}"}


    times_match = np.array_equal(orig_times_sorted, combined_times_sorted)

    verification_info: Dict[str, Any] = {
        "original_total_time_points": len(orig_times_sorted),
        "combined_total_time_points": len(combined_times_sorted),
        "time_values_match": bool(times_match),
        "per_file_time_counts": file_time_counts,
        "original_time_range": (
            str(orig_times_sorted[0]) if orig_times_sorted.size > 0 else None,
            str(orig_times_sorted[-1]) if orig_times_sorted.size > 0 else None,
        ),
        "combined_time_range": (
            str(combined_times_sorted[0]) if combined_times_sorted.size > 0 else None,
            str(combined_times_sorted[-1]) if combined_times_sorted.size > 0 else None,
        ),
    }
    return times_match, verification_info


def save_json_output(data: Dict[str, Any], path: Path) -> None:
    """
    Save a dictionary to a JSON file with pretty printing.

    Ensures the parent directory exists.

    Parameters
    ----------
    data : Dict[str, Any]
        The dictionary to save.
    path : Path
        The file path to save the JSON to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def main() -> None:
    """
    Main entry point for the script.

    Parses command-line arguments, finds NetCDF files, generates individual
    Kerchunk references, combines them, saves the result, and optionally
    performs metadata consolidation and time consistency verification.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate a consolidated JSON reference for Radklim NetCDF datasets "
            "using Kerchunk."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Example:\n"
            "  python3 %(prog)s /data/radklim/input /data/radklim/output \\\n"
            "    --output-name my_combined_ref.json \\\n"
            "    --drop-variables crs rotated_pole \\\n"
            "    --concat-dims time \\\n"
            "    --identical-dims y x"
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory with year-named subdirectories containing NetCDF (.nc) files.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where the combined reference JSON will be saved.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="radklim_combined_reference.json",
        help=(
            "Filename for the combined reference JSON.\n"
            "(default: radklim_combined_reference.json)"
        ),
    )
    parser.add_argument(
        "--drop-variables",
        nargs="*",
        default=["crs"],
        metavar="VAR",
        help="Space-separated list of variables to drop from each reference.\n(default: crs)",
    )
    parser.add_argument(
        "--concat-dims",
        nargs="+",
        default=["time"],
        metavar="DIM",
        help=(
            "Space-separated list of dimensions to concatenate over in the "
            "combined reference.\n(default: time)"
        ),
    )
    parser.add_argument(
        "--identical-dims",
        nargs="+",
        default=["y", "x"],
        metavar="DIM",
        help="Space-separated list of dimensions identical across all files.\n(default: y x)",
    )
    parser.add_argument(
        "--no-consolidate",
        action="store_true",
        help="Skip final Zarr metadata consolidation of the combined reference.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip time consistency verification.",
    )
    args = parser.parse_args()

    print(f"Starting Radklim reference generation for input: {args.input_dir}")

    try:
        year_dirs = find_year_directories(args.input_dir)
        if not year_dirs:
            raise FileNotFoundError(
                f"No valid year directories (e.g., '2001') found in {args.input_dir}"
            )
        print(f"Found {len(year_dirs)} year directories.")

        nc_files = collect_netcdf_files(year_dirs)
        if not nc_files:
            raise FileNotFoundError(
                f"No NetCDF (.nc) files found within the year directories in {args.input_dir}"
            )
        print(f"Collected {len(nc_files)} NetCDF files for processing.")

        print("Generating individual Kerchunk references...")
        individual_refs = generate_kerchunk_references(nc_files, args.drop_variables)
        print(f"Generated {len(individual_refs)} individual references.")

        print("Combining Kerchunk references...")
        combined_ref_dict = combine_kerchunk_references(
            individual_refs, args.concat_dims, args.identical_dims
        )
        print("Successfully combined references.")

        combined_ref_path = args.output_dir / args.output_name
        print(f"Saving combined reference to: {combined_ref_path}")
        save_json_output(combined_ref_dict, combined_ref_path)

        if not args.no_consolidate:
            print("Consolidating metadata for the combined reference...")
            consolidate_metadata_on_file(combined_ref_path)
            print("Metadata consolidation complete.")
        else:
            print("Skipping metadata consolidation as requested.")

        verification_report_data: Dict[str, Any] = {}
        if not args.skip_verify:
            print("Performing time consistency verification...")
            match, report = verify_time_consistency(nc_files, combined_ref_path)
            verification_report_data = report
            report_path = args.output_dir / "time_verification_report.json"
            save_json_output(report, report_path)
            if not match:
                raise RuntimeError(
                    "Time consistency verification FAILED. "
                    f"Report saved to {report_path}"
                )
            print(f"Time consistency verification PASSED. Report saved to {report_path}")
        else:
            print("Skipping time consistency verification as requested.")

        print("\n Radklim Kerchunk reference generation complete.")
        print(f"  Input directory:    {args.input_dir.resolve()}")
        print(f"  Processed NetCDF files: {len(nc_files)}")
        print(f"  Output reference:   {combined_ref_path.resolve()}")
        if verification_report_data:
            status = "PASSED" if verification_report_data.get("time_values_match") else "FAILED"
            print(f"  Time verification:  {status} (see time_verification_report.json)")
        print()

    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as e:
        print(f"\n Error during reference generation: {e}")
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"\n An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()