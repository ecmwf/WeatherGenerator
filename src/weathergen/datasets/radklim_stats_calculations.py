#!/usr/bin/env python

import xarray as xr
import numpy as np
import pandas as pd
import glob
import json
import argparse
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Union

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
_logger = logging.getLogger(__name__)

def calculate_and_save_stats_file_by_file(
    input_glob_pattern: str,
    variables: List[str],
    output_stats_file: str,
    start_date_str: Optional[str] = None,
    end_date_str: Optional[str] = None,
    fail_on_error_per_file: bool = False 
):
    """
    Calculates global mean, std, and count for specified variables by processing
    NetCDF files one by one, then saves them to a JSON file.
    """
    _logger.info(f"Starting global statistics calculation (file-by-file) for pattern: {input_glob_pattern}")
    _logger.info(f"Variables to process: {variables}")
    if start_date_str: _logger.info(f"Overall start date filter: {start_date_str}")
    if end_date_str: _logger.info(f"Overall end date filter: {end_date_str}")

    all_input_files = sorted(glob.glob(input_glob_pattern))
    if not all_input_files:
        _logger.error(f"No files found for pattern: {input_glob_pattern}")
        return False
    _logger.info(f"Found {len(all_input_files)} total files to process.")

    # Initialize accumulators for each variable
    # Format: {var_name: {'sum_x': 0.0, 'sum_x_sq': 0.0, 'count_n': 0}}
    accumulated_stats = {var: {'sum_x': 0.0, 'sum_x_sq': 0.0, 'count_n': 0} for var in variables}

    # Convert overall date filters to pd.Timestamp once
    overall_start_ts = pd.Timestamp(start_date_str) if start_date_str else None
    overall_end_ts = pd.Timestamp(end_date_str) if end_date_str else None

    for i, file_path in enumerate(all_input_files):
        _logger.info(f"Processing file {i+1}/{len(all_input_files)}: {file_path}")
        try:
            # Open one file at a time. Chunking here is less critical for memory
            # as we .load() the variable data, but can help xarray read.
            with xr.open_dataset(file_path, chunks={'time': 'auto'}) as ds_single_file:
                
                ds_to_process = ds_single_file
                # Apply overall date filtering to this single file's data
                if overall_start_ts:
                    ds_to_process = ds_to_process.sel(time=slice(overall_start_ts, None))
                if overall_end_ts:
                    ds_to_process = ds_to_process.sel(time=slice(None, overall_end_ts))

                if ds_to_process.time.size == 0:
                    _logger.debug(f"  No data in {Path(file_path).name} within the overall date range. Skipping.")
                    continue
                
                for var_name in variables:
                    if var_name not in ds_to_process:
                        _logger.warning(f"  Variable '{var_name}' not found in file {Path(file_path).name}. Skipping for this file.")
                        continue
                    
                    _logger.debug(f"  Calculating partial stats for '{var_name}' in {Path(file_path).name}...")
                    # Load data for the variable for this file as float64 for precision
                    var_data_np = ds_to_process[var_name].load().data.astype(np.float64)
                    
                    # Filter out NaNs for this file's data
                    valid_data = var_data_np[~np.isnan(var_data_np)]
                    
                    if valid_data.size > 0:
                        current_sum = np.sum(valid_data)
                        current_sum_sq = np.sum(valid_data**2)
                        current_count = valid_data.size

                        accumulated_stats[var_name]['sum_x'] += current_sum
                        accumulated_stats[var_name]['sum_x_sq'] += current_sum_sq
                        accumulated_stats[var_name]['count_n'] += current_count
                        _logger.debug(f"    {var_name}: sum={current_sum:.2f}, count={current_count}, sum_sq={current_sum_sq:.2f}")
                    else:
                        _logger.debug(f"    {var_name}: No valid data points in this file after filtering.")
        
        except Exception as e_file:
            _logger.error(f"Error processing file {file_path}: {e_file}", exc_info=True)
            if fail_on_error_per_file:
                _logger.error("Stopping due to fail_on_error_per_file=True.")
                return False
            _logger.warning(f"Skipping file {file_path} due to error.")
            continue 

    # Calculate final global statistics
    final_stats_to_save = {}
    _logger.info("\n--- Global Statistics (Aggregated from File-by-File Processing) ---")
    all_successful = True
    for var_name, acc_data in accumulated_stats.items():
        count_n = acc_data['count_n']
        sum_x = acc_data['sum_x']
        sum_x_sq = acc_data['sum_x_sq']

        if count_n > 0:
            mean = sum_x / count_n
            # Variance = E[X^2] - (E[X])^2 = (sum_x_sq / count_n) - mean^2
            variance = (sum_x_sq / count_n) - (mean**2)
            if variance < 0:
                if np.isclose(variance, 0): variance = 0.0
                else: _logger.warning(f"Computed negative variance ({variance}) for {var_name}. Clamping to 0."); variance = 0.0
            std_dev = np.sqrt(variance)
            
            final_stats_to_save[var_name] = {'mean': float(mean), 'std': float(std_dev), 'count': int(count_n)}
            _logger.info(f"Variable: {var_name}")
            _logger.info(f"  Global Mean: {mean:.6f}")
            _logger.info(f"  Global Std Dev: {std_dev:.6f}")
            _logger.info(f"  Total Valid Points: {count_n}")
        else:
            _logger.warning(f"No valid data points found for variable '{var_name}' across all files. Stats will be NaN.")
            final_stats_to_save[var_name] = {'mean': float('nan'), 'std': float('nan'), 'count': 0}
            all_successful = False # Mark as partially failed if a variable has no data

    # Save statistics to JSON file
    try:
        output_dir = os.path.dirname(output_stats_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            _logger.info(f"Created output directory for stats file: {output_dir}")
        with open(output_stats_file, 'w') as f:
            json.dump(final_stats_to_save, f, indent=4)
        _logger.info(f"Global statistics saved to: {output_stats_file}")
    except Exception as e:
        _logger.error(f"Failed to save statistics file: {e}", exc_info=True)
        return False
        
    return all_successful


def main():
    parser = argparse.ArgumentParser(description="Calculate global statistics from RADKLIM NetCDF files (file-by-file).")
    
    parser.add_argument("input_glob_pattern", type=str,
                        help="Glob pattern for input NetCDF files (e.g., 'data_directory/*/*.nc')")
    parser.add_argument("output_stats_file", type=str,
                        help="Path to save the output JSON statistics file (e.g., 'radklim_global_stats.json')")
    parser.add_argument("--variables", nargs='+', default=['RR'],
                        help="List of variables to calculate statistics for (default: ['RR'])")
    parser.add_argument("--start_date", type=str, default=None,
                        help="Overall start date filter (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS), inclusive.")
    parser.add_argument("--end_date", type=str, default=None,
                        help="Overall end date filter (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS), inclusive.")
    parser.add_argument("--fail_on_file_error", action="store_true", # Renamed for clarity
                        help="If set, the script will stop if an error occurs while processing any single file.")
    
    args = parser.parse_args()
    
    success = calculate_and_save_stats_file_by_file(
        args.input_glob_pattern,
        args.variables,
        args.output_stats_file,
        args.start_date,
        args.end_date,
        fail_on_error_per_file=args.fail_on_file_error
    )
    
    if success:
        _logger.info("Global statistics calculation finished successfully.")
    else:
        _logger.error("Global statistics calculation encountered errors or failed for some variables/files.")
    _logger.info("Script finished.")

if __name__ == "__main__":
    main()