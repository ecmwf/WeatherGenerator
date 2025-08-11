# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import argparse
import fnmatch
import json
import logging
import os

import pandas as pd
import yaml

from config import load_model_config

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def truncate_value(value, max_length=50):
    """
    Truncate long string values to reduce table width.
    """
    if isinstance(value, str) and len(value) > max_length:
        return value[: max_length - 3] + "..."
    return value


def read_configs_from_list(list_yml):
    """
    - Reads a YAML file containing config file paths and always show patterns,
    loads and flattens each config.
    - Special-cases the 'streams' key to unpack its contents into separate rows.
    - Returns a tuple of (configs dict, always_show_patterns list).
    """
    with open(list_yml) as f:
        yaml_data = yaml.safe_load(f)

    # Expect dict format with run_ids key
    if not isinstance(yaml_data, dict):
        raise ValueError("YAML file must contain a dict with 'run_ids' key.")

    if "run_ids" not in yaml_data:
        raise ValueError("YAML file must contain 'run_ids' key with list of config file paths.")

    config_files = yaml_data["run_ids"]
    always_show_patterns = yaml_data.get("always_show_patterns", [])

    if not isinstance(config_files, list):
        raise ValueError("'run_ids' must be a list of config file paths.")

    configs = {}
    for path in config_files:
        with open(path) as f:
            cfg = json.load(f)
        # Use run_id from config file, fallback to filename if not present
        run_id = cfg.get("run_id", os.path.splitext(os.path.basename(path))[0])
        flat_cfg = flatten_dict(cfg)
        # Special case: if 'streams' is a key and is a dict or list, unpack it
        # if "streams" in cfg:
        #     streams_val = cfg["streams"]
        #     # If it's a list, unpack each dict in the list
        #     if isinstance(streams_val, list):
        #         for i, stream in enumerate(streams_val):
        #             if isinstance(stream, dict):
        #                 for k, v in stream.items():
        #                     flat_cfg[f"streams[{i}].{k}"] = v
        #     elif isinstance(streams_val, dict):
        #         for k, v in streams_val.items():
        #             flat_cfg[f"streams.{k}"] = v
        #     # Remove the original 'streams' key if present
        #     if "streams" in flat_cfg:
        #         del flat_cfg["streams"]
        configs[run_id] = flat_cfg
    return configs, always_show_patterns


def flatten_dict(d, parent_key="", sep="."):
    """
    Recursively flattens a nested dictionary, joining keys with sep.
    Returns a flat dictionary with compound keys.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def key_matches_patterns(key: str, patterns: list) -> bool:
    """
    Check if a key matches any of the wildcard patterns.
    """
    if not patterns:
        return False
    return any(fnmatch.fnmatch(key, pattern) for pattern in patterns)


def build_config_dataframe(
    configs: dict, max_value_length: int = 50, always_show_patterns: list = None
) -> pd.DataFrame:
    """
    Build a DataFrame with all config keys as rows and run_ids as columns.
    Filters out rows where all values are identical to reduce table size.
    Truncates long values to keep table compact.
    Keys matching always_show_patterns are always included even if identical.
    """
    if always_show_patterns is None:
        always_show_patterns = []

    all_keys = sorted({k for conf in configs.values() for k in conf})
    run_ids = list(configs.keys())
    data = {k: [configs[run_id].get(k, "") for run_id in run_ids] for k in all_keys}
    df = pd.DataFrame(data, index=run_ids).T

    # Truncate long values
    df = df.map(lambda x: truncate_value(x, max_value_length))

    # Filter out rows where all values are identical (no differences)
    # But keep rows that match always_show_patterns
    str_df = df.astype(str)
    varying_rows = str_df.apply(lambda row: len(set(row)) > 1, axis=1)
    always_show_rows = df.index.to_series().apply(
        lambda key: key_matches_patterns(key, always_show_patterns)
    )
    keep_rows = varying_rows | always_show_rows
    df_filtered = df[keep_rows]

    return df_filtered


def highlight_row(row: pd.Series) -> pd.Series:
    """
    Bold all values in a row if there are any differences between values.
    """
    str_row = row.astype(str)
    unique = set(str_row)
    if len(unique) <= 1:
        return row
    # If there are differences, highlight all values
    return pd.Series([f"**{v}**" if v != "" else v for v in row], index=row.index)


def row_has_bold(row: pd.Series) -> bool:
    """
    Return True if any value in the row is bolded.
    """
    return any(isinstance(v, str) and v.startswith("**") for v in row)


def configs_to_markdown_table(
    configs: dict, max_value_length: int = 50, always_show_patterns: list = None
) -> str:
    """
    Generate a markdown table comparing all config parameters across runs.
    """
    df = build_config_dataframe(configs, max_value_length, always_show_patterns)
    df_highlighted = df.apply(highlight_row, axis=1)
    # Move rows with any bolded value to the top
    bold_mask = df_highlighted.apply(row_has_bold, axis=1)
    df_highlighted = pd.concat([df_highlighted[bold_mask], df_highlighted[~bold_mask]])
    return df_highlighted.to_markdown(tablefmt="github")


def main():
    parser = argparse.ArgumentParser(
        description="Compare WeatherGenerator configs and output markdown table."
    )
    parser.add_argument(
        "config", help="Path to YAML file listing run_ids and always_show_patterns."
    )
    parser.add_argument(
        "output", nargs="?", default="configs.md", help="Output markdown file path."
    )
    parser.add_argument(
        "--max-length", type=int, default=50, help="Maximum length for config values."
    )
    parser.add_argument(
        "--always-show",
        action="append",
        default=None,
        help="Override YAML always_show_patterns (can be repeated).",
    )

    args = parser.parse_args()

    # Read YAML config list
    with open(args.config) as f:
        yaml_data = yaml.safe_load(f)

    config_files = yaml_data["run_ids"]
    yaml_always_show_patterns = yaml_data.get("always_show_patterns", [])

    # Load configs using load_model_config from config module
    configs = {}
    for path in config_files:
        cfg = load_model_config(path)
        run_id = cfg.get("run_id", os.path.splitext(os.path.basename(path))[0])
        flat_cfg = flatten_dict(cfg)
        configs[run_id] = flat_cfg

    # CLI patterns override YAML patterns if provided
    final_patterns = args.always_show if args.always_show else yaml_always_show_patterns

    # Generate markdown table
    md_table = configs_to_markdown_table(configs, args.max_length, final_patterns)

    # Write output
    if args.output:
        with open(args.output, "w") as f:
            f.write(md_table)
        _logger.info(f"Table written to {args.output}")
        row_count = len(md_table.split("\n")) - 3
        pattern_info = f" (patterns: {', '.join(final_patterns)})" if final_patterns else ""
        _logger.info(f"Filtered to {row_count} rows{pattern_info}")
    else:
        _logger.info(md_table)


if __name__ == "__main__":
    main()
