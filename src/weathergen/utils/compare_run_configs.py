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
import logging
import os

import pandas as pd
import yaml
from omegaconf import OmegaConf

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
        "--max-length", type=int, default=30, help="Maximum length for config values."
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
    for item in config_files:
        # Handle both formats: [run_id, path] or just path
        if isinstance(item, list) and len(item) == 2:
            run_id, path = item
        else:
            # If it's just a path, extract run_id from filename
            path = item
            run_id = os.path.splitext(os.path.basename(path))[0]

        _logger.info(f"Loading config for run_id: {run_id} from {path}")
        cfg = load_model_config(run_id, None, path)

        # Override run_id if present in config, otherwise use the one we determined
        actual_run_id = cfg.get("run_id", run_id)

        # Debug: log the original config structure
        _logger.debug(f"Original config keys: {list(cfg.keys())}")
        if "streams" in cfg:
            _logger.debug(f"Streams type: {type(cfg['streams'])}, value: {cfg['streams']}")

        # Special case: if 'streams' is a key and is a dict or list, unpack it BEFORE flattening
        if "streams" in cfg:
            streams_val = cfg["streams"]
            _logger.debug(f"Streams type: {type(streams_val)}")

            # Convert OmegaConf objects to regular Python objects
            if hasattr(streams_val, "_content"):  # OmegaConf object
                streams_val = OmegaConf.to_object(streams_val)
                _logger.debug(f"Converted streams to Python object: {type(streams_val)}")

            # If it's a list, unpack each dict in the list
            if isinstance(streams_val, list):
                for i, stream in enumerate(streams_val):
                    if isinstance(stream, dict):
                        for k, v in stream.items():
                            cfg[f"streams[{i}].{k}"] = v
                    else:
                        cfg[f"streams[{i}]"] = stream
            elif isinstance(streams_val, dict):
                for k, v in streams_val.items():
                    cfg[f"streams.{k}"] = v
            else:
                # If streams is not a dict or list, keep it as is
                cfg["streams.value"] = streams_val
            # Remove the original 'streams' key
            del cfg["streams"]
            streams_keys = [k for k in cfg.keys() if k.startswith("streams")]
            _logger.debug(f"After streams processing: {streams_keys}")

        # Now flatten the modified config
        flat_cfg = flatten_dict(cfg)
        flattened_streams_keys = [k for k in flat_cfg.keys() if "streams" in k]
        _logger.debug(f"Final flattened keys with 'streams': {flattened_streams_keys}")
        configs[actual_run_id] = flat_cfg

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
