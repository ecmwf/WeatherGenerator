#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "pandas",
#   "tabulate",
#   "pyyaml",
#   "omegaconf",
#   "weathergen",
# ]
# [tool.uv.sources]
# weathergen = { path = "../../../" }
# ///

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
from pathlib import Path

import pandas as pd
import yaml
from omegaconf import OmegaConf

from weathergen.common.config import load_run_config


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
    """Build DataFrame with configs, filtering identical rows unless in always_show_patterns."""
    always_show_patterns = always_show_patterns or []

    all_keys = sorted({k for conf in configs.values() for k in conf})
    run_ids = list(configs.keys())
    data = {k: [configs[run_id].get(k, "") for run_id in run_ids] for k in all_keys}
    df = pd.DataFrame(data, index=run_ids).T

    # Truncate and filter
    df = df.map(lambda x: truncate_value(x, max_value_length))
    varying_rows = df.astype(str).apply(lambda row: len(set(row)) > 1, axis=1)
    always_show_rows = df.index.to_series().apply(
        lambda key: key_matches_patterns(key, always_show_patterns)
    )
    return df[varying_rows | always_show_rows]


def highlight_row(row: pd.Series) -> pd.Series:
    """Bold all values in a row if there are differences."""
    if len(set(row.astype(str))) <= 1:
        return row
    return pd.Series([f"**{v}**" if v != "" else v for v in row], index=row.index)


def row_has_bold(row: pd.Series) -> bool:
    """Return True if any value in the row is bolded."""
    return any(isinstance(v, str) and v.startswith("**") for v in row)


def configs_to_markdown_table(
    configs: dict, max_value_length: int = 50, always_show_patterns: list = None
) -> str:
    """Generate a markdown table comparing all config parameters across runs."""
    df = build_config_dataframe(configs, max_value_length, always_show_patterns)
    df_highlighted = df.apply(highlight_row, axis=1)
    # Move rows with bold values to the top
    bold_mask = df_highlighted.apply(row_has_bold, axis=1)
    df_sorted = pd.concat([df_highlighted[bold_mask], df_highlighted[~bold_mask]])
    return df_sorted.to_markdown(tablefmt="github")


def process_streams(cfg: dict | None):
    """Process and flatten streams configuration."""
    if "streams" not in cfg:
        return

    streams_val = cfg["streams"]

    # Convert OmegaConf objects to regular Python objects
    if hasattr(streams_val, "_content"):
        streams_val = OmegaConf.to_object(streams_val)

    # Unpack streams based on type
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
        cfg["streams.value"] = streams_val

    del cfg["streams"]


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Compare WeatherGenerator configs and output markdown table."
    )
    parser.add_argument("-r1", "--run_id_1", required=False)
    parser.add_argument("-r2", "--run_id_2", required=False)
    parser.add_argument(
        "-m1",
        "--model_directory_1",
        type=Path,
        default=Path("models/"),
        help="Path to model directory for -r1/--run_id_1",
    )
    parser.add_argument(
        "-m2",
        "--model_directory_2",
        type=Path,
        default=Path("models/"),
        help="Path to model directory for -r2/--run_id_2",
    )
    parser.add_argument(
        "--config",
        default="config/compare_config_list.yml",
        help="Path to YAML file listing run_ids and always_show_patterns.",
    )
    parser.add_argument(
        "output", nargs="?", default="reports/compare_configs.md", help="Output markdown file path."
    )
    parser.add_argument(
        "--max-length", type=int, default=30, help="Maximum length for config values."
    )
    parser.add_argument(
        "--show",
        type=str,
        default=[],
        help=(
            "Put '*' to show all parameters, or leave empty to only show changed parameters. "
            "Use for example 'ae_global' to show all parameters starting with 'ae_global'."
        ),
    )

    args = parser.parse_args()

    if args.run_id_1 and args.run_id_2:
        config_files = [
            [args.run_id_1, args.model_directory_1],
            [args.run_id_2, args.model_directory_2],
        ]
        yaml_always_show_patterns = args.show if args.show else []
    # Read YAML config list if exists
    elif Path(args.config).exists():
        with open(args.config) as f:
            yaml_data = yaml.safe_load(f)

        config_files = yaml_data["run_ids"]
        yaml_always_show_patterns = yaml_data.get("always_show_patterns", [])
    else:
        # error: pass config or command line arguments
        logger.error(
            "Please provide a config list (.yml format) or specify two run IDs "
            "and their model directories."
        )
        return
    # Load configs using load_run_config from config module
    configs = {}
    for item in config_files:
        # Handle both formats: [run_id, path] or just path
        if isinstance(item, list) and len(item) == 2:
            run_id, path = item
        else:
            path = item
            run_id = os.path.splitext(os.path.basename(path))[0]

        logger.info(f"Loading config for run_id: {run_id} from {path}")
        try:
            cfg = load_run_config(run_id=run_id, mini_epoch=None, model_path=path)
        except Exception:
            logger.warning(
                f"Failed to load config for run_id: {run_id} from {path}",
                "Assuming mini_epoch=0 and retrying.",
            )
            cfg = load_run_config(run_id=run_id, mini_epoch=0, model_path=path)
        actual_run_id = cfg.get("run_id", run_id)

        # Process streams and flatten
        process_streams(cfg)
        flat_cfg = flatten_dict(cfg)
        configs[actual_run_id] = flat_cfg

    # Generate markdown table
    md_table = configs_to_markdown_table(configs, args.max_length, yaml_always_show_patterns)

    # Prepare output file name with run ids
    run_ids = [str(rid) for rid in configs.keys()]
    run_ids_str = "_".join(run_ids)
    output_path = args.output
    # Ensure 'reports' folder exists
    reports_dir = os.path.dirname(output_path) or "reports"
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir, exist_ok=True)
    # Insert run ids into filename before extension
    base, ext = os.path.splitext(os.path.basename(output_path))
    output_file = os.path.join(reports_dir, f"{base}_{run_ids_str}{ext}")

    # Write output
    with open(output_file, "w") as f:
        f.write(md_table)
    logger.info(f"Table written to {output_file}")
    row_count = len(md_table.split("\n")) - 3
    pattern_info = (
        f" (patterns: {', '.join(yaml_always_show_patterns)})" if yaml_always_show_patterns else ""
    )
    logger.info(f"Filtered to {row_count} rows{pattern_info}")


if __name__ == "__main__":
    main()
