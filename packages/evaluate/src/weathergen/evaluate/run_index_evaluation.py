#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "weathergen-evaluate",
#   "weathergen-common",
# ]
# [tool.uv.sources]
# weathergen-evaluate = { path = "../../../../../packages/evaluate" }
# ///

# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""CLI entry point for climate-index evaluation (e.g. NAM), separate from
`run_evaluation.py`'s scores/plotting pipeline. No plotting, no MLflow push
in v1."""

# Standard library
import argparse
import logging
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

# Local application / package
from weathergen.common.logger import init_loggers
from weathergen.common.paths import _REPO_ROOT

from weathergen.evaluate.indices.index_orchestration import (
    calc_indices_per_stream,
    index_list_to_json,
)
from weathergen.evaluate.run_evaluation import get_reader
from weathergen.evaluate.utils.dict_utils import parse_metric_params

_logger = logging.getLogger(__name__)


def evaluate_indices() -> None:
    """Entry point for the index evaluation script."""
    evaluate_indices_from_args(sys.argv[1:])


def evaluate_indices_from_args(argl: list[str]) -> None:
    """Parse CLI args and run index evaluation.

    Parameters
    ----------
    argl : list[str]
        List of arguments passed from the terminal.
    """
    init_loggers()
    parser = argparse.ArgumentParser(
        description="Climate-index evaluation of WeatherGenerator runs."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the configuration yaml file. e.g. config/evaluate/config_indices.yml",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        default=[],
        help=(
            "Overwrite individual config options."
            " Individual items should be of the form: parent_obj.nested_obj=value."
            " NOTE: cannot be used for run_ids (use --run-ids instead)."
        ),
    )
    parser.add_argument(
        "--run-ids",
        nargs="+",
        default=None,
        help=(
            "Filter run_ids from the config to only these."
            " E.g. --run-ids wu4wy9os fy6fgscn so67dku1"
        ),
    )

    args = parser.parse_args(argl)
    if args.config:
        config = Path(args.config)
    else:
        _logger.info(
            "No config file provided, using the default template config (please edit accordingly)"
        )
        config = Path(_REPO_ROOT / "config" / "evaluate" / "config_indices.yml")

    cf = OmegaConf.load(config)
    assert isinstance(cf, DictConfig)

    # Disable struct flag so that --options and --run-ids can freely modify keys.
    OmegaConf.set_struct(cf, False)

    if args.options:
        cli_items = [item for item in args.options if not item.startswith("run_ids=")]
        if len(cli_items) != len(args.options):
            _logger.warning(
                "run_ids= in --options is not supported (it's a dict, not a list). "
                "Use --run-ids instead. Ignoring run_ids= items."
            )
        if cli_items:
            cli_overwrite = OmegaConf.from_cli(cli_items)
            cf = OmegaConf.merge(cf, cli_overwrite)
            _logger.info(f"Applied --options overwrites: {cli_items}")

    if args.run_ids:
        existing = cf.get("run_ids", {})
        cf.run_ids = {k: existing.get(k, {}) for k in args.run_ids}
        _logger.info(f"Overwritten run_ids to: {args.run_ids}")

    evaluate_indices_from_config(cf)


def evaluate_indices_from_config(cfg: DictConfig) -> None:
    """Main function that computes and stores climate indices for all configured runs.

    Parameters
    ----------
    cfg : DictConfig
        Configuration loaded from a config_indices.yml-shaped file.
    """
    runs = cfg.run_ids
    _logger.info(f"Detected {len(runs)} runs")
    private_paths = cfg.get("private_paths")
    default_streams = cfg.get("default_streams", {})
    max_workers = cfg.get("max_workers")

    for run_id, run in runs.items():
        if "streams" not in run:
            run["streams"] = default_streams
        if max_workers is not None and "max_workers" not in run:
            run["max_workers"] = max_workers

        reader = get_reader(run.get("type", "zarr"), run, run_id, private_paths)

        for stream in run.get("streams", {}):
            stream_dict = reader.get_stream(stream)
            if not stream_dict:
                _logger.info(f"Stream {stream} not found for run {run_id}. Skipping.")
                continue

            indices_cfg = stream_dict.get("indices")
            if not indices_cfg:
                _logger.debug(f"No indices configured for {run_id} - {stream}. Skipping.")
                continue

            indices_dict = parse_metric_params(indices_cfg)
            computed = calc_indices_per_stream(reader, stream, indices_dict)
            if computed:
                index_list_to_json(reader, stream, computed)


if __name__ == "__main__":
    evaluate_indices()
