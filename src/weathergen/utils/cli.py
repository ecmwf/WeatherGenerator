import argparse
from pathlib import Path

import pandas as pd


def get_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    _add_general_arguments(parser)

    return parser


def get_continue_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)

    _add_general_arguments(parser)
    _add_model_loading_params(parser)

    parser.add_argument(
        "--finetune-forecast",
        action="store_true",
        help=(
            "Fine tune for forecasting. It overwrites some of the Config settings. "
            "Overwrites specified with --config take precedence."
        ),
    )

    return parser


def get_inference_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)

    _add_model_loading_params(parser)
    _add_general_arguments(parser)

    parser.add_argument(
        "--start-date",
        "-start",
        type=_format_date,
        default="2022-10-01",
        help="Start date for inference. Format must be parsable with pd.to_datetime.",
    )
    parser.add_argument(
        "--end-date",
        "-end",
        type=_format_date,
        default="2022-12-01",
        help="End date for inference. Format must be parsable with pd.to_datetime.",
    )
    parser.add_argument(
        "--samples", type=int, default=10000000, help="Number of inference samples."
    )
    parser.add_argument(  # behaviour changed => implies default=False
        "--save-samples",
        type=bool,
        default=True,
        help="Toggle saving of samples from inference. Default True",
    )
    parser.add_argument(
        "--streams-output",
        nargs="+",
        help="Output streams during inference.",
    )

    return parser


def _format_date(date: str) -> str:
    try:
        parsed = pd.to_datetime(date, errors="raise")
    except (pd.errors.ParserError, ValueError) as e:
        msg = f"Can not parse a valid date from input: {date}, with type {type(date)}."
        raise ValueError(msg) from e

    return parsed.strftime("%Y%m%d%H%M")


def _add_general_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--private-config",
        type=Path,
        default=None,
        help=(
            "Path to the private configuration file that includes platform specific information "
            " like paths."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        nargs="*",
        default=[],
        help="Optional experiment specfic configuration files in ascending order of precedence.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help=(
            "The run id for this run."
            " All artifacts (models, metrics, ...) will be stored under this run_id."
            " If not provided, a new run_id will be created"
        ),
    )
    parser.add_argument(
        "--options",
        nargs="+",
        default=[],
        help=(
            "Overwrite individual config options."
            " This takes precedence over overwrites passed via --config or --finetune_forecast."
            " Individual items should be of the form: parent_obj.nested_obj=value"
        ),
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        help=(
            "Path to the base configuration file."
            "If not provided, ./config/default_config.yml is used."
        ),
    )


def _add_model_loading_params(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-id",
        "--from-run-id",
        required=True,
        help=(
            "Start inference or continue training from the WeatherGenerator"
            " model with the given run id."
        ),
    )
    parser.add_argument(
        "-e",
        "--mini-epoch",
        type=int,
        default=-1,
        help=(
            "Mini-epoch of pretrained WeatherGenerator model used"
            " (Default -1 corresponds to the last checkpoint)."
        ),
    )
    parser.add_argument(
        "--reuse-run-id",
        action="store_true",
        help="Use the id given via --from-run-id also for the current run. "
        "The storage location for artifacts will be reused as well. "
        "This might overwrite artifacts from previous runs.",
    )
