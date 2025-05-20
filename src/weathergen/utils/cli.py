import argparse
from pathlib import Path


def get_train_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    _add_general_arguments(parser)

    return parser


def get_continue_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    _add_general_arguments(parser)
    _add_model_loading_params(parser)
    
    parser.add_argument(
        "--finetune_forecast",
        action="store_true",
        help="Fine tune for forecasting. It overwrites some of the Config settings. Overwrites specified with --config take precedence.",
    )

    return parser


def get_evaluate_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    _add_model_loading_params(parser)
    _add_general_arguments(parser)

    parser.add_argument(
        "--start_date",
        "-start",
        type=str,
        required=False,
        default="2022-10-01",
        help="Start date for evaluation. Format must be parsable with pd.to_datetime.",
    )
    parser.add_argument(
        "--end_date",
        "-end",
        type=str,
        required=False,
        default="2022-12-01",
        help="End date for evaluation. Format must be parsable with pd.to_datetime.",
    )
    parser.add_argument(
        "--samples", type=int, default=10000000, help="Number of evaluation samples."
    )
    parser.add_argument(
        "--save_samples", type=bool, default=True, help="Save samples from evaluation."
    )
    parser.add_argument(
        "--analysis_streams_output",
        nargs="+",
        default=["ERA5"],
        help="Analysis output streams during evaluation.",
    )

    return parser

def _add_general_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--private_config",
        type=Path,
        default=None,
        help="Path to the private configuration file that includes platform specific information like paths.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        nargs="+",
        help="Optional experiment specfic configuration files in ascending order of precedence.",
    )
    parser.add_argument(
        "--run_id",
        default=False,
        help="Store training and evaluation artifacts under the given same run_id",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="Overwrite individual config options. This takes precedence over overwrites passed via --config or --finetune_forecast. Individual items should be of the form: parent_obj.nested_obj=value",
    )


def _add_model_loading_params(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-id",
        "--run_id_base",
        required=True,
        help="run id of the pretrained WeatherGenerator model to be used.",
    )
    parser.add_argument(
        "-e",
        "--epoch",
        type=int,
        default=-1,
        help="Epoch of pretrained WeatherGenerator model used (Default -1 corresponds to the last checkpoint).",
    )
