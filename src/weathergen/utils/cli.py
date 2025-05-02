import argparse
from pathlib import Path


def get_train_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--private_config",
        type=Path,
        default=None,
        help="Path to private configuration file for paths",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional experiment specfic configuration file",
    )
    
    return parser

def get_continue_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument(
        "-id",
        "--run_id",
        type=str,
        required=True,
        help="run id of to be continued",
    )
    parser.add_argument(
        "-e",
        "--epoch",
        type=int,
        required=False,
        default=-1,
        help="epoch where to continue run",
    )
    parser.add_argument(
        "-n",
        "--run_id_new",
        type=bool,
        required=False,
        default=False,
        help="create new run id for cont'd run",
    )
    parser.add_argument(
        "--private_config",
        type=Path,
        default=None,
        help="Path to private configuration file for paths.",
    )
    parser.add_argument(
        "--finetune_forecast",
        action="store_true",
        help="Fine tune for forecasting. It overwrites some of the Config settings.",
    )
    
    return parser

def get_evaluate_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="Run/model id of pretrained WeatherGenerator model.",
    )
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
        "--epoch",
        type=int,
        default=None,
        help="Epoch of pretrained WeatherGenerator model used for evaluation (Default None corresponds to the last checkpoint).",
    )
    parser.add_argument(
        "--forecast_steps",
        type=int,
        default=None,
        help="Number of forecast steps for evaluation. Uses attribute from config when None is set.",
    )
    parser.add_argument(
        "--samples", type=int, default=10000000, help="Number of evaluation samples."
    )
    parser.add_argument(
        "--shuffle", type=bool, default=False, help="Shuffle samples from evaluation."
    )
    parser.add_argument(
        "--save_samples", type=bool, default=True, help="Save samples from evaluation."
    )
    parser.add_argument(
        "--analysis_streams_output",
        type=list,
        default=["ERA5"],
        help="Analysis output streams during evaluation.",
    )
    parser.add_argument(
        "--private_config",
        type=Path,
        default=None,
        help="Path to private configuration file for paths.",
    )
    parser.add_argument(
        "--eval_run_id",
        type=str,
        required=False,
        dest="eval_run_id",
        help="(optional) if specified, uses the provided run id to store the evaluation results",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional experiment specfic configuration file",
    )
    
    return parser