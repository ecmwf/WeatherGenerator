import pathlib

import pytest

import weathergen.utils.cli as cli

DATE_FORMATS = ["2022-12-01T00:00:00", "20221201", "2022-12-01", "12.01.2022"]
EXPECTED_DATE_STR = "202212010000"
MODEL_LOADING_ARGS = ["from_run_id", "epoch", "reuse_run_id"]
GENERAL_ARGS = ["config", "private_config", "options", "run_id"]
MODEL_LOADING_PARSERS = [cli.get_continue_parser(), cli.get_evaluate_parser()]
BASIC_ARGLIST = ["--from_run_id", "test123"]


@pytest.fixture
def evaluate_parser():
    return cli.get_evaluate_parser()


def test_private_config_is_path():
    argl = ["--private_config", "foo/bar"]

    args = cli.get_train_parser().parse_args(argl)

    assert args.private_config.name == "bar"


@pytest.mark.parametrize("files", [["foo/bar"], ["foo/bar", "baz"]])
def test_config_is_pathes(files):
    args = cli.get_train_parser().parse_args(["--config"] + files)

    assert all([isinstance(file, pathlib.Path) for file in args.config])


@pytest.mark.parametrize("overwrites", [["foo=/bar", "baz.foo=1"], ["foo=2"]])
def test_options(overwrites):
    args = cli.get_train_parser().parse_args(["--options"] + overwrites)

    assert all([overwrite in args.options for overwrite in overwrites])


def test_train_general_has_params():
    args = cli.get_train_parser().parse_args([])

    assert all([arg in vars(args).keys() for arg in GENERAL_ARGS])


@pytest.mark.parametrize("parser", MODEL_LOADING_PARSERS)
def test_general_has_params(parser):
    args = parser.parse_args(BASIC_ARGLIST)

    assert all([arg in vars(args).keys() for arg in GENERAL_ARGS])


@pytest.mark.parametrize("parser", MODEL_LOADING_PARSERS)
def test_model_loading_has_params(parser):
    args = parser.parse_args(BASIC_ARGLIST)

    assert all([arg in vars(args).keys() for arg in MODEL_LOADING_ARGS])


@pytest.mark.parametrize("streams", [["ERA5", "FOO"], ["BAR"]])
def test_evaluate_analysis_streams_output(evaluate_parser, streams):
    arglist = BASIC_ARGLIST + ["--analysis_streams_output", *streams]
    args = evaluate_parser.parse_args(arglist)

    assert args.analysis_streams_output == streams


def test_evaluate_analysis_streams_output_empty(evaluate_parser):
    arglist = BASIC_ARGLIST + ["--analysis_streams_output", *[]]

    with pytest.raises(SystemExit):
        evaluate_parser.parse_args(arglist)


def test_evaluate_defaults(evaluate_parser):
    default_args = [
        "start_date",
        "end_date",
        "samples",
        "analysis_streams_output",
        "epoch",
        "private_config",
    ]
    default_values = [evaluate_parser.get_default(arg) for arg in default_args]
    # apply custom type
    default_values[:2] = [cli._format_date(date) for date in default_values[:2]]

    args = evaluate_parser.parse_args(BASIC_ARGLIST)

    assert all(
        [
            getattr(args, arg) == default_value
            for arg, default_value in zip(default_args, default_values, strict=True)
        ]
    )


@pytest.mark.parametrize("date", DATE_FORMATS)
def test_evaluate_start_date(evaluate_parser, date):
    args = evaluate_parser.parse_args(BASIC_ARGLIST + ["--start_date", date])

    assert args.start_date == EXPECTED_DATE_STR


def test_evaluate_start_date_invalid(evaluate_parser):
    with pytest.raises(SystemExit):
        evaluate_parser.parse_args(BASIC_ARGLIST + ["--start_date", "foobar"])


@pytest.mark.parametrize("date", DATE_FORMATS)
def test_evaluate_end_date(evaluate_parser, date):
    args = evaluate_parser.parse_args(BASIC_ARGLIST + ["--end_date", date])

    assert args.end_date == EXPECTED_DATE_STR


def test_evaluate_end_date_invalid(evaluate_parser):
    with pytest.raises(SystemExit):
        evaluate_parser.parse_args(BASIC_ARGLIST + ["--end_date", "foobar"])
