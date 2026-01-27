import pathlib

import pytest

import weathergen.utils.cli as cli

DATE_FORMATS = ["2022-12-01T00:00:00", "20221201", "2022-12-01", "12.01.2022"]
EXPECTED_DATE_STR = "202212010000"
MODEL_LOADING_ARGS = ["from-run-id", "mini-epoch", "reuse-run-id"]
GENERAL_ARGS = ["config", "private-config", "options", "run-id"]
MODEL_LOADING_PARSERS = [cli.get_continue_parser(), cli.get_inference_parser()]
BASIC_ARGLIST = ["--from-run-id", "test123"]


@pytest.fixture
def inference_parser():
    return cli.get_inference_parser()


def test_private_config_is_path():
    argl = ["--private-config", "foo/bar"]

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
def test_inference_streams_output(inference_parser, streams):
    arglist = BASIC_ARGLIST + ["--streams-output", *streams]
    args = inference_parser.parse_args(arglist)

    assert args.streams_output == streams


def test_inference_streams_output_empty(inference_parser):
    arglist = BASIC_ARGLIST + ["--streams-output", *[]]

    with pytest.raises(SystemExit):
        inference_parser.parse_args(arglist)


def test_inference_defaults(inference_parser):
    default_args = [
        "start-date",
        "end-date",
        "samples",
        "streams-output",
        "mini-epoch",
        "private-config",
    ]
    default_values = [inference_parser.get_default(arg) for arg in default_args]
    # apply custom type
    default_values[:2] = [cli._format_date(date) for date in default_values[:2]]

    args = inference_parser.parse_args(BASIC_ARGLIST)

    assert all(
        [
            getattr(args, arg) == default_value
            for arg, default_value in zip(default_args, default_values, strict=True)
        ]
    )


@pytest.mark.parametrize("date", DATE_FORMATS)
def test_inference_start_date(inference_parser, date):
    args = inference_parser.parse_args(BASIC_ARGLIST + ["--start-date", date])

    assert args.start_date == EXPECTED_DATE_STR


def test_inference_start_date_invalid(inference_parser):
    with pytest.raises(SystemExit):
        inference_parser.parse_args(BASIC_ARGLIST + ["--start-date", "foobar"])


@pytest.mark.parametrize("date", DATE_FORMATS)
def test_inference_end_date(inference_parser, date):
    args = inference_parser.parse_args(BASIC_ARGLIST + ["--end-date", date])

    assert args.end_date == EXPECTED_DATE_STR


def test_inference_end_date_invalid(inference_parser):
    with pytest.raises(SystemExit):
        inference_parser.parse_args(BASIC_ARGLIST + ["--end-date", "foobar"])
