import pathlib

import pytest

import weathergen.utils.cli as cli

DATE_FORMATS = ["2022-12-01T00:00:00", "20221201", "2022-12-01", "12.01.2022"]
EXPECTED_DATE_STR = "202212010000"
# argparse stores dashed flags (--from-run-id) under underscored names (args.from_run_id)
MODEL_LOADING_ARGS = ["from_run_id", "mini_epoch", "reuse_run_id"]
GENERAL_ARGS = ["config", "private_config", "options", "run_id", "base_config"]
MODEL_LOADING_PARSERS = [cli.get_continue_parser(), cli.get_inference_parser()]
BASIC_ARGLIST = ["--from-run-id", "test123"]


@pytest.fixture
def inference_parser():
    return cli.get_inference_parser()


def test_private_config_is_path():
    argl = ["--private-config", "foo/bar"]

    args = cli.get_train_parser().parse_args(argl)

    assert args.private_config.name == "bar"


def test_base_config_is_path():
    args = cli.get_train_parser().parse_args(["--base-config", "foo/bar.yml"])

    assert args.base_config == pathlib.Path("foo/bar.yml")


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


@pytest.mark.parametrize("parser", MODEL_LOADING_PARSERS)
def test_model_loading_requires_from_run_id(parser):
    with pytest.raises(SystemExit):
        parser.parse_args([])


@pytest.mark.parametrize("parser", MODEL_LOADING_PARSERS)
@pytest.mark.parametrize("flag", ["--from-run-id", "-id"])
def test_model_loading_from_run_id(parser, flag):
    args = parser.parse_args([flag, "abc456", "--reuse-run-id", "-e", "3"])

    assert args.from_run_id == "abc456"
    assert args.reuse_run_id
    assert args.mini_epoch == 3


def test_continue_finetune_forecast():
    parser = cli.get_continue_parser()

    assert not parser.parse_args(BASIC_ARGLIST).finetune_forecast
    assert parser.parse_args(BASIC_ARGLIST + ["--finetune-forecast"]).finetune_forecast


def test_inference_defaults(inference_parser):
    expected_defaults = {
        "from_run_id": "test123",
        "mini_epoch": -1,
        "reuse_run_id": False,
        "private_config": None,
        "config": [],
        "run_id": None,
        "options": [],
        "base_config": None,
    }

    args = inference_parser.parse_args(BASIC_ARGLIST)

    assert vars(args) == expected_defaults


@pytest.mark.parametrize("date", DATE_FORMATS)
def test_format_date(date):
    assert cli._format_date(date) == EXPECTED_DATE_STR


def test_format_date_invalid():
    with pytest.raises(ValueError):
        cli._format_date("foobar")
