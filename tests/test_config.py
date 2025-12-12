import pathlib
import tempfile

import pytest
from omegaconf import OmegaConf

import weathergen.common.config as config

TEST_RUN_ID = "test123"
SECRET_COMPONENT = "53CR3T"
DUMMY_PRIVATE_CONF = {
    "data_path_anemoi": "/path/to/anmoi/data",
    "data_path_obs": "/path/to/observation/data",
    "secrets": {
        "my_big_secret": {
            "my_secret_id": f"{SECRET_COMPONENT}01234",
            "my_secret_access_key": SECRET_COMPONENT,
        }
    },
}

DUMMY_OVERWRITES = [("num_mini_epochs", 42), ("healpix_level", 42)]

DUMMY_STREAM_CONF = {
    "ERA5": {
        "type": "anemoi",
        "filenames": ["aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8.zarr"],
        "source": ["u_", "v_", "10u", "10v"],
        "target": ["10u", "10v"],
        "loss_weight": 1.0,
        "diagnostic": False,
        "masking_rate": 0.6,
        "masking_rate_none": 0.05,
        "token_size": 32,
        "embed2": {
            "net": "transformer",
            "num_tokens": 1,
            "num_heads": 4,
            "dim_embed": 128,
            "num_blocks": 2,
        },
        "embed_target_coords": {"net": "linear", "dim_embed": 128},
        "target_readout": {
            "type": "obs_value",  # token or obs_value
            "num_layers": 2,
            "num_heads": 4,
            # "sampling_rate" : 0.2
        },
        "pred_head": {"ens_size": 1, "num_layers": 1},
    }
}
DUMMY_MULTIPLE_STREAM_CONF = DUMMY_STREAM_CONF | {"FOO": DUMMY_STREAM_CONF["ERA5"]}

VALID_STREAMS = [
    (pathlib.Path("test.yml"), DUMMY_STREAM_CONF),
    (pathlib.Path("foo/test.yml"), DUMMY_STREAM_CONF),
    (pathlib.Path("bar/foo/test.yml"), DUMMY_STREAM_CONF),
]

EXCLUDED_STREAMS = [
    (pathlib.Path(".test.yml"), DUMMY_STREAM_CONF),
    (pathlib.Path("#test.yml"), DUMMY_STREAM_CONF),
]


def contains_keys(super_config, sub_config):
    keys_present = [key in super_config.keys() for key in sub_config.keys()]

    return all(keys_present)


def contains_values(super_config, sub_config):
    correct_values = [super_config[key] == value for key, value in sub_config.items()]

    return all(correct_values)


def contains(super_config, sub_config):
    return contains_keys(super_config, sub_config) and contains_values(super_config, sub_config)


def is_equal(config1, config2):
    return contains(config1, config2) and contains(config2, config1)


@pytest.fixture
def shared_working_dir():
    with tempfile.TemporaryDirectory(prefix="models") as temp_dir:
        yield temp_dir


def write_stream_file(write_path, content: str):
    write_path.parent.mkdir(parents=True, exist_ok=True)
    with open(write_path, "a") as stream_file:
        stream_file.write(content)


def get_expected_config(key, config):
    config = OmegaConf.create(config)
    config.name = key
    return config


@pytest.fixture
def streams_dir():
    with tempfile.TemporaryDirectory(prefix="streams") as temp_dir:
        yield pathlib.Path(temp_dir)


@pytest.fixture
def private_conf(shared_working_dir):
    cf = OmegaConf.create(DUMMY_PRIVATE_CONF)
    cf.path_shared_working_dir = shared_working_dir
    cf.path_shared_slurm_dir = shared_working_dir

    return cf


@pytest.fixture
def private_config_file(private_conf):
    with tempfile.NamedTemporaryFile("w+") as temp:
        temp.write(OmegaConf.to_yaml(private_conf))
        temp.flush()
        yield pathlib.Path(temp.name)


@pytest.fixture
def overwrite_dict(request):
    key, value = request.param
    return {key: value}


@pytest.fixture
def overwrite_config(overwrite_dict):
    return OmegaConf.create(overwrite_dict)


@pytest.fixture
def overwrite_file(overwrite_config):
    # TODO should this be "w+t" instead of "w"?
    with tempfile.NamedTemporaryFile("w+") as temp:
        temp.write(OmegaConf.to_yaml(overwrite_config))
        temp.flush()
        yield pathlib.Path(temp.name)


@pytest.fixture
def config_fresh(private_config_file):
    cf = config.load_merge_configs(private_config_file, None, None)
    cf = config.set_run_id(cf, TEST_RUN_ID, False)
    cf.data_loader_rng_seed = 42

    return cf


def test_contains_private(config_fresh):
    sanitized_private_conf = DUMMY_PRIVATE_CONF.copy()
    del sanitized_private_conf["secrets"]
    assert contains_keys(config_fresh, sanitized_private_conf)


def test_is_paths_set(config_fresh):
    paths = {"model_path": "foo", "run_path": "bar"}

    assert contains_keys(config_fresh, paths)


@pytest.mark.parametrize("overwrite_dict", DUMMY_OVERWRITES, indirect=True)
def test_load_with_overwrite_dict(overwrite_dict, private_config_file):
    cf = config.load_merge_configs(private_config_file, None, None, overwrite_dict)

    assert contains(cf, overwrite_dict)


@pytest.mark.parametrize("overwrite_dict", DUMMY_OVERWRITES, indirect=True)
def test_load_with_overwrite_config(overwrite_config, private_config_file):
    cf = config.load_merge_configs(private_config_file, None, None, overwrite_config)

    assert contains(cf, overwrite_config)


@pytest.mark.parametrize("overwrite_dict", DUMMY_OVERWRITES, indirect=True)
def test_load_with_overwrite_file(private_config_file, overwrite_file):
    sub_cf = OmegaConf.load(overwrite_file)
    cf = config.load_merge_configs(private_config_file, None, None, overwrite_file)

    assert contains(cf, sub_cf)


def test_load_with_stream_in_overwrite(private_config_file, streams_dir, mocker):
    overwrite = {"streams_directory": streams_dir}
    stub = mocker.patch("weathergen.common.config.load_streams", return_value=streams_dir)

    config.load_merge_configs(private_config_file, None, None, overwrite)

    stub.assert_called_once_with(streams_dir)


def test_load_multiple_overwrites(private_config_file):
    overwrites = [{"foo": 1, "bar": 1, "baz": 1}, {"foo": 2, "bar": 2}, {"foo": 3}]

    expected = {"foo": 3, "bar": 2, "baz": 1}
    cf = config.load_merge_configs(private_config_file, None, None, *overwrites)

    assert contains(cf, expected)


@pytest.mark.parametrize("mini_epoch", [None, 0, 1, 2, -1])
def test_load_existing_config(mini_epoch, private_config_file, config_fresh):
    test_num_mini_epochs = 3000

    config_fresh.num_mini_epochs = test_num_mini_epochs  # some specific change
    config.save(config_fresh, mini_epoch)

    cf = config.load_merge_configs(private_config_file, config_fresh.run_id, mini_epoch)

    assert cf.num_mini_epochs == test_num_mini_epochs


@pytest.mark.parametrize("options,cf", [(["foo=1", "bar=2"], {"foo": 1, "bar": 2}), ([], {})])
def test_from_cli(options, cf):
    parsed_config = config.from_cli_arglist(options)

    assert parsed_config == OmegaConf.create(cf)


@pytest.mark.parametrize(
    "run_id,reuse,expected",
    [
        (None, False, "generated"),
        ("new_id", False, "new_id"),
        (None, True, TEST_RUN_ID),
        ("new_id", True, TEST_RUN_ID),
    ],
)
def test_set_run_id(config_fresh, run_id, reuse, expected, mocker):
    mocker.patch("weathergen.common.config.get_run_id", return_value="generated")

    config_fresh = config.set_run_id(config_fresh, run_id, reuse)

    assert config_fresh.run_id == expected


def test_print_cf_no_secrets(config_fresh):
    output = config.format_cf(config_fresh)

    assert "53CR3T" not in output and "secrets" not in config_fresh.keys()


@pytest.mark.parametrize("rel_path,cf", VALID_STREAMS)
def test_load_streams(streams_dir, rel_path, cf):
    expected = get_expected_config(*[*cf.items()][0])
    write_stream_file(streams_dir / rel_path, OmegaConf.to_yaml(cf))

    streams = config.load_streams(streams_dir)

    assert all(is_equal(stream, expected) for stream in streams)


@pytest.mark.parametrize("rel_path,cf", EXCLUDED_STREAMS)
def test_load_streams_exclude_files(streams_dir, rel_path, cf):
    write_stream_file(streams_dir / rel_path, OmegaConf.to_yaml(cf))

    streams = config.load_streams(streams_dir)

    assert streams == []


def test_load_empty_stream(streams_dir):
    write_stream_file(streams_dir / "empty.yml", "")

    streams = config.load_streams(streams_dir)
    assert streams == []


def test_load_malformed_stream(streams_dir):
    write_stream_file(streams_dir / "error.yml", "ae:{")

    with pytest.raises(ValueError):
        config.load_streams(streams_dir)


@pytest.mark.parametrize("rel_path,cf", [("test.yml", DUMMY_MULTIPLE_STREAM_CONF)])
def test_load_multiple_streams_len(streams_dir, rel_path, cf):
    write_stream_file(streams_dir / rel_path, OmegaConf.to_yaml(cf))

    streams = config.load_streams(streams_dir)

    assert len(streams) == len(cf)


@pytest.mark.parametrize("rel_path,cf", [("test.yml", DUMMY_MULTIPLE_STREAM_CONF)])
def test_load_multiple_streams_content(streams_dir, rel_path, cf):
    expected = [get_expected_config(name, conf) for name, conf in cf.items()]
    write_stream_file(streams_dir / rel_path, OmegaConf.to_yaml(cf))

    streams = config.load_streams(streams_dir)

    assert all(
        is_equal(stream, stream_e) for stream, stream_e in zip(streams, expected, strict=True)
    )


def test_load_duplicate_streams(streams_dir):
    write_stream_file(streams_dir / "foo.yml", OmegaConf.to_yaml(DUMMY_STREAM_CONF))
    write_stream_file(streams_dir / "bar.yml", OmegaConf.to_yaml(DUMMY_STREAM_CONF))

    with pytest.raises(ValueError):
        config.load_streams(streams_dir)


def test_load_duplicate_streams_same_file(streams_dir):
    write_stream_file(streams_dir / "foo.yml", OmegaConf.to_yaml(DUMMY_STREAM_CONF))
    write_stream_file(streams_dir / "foo.yml", OmegaConf.to_yaml(DUMMY_STREAM_CONF))

    with pytest.raises(ValueError):
        config.load_streams(streams_dir)


@pytest.mark.parametrize("mini_epoch", [None, 0, 1, 2, -1])  # maybe add -5 as test case
def test_save(mini_epoch, config_fresh):
    config.save(config_fresh, mini_epoch)

    cf = config.load_run_config(config_fresh.run_id, mini_epoch, config_fresh.model_path)
    assert is_equal(cf, config_fresh)
