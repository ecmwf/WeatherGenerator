import pathlib
import tempfile

import pytest
from omegaconf import OmegaConf

import weathergen.utils.config as config

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

DUMMY_OVERWRITES = [("num_epochs", 42), ("healpix_level", 42)]


def contains_keys(super_config, sub_config):
    keys_present = [key in super_config.keys() for key in sub_config.keys()]

    return all(keys_present)

def contains_values(super_config, sub_config):
    correct_values = [super_config[key] == value for key, value in sub_config.items()]

    return all(correct_values)

def contains(super_config, sub_config):
    return contains_keys(super_config, sub_config) and contains_values(super_config, sub_config)

@pytest.fixture
def models_dir():
    with tempfile.TemporaryDirectory(prefix="models") as temp_dir:
        yield temp_dir


@pytest.fixture
def private_conf(models_dir):
    cf = OmegaConf.create(DUMMY_PRIVATE_CONF)
    cf.model_path = models_dir
    return cf


@pytest.fixture
def private_config_file(private_conf):
    temp_file = tempfile.NamedTemporaryFile("w+", delete=False)
    temp_file.write(OmegaConf.to_yaml(private_conf))
    temp_file.flush()
    yield pathlib.Path(temp_file.name)


@pytest.fixture
def overwrite_dict(request):
    key, value = request.param
    return {key: value}


@pytest.fixture
def overwrite_config(overwrite_dict):
    return OmegaConf.create(overwrite_dict)


@pytest.fixture
def overwrite_file(overwrite_config):
    temp_file = tempfile.NamedTemporaryFile("w+", delete=False)
    temp_file.write(OmegaConf.to_yaml(overwrite_config))
    temp_file.flush()
    yield pathlib.Path(temp_file.name)

@pytest.fixture
def config_fresh(private_config_file):
    cf = config.load_config(private_config_file, None, None)
    cf.data_loader_rng_seed = 42
    
    return cf


def test_contains_private(config_fresh):
    assert contains_keys(config_fresh, DUMMY_PRIVATE_CONF)

@pytest.mark.parametrize("overwrite_dict", DUMMY_OVERWRITES, indirect=True)
def test_load_with_overwrite_dict(overwrite_dict, private_config_file):
    cf = config.load_config(private_config_file, None, None, overwrite_dict)

    assert contains(cf, overwrite_dict)

@pytest.mark.parametrize("overwrite_dict", DUMMY_OVERWRITES, indirect=True)
def test_load_with_overwrite_config(overwrite_config, private_config_file):
    cf = config.load_config(private_config_file, None, None, overwrite_config)

    assert contains(cf, overwrite_config)

@pytest.mark.parametrize("overwrite_dict", DUMMY_OVERWRITES, indirect=True)
def test_load_with_overwrite_file(private_config_file, overwrite_file):
    sub_cf = OmegaConf.load(overwrite_file)
    cf = config.load_config(private_config_file, None, None, overwrite_file)

    assert contains(cf, sub_cf)

def test_load_multiple_overwrites(private_config_file):
    overwrites = [
        {"foo": 1, "bar": 1, "baz": 1},
        {"foo": 2, "bar": 2},
        {"foo": 3}
    ]
    
    expected = {"foo": 3, "bar": 2, "baz": 1}
    cf = config.load_config(private_config_file, None, None, *overwrites)
    
    assert contains(cf, expected)

@pytest.mark.parametrize("epoch", [None, 0, 1, 2, -1])
def test_load_existing_config(epoch, private_config_file, config_fresh):
    test_run_id = "test123"
    test_num_epochs = 3000
    
    config_fresh.run_id = test_run_id  # done in trainer
    config_fresh.num_epochs = test_num_epochs # some specific change
    config.save(config_fresh, epoch)
    
    cf = config.load_config(private_config_file, test_run_id, epoch)
    
    assert cf.num_epochs == test_num_epochs
    

def test_from_cli():
    args = ["foo=1", "bar=2"]

    parsed_config = config.from_cli_arglist(args)
    assert parsed_config == OmegaConf.create({"foo": 1, "bar": 2})


def test_print_cf_no_secrets(config_fresh):
    output = config._format_cf(config_fresh)
    print(output)

    assert "53CR3T" not in output

@pytest.mark.xfail # not implemented yet
def test_load_streams():
    pass


@pytest.mark.xfail # not implemented yet
def test_save():
    pass
