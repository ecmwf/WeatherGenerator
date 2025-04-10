def pytest_addoption(parser):
    """setup parser"""
    parser.addoption("--run_id", action="store", default="name of the run to be tested.")


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.run_id
    if "run_id" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("run_id", [option_value])
