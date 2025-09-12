import importlib

import importlib.util
import sys
from pathlib import Path
from functools import lru_cache


@lru_cache()
def _get_private_module():
    """
    Loads all the private functions from the private repo."""
    file_path = Path(__file__).parent / "../../../../WeatherGenerator-private/hpc/platform-env.py"
    file_path = file_path.resolve()

    # Module name (can be anything unique)
    module_name = "platform_env"

    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    _module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = _module
    spec.loader.exec_module(_module)
    return _module
    
_module = _get_private_module()

get_hpc = _module.get_hpc
get_hpc_user = _module.get_hpc_user
get_hpc_config = _module.get_hpc_config
