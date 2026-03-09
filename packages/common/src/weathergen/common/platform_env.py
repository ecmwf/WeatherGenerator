"""
Platform environment configuration for WeatherGenerator.

These are loaded from secrets in the private repository.
"""

import importlib
import importlib.util
from functools import lru_cache
import os
from typing import Protocol
from zipfile import Path

from weathergen.common.config import _REPO_ROOT


class PlatformEnv(Protocol):
    """
    Interface for platform environment configuration.
    """

    def get_hpc(self) -> str | None: ...

    def get_hpc_user(self) -> str | None: ...

    def get_hpc_user_org(self) -> str | None: ...

    def get_hpc_config(self) -> str | None: ...

    def get_hpc_certificate(self) -> str | None: ...


@lru_cache(maxsize=1)
def get_platform_env() -> PlatformEnv:
    """
    Loads the platform environment module from the private repository.
    """
    if "WEATHERGEN_PRIVATE_REPO_PATH" in os.environ:
        env_repo_path = Path(os.environ["WEATHERGEN_PRIVATE_REPO_PATH"])
        env_script_path = env_repo_path / "hpc" / "platform-env.py"
    else:
        env_script_path = _REPO_ROOT.parent / "WeatherGenerator-private" / "hpc" / "platform-env.py"
    spec = importlib.util.spec_from_file_location("platform_env", env_script_path)
    platform_env = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(platform_env)  # type: ignore
    return platform_env  # type: ignore
