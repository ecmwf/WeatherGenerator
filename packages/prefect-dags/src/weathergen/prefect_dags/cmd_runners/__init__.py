"""
A module for running command on different HPCs.

This module is independent from Prefect. It focuses on
abstracting away the details of communicating with the various HPCs,
with a focus on EuropHPC.
"""

from weathergen.prefect_dags.cmd_runners._cineca import (
    CinecaCommandRunner,
    CinecaHpc,
    CinecaSshContext,
)
from weathergen.prefect_dags.cmd_runners._cscs_firecrest import (
    CscsFirecrestCommandRunner,
    CscsFirecrestContext,
    CscsHpc,
)
from weathergen.prefect_dags.cmd_runners._ecmwf_ecaccess import (
    EcmwfEcaccessCommandRunner,
    EcmwfEcaccessContext,
)
from weathergen.prefect_dags.cmd_runners._exec_cmd import (
    CmdContext,
    get_command_runner,
    run_cmd,
    slurm_account,
)
from weathergen.prefect_dags.cmd_runners._generic import GenericContext, GenericSshCommandRunner
from weathergen.prefect_dags.cmd_runners._jsc import (
    JscHpc,
    JscUnicoreCommandRunner,
    JscUnicoreContext,
)
from weathergen.prefect_dags.cmd_runners._local import LocalCommandRunner, LocalContext
from weathergen.prefect_dags.cmd_runners._simple import SimpleSshCommandRunner, SimpleSshContext
from weathergen.prefect_dags.cmd_runners._types import Command, CommandResult, CommandRunner

__all__ = [
    "Command",
    "CommandResult",
    "CommandRunner",
    "LocalCommandRunner",
    "LocalContext",
    "GenericSshCommandRunner",
    "GenericContext",
    "SimpleSshCommandRunner",
    "SimpleSshContext",
    "EcmwfEcaccessCommandRunner",
    "EcmwfEcaccessContext",
    "run_cmd",
    "CmdContext",
    "get_command_runner",
    "slurm_account",
    "CscsFirecrestCommandRunner",
    "CscsFirecrestContext",
    "CscsHpc",
    "CinecaCommandRunner",
    "CinecaSshContext",
    "CinecaHpc",
    "JscUnicoreCommandRunner",
    "JscUnicoreContext",
    "JscHpc",
]
