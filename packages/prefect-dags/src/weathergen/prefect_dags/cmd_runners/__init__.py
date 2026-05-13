"""
A module for running command on different HPCs.

This module is independent from Prefect. It focuses on
abstracting away the details of communicating with the various HPCs,
with a focus on EuropHPC.
"""

from weathergen.prefect_dags.cmd_runners._cscs_firecrest import (
    CscsFirecrestCommandRunner,
    CscsFirecrestContext,
    CscsHpc,
)
from weathergen.prefect_dags.cmd_runners._ecmwf import EcmwfSshCommandRunner, EcmwfSshContext
from weathergen.prefect_dags.cmd_runners._exec_cmd import CmdContext, get_command_runner, run_cmd
from weathergen.prefect_dags.cmd_runners._generic import GenericContext, GenericSshCommandRunner
from weathergen.prefect_dags.cmd_runners._local import LocalCommandRunner, LocalContext
from weathergen.prefect_dags.cmd_runners._types import Command, CommandResult, CommandRunner

__all__ = [
    "Command",
    "CommandResult",
    "CommandRunner",
    "LocalCommandRunner",
    "LocalContext",
    "GenericSshCommandRunner",
    "GenericContext",
    "EcmwfSshCommandRunner",
    "EcmwfSshContext",
    "run_cmd",
    "CmdContext",
    "get_command_runner",
    "CscsFirecrestCommandRunner",
    "CscsFirecrestContext",
    "CscsHpc",
]
