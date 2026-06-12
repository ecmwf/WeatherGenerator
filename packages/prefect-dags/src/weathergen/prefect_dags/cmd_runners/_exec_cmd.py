"""
Executing commands in an asynchronous way.
"""

import asyncio
import functools
import logging
from typing import Any

from weathergen.prefect_dags.cmd_runners._cineca import CinecaCommandRunner, CinecaSshContext
from weathergen.prefect_dags.cmd_runners._cscs_firecrest import (
    CscsFirecrestCommandRunner,
    CscsFirecrestContext,
)
from weathergen.prefect_dags.cmd_runners._ecmwf_ecaccess import (
    EcmwfEcaccessCommandRunner,
    EcmwfEcaccessContext,
)
from weathergen.prefect_dags.cmd_runners._generic import GenericContext, GenericSshCommandRunner
from weathergen.prefect_dags.cmd_runners._jsc import JscUnicoreCommandRunner, JscUnicoreContext
from weathergen.prefect_dags.cmd_runners._local import LocalCommandRunner, LocalContext
from weathergen.prefect_dags.cmd_runners._simple import SimpleSshCommandRunner, SimpleSshContext
from weathergen.prefect_dags.cmd_runners._types import Command, CommandResult, CommandRunner
from weathergen.prefect_dags.result import OpError, Result, is_err

"""
The context contains all the information necessary to run a command
on a given environment (e.g. local machine, remote server, etc.).

Expectations:
- The context should contain all necessary information to run a command, such as
  authentication credentials, hostnames, etc.
- The context should be serializable, as it may need to be passed between
  different parts of the system (e.g. from a Prefect task to a command runner).
- The context should be immutable (although the files it points to do *not* need to
be immutable). This is important to detect when credentials are updated
and to invalidate cached clients if needed.
"""
type CmdContext = (
    LocalContext
    | GenericContext
    | SimpleSshContext
    | CscsFirecrestContext
    | EcmwfEcaccessContext
    | CinecaSshContext
    | JscUnicoreContext
)


def slurm_account(context: CmdContext) -> str | None:
    """
    Extracts a slurm account if specified in the context.
    """
    match context:
        case CscsFirecrestContext():
            return context.account
        case SimpleSshContext():
            return context.account
        case EcmwfEcaccessContext():
            return context.account
        case CinecaSshContext():
            return context.account
        case JscUnicoreContext():
            return context.account
        case _:
            return None


def get_command_runner(context: CmdContext) -> CommandRunner | Exception:
    match context:
        case LocalContext():
            return LocalCommandRunner()
        case SimpleSshContext():
            return SimpleSshCommandRunner(context)
        case EcmwfEcaccessContext():
            return EcmwfEcaccessCommandRunner(context)
        case GenericContext():
            return GenericSshCommandRunner(context)
        case CscsFirecrestContext():
            return CscsFirecrestCommandRunner(context)
        case CinecaSshContext():
            return CinecaCommandRunner(context)
        case JscUnicoreContext():
            return JscUnicoreCommandRunner(context)
        case _:
            return ValueError(f"Unsupported context type: {type(context)}")


async def run_blocking(fn, *args, **kwargs):
    """Run a blocking function in the default thread executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


async def run_cmd(
    context: CmdContext, cmd: Command, logger: logging.Logger | Any | None = None
) -> Result[CommandResult]:
    """
    Runs a command on the specified HPC using the provided context.

    This function is asynchronous and it will not block the main thread.

    It abstracts away details about logging, error handlingl, etc.
    """
    logger = logger or logging.getLogger(__name__)
    runner_or_err = get_command_runner(context)
    if isinstance(runner_or_err, Exception):
        return OpError(err=runner_or_err)
    runner = runner_or_err
    raw_data = await run_blocking(runner.run, cmd, logger)
    if is_err(raw_data) or isinstance(raw_data, CommandResult):
        return raw_data
    else:
        return OpError(
            err=ValueError(f"Unexpected return type from command runner: {type(raw_data)}")
        )
