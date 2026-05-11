"""
High level interface to running commands on an HPC.
"""

from pathlib import Path

from prefect.utilities.asyncutils import run_coro_as_sync

from weathergen.prefect_dags.cmd_runners import CmdContext, Command, CommandResult, run_cmd
from weathergen.prefect_dags.prefect_logging import get_run_logger
from weathergen.prefect_dags.prefect_wrapper import task
from weathergen.prefect_dags.result import Result, unwrap


@task
def run(
    ctx: CmdContext,
    command: str | list[str],
    working_directory: str | Path | None = None,
    env_vars: dict[str, str] | None = None,
) -> CommandResult:
    """
    Runs a command on the given HPC, and returns the result (success, failure, etc.)

    """
    return run_coro_as_sync(_run_async(ctx, command, working_directory, env_vars))


@task
def run_try(
    context: CmdContext,
    command: str | list[str],
    working_directory: str | Path | None = None,
    env_vars: dict[str, str] | None = None,
) -> Result[CommandResult]:
    """
    Runs a command on the given HPC, and returns the result (success, failure, etc.)

    """
    return run_coro_as_sync(_run_try_async(context, command, working_directory, env_vars))


async def _run_async(
    ctx: CmdContext,
    command: str | list[str],
    working_directory: str | Path | None,
    env_vars: dict[str, str] | None,
) -> CommandResult:
    logger = get_run_logger()
    cmd = Command(
        command=command,
        working_directory=working_directory,
        env_vars=env_vars,
    )
    result = await run_cmd(ctx, cmd, logger=logger)
    return unwrap(result)


async def _run_try_async(
    context: CmdContext,
    command: str | list[str],
    working_directory: str | Path | None,
    env_vars: dict[str, str] | None,
) -> Result[CommandResult]:
    logger = get_run_logger()
    cmd = Command(
        command=command,
        working_directory=working_directory,
        env_vars=env_vars,
    )
    return await run_cmd(context, cmd, logger=logger)
