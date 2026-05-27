"""
High level interface to running commands on an HPC.
"""

import json
import shlex
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


_MAX_OUTPUT_BYTES = 100_000
_MAX_OUTPUT_LINES = 100


def get_head_tail_logs(ctx: CmdContext, stdout_path: str, stderr_path: str | None) -> dict:
    # Combine into a json output that we can easily inspect.
    # Important: each head and tail should be truncated to _MAX_OUTPUT_BYTES and _MAX_OUTPUT_LINES to avoid overwhelming
    # the ssh command with too much output
    # The output should look like: ">>>> {head_out: '...', tail_out: '...', head_err: '...', tail_err: '...'}"
    # head_err, tail_err are null if stderr_path is None.
    # The prefix allows easy parsing of the output given that many other lines are added by the schedulers.
    header = ">>>> "
    stdout_q = shlex.quote(stdout_path)
    # awk replaces \ " \t \r and joins lines with literal \n so the result is a valid JSON string body.
    escape_awk = (
        r"""awk 'BEGIN{ORS=""} """
        r"""{gsub(/\\/,"\\\\"); gsub(/"/,"\\\""); gsub(/\t/,"\\t"); gsub(/\r/,"\\r"); """
        r"""if(NR>1)printf "\\n"; printf "%s",$0}'"""
    )
    prelude = f"MAX_LINES={_MAX_OUTPUT_LINES}; MAX_BYTES={_MAX_OUTPUT_BYTES}; "
    out_extract = (
        f'ho=$(head -n "$MAX_LINES" {stdout_q} 2>/dev/null | head -c "$MAX_BYTES" | {escape_awk}); '
        f'to=$(tail -n "$MAX_LINES" {stdout_q} 2>/dev/null | tail -c "$MAX_BYTES" | {escape_awk}); '
    )
    if stderr_path is None:
        cmd = (
            prelude
            + out_extract
            + f'printf \'{header}{{"head_out":"%s","tail_out":"%s","head_err":null,"tail_err":null}}\\n\''
            + ' "$ho" "$to"'
        )
    else:
        stderr_q = shlex.quote(stderr_path)
        err_extract = (
            f'he=$(head -n "$MAX_LINES" {stderr_q} 2>/dev/null | head -c "$MAX_BYTES" | {escape_awk}); '
            f'te=$(tail -n "$MAX_LINES" {stderr_q} 2>/dev/null | tail -c "$MAX_BYTES" | {escape_awk}); '
        )
        cmd = (
            prelude
            + out_extract
            + err_extract
            + f'printf \'{header}{{"head_out":"%s","tail_out":"%s","head_err":"%s","tail_err":"%s"}}\\n\''
            + ' "$ho" "$to" "$he" "$te"'
        )
    res = run(ctx, command=cmd)
    assert res.stdout
    # Parse the line using json (expect a single line with this header)
    for line in res.stdout.splitlines():
        if line.startswith(header):
            return json.loads(line[len(header) :])
    raise ValueError(f"Could not find '{header}' line in output: {res.stdout!r}")


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
