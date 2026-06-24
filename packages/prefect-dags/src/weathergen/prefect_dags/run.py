"""
High level interface to running commands on an HPC.
"""

import json
from pathlib import Path

from prefect.utilities.asyncutils import run_coro_as_sync

from weathergen.prefect_dags.cmd_runners import CmdContext, Command, CommandResult, run_cmd
from weathergen.prefect_dags.cmd_runners._types import quote_path
from weathergen.prefect_dags.prefect import get_run_logger, task
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


def get_head_tail_logs(
    ctx: CmdContext, stdout_path: str, stderr_path: str | None
) -> dict[str, str]:
    header = ">>>> "
    stdout_q = quote_path(stdout_path)
    # awk replaces \ " \t \r and joins lines with literal \n so
    # the result is a valid JSON string body. It also enforces the byte/char
    # limit itself instead of relying on `head -c` / `tail -c`, which are not
    # portable on every HPC login node.
    escape_awk = (
        r"""awk -v max="$MAX_BYTES" 'BEGIN{ORS=""; used=0} """
        r"""{s=$0; gsub(/\\/,"\\\\",s); gsub(/"/,"\\\"",s); """
        r"""gsub(/\011/,"\\t",s); gsub(/\015/,"\\r",s); """
        r"""if(NR>1)s="\\n" s; remaining=max-used; if(remaining<=0)exit; """
        r"""if(length(s)>remaining){printf "%s", substr(s,1,remaining); exit} """
        r"""printf "%s", s; used+=length(s)}'"""
    )
    prelude = f"MAX_LINES={_MAX_OUTPUT_LINES}; MAX_BYTES={_MAX_OUTPUT_BYTES}; "
    helpers = (
        f"json_escape() {{ {escape_awk}; }}; "
        'json_head() { head -n "$MAX_LINES" "$1" | json_escape; }; '
        'json_tail() { tail -n "$MAX_LINES" "$1" | json_escape; }; '
    )
    out_extract = f"ho=$(json_head {stdout_q}); to=$(json_tail {stdout_q}); "
    if stderr_path is None:
        cmd = (
            prelude
            + helpers
            + out_extract
            + (
                f'printf \'{header}{{"head_out":"%s","tail_out":"%s",'
                '"head_err":null,"tail_err":null}\\n\''
            )
            + ' "$ho" "$to"'
        )
    else:
        stderr_q = quote_path(stderr_path)
        err_extract = f"he=$(json_head {stderr_q}); te=$(json_tail {stderr_q}); "
        cmd = (
            prelude
            + helpers
            + out_extract
            + err_extract
            + (
                f'printf \'{header}{{"head_out":"%s","tail_out":"%s",'
                '"head_err":"%s","tail_err":"%s"}\\n\''
            )
            + ' "$ho" "$to" "$he" "$te"'
        )
    res = run(ctx, command=cmd)
    if res.stderr.strip():
        get_run_logger().warning(f"get_head_tail_logs stderr: {res.stderr.strip()}")
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
