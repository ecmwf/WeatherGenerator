"""
Local command runner implementation for running shell commands on the local machine.
"""

import os
import subprocess
from dataclasses import dataclass
from logging import Logger
from pathlib import Path

from weathergen.prefect_dags.cmd_runners._types import (
    Command,
    CommandResult,
    CommandRunner,
)
from weathergen.prefect_dags.result import OpError, Result


@dataclass
class LocalContext:
    """
    All the information needed to run a command on a local machine.

    Nothing is required for a local runner.
    """

    pass


class LocalCommandRunner(CommandRunner):
    name = "local"
    hpc = "local"

    def run(self, cmd: Command, logger: Logger) -> Result[CommandResult]:
        # Merge onto os.environ rather than replacing it: replacing wipes PATH,
        # HOME, etc. and breaks most commands. None inherits the parent env unchanged.
        env = {**os.environ, **cmd.env_vars} if cmd.env_vars is not None else None
        logger.info(
            f"Running local command: {cmd.command} with env vars: {cmd.env_vars} "
            f"and working directory: {cmd.working_directory}"
        )
        # subprocess does not expand ~ in cwd; locally Python can do it for us
        # (remote runners leave that to the remote shell instead).
        cwd = Path(cmd.working_directory).expanduser() if cmd.working_directory else None
        try:
            proc = subprocess.run(
                cmd.command,
                # shell=True for str (so pipes/redirects work), shell=False for
                # list[str] (safer argv form, no shell parsing). Inferred from the type.
                shell=isinstance(cmd.command, str),
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                # Don't raise on non-zero: caller decides what counts as failure
                # via the returned CommandResult.return_code.
                check=False,
            )
        except Exception as e:
            # Only Python-level failures (e.g. FileNotFoundError from a missing
            # executable when shell=False) become OpError. A process that
            # ran to completion — even with non-zero exit — returns CommandResult.
            logger.error(f"Command failed with error: {e}")
            return OpError(err=e)
        return CommandResult(
            stdout=proc.stdout,
            stderr=proc.stderr,
            return_code=proc.returncode,
        )
