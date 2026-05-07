"""
SSH command runner for ECMWF / Atos clusters that authenticate via Teleport.

Unlike GenericSshCommandRunner, this runner shells out to the system `ssh`
client so that it honors ~/.ssh/config — including host aliases (e.g.
"hpc-login"), IdentityFile, CertificateFile, and ProxyCommand. Teleport's
tsh-based cert auth flow relies on these, and paramiko cannot replicate it.
"""

import shlex
import subprocess
from dataclasses import dataclass
from logging import Logger

from weathergen.prefect_dags.cmd_runners._types import (
    Command,
    CommandResult,
    CommandRunner,
)
from weathergen.prefect_dags.result import OpError, Result


@dataclass
class EcmwfSshContext:
    """
    Connection info for ECMWF / Atos clusters reached through Teleport.

    `host` is the Host alias from ~/.ssh/config (e.g. "hpc-login", "hpc2020"),
    not a real DNS name. The system ssh client resolves it and applies the
    matching config block (user, identity, certificate, proxy command).
    """

    host: str


class EcmwfSshCommandRunner(CommandRunner):
    """
    CommandRunner that executes commands on ECMWF / Atos clusters via the system ssh client.

    It assumes the standard Teleport authentication flow is used.

    TODO: documentation to ECMWF teleport setup.
    """
    name = "ecmwf_ssh"
    _ctx: EcmwfSshContext

    def __init__(self, context: EcmwfSshContext):
        self._ctx = context

    def run(self, cmd: Command, logger: Logger) -> Result[CommandResult]:
        # ssh runs a non-login, non-interactive shell on the remote, so
        # working_directory and env_vars from the local Command don't
        # propagate — prefix them onto the command line ourselves.
        parts: list[str] = []
        if cmd.working_directory is not None:
            parts.append(f"cd {shlex.quote(str(cmd.working_directory))} &&")
        if cmd.env_vars:
            for k, v in cmd.env_vars.items():
                parts.append(f"export {k}={shlex.quote(v)};")
        if isinstance(cmd.command, str):
            parts.append(cmd.command)
        else:
            parts.append(shlex.join(cmd.command))
        remote_cmd = " ".join(parts)

        # BatchMode=yes prevents hangs on password / keyboard-interactive
        # prompts; if the Teleport cert is expired the call fails fast and
        # the user can refresh it with `tsh login`.
        argv = ["ssh", "-o", "BatchMode=yes", self._ctx.host, remote_cmd]
        logger.info(f"Running: {' '.join(shlex.quote(a) for a in argv)}")
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, check=False)
        except Exception as e:
            logger.error(f"SSH command failed: {e}")
            return OpError(err=e)

        return CommandResult(
            stdout=proc.stdout,
            stderr=proc.stderr,
            return_code=proc.returncode,
        )
