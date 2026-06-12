"""
SSH command runner for ECMWF / Atos clusters that authenticate via Teleport.

Unlike GenericSshCommandRunner, this runner shells out to the system `ssh`
client so that it honors ~/.ssh/config — including host aliases (e.g.
"hpc-login"), IdentityFile, CertificateFile, and ProxyCommand. Teleport's
tsh-based cert auth flow relies on these, and paramiko cannot replicate it.

TODO: check if this is still needed for ECMWF clusters or if the generic
SSH runner can be used now.
"""

import shlex
import subprocess
from dataclasses import dataclass
from logging import Logger

from weathergen.prefect_dags.cmd_runners._generic import (
    ConnectionClosedError,
    is_connection_closed,
)
from weathergen.prefect_dags.cmd_runners._types import (
    Command,
    CommandResult,
    CommandRunner,
    quote_path,
)
from weathergen.prefect_dags.result import OpError, Result


@dataclass
class SimpleSshContext:
    """
    Connection info for ECMWF / Atos clusters reached through Teleport.

    `host` is the Host alias from ~/.ssh/config (e.g. "hpc-login", "hpc2020"),
    not a real DNS name. The system ssh client resolves it and applies the
    matching config block (user, identity, certificate, proxy command).
    """

    host: str
    account: str | None = None


class SimpleSshCommandRunner(CommandRunner):
    """
    CommandRunner that executes commands via the system ssh client, assuming
    the connection has been set separately through SSH.

    This interface directly passes batch SSH commands.

    This is the case for:
        - ECMWF ATOS/HPC2020 with teleport
        - most other HPCs when following the standard interactive protocol
    """

    name = "ecmwf_ssh"
    _ctx: SimpleSshContext

    def __init__(self, context: SimpleSshContext):
        self._ctx = context
        # No separate "hpc" name on ECMWF contexts — the ssh host alias
        # ("hpc-login", "hpc2020", ...) already identifies the cluster.
        self.hpc = context.host

    def run(self, cmd: Command, logger: Logger) -> Result[CommandResult]:
        # ssh runs a non-login, non-interactive shell on the remote, so
        # working_directory and env_vars from the local Command don't
        # propagate — prefix them onto the command line ourselves.
        parts: list[str] = []
        if cmd.working_directory is not None:
            parts.append(f"cd {quote_path(cmd.working_directory)} &&")
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

        result = CommandResult(
            stdout=proc.stdout,
            stderr=proc.stderr,
            return_code=proc.returncode,
        )
        if is_connection_closed(result):
            err = ConnectionClosedError(result)
            logger.error(f"SSH connection closed unexpectedly: {err}")
            return OpError(err=err)
        return result
