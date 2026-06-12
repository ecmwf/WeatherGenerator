"""
SSH command runner for CINECA HPCs (e.g. Leonardo), authenticated with a
short-lived step-ca certificate.

User workflow (manual, follows the official Leonardo guide):
- https://docs.hpc.cineca.it/hpc/leonardo.html#leonardo-card
- https://docs.hpc.cineca.it/hpc/leonardo.html#access-to-the-system

1. Mint a certificate via step-ca (max 4 days validity):

    step ssh certificate 'YOUR_EMAIL' \\
        --provisioner cineca-hpc \\
        ~/.ssh/leonardo_key \\
        --no-agent --no-password --insecure \\
        --force --not-after 48h

   This writes the private key to ``~/.ssh/leonardo_key`` and the matching
   certificate to ``~/.ssh/leonardo_key-cert.pub``.

2. Use the runner via context:

    CinecaSshContext(hpc="leonardo", user="my_cineca_user")

The lower-level reference implementation lives in
``weathergen.prefect_dags.cineca``.
"""

import shlex
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Literal

import paramiko

from weathergen.prefect_dags.cmd_runners._generic import (
    ConnectionClosedError,
    _load_private_key,
    is_connection_closed,
)
from weathergen.prefect_dags.cmd_runners._types import (
    Command,
    CommandResult,
    CommandRunner,
    quote_path,
)
from weathergen.prefect_dags.result import OpError, Result

type CinecaHpc = Literal["leonardo"]


# Per-HPC defaults: (private key path, certificate path, login host).
# Matches the paths the official `step ssh certificate` recipe writes to.
_CINECA_DEFAULTS: dict[CinecaHpc, tuple[Path, Path, str]] = {
    "leonardo": (
        Path.home() / ".ssh" / "leonardo_key",
        Path.home() / ".ssh" / "leonardo_key-cert.pub",
        "login.leonardo.cineca.it",
    ),
}


@dataclass
class CinecaSshContext:
    """
    Connection info for CINECA clusters reached via step-ca SSH certificate.

    ``hpc``: one of the supported CINECA systems (currently ``"leonardo"``).
    ``user``: CINECA username used as the SSH login.
    ``key_path``: private key file. Defaults to the HPC-specific path:
        - leonardo: ``~/.ssh/leonardo_key``
    ``certificate_path``: SSH certificate file. Defaults to the HPC-specific
        path:
        - leonardo: ``~/.ssh/leonardo_key-cert.pub``
    ``hostname``: login node. Defaults to the HPC-specific hostname:
        - leonardo: ``login.leonardo.cineca.it``
    ``account``: Slurm account to charge (``sbatch --account``). Optional —
        leave unset to use the cluster default.
    """

    hpc: CinecaHpc
    user: str
    key_path: str | Path | None = None
    certificate_path: str | Path | None = None
    hostname: str | None = None
    account: str | None = None
    port: int = 22


class CinecaCommandRunner(CommandRunner):
    """
    CommandRunner for CINECA clusters via paramiko + step-issued SSH cert.

    Each ``run()`` opens a fresh SSH connection, executes the command, and
    closes the connection. Credentials (key + cert) are re-read on every
    submission so step-ca cert rotations are picked up automatically.
    """

    name = "cineca_ssh"
    _ctx: CinecaSshContext

    def __init__(self, context: CinecaSshContext):
        self._ctx = context
        self.hpc = context.hpc

    def run(self, cmd: Command, logger: Logger) -> Result[CommandResult]:
        try:
            return self._run(cmd, logger)
        except Exception as e:
            # Per the CommandRunner contract: errors are returned, not raised.
            logger.error(f"SSH command failed: {e}")
            return OpError(err=e)

    def _run(self, cmd: Command, logger: Logger) -> Result[CommandResult]:
        ctx = self._ctx
        default_key, default_cert, default_host = _CINECA_DEFAULTS[ctx.hpc]

        key_path = Path(ctx.key_path).expanduser() if ctx.key_path else default_key
        cert_path = (
            Path(ctx.certificate_path).expanduser() if ctx.certificate_path else default_cert
        )
        hostname = ctx.hostname or default_host

        if not key_path.exists():
            return OpError(err=FileNotFoundError(f"Private key not found: {key_path}"))
        if not cert_path.exists():
            return OpError(err=FileNotFoundError(f"Certificate not found: {cert_path}"))

        pkey = _load_private_key(key_path, logger, cert_pub_path=cert_path)

        # SSH exec_command runs a non-login, non-interactive shell on the
        # remote, so working_directory and env_vars from the local Command
        # don't propagate — prefix them onto the command line ourselves.
        remote_cmd = _build_remote_command(cmd)

        client = paramiko.SSHClient()
        # Login-node host keys rotate on Leonardo; accept them.
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            logger.info(f"Connecting to {ctx.user}@{hostname}:{ctx.port} ...")
            client.connect(
                hostname=hostname,
                port=ctx.port,
                username=ctx.user,
                pkey=pkey,
                # We loaded an explicit key + cert; don't let paramiko also
                # probe the agent / default keys (that produces confusing
                # auth failures).
                look_for_keys=False,
                allow_agent=False,
            )

            logger.info(f"Executing remote command: {remote_cmd}")
            _stdin, stdout, stderr = client.exec_command(remote_cmd)

            # Read output BEFORE waiting for exit status to avoid deadlock:
            # if the command fills the SSH buffer, recv_exit_status() blocks
            # forever waiting for a command that is itself blocked writing.
            out = stdout.read().decode(errors="replace")
            err = stderr.read().decode(errors="replace")
            exit_status = stdout.channel.recv_exit_status()
            logger.info(f"Remote command finished with exit status {exit_status}")

            result = CommandResult(stdout=out, stderr=err, return_code=exit_status)
            if is_connection_closed(result):
                err_exc = ConnectionClosedError(result)
                logger.error(f"SSH connection closed unexpectedly: {err_exc}")
                return OpError(err=err_exc)
            return result
        finally:
            client.close()


def _build_remote_command(cmd: Command) -> str:
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
    return " ".join(parts)
