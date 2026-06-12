"""
Generic ssh command runner, when there is no specific logic for a given HPC.  This is used as a
fallback by the higher-level run_cmd function.
"""

import io
import shlex
from dataclasses import dataclass
from logging import Logger
from pathlib import Path

import paramiko
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    load_ssh_private_key,
)

from weathergen.prefect_dags.cmd_runners._types import (
    Command,
    CommandResult,
    CommandRunner,
    quote_path,
)
from weathergen.prefect_dags.result import OpError, Result


@dataclass
class GenericContext:
    hpc: str
    host: str
    user: str
    email: str | None = None
    ssh_key_path: str | None = None
    port: int = 22


class ConnectionClosedError(Exception):
    """
    Raised when it is detected that the SSH connection has terminated.
    """

    result: CommandResult

    def __init__(self, result: CommandResult) -> None:
        self.result = result
        super().__init__(f"SSH connection closed unexpectedly. Last result: {result}")


# ssh / sshd / paramiko emit connection diagnostics at the very top of stderr.
# Bound the scan so a long remote command can't trip the heuristic with a
# stray phrase deep in its output.
_CONNECTION_CLOSED_HEAD_LINES = 5


def is_connection_closed(result: CommandResult) -> bool:
    """
    Heuristic to detect if an SSH connection died mid-command, distinguishing
    a remote-side disconnect from a legitimate non-zero exit in the user's
    command.

    Signals, in order of confidence:

    1. ``return_code == -1`` — paramiko's ``recv_exit_status()`` returns -1
       when the channel was closed before the remote sent an exit-status
       message. Effectively a smoking gun for a dropped connection.
    2. ``return_code == 255`` — ssh(1)'s reserved code for connection
       failures. A user command could theoretically exit 255, but in
       practice that's vanishingly rare compared to a real SSH error.
    3. A connection-failure phrase in **the first few lines of stderr**.
       stdout is excluded because commands that legitimately print these
       strings as data (e.g. ``cat /var/log/auth.log``) would otherwise be
       flagged. Only the head of stderr is scanned because ssh/sshd emit
       these messages first — anything further down belongs to the remote
       command and could trip the heuristic by accident.

    Matching is case-insensitive — OpenSSH and paramiko don't agree on the
    capitalization of these messages.
    """
    if result.return_code in (-1, 255):
        return True

    # Phrases observed in practice from OpenSSH client, sshd, and paramiko.
    # Lower-cased here; the haystack is lower-cased once below.
    closed_phrases = (
        "connection closed by remote host",
        "connection reset by peer",
        "connection refused",
        "connection timed out",
        "broken pipe",
        "client_loop: send disconnect",
        "kex_exchange_identification:",
        "ssh_exchange_identification:",
        "no route to host",
        "network is unreachable",
        "ssh session terminated",
    )
    head_lines = (result.stderr or "").splitlines()[:_CONNECTION_CLOSED_HEAD_LINES]
    haystack = "\n".join(head_lines).lower()
    return any(phrase in haystack for phrase in closed_phrases)


class GenericSshCommandRunner(CommandRunner):
    name = "generic_ssh"
    _ctx: GenericContext

    def __init__(self, context: GenericContext):
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

        # Build the remote command line. SSH exec_command runs a non-login,
        # non-interactive shell, so env_vars and cwd from the local Command
        # don't propagate — we prefix them onto the line ourselves.
        remote_cmd = _build_remote_command(cmd)

        pkey = None
        if ctx.ssh_key_path is not None:
            key_path = Path(ctx.ssh_key_path)
            if not key_path.exists():
                return OpError(err=FileNotFoundError(f"Private key not found: {key_path}"))
            pkey = _load_private_key(key_path, logger)

        client = paramiko.SSHClient()
        # Host keys can rotate across login nodes on HPC clusters; accept them.
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            logger.info(f"Connecting to {ctx.user}@{ctx.host}:{ctx.port} ...")
            client.connect(
                hostname=ctx.host,
                port=ctx.port,
                username=ctx.user,
                pkey=pkey,
                # If we loaded an explicit key, don't also probe the agent /
                # default keys — that can produce confusing auth failures.
                look_for_keys=pkey is None,
                allow_agent=pkey is None,
            )

            logger.info(f"Executing remote command: {remote_cmd}")
            _stdin, stdout, stderr = client.exec_command(remote_cmd)

            # Read output BEFORE waiting for exit status to avoid deadlock:
            # if the command fills the SSH buffer, recv_exit_status() will
            # block forever waiting for a command that is itself blocked on
            # writing to a full buffer.
            out = stdout.read().decode(errors="replace")
            err = stderr.read().decode(errors="replace")
            exit_status = stdout.channel.recv_exit_status()
            logger.info(f"Remote command finished with exit status {exit_status}")

            return CommandResult(stdout=out, stderr=err, return_code=exit_status)
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


def _load_private_key(
    key_path: Path,
    logger: Logger,
    cert_pub_path: Path | None = None,
) -> paramiko.PKey:
    # Paramiko 4.0 can't parse OpenSSH-format ECDSA keys directly. Re-serialize
    # to traditional PEM via cryptography, then dispatch to the matching paramiko
    # key class. This works uniformly across RSA / Ed25519 / ECDSA / DSA.
    with open(key_path, "rb") as f:
        crypto_key = load_ssh_private_key(f.read(), password=None)
    pem_text = crypto_key.private_bytes(
        Encoding.PEM,
        PrivateFormat.TraditionalOpenSSL,
        NoEncryption(),
    ).decode()

    if isinstance(crypto_key, ec.EllipticCurvePrivateKey):
        pkey: paramiko.PKey = paramiko.ECDSAKey(file_obj=io.StringIO(pem_text))
    elif isinstance(crypto_key, ed25519.Ed25519PrivateKey):
        pkey = paramiko.Ed25519Key(file_obj=io.StringIO(pem_text))
    elif isinstance(crypto_key, rsa.RSAPrivateKey):
        pkey = paramiko.RSAKey(file_obj=io.StringIO(pem_text))
    else:
        # DSA was removed in paramiko 4.x; not worth supporting.
        raise ValueError(f"Unsupported private key type: {type(crypto_key).__name__}")

    # Optional companion certificate (e.g. issued by step-ca on CINECA Leonardo).
    # When `cert_pub_path` is given, use it as-is; otherwise fall back to the
    # `<key_path>-cert.pub` convention.
    cert_pub = cert_pub_path if cert_pub_path is not None else Path(f"{key_path}-cert.pub")
    if cert_pub.exists():
        pkey.load_certificate(cert_pub.read_text().strip())
        logger.info(f"Loaded SSH certificate: {cert_pub}")

    return pkey
