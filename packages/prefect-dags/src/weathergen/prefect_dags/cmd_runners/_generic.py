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
        parts.append(f"cd {shlex.quote(str(cmd.working_directory))} &&")
    if cmd.env_vars:
        for k, v in cmd.env_vars.items():
            parts.append(f"export {k}={shlex.quote(v)};")
    if isinstance(cmd.command, str):
        parts.append(cmd.command)
    else:
        parts.append(shlex.join(cmd.command))
    return " ".join(parts)


def _load_private_key(key_path: Path, logger: Logger) -> paramiko.PKey:
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
    cert_pub = Path(f"{key_path}-cert.pub")
    if cert_pub.exists():
        pkey.load_certificate(cert_pub.read_text().strip())
        logger.info(f"Loaded SSH certificate: {cert_pub}")

    return pkey
