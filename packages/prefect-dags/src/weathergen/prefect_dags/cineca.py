""" """

from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import paramiko
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    load_ssh_private_key,
)

# ---------------------------------------------------------------------------
# Default paths – override via env vars or function arguments
# ---------------------------------------------------------------------------
_DEFAULT_KEY_PATH = Path.home() / ".ssh" / "leonardo_key"
_DEFAULT_HOST = "login.leonardo.cineca.it"
_DEFAULT_PORT = 22

logger = logging.getLogger(__name__)


def run_command_cineca(
    command: str,
    username: str | None = None,
    email: str | None = None,
    key_path: str | os.PathLike | None = None,
    hostname: str = _DEFAULT_HOST,
    port: int = _DEFAULT_PORT,
    timeout: float | None = None,
    logger: logging.Logger | None = None,
) -> str:
    """SSH into Leonardo, execute *command*, block until it finishes, and
    return its combined stdout+stderr as a string.

    Parameters
    ----------
    command:
        The shell command to run on the remote host
        (e.g. ``"squeue -u $USER"`` or ``"sbatch my_job.sh"``).
    username:
        Your CINECA cluster username.  Falls back to the env-var
        ``LEONARDO_USER``.
    email:
        The e-mail address tied to your CINECA account.  Falls back to
        ``LEONARDO_EMAIL``.  Only needed when *auto_renew* is True.
    key_path:
        Path to the private key created by ``step ssh certificate``.
        Defaults to ``~/.ssh/leonardo_key``.
    hostname:
        Leonardo login node.  Defaults to ``login.leonardo.cineca.it``.
    port:
        SSH port (default 22).
    timeout:
        Optional timeout in seconds for the remote command.

    Returns
    -------
    str
        The combined stdout and stderr produced by the remote command.

    Raises
    ------
    paramiko.AuthenticationException
        If the certificate is expired / invalid and *auto_renew* is False.
    paramiko.SSHException
        On any other SSH-level error.
    RuntimeError
        If the remote command exits with a non-zero status (the exception
        message contains the captured output).
    """
    username = username or os.environ.get("LEONARDO_USER")
    email = email or os.environ.get("LEONARDO_EMAIL")
    key_path = Path(key_path) if key_path else _DEFAULT_KEY_PATH
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.info("Using key path: %s", key_path)

    if not username:
        raise ValueError("Provide 'username' or set the LEONARDO_USER environment variable.")

    # ------------------------------------------------------------------
    # Load the private key + certificate and connect
    # ------------------------------------------------------------------
    cert_pub = Path(f"{key_path}-cert.pub")
    if not key_path.exists():
        raise FileNotFoundError(f"Private key not found: {key_path}")
    if not cert_pub.exists():
        raise FileNotFoundError(f"Certificate not found: {cert_pub}")

    # Paramiko 4.0 can't parse OpenSSH-format ECDSA keys directly.
    # Work around by re-serializing to traditional PEM via cryptography.
    with open(key_path, "rb") as f:
        crypto_key = load_ssh_private_key(f.read(), password=None)
    pem_text = crypto_key.private_bytes(
        Encoding.PEM,
        PrivateFormat.TraditionalOpenSSL,
        NoEncryption(),
    ).decode()
    pkey = paramiko.ECDSAKey(file_obj=io.StringIO(pem_text))

    # Read the certificate data and attach it to the key
    # Paramiko >= 3.1 supports cert-based auth via key.load_certificate
    cert_data = cert_pub.read_text().strip()
    pkey.load_certificate(cert_data)

    client = paramiko.SSHClient()
    # Leonardo's host keys rotate across login nodes; accept them.
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        logger.info("Connecting to %s@%s:%d with certificate auth...", username, hostname, port)
        client.connect(
            hostname=hostname,
            port=port,
            username=username,
            pkey=pkey,
            look_for_keys=False,
            allow_agent=False,
            timeout=timeout,
        )

        stdin, stdout, stderr = client.exec_command(command, timeout=timeout)

        logger.info("Executing command: %s", command)

        # Read output BEFORE waiting for exit status to avoid deadlock:
        # if the command fills the SSH buffer, recv_exit_status() will
        # block forever waiting for a command that is itself blocked on
        # writing to a full buffer.
        out = stdout.read().decode(errors="replace")
        err = stderr.read().decode(errors="replace")
        combined = (out + err).strip()
        logger.info("Command output:\n%s", combined)

        exit_status = stdout.channel.recv_exit_status()
        logger.info("Command finished with exit status %d", exit_status)

        if exit_status != 0:
            raise RuntimeError(f"Remote command exited with status {exit_status}:\n{combined}")

        return combined

    finally:
        logger.info("Closing SSH connection")
        client.close()


# ---------------------------------------------------------------------------
# Convenience CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    logger.info("paramiko version: %s", paramiko.__version__)

    parser = argparse.ArgumentParser(
        description="Run a command on CINECA Leonardo via SSH certificate auth."
    )
    parser.add_argument("command", help="Remote command to execute")
    parser.add_argument("--user", default=None, help="CINECA username")
    parser.add_argument("--email", default=None, help="CINECA email")
    parser.add_argument("--key", default=None, help="Path to step SSH private key")
    args = parser.parse_args()

    output = run_command_cineca(
        args.command,
        username=args.user,
        email=args.email,
        key_path=args.key,
    )
    print(output)
