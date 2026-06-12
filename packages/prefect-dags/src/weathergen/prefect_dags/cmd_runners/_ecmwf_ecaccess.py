"""
Command runner for ECMWF / Atos clusters via the ECaccess SOAP gateway.

Unlike EcmwfSshCommandRunner, this runner does not need an interactive
Teleport / tsh session. It authenticates with a long-lived `.eccert.crt`
certificate and submits each command as an ECaccess job through the
public boaccess gateway.

Trade-off: every `run()` call is an ECaccess job submission + poll, so
even trivial commands take seconds. Use this when you need to drive an
HPC from somewhere without ssh access (CI, a remote scheduler), not as
a drop-in replacement for ssh.
"""

import shlex
import tempfile
import time
from dataclasses import dataclass
from logging import Logger
from pathlib import Path

from weathergen.prefect_dags.cmd_runners._types import (
    Command,
    CommandResult,
    CommandRunner,
    quote_path,
)
from weathergen.prefect_dags.cmd_runners.ecmwf_ecaccess_perl import (
    DEFAULT_CERT,
    DEFAULT_GATEWAY,
    ECaccessClient,
)
from weathergen.prefect_dags.result import OpError, Result

# ECaccess job status values. INIT/WAIT/EXEC are non-terminal; DONE/STOP are.
_TERMINAL_STATES = {"DONE", "STOP"}

# How often to poll for job completion. ECaccess jobs are not interactive;
# anything below a few seconds just hammers the gateway for no benefit.
_POLL_INTERVAL_SECONDS = 10.0


@dataclass
class EcmwfEcaccessContext:
    """
    Connection info for ECMWF / Atos clusters reached through
    the ECaccess SOAP API.
    """

    # Path to the .eccert.crt X509 certificate. None falls back to
    # ~/.eccert.crt, matching the ECaccess Perl toolkit default.
    cert_path: str | Path | None = None
    # ECaccess gateway hostname. None falls back to boaccess.ecmwf.int.
    gateway: str | None = None
    # Slurm account to bill for jobs submitted through this runner.
    account: str | None = None


class EcmwfEcaccessCommandRunner(CommandRunner):
    """
    CommandRunner that executes commands on ECMWF / Atos clusters via the
    ECaccess SOAP API (cert-based auth, no ssh required).

    Each `run()` wraps the command in a small bash script, submits it as
    an ECaccess job, polls until it reaches a terminal state, and returns
    the captured stdout / stderr.
    """

    name = "ecmwf_ecaccess"
    # TODO: surface this via the context once we drive more than one cluster.
    hpc = "hpc2020"
    _ctx: EcmwfEcaccessContext

    def __init__(self, context: EcmwfEcaccessContext):
        self._ctx = context

    def run(self, cmd: Command, logger: Logger) -> Result[CommandResult]:
        script = _build_script(cmd)
        cert_path = (
            Path(self._ctx.cert_path).expanduser()
            if self._ctx.cert_path is not None
            else DEFAULT_CERT
        )
        gateway = self._ctx.gateway or DEFAULT_GATEWAY

        client = ECaccessClient(cert_path=cert_path, https_host=gateway, http_host=gateway)
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".sh", prefix="ecaccess_cmd_", delete=False
            ) as tmp:
                tmp.write(script)
                script_path = tmp.name

            try:
                logger.info(f"Submitting ECaccess job to {gateway}")
                job_id = client.submit_job(script_path, no_directives=True)
                logger.info(f"ECaccess job submitted: jobId={job_id}")

                status = _wait_for_terminal(client, job_id, logger)

                stdout = client.get_job_output(job_id, which="output").decode(
                    "utf-8", errors="replace"
                )
                stderr = client.get_job_output(job_id, which="error").decode(
                    "utf-8", errors="replace"
                )
            finally:
                Path(script_path).unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"ECaccess command failed: {e}")
            return OpError(err=e)
        finally:
            client.release_token()

        # ECaccess does not surface the underlying slurm exit code in the
        # job listing — we only get the terminal state. Map DONE to 0 and
        # any other terminal state to 1 so callers can still branch on
        # success/failure via return_code.
        return CommandResult(
            stdout=stdout,
            stderr=stderr,
            return_code=0 if status == "DONE" else 1,
        )


def _build_script(cmd: Command) -> str:
    """
    Wrap `cmd` in a bash script. ECaccess uploads and submits a script
    file, so we materialize working_directory / env_vars as `cd` and
    `export` statements at the top.
    """
    lines = ["#!/bin/bash", "set -e"]
    if cmd.working_directory is not None:
        lines.append(f"cd {quote_path(cmd.working_directory)}")
    if cmd.env_vars:
        for k, v in cmd.env_vars.items():
            lines.append(f"export {k}={shlex.quote(v)}")
    if isinstance(cmd.command, str):
        lines.append(cmd.command)
    else:
        lines.append(shlex.join(cmd.command))
    return "\n".join(lines) + "\n"


def _wait_for_terminal(client: ECaccessClient, job_id: str, logger: Logger) -> str:
    """Poll until the ECaccess job reaches DONE or STOP, then return that status."""
    last_status = ""
    while True:
        jobs = client.list_jobs(job_id=job_id)
        if not jobs:
            # The job vanished from the listing (e.g. it was deleted out from
            # under us). Treat as a failure rather than spinning forever.
            raise RuntimeError(f"ECaccess job {job_id} not found while polling")
        status = str(jobs[0].get("status", ""))
        if status != last_status:
            logger.info(f"ECaccess job {job_id} status: {status}")
            last_status = status
        if status in _TERMINAL_STATES:
            return status
        time.sleep(_POLL_INTERVAL_SECONDS)
