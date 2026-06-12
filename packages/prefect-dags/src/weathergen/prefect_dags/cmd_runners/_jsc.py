"""
UNICORE command runner for JSC HPCs (JUPITER, JURECA, JUWELS booster).

Commands are submitted through the JSC UNICORE REST gateway
(https://unicore.fz-juelich.de) and executed interactively on a login node
(UC_PREFER_INTERACTIVE_EXECUTION). Authentication uses a bearer token,
read by default from ``~/.jsc_unicore_token``.

Usage via context:

    JscUnicoreContext(hpc="jupiter", project="weatherai")

The lower-level reference implementation (``run_command_jsc``) is kept for
the legacy pollers in ``weathergen.jsc_slurm_poller``.
"""

import contextlib
import logging
import re
import shlex
import time
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any, Literal

import pyunicore.client as uc_client
import pyunicore.credentials as uc_credentials

from weathergen.prefect_dags.cmd_runners._types import (
    Command,
    CommandResult,
    CommandRunner,
    quote_path,
)
from weathergen.prefect_dags.result import OpError, Result

type JscHpc = Literal["jupiter", "jureca", "juwels"]

REGISTRY_URL = "https://unicore.fz-juelich.de/FZJ/rest/registries/default_registry"

DEFAULT_TOKEN_FILE = Path("~/.jsc_unicore_token").expanduser()

# Known UNICORE core endpoints per JSC site. These are stable; use
# `discover_sites` to re-query the registry if they ever change.
_JSC_SITES: dict[str, str] = {
    "JUPITER": "https://unicore.fz-juelich.de/JUPITER/rest/core",
    "JURECA": "https://unicore.fz-juelich.de/JURECA/rest/core",
    "JUWELS": "https://unicore.fz-juelich.de/JUWELS/rest/core",
}

# Maps our HPC names to the UNICORE site names.
_VALID_HPCS: dict[JscHpc, str] = {
    "jupiter": "JUPITER",
    "jureca": "JURECA",
    "juwels": "JUWELS",
}

# UNICORE terminal job statuses.
_TERMINAL_STATUSES = ("SUCCESSFUL", "FAILED", "DONE")

# Name of the inline-imported script holding the actual command in the job
# sandbox (see JscUnicoreCommandRunner._run).
_COMMAND_SCRIPT = "wg_command.sh"


def discover_sites(credential) -> dict[str, str]:
    """Query the JSC UNICORE registry and return {site_name: base_url}."""
    import requests

    resp = requests.get(
        REGISTRY_URL,
        headers={
            "Accept": "application/json",
            "Authorization": credential.get_auth_header(),
        },
    )
    resp.raise_for_status()
    sites = {}
    for entry in resp.json().get("entries", []):
        href = entry.get("href", "")
        if entry.get("type", "") == "TargetSystemFactory":
            m = re.match(r"(https://\S+/rest/core).*", href)
            n = re.match(r"https://\S+/(\S+)/rest/core", href)
            if m and n:
                sites[n.group(1)] = m.group(1)
    return sites


@dataclass
class JscUnicoreContext:
    """
    Connection info for JSC clusters reached via the UNICORE REST gateway.

    ``hpc``: one of the supported JSC systems
        (``"jupiter"``, ``"jureca"``, ``"juwels-booster"``).
    ``project``: the JSC project (UNICORE job ``Project`` field).
    ``token``: bearer token. A ``Path`` (or None) points to a file containing
        the token (default: ``~/.jsc_unicore_token``, re-read on every
        submission so rotations are picked up); a ``str`` is the literal
        token value.
    ``account``: Slurm account to charge (``sbatch --account``). Optional —
        leave unset to use the cluster default.
    """

    hpc: JscHpc
    project: str
    token: str | Path | None = None
    account: str | None = None


class JscUnicoreCommandRunner(CommandRunner):
    """
    CommandRunner for JSC clusters via pyunicore.

    Each ``run()`` creates a fresh UNICORE client (the token file is re-read
    on every submission), submits the command as an interactive job on a
    login node, polls until it reaches a terminal status while streaming
    stdout to the logger, and returns the captured stdout/stderr/exit code.
    """

    name = "jsc_unicore"
    _ctx: JscUnicoreContext

    def __init__(self, context: JscUnicoreContext):
        self._ctx = context
        self.hpc = context.hpc

    def run(self, cmd: Command, logger: Logger) -> Result[CommandResult]:
        try:
            return self._run(cmd, logger)
        except Exception as e:
            # Per the CommandRunner contract: errors are returned, not raised.
            logger.error(f"UNICORE command failed: {e}")
            return OpError(err=e)

    def _run(self, cmd: Command, logger: Logger) -> Result[CommandResult]:
        ctx = self._ctx
        client = _make_client(ctx.token, ctx.hpc)

        # UNICORE runs the executable in the job sandbox directory, so
        # materialize working_directory as a `cd` prefix. Env vars go through
        # the job description's Environment instead of `export` lines.
        remote_cmd = _build_remote_command(cmd)
        env = {"UC_PREFER_INTERACTIVE_EXECUTION": "true", **(cmd.env_vars or {})}
        # The Executable string is embedded by UNICORE/X inside a
        # double-quoted shell assignment (`UC_EXECUTABLE="..."`) in its
        # generated wrapper script, so any quoting in the command breaks the
        # wrapper. Ship the command verbatim as an inline-imported script in
        # the job sandbox and keep Executable/Arguments trivially shell-safe.
        job_desc: dict[str, Any] = {
            "Executable": "/bin/bash",
            "Arguments": [_COMMAND_SCRIPT],
            "Environment": env,
            "Imports": [
                {
                    "From": "inline://dummy",
                    "To": _COMMAND_SCRIPT,
                    "Data": remote_cmd + "\n",
                }
            ],
        }
        if ctx.project:
            job_desc["Project"] = ctx.project

        logger.info(f"Submitting UNICORE job on {ctx.hpc}: {remote_cmd}")
        return _submit_and_wait(client, job_desc, logger)


def _build_remote_command(cmd: Command) -> str:
    parts: list[str] = []
    if cmd.working_directory is not None:
        parts.append(f"cd {quote_path(cmd.working_directory)} &&")
    if isinstance(cmd.command, str):
        parts.append(cmd.command)
    else:
        parts.append(shlex.join(cmd.command))
    return " ".join(parts)


def _make_client(token: str | Path | None, hpc: JscHpc) -> uc_client.Client:
    credential = uc_credentials.BearerToken(_load_token(token))
    site = _VALID_HPCS.get(hpc)
    if not site:
        raise ValueError(f"Invalid HPC '{hpc}'. Valid options are: {', '.join(_VALID_HPCS.keys())}")
    sites = _JSC_SITES  # discover_sites(credential)
    if site not in sites:
        raise ValueError(f"Site '{site}' not found in registry")
    return uc_client.Client(credential, site_url=sites[site], check_authentication=False)


def _load_token(token: str | Path | None) -> str:
    """Resolve the bearer token: Path (or None -> default file) reads the
    file; a plain str is the literal token value."""
    if token is None:
        token = DEFAULT_TOKEN_FILE
    if isinstance(token, Path):
        with open(token.expanduser()) as f:
            return f.read().strip()
    return token


def _job_status(job: uc_client.Job) -> str:
    # `Job.properties` is typed as Optional[dict] because it's set lazily on
    # first access, but a job we just submitted always has properties.
    props = job.properties
    if not isinstance(props, dict):
        raise RuntimeError(f"Job {job.job_id} has no properties")
    return props["status"]


def _job_exit_code(job: uc_client.Job, status: str) -> int:
    """Best-effort exit code: UNICORE exposes it as `exitCode` when the
    batch system reports one; fall back on the terminal status otherwise."""
    props = job.properties
    raw = props.get("exitCode") if isinstance(props, dict) else None
    try:
        return int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0 if status in ("SUCCESSFUL", "DONE") else 1


def _read_remote_text(wd: uc_client.Storage, path: str) -> str:
    # `Storage.stat()` returns PathFile | PathDir; only PathFile has .raw().
    # /stdout and /stderr are always files, but the static type doesn't know that.
    entry = wd.stat(path)
    if not isinstance(entry, uc_client.PathFile):
        raise RuntimeError(f"Expected a file at {path}, got {type(entry).__name__}")
    return entry.raw().read().decode("utf-8", errors="replace")


def _submit_and_wait(
    client: uc_client.Client,
    job_desc: dict[str, Any],
    logger: logging.Logger,
    poll_interval: float = 2.0,
) -> CommandResult:
    """Submit a UNICORE job, poll until terminal while streaming stdout to
    the logger, and return the captured output and exit code."""
    job = client.new_job(job_description=job_desc)
    try:
        logger.info(f"Job ID:     {job.job_id}")
        logger.info(f"Job Status: {_job_status(job)}")
        # Fast until here, working dir is slow.
        logger.info(f"Working dir: {job.working_dir}")
        logger.info(f"Job URL:    {job.resource_url}")

        wd = job.working_dir
        stdout_offset = 0
        stdout_parts: list[str] = []

        def _drain_stdout() -> None:
            nonlocal stdout_offset
            with contextlib.suppress(Exception):
                content = _read_remote_text(wd, "/stdout")
                if len(content) > stdout_offset:
                    new_content = content[stdout_offset:]
                    stdout_parts.append(new_content)
                    logger.info(new_content)
                    stdout_offset = len(content)

        while _job_status(job) not in _TERMINAL_STATUSES:
            logger.info(f"Status:     {_job_status(job)}")
            _drain_stdout()
            time.sleep(poll_interval)

        status = _job_status(job)
        logger.info(f"Status:     {status}")
        # Read any remaining stdout.
        _drain_stdout()

        stderr_text = ""
        with contextlib.suppress(Exception):
            stderr_text = _read_remote_text(wd, "/stderr")
        if stderr_text.strip():
            logger.warning(stderr_text)

        return CommandResult(
            stdout="".join(stdout_parts),
            stderr=stderr_text,
            return_code=_job_exit_code(job, status),
        )
    finally:
        with contextlib.suppress(Exception):
            job.delete()


def run_command_jsc(
    token: Path | str,
    hpc: JscHpc,
    project: str,
    command: str,
    logger: logging.Logger | None = None,
) -> str:
    """Submit a command via UNICORE and return its stdout.

    Legacy interface kept for the pollers: raises RuntimeError when the job
    fails. Prefer `JscUnicoreContext` + `run_cmd` for new code.
    """
    log = logger or logging.getLogger(__name__)
    client = _make_client(token, hpc)
    job_desc: dict[str, Any] = {
        "Executable": command,
        "Environment": {"UC_PREFER_INTERACTIVE_EXECUTION": "true"},
    }
    if project:
        job_desc["Project"] = project
    log.info(f"Submitting: {command}")
    result = _submit_and_wait(client, job_desc, log)
    if result.return_code != 0:
        raise RuntimeError(f"Job failed with return code {result.return_code} on {hpc}")
    return result.stdout
