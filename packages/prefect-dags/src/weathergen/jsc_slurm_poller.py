import asyncio
import functools
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyunicore.client as uc_client
import pyunicore.credentials as uc_credentials
from prefect import flow, task
from prefect.variables import Variable

from weathergen.prefect_dags.cineca import run_command_cineca
from weathergen.prefect_dags.prefect_logging import get_run_logger

log = logging.getLogger("weathergen.prefect.jsc_slurm_poller")

REGISTRY_URL = "https://unicore.fz-juelich.de/FZJ/rest/registries/default_registry"
DEFAULT_TOKEN_FILE = Path(os.path.expanduser("~/.jsc_unicore_token"))

# TODO: move somewhere else
type HpcName = str


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


_JSC_SITES = {
    "JUPITER": "https://unicore.fz-juelich.de/JUPITER/rest/core",
    "JURECA": "https://unicore.fz-juelich.de/JURECA/rest/core",
    "JUWELS": "https://unicore.fz-juelich.de/JUWELS/rest/core",
}


_VALID_HPCS: dict[HpcName, str] = {
    "jupiter": "JUPITER",
    "jureca": "JURECA",
    "juwels-booster": "JUWELS",
}


def run_command_jsc(
    token: Path | str, hpc: HpcName, project: str, command: str, logger: logging.Logger
) -> str:
    """Submit a command via UNICORE and print its output."""
    if isinstance(token, Path):
        with open(token) as f:
            tk_value = f.read().strip()
            credential = uc_credentials.BearerToken(tk_value)
    else:
        credential = uc_credentials.BearerToken(token)
    logger.info(f"Creential: {credential}")
    site = _VALID_HPCS.get(hpc)
    logger.info("site: %s", site)
    print("credential:", credential)
    print("site:", site)
    if not site:
        raise ValueError(f"Invalid HPC '{hpc}'. Valid options are: {', '.join(_VALID_HPCS.keys())}")
    sites = _JSC_SITES  # discover_sites(credential)
    print("discovered sites:", sites)
    logger.info(f"Discovered sites: {sites}")
    if site not in sites:
        raise ValueError(f"Site '{site}' not found in registry")
    client = uc_client.Client(credential, site_url=sites[site], check_authentication=False)
    return _run_command_jsc(client, command, project=project)


def _job_status(job: uc_client.Job) -> str:
    # `Job.properties` is typed as Optional[dict] because it's set lazily on
    # first access, but a job we just submitted always has properties.
    props = job.properties
    if not isinstance(props, dict):
        raise RuntimeError(f"Job {job.job_id} has no properties")
    return props["status"]


def _read_remote_text(wd: uc_client.Storage, path: str) -> str:
    # `Storage.stat()` returns PathFile | PathDir; only PathFile has .raw().
    # /stdout and /stderr are always files, but the static type doesn't know that.
    entry = wd.stat(path)
    if not isinstance(entry, uc_client.PathFile):
        raise RuntimeError(f"Expected a file at {path}, got {type(entry).__name__}")
    return entry.raw().read().decode("utf-8", errors="replace")


def _run_command_jsc(
    client: uc_client.Client,
    command: str,
    project: str,
    poll_interval: float = 2.0,
    logger: logging.Logger | None = None,
) -> str:
    """Submit a command via UNICORE and print its output."""
    log = logger or logging.getLogger(__name__)
    job_desc = {
        "Executable": command,
        "Environment": {"UC_PREFER_INTERACTIVE_EXECUTION": "true"},
    }
    # TODO I should not have to do that
    if project:
        job_desc["Project"] = project

    log.info("Submitting: %s", command)
    job = client.new_job(job_description=job_desc)
    try:
        log.info("Job ID:     %s", job.job_id)
        log.info("Job Status: %s", _job_status(job))
        # Fast until here, working dir is slow.
        log.info("Working dir: %s", job.working_dir)
        log.info("Job URL:    %s", job.resource_url)

        wd = job.working_dir
        stdout_offset = 0
        stdout_parts: list[str] = []

        while _job_status(job) not in ("SUCCESSFUL", "FAILED", "DONE"):
            log.info("Status:     %s", _job_status(job))
            try:
                content = _read_remote_text(wd, "/stdout")
                if len(content) > stdout_offset:
                    new_content = content[stdout_offset:]
                    stdout_parts.append(new_content)
                    log.info(new_content)
                    stdout_offset = len(content)
            except Exception:
                pass
            time.sleep(poll_interval)

        status = _job_status(job)
        log.info("Status:     %s", status)

        # Read any remaining stdout
        try:
            content = _read_remote_text(wd, "/stdout")
            if len(content) > stdout_offset:
                new_content = content[stdout_offset:]
                stdout_parts.append(new_content)
                log.info(new_content)
        except Exception:
            pass

        try:
            stderr_text = _read_remote_text(wd, "/stderr")
            if stderr_text.strip():
                log.warning(stderr_text)
        except Exception:
            pass

        if status == "FAILED":
            raise RuntimeError(f"Job failed: {job.resource_url}")

        return "".join(stdout_parts)
    finally:
        try:
            job.delete()
        except Exception:
            pass


async def run_blocking(fn, *args, **kwargs):
    """Run a blocking function in the default thread executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


@dataclass
class SlurmJobs:
    hpc: HpcName
    job_name: str
    job_id: str
    job_status: str


TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"}


def _get_status(s) -> str:
    s = s.upper()
    for pattern in ("RUNNING", "PENDING", "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"):
        if pattern in s:
            return pattern
    return "UNKNOWN"


_JOB_PREFIX = "weathergen"


def _get_queue_command(jobs_prefix: str | None) -> str:
    job_prefix = jobs_prefix or _JOB_PREFIX
    #     return f"""
    # {{
    # squeue -a -h -o "%.40j %i %T"
    # }} | grep {job_prefix}
    # """
    return f"""
{{  
squeue -a -h -o "%.40j %i %T"  
sacct -S now-3days -E now -a --format=JobName%-40,JobID,State%20  
}} | grep {job_prefix}
"""


def _parse_slurm_output(raw_data: str, hpc: HpcName, logger: logging.Logger) -> list[SlurmJobs]:
    jobs = []
    for line in raw_data.strip().splitlines():
        parts = line.split()
        if len(parts) < 3:
            logger.warning("Unexpected line format (skipping): %s", line)
            continue
        job_name = parts[0]
        job_id = parts[1]
        job_status = _get_status(" ".join(parts[2:]))
        sj = SlurmJobs(hpc=hpc, job_name=job_name, job_id=job_id, job_status=job_status)
        logger.info("Parsed job: %s", sj)
        jobs.append(sj)
    return jobs


# TODO: hardcoded to leonardo
@task(retries=0, retry_delay_seconds=10, task_run_name="cineca_slurm_poller-leonardo")
async def cineca_slurm_poller(logger: logging.Logger | None = None) -> list[SlurmJobs]:
    log = logger or get_run_logger()
    command = _get_queue_command(None)
    raw_data = await run_blocking(run_command_cineca, command, logger=log, username="thunter0")
    log.info("Raw data:\n%s", raw_data)
    jobs = _parse_slurm_output(raw_data, "cineca", log)
    return jobs


@task(retries=0, retry_delay_seconds=10, task_run_name="jsc_slurm_poller-{hpc}")
async def jsc_slurm_poller(hpc: HpcName, logger: logging.Logger | None = None) -> list[SlurmJobs]:
    # TODO I should not have to do that
    log = logger or get_run_logger()
    saved_token = await Variable.aget("JSC_UNICORE_TOKEN", default=None)
    if not saved_token:
        token = DEFAULT_TOKEN_FILE
    else:
        token = saved_token
    log.info(f"Using token from {'variable' if saved_token else 'file'}: {token}")
    command = _get_queue_command(None)
    # Important to pass the logger here.
    # The code is runnin on another thread and prefect will not see the logs.
    raw_data = await run_blocking(
        run_command_jsc, token, hpc, project="weatherai", command=command, logger=log
    )
    log.info("Raw data:\n%s", raw_data)
    jobs = _parse_slurm_output(raw_data, hpc, log)
    return jobs


@dataclass
class PrefectSlurmJobInfo:
    data: SlurmJobs
    updated_at: str


async def _update_job_variables(statuses: list[SlurmJobs]):
    now = datetime.now(UTC).isoformat()

    for sj in statuses:
        var_name = f"slurm_job_status_{sj.hpc}_{sj.job_id}"
        info = PrefectSlurmJobInfo(data=sj, updated_at=now)
        serial = json.dumps(asdict(info))

        await Variable.aset(var_name, serial, overwrite=True)

        # Tag management: move to "completed" tag when terminal
        if info.data.job_status in TERMINAL_STATES:
            await Variable.aset(
                var_name,
                serial,
                tags=["completed"],
                overwrite=True,
            )
            log.info(f"Job {sj.hpc}:{sj.job_id} reached terminal state: {info.data.job_status}")
        else:
            log.info(f"Job {sj.job_id} is in non-terminal state: {info.data.job_status}")


def _flow_run_name(base: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{base}-{timestamp}"


@flow(log_prints=True, flow_run_name=_flow_run_name("slurm_queue_poller_jsc"))
async def slurm_queue_poller_jsc():
    # TODO I should not have to do that
    log = get_run_logger()
    for hpc in ["jupiter", "juwels-booster"]:
        log.info(f"Polling SLURM queue for HPC: {hpc}")
        statuses = await jsc_slurm_poller(hpc, logger=log)
        await _update_job_variables(statuses)
        log.info(f"Updated variables for {len(statuses)} jobs from {hpc}")


@flow(log_prints=True, flow_run_name=_flow_run_name("slurm_queue_poller_cineca"))
async def slurm_queue_poller_cineca():
    log: logging.Logger = get_run_logger()
    hpc = "leonardo"
    log.info(f"Polling SLURM queue for HPC: {hpc}")
    statuses = await cineca_slurm_poller(logger=log)
    await _update_job_variables(statuses)
    log.info(f"Updated variables for {len(statuses)} jobs from {hpc}")


@dataclass
class CinecaContext:
    """
    Context information for Cineca HPC.
    The critical inforamtion that is required to execute commands
    on a Cineca cluster.
    """

    username: str
    ssh_key_path: str | os.PathLike


@dataclass
class JscContext:
    """
    Context information for JSC HPCs.
    """

    token: str | os.PathLike
    project: str


# Any context to run something.
type HpcContext = CinecaContext | JscContext


async def run_command_on_hpc(
    hpc: HpcName, context: HpcContext, command: str, logger: logging.Logger | None = None
) -> str:
    """
    Runs a command on the specified HPC using the provided context.
    """
    logger = logger or logging.getLogger(__name__)
    match context:
        case CinecaContext(username=username, ssh_key_path=ssh_key_path):
            return await run_blocking(
                run_command_cineca, command, username=username, key_path=ssh_key_path, logger=logger
            )
        case JscContext(token=token, project=project):
            return await run_blocking(run_command_jsc, token, hpc, project, command, logger=logger)
        case _:
            raise ValueError(f"Unsupported context type: {type(context)}")
