from prefect import flow, task, variables
from prefect.variables import Variable
from prefect.tasks import task_input_hash
import asyncio, json, functools
from datetime import datetime, timezone
from prefect.logging import get_run_logger
import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path

import pyunicore.client as uc_client
import pyunicore.credentials as uc_credentials
from dataclasses import dataclass

log = logging.getLogger("weathergen.prefect.jsc_slurm_poller")
print("logger:", log)

REGISTRY_URL = (
    "https://unicore.fz-juelich.de/FZJ/rest/registries/default_registry"
)
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

_VALID_HPCS: dict[HpcName, str] = {
    "jupiter": "JUPITER",
    "jureca": "JURECA",
    "juwels-booster": "JUWELS"
}

def run_command(token: Path|str, hpc: HpcName, project: str, command: str, logger: logging.Logger) -> str:
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
    sites = discover_sites(credential)
    print("discovered sites:", sites)
    logger.info(f"Discovered sites: {sites}")
    if site not in sites:
        raise ValueError(f"Site '{site}' not found in registry")
    client = uc_client.Client(credential, site_url=sites[site], check_authentication=False)
    return _run_command(client, command, project=project)

def _run_command(client: uc_client.Client, command: str, project: str, poll_interval: float = 2.0, logger: logging.Logger | None = None) -> str:
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
        log.info("Job Status: %s", job.properties["status"])
        # Fast until here, working dir is slow.
        log.info("Working dir: %s", job.working_dir)
        log.info("Job URL:    %s", job.resource_url)

        wd = job.working_dir
        stdout_offset = 0
        stdout_parts: list[str] = []

        while job.properties["status"] not in ("SUCCESSFUL", "FAILED", "DONE"):
            log.info("Status:     %s", job.properties["status"])
            try:
                content = wd.stat("/stdout").raw().read().decode("utf-8", errors="replace")
                if len(content) > stdout_offset:
                    new_content = content[stdout_offset:]
                    stdout_parts.append(new_content)
                    log.info(new_content)
                    stdout_offset = len(content)
            except Exception:
                pass
            time.sleep(poll_interval)

        status = job.properties["status"]
        log.info("Status:     %s", status)

        # Read any remaining stdout
        try:
            content = wd.stat("/stdout").raw().read().decode("utf-8", errors="replace")
            if len(content) > stdout_offset:
                new_content = content[stdout_offset:]
                stdout_parts.append(new_content)
                log.info(new_content)
        except Exception:
            pass

        try:
            stderr_text = wd.stat("/stderr").raw().read().decode("utf-8", errors="replace")
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

@task(retries=0, retry_delay_seconds=10)
async def jsc_slurm_poller(hpc:HpcName, jobs_prefix = "weathergen") -> list[SlurmJobs]:
    # TODO I should not have to do that
    log = get_run_logger()
    saved_token = await Variable.aget("JSC_UNICORE_TOKEN", default=None)
    if not saved_token:
        token= DEFAULT_TOKEN_FILE
    else:
        token = saved_token
    log.info(f"Using token from {'variable' if saved_token else 'file'}: {token}")
    command = f"""
{{  
squeue -a -h -o "%.40j %i %T"  
sacct -S now-5days -E now -a --format=JobName%-40,JobID,State%20  
}} | grep {jobs_prefix}
"""
    # Important to pass the logger here.
    # The code is runnin on another thread and prefect will not see the logs.
    raw_data = await run_blocking(run_command, token, hpc, project="weatherai", command=command, logger=log)
    log.info("Raw data:\n%s", raw_data)
    # Parse the output into SlurmJobs
    # Each line should have the format: JobName JobID State
    jobs = []
    for line in raw_data.strip().splitlines():
        parts = line.split()
        if len(parts) < 3:
            log.warning("Unexpected line format (skipping): %s", line)
            continue
        job_name = parts[0]
        job_id = parts[1]
        job_status = _get_status(" ".join(parts[2:]))
        sj = SlurmJobs(hpc=hpc, job_name=job_name, job_id=job_id, job_status=job_status)
        log.info("Parsed job: %s", sj)
        jobs.append(sj)
    return jobs

@dataclass
class PrefectSlurmJobInfo:
    data: SlurmJobs
    updated_at: str


def _update_job_variables(statuses: list[SlurmJobs]):
    now = datetime.now(timezone.utc).isoformat()
    
    for sj in statuses:
        var_name = f"slurm_job_status_{sj.job_id}"
        info = PrefectSlurmJobInfo(data=sj, updated_at=now)
        
        Variable.set(var_name, json.dumps(info), overwrite=True)
        
        # Tag management: move to "completed" tag when terminal
        if info.data.job_status in TERMINAL_STATES:
            Variable.set(
                var_name,
                json.dumps(info),
                tags=["completed"],
                overwrite=True,
            )
            log.info(f"Job {sj.job_id} reached terminal state: {info.data.job_status}")
        else:
            log.info(f"Job {sj.job_id} is in non-terminal state: {info.data.job_status}")


@flow(log_prints=True)
async def jsc_slurm_queue_poller(jobs_prefix = "weathergen"):
    # TODO I should not have to do that
    log = get_run_logger()
    for hpc in [
        # "jupiter",
          "juwels-booster"
          ]:
        log.info(f"Polling SLURM queue for HPC: {hpc}")
        statuses = await jsc_slurm_poller(hpc, jobs_prefix)
        _update_job_variables(statuses)
        log.info(f"Updated variables for {len(statuses)} jobs from {hpc}")
