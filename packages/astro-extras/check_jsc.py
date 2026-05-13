#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["pyunicore"]
# [tool.uv.sources]
# ///
"""
Run a command on a JSC UNICORE site using a saved API token.

Usage:
    python check_jsc.py [--token FILE] [--site SITE] [--project PROJECT] COMMAND

Examples:
    python check_jsc.py whoami
    python check_jsc.py --site JURECA 'squeue -u $USER'
    python check_jsc.py --token /tmp/my_token --site JUPITER hostname
"""

import argparse
import logging
import os
import re
import sys
import time
from dataclasses import dataclass

import pyunicore.client as uc_client
import pyunicore.credentials as uc_credentials

log = logging.getLogger(__name__)

REGISTRY_URL = (
    # "https://unicore.fz-juelich.de/FZJ/rest/registries/default_registry"
    "https://unicore.fz-juelich.de/JUPITER/rest/registries/default_registry"
)

DEFAULT_TOKEN_FILE = os.path.expanduser("~/.jsc_unicore_token")


def discover_sites(credential):
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
    log.info("Registry response: %s", resp.text)
    for entry in resp.json().get("entries", []):
        href = entry.get("href", "")
        if entry.get("type", "") == "TargetSystemFactory":
            m = re.match(r"(https://\S+/rest/core).*", href)
            n = re.match(r"https://\S+/(\S+)/rest/core", href)
            if m and n:
                sites[n.group(1)] = m.group(1)
    log.info("Discovered sites: %s", sites)
    return sites


@dataclass
class SlurmJobInfo:
    """
    Holds information about a submitted UNICORE job that can be used to query its status or
    cancel it later.
    """

    slurm_job_id: str


def run_command(
    client: uc_client.Client, command: str, project: str, poll_interval: float = 2.0
) -> str:
    """Submit a command via UNICORE and print its output."""
    job_desc = {
        "Executable": command,
        "Environment": {"UC_PREFER_INTERACTIVE_EXECUTION": "true"},
    }
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


def launch_slurm(
    client: uc_client.Client,
    command: str,
    project: str,
    job_name: str | None = None,
) -> SlurmJobInfo:
    """Submit a command via sbatch and return the SLURM job info."""
    sbatch_args = ""
    if job_name:
        sbatch_args += f" --job-name={job_name}"

    sbatch_command = f"sbatch{sbatch_args} <<'SLURM_EOF'\n#!/bin/bash\n{command}\nSLURM_EOF"
    stdout = run_command(client, sbatch_command, project=project)

    # sbatch outputs "Submitted batch job <job_id>"
    match = re.search(r"Submitted batch job (\d+)", stdout)
    if not match:
        raise RuntimeError(f"Failed to parse SLURM job ID from sbatch output: {stdout.strip()}")

    slurm_job_id = match.group(1)
    log.info("SLURM job submitted: %s", slurm_job_id)
    return SlurmJobInfo(slurm_job_id=slurm_job_id)


def main():
    parser = argparse.ArgumentParser(
        description="Run a command on a JSC UNICORE site.",
    )
    parser.add_argument(
        "--token",
        default=DEFAULT_TOKEN_FILE,
        help=f"Path to token file (default: {DEFAULT_TOKEN_FILE})",
    )
    parser.add_argument(
        "--site",
        default="JUDAC",
        help="UNICORE site name, e.g. JURECA, JUWELS, JUPITER (default: JUDAC)",
    )
    parser.add_argument(
        "--project",
        help="Budget/project account to charge",
    )
    parser.add_argument(
        "command",
        help="Shell command to execute on the remote site",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if not os.path.exists(args.token):
        log.error("Token file not found: %s", args.token)
        sys.exit(1)

    with open(args.token) as f:
        token = f.read().strip()

    credential = uc_credentials.BearerToken(token=token)
    # sites = discover_sites(credential)
    sites = {
        "JUPITER": "https://unicore.fz-juelich.de/JUPITER/rest/core",
        "JURECA": "https://unicore.fz-juelich.de/JURECA/rest/core",
        "JUWELS": "https://unicore.fz-juelich.de/JUWELS/rest/core",
    }

    if args.site not in sites:
        log.error("Site '%s' not found. Available: %s", args.site, ", ".join(sorted(sites)))
        sys.exit(1)

    client = uc_client.Client(credential, site_url=sites[args.site], check_authentication=False)
    run_command(client, args.command, project=args.project)


if __name__ == "__main__":
    main()
