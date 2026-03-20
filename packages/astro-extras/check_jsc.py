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
import os
import re
import sys
import time

import pyunicore.client as uc_client
import pyunicore.credentials as uc_credentials

REGISTRY_URL = (
    "https://unicore.fz-juelich.de/FZJ/rest/registries/default_registry"
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
    for entry in resp.json().get("entries", []):
        href = entry.get("href", "")
        if entry.get("type", "") == "TargetSystemFactory":
            m = re.match(r"(https://\S+/rest/core).*", href)
            n = re.match(r"https://\S+/(\S+)/rest/core", href)
            if m and n:
                sites[n.group(1)] = m.group(1)
    return sites


def run_command(client, command, project=None, poll_interval=2.0):
    """Submit a command via UNICORE and print its output."""
    job_desc = {
        "Executable": command,
        "Environment": {"UC_PREFER_INTERACTIVE_EXECUTION": "true"},
    }
    if project:
        job_desc["Project"] = project

    print(f"Submitting: {command}")
    job = client.new_job(job_description=job_desc)
    print(f"Job URL:    {job.resource_url}")

    while job.properties["status"] not in ("SUCCESSFUL", "FAILED", "DONE"):
        time.sleep(poll_interval)

    status = job.properties["status"]
    print(f"Status:     {status}")

    wd = job.working_dir
    try:
        print(wd.stat("/stdout").raw().read().decode("utf-8", errors="replace"), end="")
    except Exception:
        pass

    try:
        stderr_text = wd.stat("/stderr").raw().read().decode("utf-8", errors="replace")
        if stderr_text.strip():
            print(stderr_text, end="", file=sys.stderr)
    except Exception:
        pass

    try:
        job.delete()
    except Exception:
        pass

    if status == "FAILED":
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run a command on a JSC UNICORE site.",
    )
    parser.add_argument(
        "--token", default=DEFAULT_TOKEN_FILE,
        help=f"Path to token file (default: {DEFAULT_TOKEN_FILE})",
    )
    parser.add_argument(
        "--site", default="JUDAC",
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

    if not os.path.exists(args.token):
        print(f"Error: token file not found: {args.token}", file=sys.stderr)
        sys.exit(1)

    with open(args.token) as f:
        token = f.read().strip()

    credential = uc_credentials.BearerToken(token=token)
    sites = discover_sites(credential)

    if args.site not in sites:
        print(f"Error: site '{args.site}' not found. Available: {', '.join(sorted(sites))}", file=sys.stderr)
        sys.exit(1)

    client = uc_client.Client(credential, site_url=sites[args.site], check_authentication=False)
    run_command(client, args.command, project=args.project)


if __name__ == "__main__":
    main()
