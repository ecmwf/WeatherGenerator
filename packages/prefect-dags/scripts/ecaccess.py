#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
#     "weathergen-prefect-dags",
# ]
#
# [tool.uv.sources]
# weathergen-prefect-dags = { path = "../", editable = true }
# ///
"""
CLI entry point for the ECMWF ECaccess SOAP client.

All client logic lives in weathergen.prefect_dags.cmd_runners.ecmwd_ecaccess_perl;
this script is a thin uv-runnable wrapper so the same code path is used both
from this CLI and from any other caller in the package.

Usage:
    chmod +x ecaccess.py

    # No auth needed — confirms the gateway is reachable:
    ./ecaccess.py gateway

    # Check certificate validity / list available operations:
    ./ecaccess.py ops

    # List available queues:
    ./ecaccess.py queues

    # List your jobs:
    ./ecaccess.py jobs
    ./ecaccess.py jobs 123456          # detailed view of one job

    # Submit a local Slurm job script:
    ./ecaccess.py submit my_job.sh
    ./ecaccess.py submit my_job.sh --queue hpc --name my_run

    # Submit a script already on the remote side:
    ./ecaccess.py submit my_job.sh --distant

    # Delete / restart a job:
    ./ecaccess.py delete 123456
    ./ecaccess.py restart 123456

    # Get job output / error / input:
    ./ecaccess.py get 123456
    ./ecaccess.py get 123456 --error
    ./ecaccess.py get 123456 -o result.log

Environment variables:
    ECCERT              Path to .eccert.crt  (default: ~/.eccert.crt)
    https_ecaccess      Gateway host for HTTPS control channel
                        (default: boaccess.ecmwf.int)
    http_ecaccess       Gateway host for HTTP data channel
                        (default: boaccess.ecmwf.int)
"""

from __future__ import annotations

import sys

from weathergen.prefect_dags.cmd_runners.ecmwf_ecaccess_perl import main

if __name__ == "__main__":
    sys.exit(main())
