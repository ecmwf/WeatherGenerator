#!/usr/bin/env -S uv run
# /// script
# dependencies = ["boto3<1.36", "certifi"]
# [tool.uv]
# exclude-newer = "2025-05-01T00:00:00Z"
# ///

"""Download a WeatherGenerator checkpoint without WeatherGenerator-private."""

import argparse
import logging
import os
from pathlib import Path

import boto3
import certifi

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ENDPOINT_URL = "https://object-store.os-api.cci1.ecmwf.int"
_BUCKET = "weathergenerator-dev"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Experiment run ID to download.")
    parser.add_argument(
        "--epoch",
        type=int,
        help="Checkpoint epoch to download; defaults to the latest checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "models",
        help="Directory that will contain the downloaded run directory (default: %(default)s).",
    )
    parser.add_argument("--profile", help="Optional AWS profile name.")
    parser.add_argument("--bucket", default=os.environ.get("WEATHERGEN_S3_BUCKET", _BUCKET))
    parser.add_argument(
        "--endpoint-url",
        default=os.environ.get("WEATHERGEN_S3_ENDPOINT", _ENDPOINT_URL),
    )
    args = parser.parse_args()

    session = boto3.Session(profile_name=args.profile)
    s3 = session.client(
        "s3",
        region_name="us-east-1",
        endpoint_url=args.endpoint_url,
        verify=certifi.where(),
    )
    output_path = args.output_dir.expanduser().resolve() / args.run_id
    output_path.mkdir(parents=True, exist_ok=True)

    if args.epoch is None:
        checkpoint_key = "models/latest.chkpt"
        checkpoint_name = f"{args.run_id}_latest.chkpt"
    else:
        checkpoint_key = f"models/epoch{args.epoch:04d}.chkpt"
        checkpoint_name = f"{args.run_id}_chkpt{args.epoch:05d}.chkpt"

    artifacts = {
        "config.json": output_path / f"model_{args.run_id}.json",
        checkpoint_key: output_path / checkpoint_name,
    }
    for key, destination in artifacts.items():
        source = f"experiments/{args.run_id}/{key}"
        temporary = destination.with_suffix(destination.suffix + ".download")
        logger.info("Downloading s3://%s/%s", args.bucket, source)
        s3.download_file(args.bucket, source, str(temporary))
        temporary.replace(destination)

    logger.info("Downloaded model artifacts to %s", output_path)


if __name__ == "__main__":
    main()