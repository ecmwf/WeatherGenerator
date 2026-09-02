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
import time
import uuid
from pathlib import Path

import boto3
import certifi
from botocore import UNSIGNED
from botocore.config import Config

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ENDPOINT_URL = "https://object-store.os-api.cci1.ecmwf.int"
_BUCKET = "weathergenerator-dev"
_BUCKET_RELEASE = "weathergenerator-release"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _refresh_atime_if_size_matches(path: Path, expected_size: int) -> bool:
    if not path.is_file():
        return False

    file_stat = path.stat()
    if file_stat.st_size != expected_size:
        return False

    os.utime(path, ns=(time.time_ns(), file_stat.st_mtime_ns))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Experiment run ID to download.")
    parser.add_argument("--source", choices=("dev", "release"), default="dev")
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
    parser.add_argument("--bucket", default=os.environ.get("WEATHERGEN_S3_BUCKET"))
    parser.add_argument(
        "--endpoint-url",
        default=os.environ.get("WEATHERGEN_S3_ENDPOINT", _ENDPOINT_URL),
    )
    args = parser.parse_args()
    if args.bucket is None:
        args.bucket = _BUCKET if args.source == "dev" else _BUCKET_RELEASE

    session = boto3.Session(profile_name=args.profile)
    client_config = Config(signature_version=UNSIGNED) if args.bucket == _BUCKET_RELEASE else None
    s3 = session.client(
        "s3",
        region_name="us-east-1",
        endpoint_url=args.endpoint_url,
        verify=certifi.where(),
        config=client_config,
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
        prefix = "experiments" if args.bucket == _BUCKET else "checkpoints/atmo/main"
        source = f"{prefix}/{args.run_id}/{key}"
        expected_size = s3.head_object(Bucket=args.bucket, Key=source)["ContentLength"]

        if os.path.lexists(destination):
            if _refresh_atime_if_size_matches(destination, expected_size):
                logger.info("Skipping existing file with matching size: %s", destination)
                continue
            raise FileExistsError(
                f"Refusing to overwrite existing file with an unexpected size: {destination}"
            )

        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.download")
        try:
            logger.info("Downloading s3://%s/%s", args.bucket, source)
            s3.download_file(args.bucket, source, str(temporary))

            downloaded_size = temporary.stat().st_size
            if downloaded_size != expected_size:
                raise OSError(
                    f"Downloaded size mismatch for s3://{args.bucket}/{source}: "
                    f"expected {expected_size} bytes, got {downloaded_size}"
                )

            try:
                # Unlike rename/replace, link fails rather than overwriting a destination
                # created concurrently. The paths share a directory and filesystem.
                os.link(temporary, destination)
            except FileExistsError:
                if _refresh_atime_if_size_matches(destination, expected_size):
                    logger.info(
                        "Another process installed a file with matching size: %s", destination
                    )
                    continue
                raise FileExistsError(
                    f"Refusing to overwrite file created concurrently: {destination}"
                ) from None
        finally:
            temporary.unlink(missing_ok=True)

    logger.info("Downloaded model artifacts to %s", output_path)


if __name__ == "__main__":
    main()
