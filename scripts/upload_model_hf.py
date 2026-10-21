#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface_hub>=0.34,<2",
# ]
# ///

"""
Upload a trained model (config + checkpoint) to Hugging Face.

By default the files are taken from `models/<run_id>` in the root of this repository.

The files are uploaded to the `<run_id>/` folder of the repo with their original names, e.g.
`<run_id>/model_<run_id>_latest.json` and `<run_id>/<run_id>_latest.chkpt`, so they can be
downloaded with:
    hf download <repo-id> --include "<run_id>/*" --local-dir models

Log in first with a token that has write access to the repo:
    uvx --from huggingface_hub hf auth login

Example:
    ./scripts/upload_model_hf.py --run-id mz7oyrr5
"""

import argparse
import logging
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

_REPO_ROOT = Path(__file__).parent.parent
_DEFAULT_REPO_ID = "weathergen-dev/test-preprint-model"


def get_model_files(model_dir: Path, run_id: str, mini_epoch: int | None) -> list[Path]:
    """
    Returns the config and checkpoint files of the given mini_epoch (latest if None).
    """
    mini_epoch_id = "latest" if mini_epoch is None else f"chkpt{mini_epoch:05d}"
    files = [
        model_dir / f"model_{run_id}_{mini_epoch_id}.json",
        model_dir / f"{run_id}_{mini_epoch_id}.chkpt",
    ]
    missing = [f.name for f in files if not f.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing files in {model_dir}: {missing}")
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload the model config and checkpoint of a run to Hugging Face",
    )
    parser.add_argument("--run-id", required=True, help="The run id of the model")
    parser.add_argument(
        "--repo-id",
        default=_DEFAULT_REPO_ID,
        help=f"The Hugging Face model repo (default: {_DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--mini-epoch",
        type=int,
        help="The mini_epoch of the checkpoint to upload (default: latest)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="The directory containing the model files (default: <repo root>/models/<run-id>)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print which files would be uploaded",
    )
    args = parser.parse_args()

    model_dir = (args.model_dir or _REPO_ROOT / "models" / args.run_id).expanduser().resolve()
    _logger.info(f"Using model directory {model_dir}")

    files = get_model_files(model_dir, args.run_id, args.mini_epoch)
    for f in files:
        size_gb = f.stat().st_size / 1e9
        _logger.info(f"Will upload {f.name} ({size_gb:.2f} GB) to {args.repo_id}/{args.run_id}")
    if args.dry_run:
        sys.exit(0)

    api = HfApi()
    # Fail early if not logged in, before hashing several GB of checkpoint
    _logger.info(f"Logged in to Hugging Face as {api.whoami()['name']}")
    commit = api.create_commit(
        repo_id=args.repo_id,
        repo_type="model",
        operations=[
            CommitOperationAdd(path_in_repo=f"{args.run_id}/{f.name}", path_or_fileobj=f)
            for f in files
        ],
        commit_message=f"Upload {', '.join(f.name for f in files)}",
    )
    _logger.info(f"Uploaded to {commit.commit_url}")
