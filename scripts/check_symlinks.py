#!/usr/bin/env python3
"""Create shared-directory links and optionally replace incorrect symlinks."""

import argparse
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Create symlinks to shared directories")
    parser.add_argument("working_dir", type=Path, help="path to the shared working directory")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="automatically remove and recreate any incorrect symlinks",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    os.chdir(Path(__file__).resolve().parent.parent)

    for name in ["logs", "models", "plots", "results"]:
        link = Path(name)
        target = args.working_dir / name
        if not target.exists():
            logger.warning(f"Target '{target}' does not exist.")
        if link.is_symlink() and link.readlink() != target:
            logger.warning(f"'{name}' is a symlink BUT IS NOT correctly linked to {target}.")
            if args.fix:
                logger.info(f"Removing incorrect symlink '{name}' and recreating it.")
                link.unlink()
            else:
                logger.warning("Run this script with --fix to remove it automatically.")
        elif link.is_symlink():
            logger.info(f"'{name}' already correctly linked to {target}, skipping.")
        elif link.exists():
            logger.warning(
                f"'{name}' exists as a file/directory (not a symlink), PLEASE REMOVE IT MANUALLY."
            )

        if not link.is_symlink() and not link.exists():
            logger.info(f"{name} -> {target}")
            link.symlink_to(target)


if __name__ == "__main__":
    main()
