#!/usr/bin/env python3

import argparse
import logging
import os

# parse command-line options
parser = argparse.ArgumentParser(description="Create symlinks to shared directories")
parser.add_argument("working_dir", help="path to the shared working directory")
parser.add_argument(
    "--fix",
    action="store_true",
    help="automatically remove and recreate any incorrect symlinks",
)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Change to the repo root directory (parent of scripts/)
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
os.chdir(repo_root)

# Create all the links
for d in ["logs", "models", "plots", "results"]:
    target = os.path.join(args.working_dir, d)
    # Check if something exists at this path (including broken symlinks)
    if os.path.islink(d):
        # It's a symlink - check if it points to the correct target
        if os.readlink(d) == target and os.path.exists(target):
            logger.info(f"'{d}' already correctly linked to {target}, skipping.")
            continue
        else:
            logger.warning(f"'{d}' is a symlink BUT IS NOT correctly linked to {target}.")
            if args.fix:
                logger.info(f"Removing incorrect symlink '{d}' and recreating it.")
                os.remove(d)
                # fall through to create the correct link below
            else:
                logger.warning("Run this script with --fix to remove it automatically.")
                continue
    elif os.path.exists(d):
        # It exists but is not a symlink (regular file or directory)
        logger.warning(
            f"'{d}' exists as a file/directory (not a symlink), PLEASE REMOVE IT MANUALLY."
        )
        continue

    # create link if we didn't continue above
    logger.info(f"{d} -> {target}")
    os.symlink(target, d)
