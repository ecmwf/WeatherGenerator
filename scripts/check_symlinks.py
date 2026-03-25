#!/usr/bin/env -S uv run
# /// script
# dependencies = [ "PyYAML", "GitPython"
# ]
# [tool.uv]
# exclude-newer = "2025-01-01T00:00:00Z"
# ///

import os
import subprocess
import sys
import logging
import yaml
import argparse

# parse command-line options
parser = argparse.ArgumentParser(description="Create symlinks to shared directories")
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

# This script creates symbolic links to the shared working directories.
# 1. Get the path of the private config of the cluster
# 2. Read the yaml and extract the path of the shared conf
# This uses the yq command equivalent in Python with PyYAML

# Run the command to get the config file path
try:
    result = subprocess.run(
        ["../WeatherGenerator-private/hpc/platform-env.py", "hpc-config"],
        capture_output=True,
        text=True,
        check=True
    )
    config_file = result.stdout.strip()
except subprocess.CalledProcessError as e:
    logger.error(f"Error running platform-env.py: {e}")
    sys.exit(1)

# Read the YAML file
try:
    with open(config_file, 'r') as f:
        data = yaml.safe_load(f)
    working_dir = data['path_shared_working_dir']
except (FileNotFoundError, yaml.YAMLError) as e:
    logger.error(f"Error reading config file {config_file}: {e}")
    sys.exit(1)

# Remove quotes
working_dir = working_dir.strip('"\'').strip()

# If the working directory does not exist, exit with an error
if not os.path.isdir(working_dir):
    logger.error(f"Working directory {working_dir} does not exist. Please check the configuration.")
    sys.exit(1)

# Ensure the working directory ends with a slash
if not working_dir.endswith('/'):
    working_dir += '/'

logger.info(f"Working directory: {working_dir}")

# Create all the links
for d in ["logs", "models", "output", "plots", "results"]:
    target = working_dir + d
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
        logger.warning(f"'{d}' exists as a file/directory (not a symlink), PLEASE REMOVE IT MANUALLY.")
        continue

    # create link if we didn't continue above
    logger.info(f"{d} -> {target}")
    os.symlink(target, d)

