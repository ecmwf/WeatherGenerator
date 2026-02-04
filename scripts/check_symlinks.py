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
import yaml

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
    print(f"Error running platform-env.py: {e}")
    sys.exit(1)

# Read the YAML file
try:
    with open(config_file, 'r') as f:
        data = yaml.safe_load(f)
    working_dir = data['path_shared_working_dir']
except (FileNotFoundError, yaml.YAMLError) as e:
    print(f"Error reading config file {config_file}: {e}")
    sys.exit(1)

# Remove quotes
working_dir = working_dir.strip('"\'').strip()

# If the working directory does not exist, exit with an error
if not os.path.isdir(working_dir):
    print(f"Working directory {working_dir} does not exist. Please check the configuration.")
    sys.exit(1)

# Ensure the working directory ends with a slash
if not working_dir.endswith('/'):
    working_dir += '/'

print(f"Working directory: {working_dir}")

# Create all the links
for d in ["logs", "models", "output", "plots", "results"]:
    target = working_dir + d
    # Check if something exists at this path (including broken symlinks)
    if os.path.islink(d):
        # It's a symlink - check if it points to the correct target
        if os.readlink(d) == target and os.path.exists(target):
            print(f"'{d}' already correctly linked to {target}, skipping.")
            continue
        else:
            print(f"'{d}' is a symlink BUT IS NOT correctly linked to {target}, PLEASE REMOVE IT MANUALLY.")
            continue
    elif os.path.exists(d):
        # It exists but is not a symlink (regular file or directory)
        print(f"'{d}' exists as a file/directory (not a symlink), PLEASE REMOVE IT MANUALLY.")
        continue
    print(f"{d} -> {target}")
    os.symlink(target, d)

