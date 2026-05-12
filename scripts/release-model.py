#!/usr/bin/env python3
"""
make_release.py - Create a release tag for a machine learning model across
 WeatherGenerator and WeatherGenerator-private repositories.

Usage:
    python make_release.py --release-name atmo-foo-bar --release-version 1.2 --run-id 12345
    python make_release.py --release-name atmo-forecast --release-version 1.2rc1 \
        --run-id 12345 --dry-run

Generated with Claude, manually reviewed and tested.        
"""

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("make_release")


# ---------------------------------------------------------------------------
# Repo paths  (script lives at WeatherGenerator/scripts/make_release.py)
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent  # WeatherGenerator/scripts/
WeatherGenerator_DIR = SCRIPT_DIR.parent  # WeatherGenerator/
WeatherGeneratorPrivate_DIR = WeatherGenerator_DIR.parent / "WeatherGenerator-private"

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

# Specific to the atmo model.
# This follows the naming conventions used in huggingface to simplify upload.
RELEASE_NAME_RE = re.compile(r"^atmo(-[A-Za-z0-9]+)+$")
# Accepts:  X.Y  with an optional rcN suffix (e.g. 1.2, 1.2rc1, 1.2rc2)
VERSION_RE = re.compile(r"^(\d+)\.(\d+)(?:(rc)(\d+))?$")


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def run(cmd: list, cwd: Path) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        details = "\n".join(filter(None, [result.stdout.strip(), result.stderr.strip()]))
        raise RuntimeError(f"Command {' '.join(cmd)!r} failed in {cwd}:\n{details}")
    return result


def git_fetch_tags(repo: Path) -> None:
    logger.info("Fetching tags in %s …", repo)
    run(["git", "fetch", "--tags", "origin"], cwd=repo)


def git_all_tags(repo: Path) -> list:
    result = run(["git", "tag", "--list"], cwd=repo)
    return [t.strip() for t in result.stdout.splitlines() if t.strip()]


def git_is_dirty(repo: Path) -> bool:
    """Return True if there are uncommitted changes to tracked files
    (ignores untracked files)."""
    result = subprocess.run(
        ["git", "diff-index", "--quiet", "HEAD", "--"],
        cwd=repo,
        capture_output=True,
    )
    return result.returncode != 0


def git_tag(repo: Path, tag: str, message: str) -> None:
    logger.info("Tagging %s with %r …", repo.name, tag)
    run(["git", "tag", "-a", tag, "-m", message], cwd=repo)


def git_push_tags(repo: Path) -> None:
    logger.info("Pushing tags in %s …", repo.name)
    run(["git", "push", "origin", "--tags"], cwd=repo)


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------


def parse_version(ver: str):
    """Return a sortable tuple, or None if ver doesn't match VERSION_RE."""
    m = VERSION_RE.match(ver)
    if not m:
        return None
    major = int(m.group(1))
    minor = int(m.group(2))
    pre_type = m.group(3)
    pre_num = int(m.group(4) or 0)
    # pre-release < final:  (0, n)  vs  (1, 0)
    pre = (0, pre_num) if pre_type else (1, 0)
    return (major, minor, pre)


def version_lt(a, b) -> bool:
    ma, mia, (poa, pna) = a
    mb, mib, (pob, pnb) = b
    return (ma, mia, poa, pna) < (mb, mib, pob, pnb)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_release_name(name: str) -> None:
    if not RELEASE_NAME_RE.match(name):
        logger.error(
            "Release name %r is invalid. "
            "Expected: atmo-<part>[-<part>...]  where each part is alphanumeric. "
            "Examples: atmo-wind, atmo-rain-v2",
            name,
        )
        sys.exit(1)


def validate_version(ver: str) -> None:
    if not VERSION_RE.match(ver):
        logger.error(
            "Version %r is invalid. Expected: X.Y or X.YrcN  e.g. 1.2  1.2rc1  2.0rc3",
            ver,
        )
        sys.exit(1)


def check_repos_exist() -> None:
    for repo in (WeatherGenerator_DIR, WeatherGeneratorPrivate_DIR):
        if not (repo / ".git").exists():
            logger.error("Not a git repository (or missing): %s", repo)
            sys.exit(1)


def check_not_dirty(repos: list) -> None:
    for repo in repos:
        if git_is_dirty(repo):
            logger.error(
                "Repository %s has uncommitted changes. "
                "Commit or stash them before making a release.",
                repo.name,
            )
            sys.exit(1)


def check_tag_absent(tags: list, full_tag: str) -> None:
    if full_tag in tags:
        logger.error("Tag %r already exists.", full_tag)
        sys.exit(1)


def check_no_smaller_version(tags: list, release_name: str, new_ver: str) -> None:
    """Refuse to release a version older than one that already exists."""
    prefix = f"{release_name}-"
    new_tuple = parse_version(new_ver)

    for tag in tags:
        if not tag.startswith(prefix):
            continue
        existing_ver = tag[len(prefix) :]
        existing_tuple = parse_version(existing_ver)
        if existing_tuple is None:
            continue
        if version_lt(new_tuple, existing_tuple):
            logger.error(
                "Version %r of %r already exists. Cannot release older version %r.",
                existing_ver,
                release_name,
                new_ver,
            )
            sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tag and release a versioned ML model across WeatherGenerator"
        " and WeatherGenerator-private repos."
    )
    parser.add_argument(
        "--release-name", required=True, metavar="NAME", help="Model name, e.g. atmo-wind-speed"
    )
    parser.add_argument(
        "--release-version",
        required=True,
        metavar="VER",
        help="Version string: X.Y or X.YrcN  e.g. 1.2 or 1.2rc1",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        metavar="RUN_ID",
        help="CI/CD run ID associated with this release",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all checks and fetch tags, but skip tagging and pushing.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()

    name = args.release_name
    ver = args.release_version
    run_id = args.run_id
    dry_run = args.dry_run
    full_tag = f"{name}-{ver}"
    repos = [WeatherGenerator_DIR, WeatherGeneratorPrivate_DIR]

    logger.info("Release name   : %s", name)
    logger.info("Release version: %s", ver)
    logger.info("Full tag       : %s", full_tag)
    logger.info("Run ID         : %s", run_id)
    logger.info("Dry run        : %s", dry_run)

    # 1 – Validate name format
    logger.info("[1/6] Validating release name …")
    validate_release_name(name)

    # 2 – Validate version format
    logger.info("[2/6] Validating version format …")
    validate_version(ver)

    # 3 – Repos exist
    logger.info("[3/6] Checking repositories …")
    check_repos_exist()

    # 4 – Fetch tags (done even in dry-run)
    logger.info("[4/6] Fetching tags from origin …")
    for repo in repos:
        git_fetch_tags(repo)

    # Use WeatherGenerator as the authoritative tag source
    all_tags = git_all_tags(WeatherGenerator_DIR)

    # 5 – Tag must not exist + no smaller version
    logger.info("[5/6] Checking version constraints …")
    check_tag_absent(all_tags, full_tag)
    check_no_smaller_version(all_tags, name, ver)

    # 6 – Repos must be clean
    logger.info("[6/6] Checking repositories are clean …")
    check_not_dirty(repos)

    if dry_run:
        logger.info("DRY RUN complete – all checks passed. No tags were created.")
        return

    # Tag
    tag_message = f"Release {name} {ver} (run-id: {run_id})"
    logger.info("Creating annotated tags …")
    for repo in repos:
        git_tag(repo, full_tag, tag_message)

    # Push
    logger.info("Pushing tags …")
    for repo in repos:
        git_push_tags(repo)

    logger.info("Release %r created and pushed successfully.", full_tag)


if __name__ == "__main__":
    main()
