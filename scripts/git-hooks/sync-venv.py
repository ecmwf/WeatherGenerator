# ruff: noqa: T201
"""Git hooks that keep the venv in sync with the dependency files.

`scripts/actions.sh sync` installs the hooks (`sync-venv.py install`) on HPCs
with a configured uv link mode. After a pull, rebase or branch switch that
changed a dependency file, they run `scripts/actions.sh sync`. A later `uv run`
then finds nothing to sync, so it does not install packages with
pyproject.toml's symlink mode.

Git runs no hook after `git reset`, `git cherry-pick`, `git stash pop` or a
merge that is committed by hand after a conflict. `uv run` then installs the
changed packages as symlinks, until the next `scripts/actions.sh sync` relinks
them.

Runs with the venv's Python (see HOOK_STUB) and the standard library only.
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEPENDENCY_FILES = ["pyproject.toml", "packages/*/pyproject.toml", ".python-version"]
HOOKS = ["post-merge", "post-rewrite", "post-checkout"]
STUB_MARKER = "weathergen-sync-venv"
HOOK_STUB = """\
#!/bin/sh
# {marker}: installed by scripts/actions.sh sync
top=$(git rev-parse --show-toplevel) || exit 0
python="$top/.venv/bin/python"
script="$top/scripts/git-hooks/sync-venv.py"
[ -x "$python" ] && [ -f "$script" ] && exec "$python" -I "$script" {hook} "$@"
exit 0
"""


def git(*args: str) -> subprocess.CompletedProcess[str]:
    """Run a git command in the repository and capture its output."""
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)


def install_hooks() -> None:
    """Write a stub for each hook into the checkout's git hooks directory.

    Hooks from elsewhere are left alone. Nothing is installed if
    core.hooksPath is set, since that directory may serve other repositories.
    """
    if git("config", "core.hooksPath").returncode == 0:
        print("Not installing the venv sync hooks: core.hooksPath is set.", file=sys.stderr)
        return
    hooks_path = git("rev-parse", "--git-path", "hooks")
    if hooks_path.returncode != 0:
        return
    hooks_dir = REPO / hooks_path.stdout.strip()
    hooks_dir.mkdir(parents=True, exist_ok=True)
    for hook in HOOKS:
        stub = hooks_dir / hook
        if stub.exists() and STUB_MARKER not in stub.read_text(errors="replace"):
            print(f"Not installing the {hook} hook: {stub} exists.", file=sys.stderr)
            continue
        stub.write_text(HOOK_STUB.format(marker=STUB_MARKER, hook=hook))
        stub.chmod(0o755)


def commit_before(hook: str, hook_args: list[str]) -> str | None:
    """Return the commit that HEAD was on before the git command, or None if there is nothing to do.

    The arguments are those git passes to the hook (see `git help hooks`).
    A rebase fires post-checkout when it starts and post-rewrite when it has
    moved commits, so a rebase with local commits is handled twice.
    """
    match hook, hook_args:
        case "post-merge", _:
            return "ORIG_HEAD"
        case "post-rewrite", ["rebase", *_]:
            return "ORIG_HEAD"
        case "post-checkout", [old_head, new_head, "1"] if old_head != new_head:
            return old_head
    return None


def dependencies_changed_since(commit: str) -> bool:
    """Return True if a dependency file differs between `commit` and HEAD."""
    if git("rev-parse", "--quiet", "--verify", f"{commit}^{{commit}}").returncode != 0:
        return False
    return git("diff", "--quiet", commit, "HEAD", "--", *DEPENDENCY_FILES).returncode == 1


def sync_venv() -> None:
    """Run `scripts/actions.sh sync`, which picks the HPC's uv link mode."""
    print("Dependencies changed: re-syncing the venv (scripts/actions.sh sync)", flush=True)
    sync = subprocess.run(["./scripts/actions.sh", "sync"], cwd=REPO, check=False)
    if sync.returncode != 0:
        print("Venv sync failed; run ./scripts/actions.sh sync before uv run.", file=sys.stderr)


def main(command: str, *args: str) -> None:
    """Install the hooks, or act as the hook `command` that git called with `args`."""
    if command == "install":
        install_hooks()
        return
    old_head = commit_before(command, list(args))
    if old_head is not None and dependencies_changed_since(old_head):
        sync_venv()


if __name__ == "__main__":
    # A failure here must not fail the git command that called the hook.
    try:
        main(*sys.argv[1:])
    except Exception as error:
        print(f"sync-venv.py: {error!r}", file=sys.stderr)
