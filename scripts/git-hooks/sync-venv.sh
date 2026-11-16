#!/bin/bash
# Re-syncs the venv after git pull (post-merge, post-rewrite) or a branch
# switch (post-checkout) when dependency files changed, so that `uv run` never
# syncs by itself with pyproject.toml's link mode instead of the HPC's.
# Installed by `scripts/actions.sh sync` on HPCs that set uv_link_mode in their
# config (JUPITER); called from small stubs in .git/hooks with the hook name
# and git's hook arguments. Never fails the git command.

hook="$1"
shift
case "$hook" in
  post-merge) old=ORIG_HEAD ;;
  post-rewrite) [[ "$1" == rebase ]] || exit 0; old=ORIG_HEAD ;;
  # Arguments: previous HEAD, new HEAD, 1 for a branch checkout.
  # During a rebase, post-rewrite syncs once at the end.
  post-checkout)
    [[ "$3" == 1 && "$1" != "$2" ]] || exit 0
    [[ -d "$(git rev-parse --git-path rebase-merge)" ]] && exit 0
    [[ -d "$(git rev-parse --git-path rebase-apply)" ]] && exit 0
    old="$1" ;;
  *) exit 0 ;;
esac

top=$(git rev-parse --show-toplevel) || exit 0
cd "$top" || exit 0
[[ -d .venv ]] || exit 0
git rev-parse -q --verify "$old" > /dev/null || exit 0
if git diff --quiet "$old" HEAD -- pyproject.toml 'packages/*/pyproject.toml' \
  .python-version; then
  exit 0
fi

echo "Dependencies changed: re-syncing the venv (scripts/actions.sh sync)"
if ! ./scripts/actions.sh sync; then
  echo "Venv sync failed; run ./scripts/actions.sh sync before uv run." >&2
fi
exit 0
