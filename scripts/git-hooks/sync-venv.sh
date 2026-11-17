#!/bin/bash
# Re-syncs the venv after a pull, rebase or branch switch that changed the
# dependencies. Called by the git hooks that scripts/actions.sh sync installs.

hook="$1"
shift
case "$hook" in
  post-merge) old=ORIG_HEAD ;;
  post-rewrite) [[ "$1" == rebase ]] || exit 0; old=ORIG_HEAD ;;
  post-checkout)
    # Arguments: previous HEAD, new HEAD, 1 for a branch checkout.
    [[ "$3" == 1 && "$1" != "$2" ]] || exit 0
    # Skip during a rebase: post-rewrite syncs once at the end.
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
