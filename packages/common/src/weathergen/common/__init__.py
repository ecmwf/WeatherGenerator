# TODO: move here:
# - better_abc
# - run_id
# - config
# - distributed
# - logger

from pathlib import Path

_REPO_ROOT = Path(
    __file__
).parent.parent.parent.parent.parent.parent

def common_function():
    return "This is a common function for weather generation."
