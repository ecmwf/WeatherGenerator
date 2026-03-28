# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
from pathlib import Path

_REPO_ROOT = Path(
    __file__
).parent.parent.parent.parent.parent.parent  # TODO use importlib for resources


def get_wg_private_path() -> Path:
    """Returns the root path of the WeatherGenerator private repository."""
    if "WEATHERGEN_PRIVATE_REPO_PATH" in os.environ:
        path = Path(os.environ["WEATHERGEN_PRIVATE_REPO_PATH"])
    else:
        path = _REPO_ROOT.parent / "WeatherGenerator-private"

    path = path.resolve()
    assert path.is_dir(), (
        f"WeatherGenerator private repo path does not exist or is not a directory: {path}"
    )
    return path
