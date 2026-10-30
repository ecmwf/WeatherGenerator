# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Forward/backward numerics regression test, see numerics.py for how references are generated.
This test must run on a GPU machine with the ERA5 data available.

Command:
WEATHERGEN_NUMERICS_REF_DIR=<dir> uv run pytest integration_tests/numerics/numerics_test.py -v -s
"""

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

import numerics  # noqa: E402

REF_DIR = os.environ.get("WEATHERGEN_NUMERICS_REF_DIR")
REPORT_DIR = os.environ.get("WEATHERGEN_NUMERICS_REPORT_DIR")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.skipif(REF_DIR is None, reason="WEATHERGEN_NUMERICS_REF_DIR is not set")
@pytest.mark.parametrize("technique", numerics.TECHNIQUES)
def test_forward_backward_numerics(technique):
    ref_dir = Path(REF_DIR)
    # a wrong path or an incomplete reference must not let the test pass silently
    if not (ref_dir / technique / "results.pt").exists():
        pytest.fail(f"no reference for {technique} in {ref_dir}")
    report = Path(REPORT_DIR) / f"{technique}.json" if REPORT_DIR else None
    mismatches = numerics.compare(technique, ref_dir, report)
    assert not mismatches, numerics.format_mismatches(technique, mismatches)
