import datetime
import os
import sys

# Make the local package importable for tests without installing the package
tests_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(tests_dir, os.pardir, "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from weathergen.evaluate.plotting.image_sort import _extract_valid_time_and_fstep_from_filename, _image_sort_key


def test_extract_valid_time_and_fstep_from_filename():
    fname = "map_run_tag_0_2023-01-01T0100_ERA5_region_var_fstep_006.png"
    dt, fstep = _extract_valid_time_and_fstep_from_filename(fname)
    assert dt == datetime.datetime(2023, 1, 1, 1, 0)
    assert fstep == 6


def test_image_sort_key_orders_by_valid_time_then_fstep():
    base = "map_run_tag_0"
    a = f"{base}_2023-01-01T0100_ERA5_region_var_fstep_006.png"
    b = f"{base}_2023-01-01T0000_ERA5_region_var_fstep_000.png"
    c = f"{base}_2023-01-01T0200_ERA5_region_var_fstep_012.png"

    shuffled = [a, c, b]
    sorted_paths = sorted(shuffled, key=_image_sort_key)
    assert sorted_paths == [b, a, c]


def test_image_sort_key_fallbacks_to_fstep_when_no_valid_time():
    base = "map_run_tag_0"
    a = f"{base}_ERA5_region_var_fstep_012.png"
    b = f"{base}_ERA5_region_var_fstep_000.png"
    c = f"{base}_ERA5_region_var_fstep_006.png"

    shuffled = [a, c, b]
    sorted_paths = sorted(shuffled, key=_image_sort_key)
    assert sorted_paths == [b, c, a]
