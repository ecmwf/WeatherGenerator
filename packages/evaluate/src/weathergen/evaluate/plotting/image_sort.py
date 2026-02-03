import datetime
import os
import re


def _extract_valid_time_and_fstep_from_filename(fname: str):
    """Extract valid_time and fstep from a plot filename.

    The filename format includes an ISO-like valid time fragment like
    YYYY-MM-DDTHHMM and a forecast step fragment like fstep_XXX.

    Returns a tuple (valid_time: datetime | None, fstep: int | None).
    """
    basename = os.path.basename(fname)

    dt_match = re.search(r"\d{4}-\d{2}-\d{2}T\d{4}", basename)
    if dt_match:
        try:
            dt = datetime.datetime.strptime(dt_match.group(), "%Y-%m-%dT%H%M")
        except Exception:
            dt = None
    else:
        dt = None

    fstep_match = re.search(r"fstep_(\d{3})", basename)
    fstep = int(fstep_match.group(1)) if fstep_match else None

    return dt, fstep


def _image_sort_key(path: str):
    """Sort key for image file paths: by valid_time (if present), then fstep, then filename."""
    dt, fstep = _extract_valid_time_and_fstep_from_filename(path)
    dt_key = dt if dt is not None else datetime.datetime.min
    fstep_key = fstep if fstep is not None else -1
    return (dt_key, fstep_key, os.path.basename(path))
