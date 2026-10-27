import logging

import numpy as np
import xarray as xr

_logger = logging.getLogger(__name__)


def _compute_ranges(spatial: dict[str, dict[str, xr.DataArray]]) -> dict:
    """Compute colour ranges for each metric/region/channel from the spatial scores.

    Parameters
    ----------
    spatial
        ``spatial[region][metric] -> DataArray(forecast_step, ipoint, channel[, ens])``
        (``MetricResult.spatial``).

    Returns
    -------
    score_ranges_dict : dict
        ``{metric: {region: {channel: {'vmin': float, 'vmax': float}}}}``, the range of
        each channel over all forecast steps (and ensemble members).
    """
    score_ranges_dict: dict = {}
    for region, by_metric in spatial.items():
        for metric, result in by_metric.items():
            if "channel" not in result.coords:
                continue
            score_ranges_dict.setdefault(metric, {}).setdefault(region, {})
            for ch in result.coords["channel"].values:
                vals = result.sel(channel=ch).values.flatten()
                vals = vals[~np.isnan(vals)]
                if vals.size == 0:
                    continue
                score_ranges_dict[metric][region][str(ch)] = {
                    "vmin": float(vals.min()),
                    "vmax": float(vals.max()),
                }

    return score_ranges_dict
