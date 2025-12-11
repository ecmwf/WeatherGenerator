# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import inspect
import logging
from dataclasses import dataclass

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from weathergen.evaluate.scores.score_utils import to_list

# from common.io import MockIO

_logger = logging.getLogger(__name__)

try:
    import xskillscore
    from xhistogram.xarray import histogram
except Exception:
    _logger.warning(
        "Could not import xskillscore and xhistogram. Thus, CRPS and "
        "rank histogram-calculations are not supported."
    )


# helper function to calculate skill score


def _get_skill_score(
    score_fcst: xr.DataArray, score_ref: xr.DataArray, score_perf: float
) -> xr.DataArray:
    """
    Calculate the skill score of a forecast data array w.r.t. a reference and a perfect score.
    Definition follows Wilks, Statistical Methods in the Atmospheric Sciences (2006),
    Chapter 7.1.4, Equation 7.4

    Parameters
    ----------
    score_fcst : xr.DataArray
        Forecast score data array
    score_ref : xr.DataArray
        Score data array of a reference forecast, e.g. a climatological mean
    score_perf : float
        Score data array of a perfect forecast, e.g. 0 for the RMSE-score

    Returns
    ----------
    skill_score : xr.DataArray
        Skill score data array
    """

    skill_score = (score_fcst - score_ref) / (score_perf - score_ref)

    return skill_score


@dataclass(frozen=True)
class VerifiedData:
    """
    # Used to ensure that the prediction and ground truth data are compatible,
    # i.e. dimensions, broadcastability.
    # This is meant to ensure that the data can be used for score calculations.
    """

    prediction: xr.DataArray
    ground_truth: xr.DataArray
    prediction_next: xr.DataArray | None
    ground_truth_next: xr.DataArray | None
    climatology: xr.DataArray | None

    def __post_init__(self):
        # Perform checks on initialization
        self._validate_dimensions()
        self._validate_broadcastability()

    # TODO: add checks for prediction_next, ground_truth_next, climatology
    def _validate_dimensions(self):
        # Ensure all dimensions in truth are in forecast (or equal)
        missing_dims = set(self.ground_truth.dims) - set(self.prediction.dims)
        if missing_dims:
            raise ValueError(
                f"Truth data has extra dimensions not found in forecast: {missing_dims}"
            )

    # TODO: add checks for prediction_next, ground_truth_next, climatology
    def _validate_broadcastability(self):
        try:
            # Attempt broadcast
            xr.broadcast(self.prediction, self.ground_truth)
        except ValueError as e:
            raise ValueError(f"Forecast and truth are not broadcastable: {e}") from e


def get_score(
    data: VerifiedData,
    score_name: str,
    agg_dims: str | list[str] = "all",
    group_by_coord: str | None = None,
    ens_dim: str = "ens",
    compute: bool = False,
    **kwargs,
) -> xr.DataArray:
    """
    Get the score for the given data and score name.
    Note that the scores are aggregated over all dimensions of the prediction data by default.

    Parameters
    ----------
    data : VerifiedData
        VerifiedData object containing prediction and ground truth data.
    score_name : str
        Name of the score to calculate.
    agg_dims : str | List[str]
        List of dimension names over which the score will be aggregated (most often averaged).
        If set to 'all', aggregation will be performed over all dimensions of the forecast data.
    ens_dim : str
        Name of the ensemble dimension in the forecast data. Only used for probabilistic scores.
    compute : bool
        If True, the score will be computed immediately. If False, the score will be returned
        as a lazy xarray DataArray, which allows for efficient graph construction and execution
    kwargs : dict
        Additional keyword arguments to pass to the score function.

    Returns
    -------
    xr.DataArray
        Calculated score as an xarray DataArray.
    """
    sc = Scores(agg_dims=agg_dims, ens_dim=ens_dim)

    score_data = sc.get_score(data, score_name, group_by_coord, **kwargs)
    if compute:
        # If compute is True, compute the score immediately
        return score_data.compute()

    return score_data


# scores class
class Scores:
    """
    Class to calculate scores and skill scores.
    """

    def __init__(
        self,
        agg_dims: str | list[str] = "all",
        ens_dim: str = "ens",
    ):
        """
        Parameters
        ----------
        agg_dims : str | List[str]
            List of dimension names over which the score will be aggregated (most often averaged).
            If set to 'all', aggregation will be performed over all dimensions of the forecast data.
        ens_dim: str
            Name of the ensemble dimension in the forecast data. Only used for probablistic scores.

        Returns
        -------
        """
        self._agg_dims = self._validate_agg_dims(agg_dims)
        self._ens_dim = self._validate_ens_dim(ens_dim)

        self.det_metrics_dict = {
            "ets": self.calc_ets,
            "pss": self.calc_pss,
            "fbi": self.calc_fbi,
            "mae": self.calc_mae,
            "l1": self.calc_l1,
            "l2": self.calc_l2,
            "mse": self.calc_mse,
            "rmse": self.calc_rmse,
            "vrmse": self.calc_vrmse,
            "bias": self.calc_bias,
            "acc": self.calc_acc,
            "froct": self.calc_froct,
            "troct": self.calc_troct,
            "fact": self.calc_fact,
            "tact": self.calc_tact,
            "grad_amplitude": self.calc_spatial_variability,
            "psnr": self.calc_psnr,
            "seeps": self.calc_seeps,
        }
        self.prob_metrics_dict = {
            "ssr": self.calc_ssr,
            "crps": self.calc_crps,
            "rank_histogram": self.calc_rank_histogram,
            "spread": self.calc_spread,
        }

    def get_score(
        self,
        data: VerifiedData,
        score_name: str,
        group_by_coord: str | None = None,
        compute: bool = False,
        **kwargs,
    ):
        """
        Calculate the score for the given data and score name.

        If data is a dask array, the score will be calculated lazily.
        This allows for efficient graph construction and execution when calculating several scores.
        Example usage:
        >>> # Initialize Scores object with aggregation dimensions
        >>> sc = Scores(agg_dims=["ipoints"])
        >>> # Collect list of scores for a given VerifiedData object
        >>> score_list = [sc(data, score_name) for score_name in ["ets", "pss", "fbi"]]
        >>> combined_metrics = xr.concat(score_list, dim="score_name")
        >>> combined_metrics["score_name"] = score_list
        >>> # Do the computation with a joint graph
        >>> combined_metrics = combined_metrics.compute()

        Parameters
        ----------
        data : VerifiedData
            VerifiedData object containing prediction and ground truth data.
        score_name : str
            Name of the score to calculate.
        compute : bool
            If True, the score will be computed immediately. If False, the score will be returned
            as a lazy xarray DataArray, which allows for efficient graph construction and execution.
        kwargs : dict
            Additional keyword arguments to pass to the score function.

        Returns
        -------
        xr.DataArray
            Calculated score as an xarray DataArray.

        """
        if score_name in self.det_metrics_dict.keys():
            f = self.det_metrics_dict[score_name]
        elif score_name in self.prob_metrics_dict.keys():
            assert self.ens_dim in data.prediction.dims, (
                f"Probablistic score {score_name} chosen, but ensemble dimension {self.ens_dim} "
                "not found in prediction data. Skipping score calculation."
            )
            return None
            f = self.prob_metrics_dict[score_name]
        else:
            raise ValueError(
                f"Unknown score chosen. Supported scores: {
                    ', '.join(self.det_metrics_dict.keys())
                    + ', '
                    + ', '.join(self.prob_metrics_dict.keys())
                }"
            )

        if self._agg_dims == "all":
            # Aggregate over all dimensions of the prediction data
            self._agg_dims = list(data.prediction.dims)
        else:
            # Check if _agg_dims is in prediction data
            for dim in self._agg_dims:
                if dim not in data.prediction.dims:
                    raise ValueError(
                        f"Average dimension '{dim}' not found in prediction data "
                        f"dimensions: {data.prediction.dims}"
                    )

        arg_names: list[str] = inspect.getfullargspec(f).args[1:]

        score_args_map = {
            "froct": ["p", "gt", "p_next", "gt_next"],
            "troct": ["p", "gt", "p_next", "gt_next"],
            "acc": ["p", "gt", "c"],
            "fact": ["p", "c"],
            "tact": ["gt", "c"],
        }

        available = {
            "p": data.prediction,
            "gt": data.ground_truth,
            "p_next": data.prediction_next,
            "gt_next": data.ground_truth_next,
            "c": data.climatology,
        }

        # assign p and gt by default if metrics do not have specific args
        keys = score_args_map.get(score_name, ["p", "gt"])
        args = {k: available[k] for k in keys}

        for an in arg_names:
            if an in kwargs:
                args[an] = kwargs[an]

        if group_by_coord is not None and self._validate_groupby_coord(data, group_by_coord):
            # Apply groupby to all DataArrays in args
            grouped_args = {
                k: (v.groupby(group_by_coord) if isinstance(v, xr.DataArray) else v)
                for k, v in args.items()
            }

            # Apply function f to each group and concatenate results
            group_names = list(next(iter(grouped_args.values())).groups.keys())
            results = []
            for name in group_names:
                group_slice = {
                    k: (v[name] if v is not None else v) for k, v in grouped_args.items()
                }
                res = f(**group_slice)
                # Add coordinate for concatenation
                res = res.expand_dims({group_by_coord: [name]})
                results.append(res)
            result = xr.concat(results, dim=group_by_coord)
        else:
            # No grouping: just call the function
            result = f(**args)

        if compute:
            return result.compute()
        else:
            return result

    def _validate_agg_dims(self, dims: str | list[str]) -> list[str] | str:
        if dims == "all":
            return dims
        if isinstance(dims, str):
            return [dims]
        if isinstance(dims, list) and all(isinstance(d, str) for d in dims):
            return dims
        raise ValueError("agg_dims must be 'all', a string, or list of strings.")

    def _validate_ens_dim(self, dim: str) -> str:
        if not isinstance(dim, str):
            raise ValueError("ens_dim must be a string.")
        return dim

    def _validate_groupby_coord(self, data: VerifiedData, group_by_coord: str | None) -> bool:
        """
        Check if the group_by_coord is present in both prediction and ground truth data
        and compatible. Raises ValueError if conditions are not met.
        If group_by_coord does not have more than one unique value in the prediction data,
        a warning is logged and the function returns False, indicating that grouping is
        not applicable.

        Parameters
        ----------
        data : VerifiedData
            VerifiedData object containing prediction and ground truth data.
        group_by_coord : str
            Name of the coordinate to group by.

        Returns
        -------
        group_by_coord : bool
            True if the group_by_coord is valid for grouping, False otherwise.
        """
        p, gt = data.prediction, data.ground_truth
        if group_by_coord not in p.coords or group_by_coord not in gt.coords:
            raise ValueError(
                f"Coordinate '{group_by_coord}' must be present in both prediction "
                "and ground truth data."
            )

        # Check if the dims associated with the groupby_coord are compatible
        dims_p = set(p.coords[group_by_coord].dims)
        dims_gt = set(gt.coords[group_by_coord].dims)
        if dims_p != dims_gt:
            raise ValueError(
                f"Coordinate '{group_by_coord}' is associated with different dimensions: "
                f"{dims_p} in prediction, {dims_gt} in ground truth."
            )

        if len(np.atleast_1d(p.coords[group_by_coord].values)) > 1:
            return True
        else:
            _logger.warning(
                f"Coordinate '{group_by_coord}' has only one unique value in prediction data. "
                "It will not be used for grouping."
            )
            return False

    def _sum(self, data: xr.DataArray) -> xr.DataArray:
        """
        Sum data over aggregation dimensions.

        Parameters
        ----------
        data : xr.DataArray
            xarray DataArray to sum over aggregation dimensions

        Returns
        -------
        xr.DataArray
            Summed data
        """
        return data.sum(dim=self._agg_dims)

    def _mean(self, data: xr.DataArray) -> xr.DataArray:
        """
        Average data over aggregation dimensions.

        Parameters
        ----------
        data : xr.DataArray
            xarray DataArray to average over aggregation dimensions

        Returns
        -------
        xr.DataArray
            Averaged data
        """
        return data.mean(dim=self._agg_dims)

    def get_2x2_event_counts(
        self,
        p: xr.DataArray,
        gt: xr.DataArray,
        thresh: float,
    ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
        """
        Get counts of 2x2 contingency tables

        Parameters
        ----------
        p: xr.DataArray
            Forecast data array
        gt: xr.DataArray
            Ground truth data array
        thresh: float
            Threshold to define event occurrence
        Returns
        -------
        tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]
            Counts of hits (a), false alarms (b), misses (c), and correct negatives (d)
        """

        a = self._sum((p >= thresh) & (gt >= thresh))
        b = self._sum((p >= thresh) & (gt >= thresh))
        c = self._sum((p < thresh) & (gt >= thresh))
        d = self._sum((p < thresh) & (gt < thresh))

        return a, b, c, d

    ### Deterministic scores

    def calc_ets(
        self,
        p: xr.DataArray,
        gt: xr.DataArray,
        thresh: float = 0.1,
    ) -> xr.DataArray:
        """
        Calculate the equitable threat score (ETS) of forecast data w.r.t. reference data.
        Parameters
        ----------
        p: xr.DataArray
            Forecast data array
        gt: xr.DataArray
            Ground truth data array
        thresh: float
            Threshold to define event occurrence
        Returns
        -------
        xr.DataArray
            Equitable threat score (ETS)
        """
        a, b, c, d = self.get_2x2_event_counts(p, gt, thresh)
        n = a + b + c + d
        ar = (a + b) * (a + c) / n  # random reference forecast

        denom = a + b + c - ar

        ets = (a - ar) / denom
        ets = ets.where(denom > 0, np.nan)

        return ets

    def calc_fbi(
        self,
        p: xr.DataArray,
        gt: xr.DataArray,
        thresh: float = 0.1,
    ) -> xr.DataArray:
        """
        Calculate the frequency bias index (FBI) of forecast data w.r.t. reference data.
        Parameters
        ----------
        p: xr.DataArray
            Forecast data array
        gt: xr.DataArray
            Ground truth data array
        thresh: float
            Threshold to define event occurrence
        Returns
        -------
        xr.DataArray
            Frequency bias index (FBI)
        """

        a, b, c, _ = self.get_2x2_event_counts(p, gt, thresh)

        denom = a + c
        fbi = (a + b) / denom

        fbi = fbi.where(denom > 0, np.nan)

        return fbi

    def calc_pss(
        self,
        p: xr.DataArray,
        gt: xr.DataArray,
        thresh: float = 0.1,
    ) -> xr.DataArray:
        """
        Calculate the Peirce skill score (PSS) of forecast data w.r.t. reference data.
        Parameters
        ----------
        p: xr.DataArray
            Forecast data array
        gt: xr.DataArray
            Ground truth data array
        thresh: float
            Threshold to define event occurrence
        Returns
        -------
        xr.DataArray
            Pierce skill score (PSS)
        """

        a, b, c, d = self.get_2x2_event_counts(p, gt, thresh)

        denom = (a + c) * (b + d)
        pss = (a * d - b * c) / denom

        pss = pss.where(denom > 0, np.nan)

        return pss

    def calc_l1(
        self,
        p: xr.DataArray,
        gt: xr.DataArray,
        scale_dims: list | None = None,
    ) -> xr.DataArray:
        """
        Calculate the L1 error norm of forecast data w.r.t. reference data.
        Note that the L1 error norm is calculated as the sum of absolute differences.

        Parameters
        ----------
        p: xr.DataArray
            Forecast data array
        gt: xr.DataArray
            Ground truth data array
        scale_dims: list | None
            List of dimensions over which the L1 score will be scaled.
            If provided, the L1 score will be divided by the product of the sizes of these
            dimensions.

        Returns
        -------
        xr.DataArray
            L1 error norm
        """
        l1 = np.abs(p - gt)

        l1 = self._sum(l1)

        if scale_dims:
            scale_dims = to_list(scale_dims)

            assert all([dim in p.dims for dim in scale_dims]), (
                f"Provided scale dimensions {scale_dims} are not all present in the prediction "
                f"data dimensions {p.dims}."
            )

            len_dims = np.array([p.sizes[dim] for dim in scale_dims])
            l1 /= np.prod(len_dims)

        return l1

    def calc_l2(
        self,
        p: xr.DataArray,
        gt: xr.DataArray,
        scale_dims: list | None = None,
        squared_l2: bool = False,
    ) -> xr.DataArray:
        """
        Calculate the L2 error norm of forecast data w.r.t. reference data.

        Parameters
        ----------
        p: xr.DataArray
            Forecast data array
        gt: xr.DataArray
            Ground truth data array
        scale_dims: list | None
            List of dimensions over which the L2 score will be scaled.
            If provided, the L2 score will be divided by the product of the sizes of these
            dimensions.
        squared_l2: bool
            If True, the L2 score will be returned as the sum of squared differences.
            If False, the L2 score will be returned as the square root of the sum of squared
            differences. Default is False, i.e. the L2 score is returned as the square root of the
            sum of squared differences.

        Returns
        -------
        xr.DataArray
            L2 error norm
        """
        l2 = np.square(p - gt)

        l2 = self._sum(l2)

        if not squared_l2:
            l2 = np.sqrt(l2)

        if scale_dims:
            scale_dims = to_list(scale_dims)

            assert all([dim in p.dims for dim in scale_dims]), (
                f"Provided scale dimensions {scale_dims} are not all present in the prediction "
                f"data dimensions {p.dims}."
            )

            len_dims = np.array([p.sizes[dim] for dim in scale_dims])
            l2 /= np.prod(len_dims)

        return l2

    def calc_mae(self, p: xr.DataArray, gt: xr.DataArray) -> xr.DataArray:
        """
        Calculate mean absolute error (MAE) of forecast data w.r.t. reference data.

        Parameters
        ----------
        p: xr.DataArray
            Forecast data array
        gt: xr.DataArray
            Ground truth data array
        """
        if self._agg_dims is None:
            raise ValueError(
                "Cannot calculate mean absolute error without aggregation dimensions "
                "(agg_dims=None)."
            )

        return self._mean(np.abs(p - gt))

    def calc_mse(self, p: xr.DataArray, gt: xr.DataArray) -> xr.DataArray:
        """
        Calculate mean squared error (MSE) of forecast data w.r.t. reference data.

        Parameters
        ----------
        p: xr.DataArray
            Forecast data array
        gt: xr.DataArray
            Ground truth data array
        Returns
        -------
        xr.DataArray
            Mean squared error (MSE)
        """
        if self._agg_dims is None:
            raise ValueError(
                "Cannot calculate mean squared error without aggregation dimensions "
                "(agg_dims=None)."
            )

        return self._mean(np.square(p - gt))

    def calc_rmse(self, p: xr.DataArray, gt: xr.DataArray) -> xr.DataArray:
        """
        Calculate root mean squared error (RMSE) of forecast data w.r.t. reference data
        Parameters
        ----------
        p: xr.DataArray
            Forecast data array
        gt: xr.DataArray
            Ground truth data array
        Returns
        -------
        xr.DataArray
            Root mean squared error (RMSE)

        """
        if self._agg_dims is None:
            raise ValueError(
                "Cannot calculate root mean squared error without aggregation dimensions "
                "(agg_dims=None)."
            )

        rmse = np.sqrt(self.calc_mse(p, gt))

        return rmse

    def calc_vrmse(self, p: xr.DataArray, gt: xr.DataArray):
        """
        Calculate variance-normalized root mean squared error (VRMSE) of forecast data w.r.t.
        reference data

        Parameters
        ----------
        p: xr.DataArray
            Forecast data array
        gt: xr.DataArray
            Ground truth data array
        """
        if self._agg_dims is None:
            raise ValueError(
                "Cannot calculate variance-normalized root mean squared error without aggregation "
                "dimensions (agg_dims=None)."
            )

        vrmse = np.sqrt(self.calc_mse(p, gt) / (gt.var(dim=self._agg_dims) + 1e-6))

        return vrmse

    @staticmethod
    def sort_by_coords(da_to_sort: xr.DataArray, da_reference: xr.DataArray) -> xr.DataArray:
        """
        Sorts one xarray.DataArray's coordinate ordering to match a reference array using KDTree.

        This method finds the nearest neighbor in `da_to_sort` for every coordinate in
        `da_reference`, effectively reordering `da_to_sort` along its indexed dimension to align
        with the sequence of coordinates in the reference.

        Parameters
        ----------
        da_to_sort : xr.DataArray
            The DataArray whose coordinate ordering needs to be matched.
            Must contain 'lat' and 'lon' coordinates and an indexed dimension (e.g., 'ipoint').
        da_reference : xr.DataArray
            The DataArray providing the target coordinate ordering (the template). Must contain
            'lat' and 'lon' coordinates.

        Returns
        -------
        xr.DataArray
            A new DataArray with the data from `da_to_sort` reordered to match the
            coordinate sequence of `da_reference`.

        Raises
        ------
        ValueError
            If any reference coordinate does not have a matching coordinate in
            `da_to_sort` within the allowed distance tolerance (1e-5).

        Notes
        -----
        The matching uses `scipy.spatial.cKDTree.query` with a strict distance threshold
        (`distance_upper_bound=1e-5`) to ensure precise one-to-one alignment.
        """

        # Extract coordinates
        ref_lats = da_reference.lat.values
        ref_lons = da_reference.lon.values
        sort_lats = da_to_sort.lat.values
        sort_lons = da_to_sort.lon.values

        # Build KDTree on coordinates to sort
        sort_coords = np.column_stack((sort_lats, sort_lons))
        tree = cKDTree(sort_coords)

        # Find nearest neighbors for reference coordinates
        ref_coords = np.column_stack((ref_lats, ref_lons))
        dist, indices = tree.query(ref_coords, distance_upper_bound=1e-5)

        # Check for unmatched coordinates
        unmatched_mask = ~np.isfinite(dist)
        if np.any(unmatched_mask):
            n_unmatched = np.sum(unmatched_mask)
            _logger.info(
                f"Found {n_unmatched} reference coordinates with no matching coordinates in array"
                "to sort. Returning NaN DataArray."
            )
            return xr.full_like(da_reference, np.nan)

        # Reorder da_to_sort to match reference ordering
        return da_to_sort.isel(ipoint=indices)

    def calc_change_rate(
        self,
        s0: xr.DataArray,
        s1: xr.DataArray,
    ) -> xr.DataArray:
        """
        Calculate the "change rate" of a data array as the mean absolute difference between two
        consecutive time steps.

        Parameters
        ----------
        s0: xr.DataArray
            Data array at time step t0
        s1: xr.DataArray
            Data array at time step t1

        Returns
        -------
        xr.DataArray
            Change rate of the data array
        """

        if s1 is None:
            return xr.full_like(s0, np.nan)
        else:
            # Sort the coordinates of subsequent time steps to match each other. Can be removed
            # once unshuffling is solved elsewhere
            s1 = self.sort_by_coords(da_to_sort=s1, da_reference=s0)
            crate = np.abs(s0 - s1.values)
            return crate

    def calc_froct(
        self,
        p: xr.DataArray,
        gt: xr.DataArray,
        p_next: xr.DataArray,
        gt_next: xr.DataArray,
    ) -> xr.DataArray:
        """
        Calculate forecast rate of change over time

        Parameters
        ----------
        p: xr.DataArray
            Forecast data array
        gt: xr.DataArray
            Ground truth data array (not used in calculation, but kept for consistency)
        p_next: xr.DataArray
            Next forecast step data array
        gt_next: xr.DataArray
            Next ground truth step data array (not used in calculation, but kept for consistency)
        Returns
        -------
        xr.DataArray
            Forecast rate of change over time
        """
        if self._agg_dims is None:
            raise ValueError(
                "Cannot calculate rate of change without aggregation dimensions (agg_dims=None)."
            )

        froct = self.calc_change_rate(p, p_next)

        froct = self._mean(froct)

        return froct

    def calc_troct(
        self,
        p: xr.DataArray,
        gt: xr.DataArray,
        gt_next: xr.DataArray,
        p_next: xr.DataArray,
    ):
        """
        Calculate target rate of change over time

        Parameters
        ----------
        p: xr.DataArray
            Forecast data array (not used in calculation, but kept for consistency)
        gt: xr.DataArray
            Ground truth data array
        p_next: xr.DataArray
            Next forecast step data array (not used in calculation, but kept for consistency)
        gt_next: xr.DataArray
            Next ground truth step data array
        Returns
        -------
        xr.DataArray
            Target rate of change over time
        """
        if self._agg_dims is None:
            raise ValueError(
                "Cannot calculate rate of change without aggregation dimensions (agg_dims=None)."
            )

        troct = self.calc_change_rate(gt, gt_next)
        troct = self._mean(troct)

        return troct

    def _calc_act(
        self,
        x: xr.DataArray,
        c: xr.DataArray,
    ):
        """
        Calculate activity metric as standard deviation of forecast or target anomaly.

        NOTE:
        The climatlogical mean data clim_mean must fit to the forecast and ground truth data.

        Parameters
        ----------
        x: xr.DataArray
            Forecast or target data array
        c: xr.DataArray
            Climatological mean data array, which is used to calculate anomalies
        """

        if c is None:
            return xr.full_like(x.sum(self._agg_dims), np.nan)

        # Calculate anomalies
        ano = x - c
        act = ano.std(dim=self._agg_dims)

        return act

    def calc_fact(
        self,
        p: xr.DataArray,
        c: xr.DataArray,
    ):
        """
        Calculate forecast activity metric as standard deviation of forecast anomaly.

        NOTE:
        The climatlogical mean data clim_mean must fit to the forecast data.

        Parameters
        ----------
        p: xr.DataArray
            Forecast data array
        c: xr.DataArray
            Climatological mean data array, which is used to calculate anomalies
        """

        return self._calc_act(p, c)

    def calc_tact(
        self,
        gt: xr.DataArray,
        c: xr.DataArray,
    ):
        """
        Calculate target activity metric as standard deviation of target anomaly.

        NOTE:
        The climatlogical mean data clim_mean must fit to the target data.

        Parameters
        ----------
        gt: xr.DataArray
            Target data array
        c: xr.DataArray
            Climatological mean data array, which is used to calculate anomalies
        """

        return self._calc_act(gt, c)

    def calc_acc(
        self,
        p: xr.DataArray,
        gt: xr.DataArray,
        c: xr.DataArray,
    ) -> xr.DataArray:
        """
        Calculate anomaly correlation coefficient (ACC).

        NOTE:
        The climatlogical mean data clim_mean must fit to the forecast and ground truth data.
        By definition, the ACC is always aggregated over the spatial dimensions.

        Parameters
        ----------
        p: xr.DataArray
            Forecast data array
        gt: xr.DataArray
            Ground truth data array
        c: xr.DataArray
            Climatological mean data array, which is used to calculate anomalies

        Returns
        -------
        xr.DataArray
            Anomaly correlation coefficient (ACC)
        """

        if c is None:
            return xr.full_like(p.sum(self._agg_dims), np.nan)

        # Calculate anomalies
        fcst_ano, obs_ano = p - c, gt - c

        # Calculate ACC over spatial dimensions (no grouping)
        acc = (fcst_ano * obs_ano).sum(self._agg_dims) / np.sqrt(
            (fcst_ano**2).sum(self._agg_dims) * (obs_ano**2).sum(self._agg_dims)
        )

        return acc

    def calc_bias(self, p: xr.DataArray, gt: xr.DataArray) -> xr.DataArray:
        """
        Calculate mean bias of forecast data w.r.t. reference data

        Parameters
        ----------
        p: xr.DataArray
            Forecast data array
        gt: xr.DataArray
            Ground truth data array
        Returns
        -------
        xr.DataArray
            Mean bias
        """
        bias = self._mean(p - gt)

        return bias

    def calc_psnr(
        self,
        p: xr.DataArray,
        gt: xr.DataArray,
        pixel_max: float = 1.0,
    ) -> xr.DataArray:
        """
        Calculate PSNR of forecast data w.r.t. reference data

        Parameters
        ----------
        p: xr.DataArray
            Forecast data array
        gt: xr.DataArray
            Ground truth data array
        pixel_max: float
            Maximum pixel value in the data. Default is 1.0.
        Returns
        -------
        xr.DataArray
            Peak signal-to-noise ratio (PSNR)
        """

        mse = self.calc_mse(p, gt)
        if np.count_nonzero(mse) == 0:
            psnr = mse
            psnr[...] = 100.0
        else:
            psnr = 20.0 * np.log10(pixel_max / np.sqrt(mse))

        return psnr

    def calc_spatial_variability(
        self,
        p: xr.DataArray,
        gt: xr.DataArray,
        order: int = 1,
        non_spatial_avg_dims: list[str] = None,
    ) -> xr.DataArray:
        """
        Calculates the ratio between the spatial variability of differental operator
        with order 1 (higher values unsupported yet) forecast and ground truth data using
        the calc_geo_spatial-method.

        NOTE:
        Requires that data is provided on a regular lat/lon-grid!

        Parameters
        ----------
        p: xr.DataArray
            Forecast data array
        gt: xr.DataArray
            Ground truth data array
        order: int
            Order of the spatial differential operator to be applied. Supported orders: 1
        non_spatial_avg_dims: List[str]
            List of dimensions over which the spatial variability ratio should be averaged.
            It must be non-spatial dimensions, i.e. not latitude or longitude.
        Returns
        -------
        xr.DataArray
            Ratio of spatial variability between forecast and ground truth data
        """

        fcst_grad = self.calc_geo_spatial_diff(p, order=order)
        ref_grd = self.calc_geo_spatial_diff(gt, order=order)

        ratio_spat_variability = fcst_grad / ref_grd

        if non_spatial_avg_dims is not None:
            ratio_spat_variability = ratio_spat_variability.mean(dim=non_spatial_avg_dims)

        return ratio_spat_variability

    def calc_seeps(
        self,
        p: xr.DataArray,
        gt: xr.DataArray,
        seeps_weights: xr.DataArray,
        t1: xr.DataArray,
        t3: xr.DataArray,
        spatial_dims: list,
    ) -> xr.DataArray:
        """
        Calculates stable equitable error in probabiliyt space (SEEPS), see Rodwell et al., 2011

        NOTE:
        Threshold arrays t1 and t3 (derived from space-time dependant climatology)
        must fit to the forecast and ground truth data.

        Parameters
        ----------
        p: xr.DataArray
            Forecast data array
        gt: xr.DataArray
            Ground truth data array
        seeps_weights: xr.DataArray
            SEEPS-parameter matrix to weight contingency table elements
        t1: xr.DataArray
            Threshold for light precipitation events
        t3: xr.DataArray
            Threshold for strong precipitation events
        spatial_dims: List[str]
            List of spatial dimensions of the data, e.g. ["lat", "lon"]
        Returns
        -------
        xr.DataArray
            SEEPS skill score (i.e. 1-SEEPS)
        """

        def seeps(ground_truth, prediction, thr_light, thr_heavy, seeps_weights):
            ob_ind = (ground_truth > thr_light).astype(int) + (ground_truth >= thr_heavy).astype(
                int
            )
            fc_ind = (prediction > thr_light).astype(int) + (prediction >= thr_heavy).astype(int)
            indices = fc_ind * 3 + ob_ind  # index of each data point in their local 3x3 matrices
            seeps_val = seeps_weights[
                indices, np.arange(len(indices))
            ]  # pick the right weight for each data point

            return 1.0 - seeps_val

        if p.ndim == 3:
            assert len(spatial_dims) == 2, (
                "Provide two spatial dimensions for three-dimensional data."
            )
            prediction, ground_truth = (
                p.stack({"xy": spatial_dims}),
                gt.stack({"xy": spatial_dims}),
            )
            seeps_weights = seeps_weights.stack({"xy": spatial_dims})
            t3 = t3.stack({"xy": spatial_dims})
            lstack = True
        elif self.prediction.ndim == 2:
            prediction, ground_truth = p, gt
            lstack = False
        else:
            raise ValueError("Data must be a two-or-three-dimensional array.")

        # check dimensioning of data
        assert prediction.ndim <= 2, (
            f"Data must be one- or two-dimensional, but has {prediction.ndim} dimensions. "
            "Check if stacking with spatial_dims may help."
        )

        if prediction.ndim == 1:
            seeps_values_all = seeps(ground_truth, prediction, t1.values, t3, seeps_weights)
        else:
            prediction, ground_truth = (
                prediction.transpose(..., "xy"),
                ground_truth.transpose(..., "xy"),
            )
            seeps_values_all = xr.full_like(prediction, np.nan)
            seeps_values_all.name = "seeps"
            for it in range(ground_truth.shape[0]):
                prediction_now, ground_truth_now = (
                    prediction[it, ...],
                    ground_truth[it, ...],
                )
                # in case of missing data, skip computation
                if np.all(np.isnan(prediction_now)) or np.all(np.isnan(ground_truth_now)):
                    continue

                seeps_values_all[it, ...] = seeps(
                    ground_truth_now,
                    prediction_now,
                    t1.values,
                    t3,
                    seeps_weights.values,
                )

        if lstack:
            seeps_values_all = seeps_values_all.unstack()

        if self._agg_dims is not None:
            seeps_values = self._mean(seeps_values_all)
        else:
            seeps_values = seeps_values_all

        return seeps_values

    ### Probablistic scores

    def calc_spread(self, p: xr.DataArray, **kwargs) -> xr.DataArray:
        """
        Calculate the spread of the forecast ensemble
        Parameters
        ----------
        p: xr.DataArray
            Forecast data array with ensemble dimension

        Returns
        -------
        xr.DataArray
            Spread of the forecast ensemble
        """
        ens_std = p.std(dim=self._ens_dim)

        return self._mean(np.sqrt(ens_std**2))

    def calc_ssr(self, p: xr.DataArray, gt: xr.DataArray) -> xr.DataArray:
        """
        Calculate the Spread-Skill Ratio (SSR) of the forecast ensemble data w.r.t. reference data

        Parameters
        ----------
        p: xr.DataArray
            Forecast data array with ensemble dimension
        gt: xr.DataArray
            Ground truth data array
        Returns
        -------
        xr.DataArray
            Spread-Skill Ratio (SSR)
        """
        ssr = self.calc_spread(p) / self.calc_rmse(p, gt)  # spread/rmse

        return ssr

    def calc_crps(
        self,
        p: xr.DataArray,
        gt: xr.DataArray,
        method: str = "ensemble",
        **kwargs,
    ) -> xr.DataArray:
        """
        Wrapper around CRPS-methods provided by xskillscore-package.
        See https://xskillscore.readthedocs.io/en/stable/api

        Parameters
        ----------
        p: xr.DataArray
            Forecast data array with ensemble dimension
        gt: xr.DataArray
            Ground truth data array
        method: str
            Method to calculate CRPS. Supported methods: ["ensemble", "gaussian"]
        kwargs: dict
            Other keyword parameters supported by respective CRPS-method from
            the xskillscore package

        Returns
        -------
        xr.DataArray
            CRPS score data array averaged over the provided dimensions
        """
        crps_methods = ["ensemble", "gaussian"]

        if method == "ensemble":
            func_kwargs = {
                "forecasts": p,
                "member_dim": self._ens_dim,
                "dim": self._agg_dims,
                **kwargs,
            }
            crps_func = xskillscore.crps_ensemble
        elif method == "gaussian":
            func_kwargs = {
                "mu": p.mean(dim=self._ens_dim),
                "sig": p.std(dim=self._ens_dim),
                "dim": self._agg_dims,
                **kwargs,
            }
            crps_func = xskillscore.crps_gaussian
        else:
            raise ValueError(
                f"Unsupported CRPS-calculation method {method} chosen."
                + f"Supported methods: {', '.join(crps_methods)}"
            )

        crps = crps_func(gt, **func_kwargs)

        return crps

    def calc_rank_histogram(
        self,
        p: xr.DataArray,
        gt: xr.DataArray,
        norm: bool = True,
        add_noise: bool = True,
        noise_fac=1.0e-03,
    ) -> xr.DataArray:
        """
        Calculate the rank histogram of the forecast data w.r.t. reference data.

        Parameters
        ----------
        p: xr.DataArray
            Forecast data array with ensemble dimension
        gt: xr.DataArray
            Ground truth data array
        norm: bool
            Flag if normalized counts should be returned. If True, the rank histogram will be
            normalized by the number of ensemble members in the forecast data.
        add_noise: bool
            Flag if a small amount of random noise should be added to the data to avoid ties in the
            rank histogram.
            This is recommended for fair computations, cf. Sec. 4.2.2 in Harris et al. 2022
        noise_fac: float
            Magnitude of random noise to be added to the data if add_noise is True.
            Default is 1.0e-03. This value is only relevant if add_noise is True

        Returns
        -------
        xr.DataArray
            Rank histogram data array averaged over the provided dimensions
        """

        # unstack stacked time-dimension beforehand if required (time may be stacked for forecast
        # data)
        ground_truth = gt
        if "time" in ground_truth.indexes:
            if isinstance(ground_truth.indexes["time"], pd.MultiIndex):
                ground_truth = ground_truth.reset_index("time")

        prediction = p
        if "time" in prediction.indexes:
            if isinstance(prediction.indexes["time"], pd.MultiIndex):
                prediction = prediction.reset_index("time")

        # perform the stacking
        obs_stacked = ground_truth.stack({"npoints": self._agg_dims})
        fcst_stacked = prediction.stack({"npoints": self._agg_dims})

        # add noise to data if desired
        if add_noise:
            if obs_stacked.chunks is None and fcst_stacked.chunks is None:
                # underlying arrays are numpy arrays -> use numpy's native random generator
                rng = np.random.default_rng()

                obs_stacked += rng.random(size=obs_stacked.shape, dtype=np.float32) * noise_fac
                fcst_stacked += rng.random(size=fcst_stacked.shape, dtype=np.float32) * noise_fac
            else:
                # underlying arrays are dask arrays -> use dask's random generator
                obs_stacked += (
                    da.random.random(size=obs_stacked.shape, chunks=obs_stacked.chunks) * noise_fac
                )
                fcst_stacked += (
                    da.random.random(size=fcst_stacked.shape, chunks=fcst_stacked.chunks)
                    * noise_fac
                )
        # preserve the other coordinates
        preserved_coords = {
            c: obs_stacked[c].values
            for c in obs_stacked.coords
            if all(dim not in {self._ens_dim, "npoints"} for dim in obs_stacked[c].dims)
        }

        # calculate ranks for all data points
        rank = (obs_stacked >= fcst_stacked).sum(dim=self._ens_dim)
        # and count occurence of rank values
        rank.name = "rank"  # name for xr.DataArray is required for histogram-method
        rank_counts = histogram(
            rank,
            dim=["npoints"],
            bins=np.arange(len(fcst_stacked[self._ens_dim]) + 2),
            block_size=None if rank.chunks is None else "auto",
        )

        # Reattach preserved coordinates by broadcasting
        for coord_name, coord_values in preserved_coords.items():
            # Only keep unique values along npoints if necessary
            if coord_name in rank_counts.coords:
                continue
            rank_counts = rank_counts.assign_coords({coord_name: coord_values})

        # Reattach preserved coordinates by broadcasting
        for coord_name, coord_values in preserved_coords.items():
            # Only keep unique values along npoints if necessary
            if coord_name in rank_counts.coords:
                continue
            rank_counts = rank_counts.assign_coords({coord_name: coord_values})

        # provide normalized rank counts if desired
        if norm:
            npoints = len(fcst_stacked["npoints"])
            rank_counts = rank_counts / npoints

        return rank_counts

    def calc_rank_histogram_xskillscore(self, p: xr.DataArray, gt: xr.DataArray) -> xr.DataArray:
        """
        Wrapper around rank_histogram-method by xskillscore-package.
        See https://xskillscore.readthedocs.io/en/stable/api
        Note: this version is found to be very slow. Use calc_rank_histogram alternatively.
        Parameters
        ----------
        p: xr.DataArray
            Forecast data array with ensemble dimension
        gt: xr.DataArray
            Ground truth data array
        Returns
        -------
        xr.DataArray
            Rank histogram data array averaged over the provided dimensions
        """
        rank_hist = xskillscore.rank_histogram(gt, p, member_dim=self._ens_dim, dim=self._agg_dims)

        return rank_hist

    @staticmethod
    def calc_geo_spatial_diff(
        scalar_field: xr.DataArray,
        order: int = 1,
        r_e: float = 6371.0e3,
        dom_avg: bool = True,
    ) -> xr.DataArray:
        """
        Calculates the amplitude of the gradient (order=1) or the Laplacian (order=2)
        of a scalar field given on a regular, geographical grid
        (i.e. dlambda = const. and dphi=const.)

        Parameters
        ----------
        scalar_field: xr.DataArray
            Scalar field as data array with latitude and longitude as coordinates
        order: int
            Order of spatial differential operator
        r_e: float
            Radius of the sphere
        dom_avg: bool
            Flag whether to return the domain-averaged amplitude or the amplitude at each
            grid point

        Returns
        -------
        xr.DataArray
            the amplitude of the gradient/laplacian at each grid point or over the whole domain
            (see dom_avg)
        """
        method = Scores.calc_geo_spatial_diff.__name__
        # sanity checks
        assert isinstance(scalar_field, xr.DataArray), (
            f"Scalar_field of {method} must be a xarray DataArray."
        )
        assert order in [1, 2], f"Order for {method} must be either 1 or 2."

        dims = list(scalar_field.dims)
        lat_dims = ["rlat", "lat", "latitude"]
        lon_dims = ["rlon", "lon", "longitude"]

        def check_for_coords(coord_names_data, coord_names_expected):
            try:
                _ = coord_names_expected.index()
            except ValueError as e:
                expected_names = ",".join(coord_names_expected)
                raise ValueError(
                    "Could not find one of the following coordinates in the"
                    + f"passed dictionary: {expected_names}"
                ) from e

        _, lat_name = check_for_coords(dims, lat_dims)
        _, lon_name = check_for_coords(dims, lon_dims)

        lat, lon = (
            np.deg2rad(scalar_field[lat_name]),
            np.deg2rad(scalar_field[lon_name]),
        )
        dphi, dlambda = lat[1].values - lat[0].values, lon[1].values - lon[0].values

        if order == 1:
            dvar_dlambda = (
                1.0 / (r_e * np.cos(lat) * dlambda) * scalar_field.differentiate(lon_name)
            )
            dvar_dphi = 1.0 / (r_e * dphi) * scalar_field.differentiate(lat_name)
            dvar_dlambda = dvar_dlambda.transpose(
                *scalar_field.dims
            )  # ensure that dimension ordering is not changed

            var_diff_amplitude = np.sqrt(dvar_dlambda**2 + dvar_dphi**2)
            if dom_avg:
                var_diff_amplitude = var_diff_amplitude.mean(dim=[lat_name, lon_name])
        else:
            raise ValueError(f"Second-order differentation is not implemenetd in {method} yet.")

        return var_diff_amplitude
