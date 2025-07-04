# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import dataclass
from typing import List, Any
import json
import dask.array as da
import numpy as np
import pandas as pd
import inspect
import xarray as xr
import numpy as np

try:
    import xskillscore
    from xhistogram.xarray import histogram
except Exception:
    _logger.warning(
        "Could not import xskillscore and xhistogram. Thus, CRPS and"
        + "rank histogram-calculations are not supported."
    )
    print()


# helper function to calculate skill score
def _get_skill_score(score_fcst: xr.DataArray, score_ref: xr.DataArray, score_perf: float) -> xr.DataArray:  
    """
    Calculate the skill score of a forecast data array w.r.t. a reference and a perfect score.
    Definition follows Wilks, Statistical Methods in the Atmospheric Sciences (2006), Chapter 7.1.4, Equation 7.4

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
    skiil_score : xr.DataArray
        Skill score data array
    """
    
    skill_score = (score_fcst - score_ref) / (score_perf - score_ref)

    return skill_score


def to_json(func):
    def wrapper(*args, **kwargs):
        da = func(*args, **kwargs)
        return json.dumps(da.to_dict(), default=str)

    return wrapper


def to_list(obj: Any) -> List:
    """
    Converts given object to list if obj is not already a list. Sets are also transformed to a list.

    Parameters
    ----------
    obj : Any
        Object to convert to list
    
    Returns
    ----------
    list 
        List containing obj, or obj itself if obj was already a list
    """
    if isinstance(obj, (set, tuple)):
        obj = list(obj)
    elif not isinstance(obj, list):
        obj = [obj]
    return obj


@dataclass(frozen=True)
class VerifiedData:
    prediction: xr.DataArray
    ground_truth: xr.DataArray

    def __post_init__(self):
        # Perform checks on initialization
        self._validate_dimensions()
        self._validate_broadcastability()

    def _validate_dimensions(self):
        # Ensure all dimensions in truth are in forecast (or equal)
        missing_dims = set(self.ground_truth.dims) - set(self.prediction.dims)
        if missing_dims:
            raise ValueError(f"Truth data has extra dimensions not found in forecast: {missing_dims}")

    def _validate_broadcastability(self):
        try:
            # Attempt broadcast
            xr.broadcast(self.prediction, self.ground_truth)
        except ValueError as e:
            raise ValueError(f"Forecast and truth are not broadcastable: {e}")


# scores class
class Scores:
    """
    Class to calculate scores and skill scores.
    """

    def __init__(
        self,
        agg_dims: str | List[str] = "all",
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
        self._agg_dims = agg_dims
        self._ens_dim = ens_dim

        self.det_metrics_dict = {
            "ets": self.calc_ets,
            "pss": self.calc_pss,
            "fbi": self.calc_fbi,
            "mae": self.calc_mae,
            "l1": self.calc_l1,
            "l2": self.calc_l2,
            "mse": self.calc_mse,
            "rmse": self.calc_rmse,
            "bias": self.calc_bias,
            "acc": self.calc_acc,
            "bias": self.calc_bias,
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


    def __call__(self, data: VerifiedData, score_name: str, **kwargs):

        if score_name in self.det_metrics_dict.keys:
            f = self.det_metrics_dict[score_name]
        elif score_name in self.prob_metrics_dict.keys:
            assert self.ens_dim in data.prediction.dims, \
                f"Probablistic score {score_name} chosen, but ensemble dimension {self.ens_dim} not found in prediction data" 
            f = self.prob_metrics_dict[score_name]
        else: 
            raise ValueError(f"Unknown score chosen. Supported scores: {', '.join(self.det_metrics_dict.keys()) 
                                                                        + ', ' + ', '.join(self.prob_metrics_dict.keys())}")
        
        # Check if avg_dims is in prediction data
        if self._agg_dims == "all":
            self._agg_dims_applied = list(data.prediction.dims)
        else:
            for dim in self._agg_dims:
                if dim not in data.prediction.dims:
                    raise ValueError(f"Average dimension '{dim}' not found in prediction data dimensions: {data.prediction.dims}")
            self._agg_dims_applied = self._agg_dims
            
        arg_names: list[str] = inspect.getfullargspec(f).args[1:]
        # Shortcut names, convenient but not really necessary.
        args = {"p": data.prediction, "gt": data.ground_truth}
        for an in arg_names:
            if an in kwargs:
                args[an] = kwargs[an]
        # Arguments to pass to the function
        print(f"{f} -> {args}")
        # Call:
        result = f(**args)
        
        return result

    @property
    def agg_dims(self):
        return self._agg_dims

    @agg_dims.setter
    def agg_dims(self, dims):
        if 
        if dims == "all":
            self._agg_dims = dims 
        else:
            agg_dims = to_list(dims)
            assert all([isinstance(dim) for dim in dims]), "All elements of the avg_dims must be strings."

            self._agg_dims = agg_dims

    @property
    def ens_dim(self):
        return self._ens_dim
    
    @ens_dim.setter
    def ens_dim(self, ens_dim):
        assert isinstance(ens_dim, str), "ens_dim must be a string."
        self._ens_dim = ens_dim

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


    def get_2x2_event_counts(self, p: xr.DataArray, gt: xr.DataArray, thresh: float) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
        """
        Get counts of 2x2 contingency tables
        """
        a = self._sum((p >= thresh) & (gt >= thresh))
        b = self._sum((p >= thresh) & (gt >= thresh))
        c = self._sum((p < thresh) & (gt >= thresh))
        d = self._sum((p < thresh) & (gt < thresh))

        return a, b, c, d

    ### Deterministic scores
    
    def calc_ets(self, p: xr.DataArray, gt: xr.DataArray, thresh=0.1):
        a, b, c, d = self.get_2x2_event_counts(p, gt, thresh)
        n = a + b + c + d
        ar = (a + b) * (a + c) / n  # random reference forecast

        denom = a + b + c - ar

        ets = (a - ar) / denom
        ets = ets.where(denom > 0, np.nan)

        return ets

    
    def calc_fbi(self, p: xr.DataArray, gt: xr.DataArray, thresh=0.1):
        a, b, c, _ = self.get_2x2_event_counts(p, gt, thresh)

        denom = a + c
        fbi = (a + b) / denom

        fbi = fbi.where(denom > 0, np.nan)

        return fbi

    
    def calc_pss(self, p: xr.DataArray, gt: xr.DataArray, thresh=0.1):
        a, b, c, d = self.get_2x2_event_counts(p, gt, thresh)

        denom = (a + c) * (b + d)
        pss = (a * d - b * c) / denom

        pss = pss.where(denom > 0, np.nan)

        return pss

    
    def calc_l1(self, p: xr.DataArray, gt: xr.DataArray, scale_dims: List = None):
        """
        Calculate the L1 error norm of forecast data w.r.t. reference data.
        Note that the L1 error norm is calculated as the sum of absolute differences.
        If scale_dims is not None, the L1 will scaled by the number of elements in the average dimensions.
        """
        l1 = self._sum(np.abs(p - gt))

        if _scale_dims:
            _scale_dims = to_list(scale_dims)

            assert all([dim in p.dims for dim in _scale_dims]), \
                f"Provided scale dimensions {_scale_dims} are not all present in the prediction data dimensions {p.dims}."
            
            len_dims = np.array([p.sizes[dim] for dim in _scale_dims])
            l1 /= np.prod(len_dims)

        return l1

    
    def calc_l2(self, p: xr.DataArray, gt: xr.DataArray, scale_dims: List = None):
        """
        Calculate the L2 error norm of forecast data w.r.t. reference data.
        Note that the L2 error norm is calculated as the sum of absolute differences.
        If scale_dims is not None, the L2 will scaled by the number of elements in the average dimensions.
        """
        l2 = self._sum(np.sqrt((np.square(p - gt))))

        if _scale_dims:
            _scale_dims = to_list(scale_dims)

            assert all([dim in p.dims for dim in _scale_dims]), \
                f"Provided scale dimensions {_scale_dims} are not all present in the prediction data dimensions {p.dims}."
            
            len_dims = np.array([p.sizes[dim] for dim in _scale_dims])
            l2 /= np.prod(len_dims)

        return l2

    
    def calc_mae(self, p: xr.DataArray, gt: xr.DataArray):
        """
        Calculate mean absolute error (MAE) of forecast data w.r.t. reference data
        """
        if self._agg_dims is None:
            raise ValueError(
                "Cannot calculate mean absolute error without aggregation dimensions (agg_dims=None)."
            )
        mae = self._mean(np.abs(p - gt))

        return mae

    def calc_mse(self, p: xr.DataArray, gt: xr.DataArray):
        """
        Calculate mean squared error (MSE) of forecast data w.r.t. reference data
        :return: MSE averaged over provided dimensions
        """
        if self._agg_dims is None:
            raise ValueError(
                "Cannot calculate mean squared error without aggregation dimensions (agg_dims=None)."
            )

        mse = self._mean(np.square(self.prediction - self.ground_truth))

        return mse

    def calc_rmse(self, p: xr.DataArray, gt: xr.DataArray):
        """
        Calculate root mean squared error (RMSE) of forecast data w.r.t. reference data
        :return: RMSE averaged over provided dimensions
        """
        if self._agg_dims is None:
            raise ValueError(
                "Cannot calculate root mean squared error without aggregation dimensions (agg_dims=None)."
            )
            raise ValueError(msg)

        rmse = np.sqrt(self.calc_mse(p, gt))

        return rmse

    
    def calc_acc(self, p: xr.DataArray, gt: xr.DataArray, clim_mean: xr.DataArray, spatial_dims: List = ["lat", "lon"]):
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
        clim_mean: xr.DataArray
            Climatological mean data array, which is used to calculate anomalies
        spatial_dims: List[str]
            Names of spatial dimensions over which ACC is calculated.
            Note: No averaging is possible over these dimensions.
        """

        # Check if spatial_dims are in the data
        spatial_dims = to_list(spatial_dims)
        for dim in spatial_dims:
            if dim not in p.dims:
                raise ValueError(f"Spatial dimension '{dim}' not found in prediction data dimensions: {p.dims}")

        fcst_ano, obs_ano = p - clim_mean, gt - clim_mean

        acc = (fcst_ano * obs_ano).sum(spatial_dims) / np.sqrt(
            fcst_ano.sum(spatial_dims) * obs_ano.sum(spatial_dims)
        )

        # Exclude spatial dimensions from averaging since ACC is always calculated over them
        if self._agg_dims is not None:
            mean_dims = [x for x in self._agg_dims if x not in spatial_dims]
            if len(mean_dims) > 0:
                acc = acc.mean(mean_dims)

        return acc

    
    def calc_bias(self, p: xr.DataArray, gt: xr.DataArray):
        """
        Calculate mean bias of forecast data w.r.t. reference data
        """

        bias = self._mean(p - gt)

        return bias

    
    def calc_psnr(self, p: xr.DataArray, gt: xr.DataArray, pixel_max: float = 1.0):
        """
        Calculate PSNR of forecast data w.r.t. reference data
        :param kwargs: known keyword argument 'pixel_max' for maximum value of data
        :return: averaged PSNR
        """

        mse = self.calc_mse(p, gt)
        if np.count_nonzero(mse) == 0:
            psnr = mse
            psnr[...] = 100.0
        else:
            psnr = 20.0 * np.log10(pixel_max / np.sqrt(mse))

        return psnr

    
    def calc_spatial_variability(self, p: xr.DataArray, gt: xr.DataArray, order: int = 1, non_spatial_avg_dims: List[str] = None):
        """
        Calculates the ratio between the spatial variability of differental operator with order 1 (highher values unsupported yest) 
        forecast and ground truth data using the calc_geo_spatial-method.
        
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
        """

        fcst_grad = self.calc_geo_spatial_diff(p, order=order)
        ref_grd = self.calc_geo_spatial_diff(gt, order=order)

        ratio_spat_variability = fcst_grad / ref_grd
        if non_spatial_avg_dims is not None:
            ratio_spat_variability = ratio_spat_variability.mean(dim=non_spatial_avg_dims)

        return ratio_spat_variability

    
    def calc_seeps(
        self, p: xr.DataArray, gt: xr.DataArray, seeps_weights: xr.DataArray, t1: xr.DataArray, 
        t3: xr.DataArray, spatial_dims: List
    ):
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
            assert (
                len(spatial_dims) == 2
            ), "Provide two spatial dimensions for three-dimensional data."
            prediction, ground_truth = p.stack(
                {"xy": spatial_dims}
            ), gt.stack({"xy": spatial_dims})
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
            f"Data must be one- or two-dimensional, but has {prediction.ndim} dimensions. Check if stacking with spatial_dims may help."
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
                prediction_now, ground_truth_now = prediction[it, ...], ground_truth[it, ...]
                # in case of missing data, skip computation
                if np.all(np.isnan(prediction_now)) or np.all(np.isnan(ground_truth_now)):
                    continue

                seeps_values_all[it, ...] = seeps(
                    ground_truth_now, prediction_now, t1.values, t3, seeps_weights.values
                )

        if lstack:
            seeps_values_all = seeps_values_all.unstack()

        if self.avg_dims is not None:
            seeps_values = self._mean(seeps_values_all)
        else:
            seeps_values = seeps_values_all

        return seeps_values

    ### Probablistic scores
    
    def calc_spread(self, p: xr.DataArray, gt: xr.DataArray):
        """
        Calculate the spread of the forecast ensemble
        :return: spread averaged over the provided dimensions
        """
        ens_std = p.std(dim=self.ens_dim)

        return self._mean(np.sqrt((ens_std**2)))

    
    def calc_ssr(self, p: xr.DataArray, gt: xr.DataArray):
        """
        Calculate the Spread-Skill Ratio (SSR) of the forecast ensemble data w.r.t. reference data
        :return: the SSR averaged over provided dimensions
        """
        ssr = self.calc_spread(p) / self.calc_rmse(p, gt)     # spread/rmse

        return ssr  

    
    def calc_crps(self, p: xr.DataArray, gt: xr.DataArray, method: str = "ensemble", **kwargs):
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
            Other keyword parameters supported by respective CRPS-method from the xskillscore package

         
        :param method: Method to calculate CRPS. Supported methods: ["ensemble", "gaussian"]
        :param kwargs: Other keyword parameters supported by respective CRPS-method
        :return: calculated CRPS
        """
        crps_methods = ["ensemble", "gaussian"]

        if method == "ensemble":
            func_kwargs = {
                "forecasts": p,
                "member_dim": self.ens_dim,
                "dim": self._agg_dims,
                **kwargs,
            }
            crps_func = xskillscore.crps_ensemble
        elif method == "gaussian":
            func_kwargs = {
                "mu": p.mean(dim=self.ens_dim),
                "sig": p.std(dim=self.ens_dim),
                "dim": self._agg_dims,
                **kwargs,
            }
            crps_func = xskillscore.crps_gaussian
        else:
            raise ValueError(
            f"Unsupported CRPS-calculation method {method} chosen. Supported methods: {', '.join(crps_methods)}"
            )   

        crps = crps_func(gt, **func_kwargs)

        return crps

    def calc_rank_histogram(self, p: xr.DataArray, gt: xr.DataArray, norm: bool = True,
                            add_noise: bool = True, noise_fac=1.0e-03):
        """
        Calculate the rank histogram of the forecast data w.r.t. reference data.

        Parameters
        ----------
        p: xr.DataArray
            Forecast data array with ensemble dimension
        gt: xr.DataArray
            Ground truth data array
        norm: bool
            Flag if normalized counts should be returned. If True, the rank histogram will be normalized by
            the number of ensemble members in the forecast data.
        add_noise: bool
            Flag if a small amount of random noise should be added to the data to avoid ties in the rank histogram.
            This is recommended for fair computations, cf. Sec. 4.2.2 in Harris et al. 2022
        noise_fac: float
            Magnitude of random noise to be added to the data if add_noise is True. Default is 1.0e-03.
            This value is only relevant if add_noise is True
        """

        # unstack stacked time-dimension beforehand if required (time may be stacked for forecast data)
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

        # calculate ranks for all data points
        rank = (obs_stacked >= fcst_stacked).sum(dim=self.ens_dim)
        # and count occurence of rank values
        rank.name = "rank"  # name for xr.DataArray is required for histogram-method
        rank_counts = histogram(
            rank,
            dim=["npoints"],
            bins=np.arange(len(fcst_stacked[self.ens_dim]) + 2),
            block_size=None if rank.chunks is None else "auto",
        )

        # provide normalized rank counts if desired
        if norm:
            npoints = len(fcst_stacked["npoints"])
            rank_counts = rank_counts / npoints

        return rank_counts

    def calc_rank_histogram_xskillscore(self, p: xr.DataArray, gt: xr.DataArray):   
        """
        Wrapper around rank_histogram-method by xskillscore-package.
        See https://xskillscore.readthedocs.io/en/stable/api
        Note: this version is found to be very slow. Use calc_rank_histogram alternatively.
        """
        rank_hist = xskillscore.rank_histogram(
            gt, p, member_dim=self.ens_dim, dim=self._agg_dims
        )

        return rank_hist

    @staticmethod
    def calc_geo_spatial_diff(
        scalar_field: xr.DataArray, order: int = 1, r_e: float = 6371.0e3, dom_avg: bool = True
    ):
        """
        Calculates the amplitude of the gradient (order=1) or the Laplacian (order=2)
        of a scalar field given on a regular, geographical grid
        (i.e. dlambda = const. and dphi=const.)
        :param scalar_field: scalar field as data array with latitude and longitude as coordinates
        :param order: order of spatial differential operator
        :param r_e: radius of the sphere
        :return: the amplitude of the gradient/laplacian at each grid point or over the whole domain (see dom_avg)
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
                i = coord_names_expected.index()
            except ValueError as e:
                expected_names = ",".join(coord_names_expected)
                raise ValueError(
                    "Could not find one of the following coordinates in the"
                    + f"passed dictionary: {expected_names}"
                ) from e

        _, lat_name = check_for_coords(dims, lat_dims)
        _, lon_name = check_for_coords(dims, lon_dims)

        lat, lon = np.deg2rad(scalar_field[lat_name]), np.deg2rad(scalar_field[lon_name])
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


# Example usage in tests
if __name__ == "__main__":
    io = MockIO(config={"dummy": True})

    sc = Scores(io, 1, "ERA5", 6)

    a = sc.calc_l1()
    print(a)
    a = sc.calc_mae()
    print(a)
