import json
import logging

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

# TODO: adjust to use utils.io.ZarrIO
from weathergen.utils.mock_io import MockIO

_logger = logging.getLogger(__name__)


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
def _get_skill_score(score_model, score_ref, score_perf):
    skill_score = (score_model - score_ref) / (score_perf - score_ref)

    return skill_score


def to_json(func):
    def wrapper(*args, **kwargs):
        da = func(*args, **kwargs)
        return json.dumps(da.to_dict(), default=str)

    return wrapper


# scores class
class Scores:
    """
    Class to calculate scores and skill scores.
    """

    def __init__(
        self,
        io: MockIO,
        sample: int,
        stream: str,
        forecast_step: int,
        avg_dims: str | list[str] = "all",
    ):
        """
        :param avg_dims: dimension or list of dimensions over which scores shall be averaged.
                         Parse 'all' to average over all data dimensions.
        :param ens_dim: name of ensemble meber dimension in prediction. Ignored if
                        determinsitic forecast is processed.
        """
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
            "ssr": self.calc_ssr,
            "grad_amplitude": self.calc_spatial_variability,
            "psnr": self.calc_psnr,
            "seeps": self.calc_seeps,
        }
        self.prob_metrics_dict = {
            "crps": self.calc_crps,
            "rank_histogram": self.calc_rank_histogram,
            "spread": self.calc_spread,
        }

        data = io.get_data(sample, stream, forecast_step)
        self.ground_truth = data.target.as_xarray()
        self.prediction = data.prediction.as_xarray()
        self.ens_dim = self.prediction.shape[-1]

        self.prob_fcst = self.ens_dim in self.prediction.dims
        self.joint_data_dims = [
            dim for dim in self.prediction.dims if dim != self.ens_dim
        ]  # excludes ensemble-dimension for probablistic forecasts
        self.avg_dims = avg_dims

        self.metrics_dict = self.prob_metrics_dict if self.prob_fcst else self.det_metrics_dict

    def __call__(self, score_name, **kwargs):
        try:
            score_func = self.metrics_dict[score_name]
        except Exception as e:
            score_family = "probablistic" if self.prob_fcst else "deterministic"
            metrics = ", ".join(self.metrics_dict.keys())
            msg = (
                f"{score_name} is not an implemented {score_family} score."
                + f"Choose one of the following: {metrics}"
            )
            raise ValueError(msg) from e

        return score_func(**kwargs)

    @property
    def avg_dims(self):
        return self._avg_dims

    @avg_dims.setter
    def avg_dims(self, dims):
        if dims is None:
            self._avg_dims = None
        elif dims == "all":
            self._avg_dims = self.joint_data_dims
            # print("Scores will be averaged across all data dimensions.")
        else:
            dim_stat = [avg_dim in self.joint_data_dims for avg_dim in dims]
            if not all(dim_stat):
                ind_bad = [i for i, x in enumerate(dim_stat) if not x]
                raise ValueError(
                    "The following dimensions for score-averaging are not "
                    + "part of the data: {}".format(", ".join(np.array(dims)[ind_bad]))
                )

            self._avg_dims = dims

    def get_2x2_event_counts(self, thresh):
        """
        Get counts of 2x2 contingency tables
        """
        a = ((self.prediction >= thresh) & (self.ground_truth >= thresh)).sum(dim=self.avg_dims)
        b = ((self.prediction >= thresh) & (self.ground_truth < thresh)).sum(dim=self.avg_dims)
        c = ((self.prediction < thresh) & (self.ground_truth >= thresh)).sum(dim=self.avg_dims)
        d = ((self.prediction < thresh) & (self.ground_truth < thresh)).sum(dim=self.avg_dims)

        return a, b, c, d

    ### Deterministic scores
    @to_json
    def calc_ets(self, thresh=0.1):
        a, b, c, d = self.get_2x2_event_counts(thresh)
        n = a + b + c + d
        ar = (a + b) * (a + c) / n  # random reference forecast

        denom = a + b + c - ar

        ets = (a - ar) / denom
        ets = ets.where(denom > 0, np.nan)

        return ets

    @to_json
    def calc_fbi(self, thresh=0.1):
        a, b, c, d = self.get_2x2_event_counts(thresh)

        denom = a + c
        fbi = (a + b) / denom

        fbi = fbi.where(denom > 0, np.nan)

        return fbi

    @to_json
    def calc_pss(self, thresh=0.1):
        a, b, c, d = self.get_2x2_event_counts(thresh)

        denom = (a + c) * (b + d)
        pss = (a * d - b * c) / denom

        pss = pss.where(denom > 0, np.nan)

        return pss

    @to_json
    def calc_l1(self, **kwargs):
        """
        Calculate the L1 error norm of forecast data w.r.t. reference data.
        L1 will be divided by the number of samples along the average dimensions.
        Similar to MAE, but provides just a number divided by number of samples along
        average dimensions.
        :return: L1-error
        """
        sum_dims = kwargs.get("sum_dims", [])

        l1 = (np.abs(self.prediction - self.ground_truth)).sum(dim=sum_dims)

        if self.avg_dims is not None:
            len_dims = np.array([self.prediction.sizes[dim] for dim in self.avg_dims])
            l1 /= np.prod(len_dims)

        return l1

    @to_json
    def calc_l2(self, **kwargs):
        """
        Calculate the L2 error norm of forecast data w.r.t. reference data.
        Similar to RMSE, but provides just a number divided by number of samples along
        average dimensions.
        :return: L2-error
        """
        sum_dims = kwargs.get("sum_dims", [])

        l2 = np.sqrt((np.square(self.prediction - self.ground_truth)).sum(dim=sum_dims))

        if self.avg_dims is not None:
            len_dims = np.array([self.prediction.sizes[dim] for dim in self.avg_dims])
            l2 /= np.prod(len_dims)

        return l2

    @to_json
    def calc_mae(self, **kwargs):
        """
        Calculate mean absolute error (MAE) of forecast data w.r.t. reference data
        :return: MAE averaged over provided dimensions
        """
        if kwargs:
            print("Passed keyword arguments to calc_mae are without effect.")

        if self.avg_dims is None:
            raise ValueError(
                "Cannot calculate mean absolute error without average dimensions (avg_dims=None)."
            )
        mae = np.abs(self.prediction - self.ground_truth).mean(dim=self.avg_dims)

        return mae

    @to_json
    def calc_mse(self, **kwargs):
        """
        Calculate mean squared error (MSE) of forecast data w.r.t. reference data
        :return: MSE averaged over provided dimensions
        """
        if kwargs:
            print("Passed keyword arguments to calc_mse are without effect.")

        if self.avg_dims is None:
            raise ValueError(
                "Cannot calculate mean squared error without average dimensions (avg_dims=None)."
            )

        mse = np.square(self.prediction - self.ground_truth).mean(dim=self.avg_dims)

        return mse

    @to_json
    def calc_rmse(self, **kwargs):
        """
        Calculate root mean squared error (RMSE) of forecast data w.r.t. reference data
        :return: RMSE averaged over provided dimensions
        """
        if self.avg_dims is None:
            msg = (
                "Cannot calculate root mean squared error without average dimensions"
                + "(avg_dims=None)."
            )
            raise ValueError(msg)

        rmse = np.sqrt(self.calc_mse(**kwargs))

        return rmse

    @to_json
    def calc_acc(self, clim_mean: xr.DataArray, spatial_dims: list = None):
        """
        Calculate anomaly correlation coefficient (ACC).
        :param clim_mean: climatological mean of the data
        :param spatial_dims: names of spatial dimensions over which ACC are calculated.
                             Note: No averaging is possible over these dimensions.
        :return acc: Averaged ACC (except over spatial_dims)
        """

        if spatial_dims is None:
            spatial_dims = ["lat", "lon"]
        fcst_ano, obs_ano = self.prediction - clim_mean, self.ground_truth - clim_mean

        acc = (fcst_ano * obs_ano).sum(spatial_dims) / np.sqrt(
            fcst_ano.sum(spatial_dims) * obs_ano.sum(spatial_dims)
        )

        if self.avg_dims is not None:
            mean_dims = [x for x in self.avg_dims if x not in spatial_dims]
            if len(mean_dims) > 0:
                acc = acc.mean(mean_dims)

        return acc

    @to_json
    def calc_bias(self, **kwargs):
        """
        Calculate mean bias of forecast data w.r.t. reference data
        :return: bias averaged over provided dimensions
        """

        if kwargs:
            print("Passed keyword arguments to calc_bias are without effect.")

        bias = self.prediction - self.ground_truth

        if self.avg_dims is not None:
            bias = bias.mean(dim=self.avg_dims)

        return bias

    @to_json
    def calc_psnr(self, **kwargs):
        """
        Calculate PSNR of forecast data w.r.t. reference data
        :param kwargs: known keyword argument 'pixel_max' for maximum value of data
        :return: averaged PSNR
        """
        pixel_max = kwargs.get("pixel_max", 1.0)

        mse = self.calc_mse()
        if np.count_nonzero(mse) == 0:
            psnr = mse
            psnr[...] = 100.0
        else:
            psnr = 20.0 * np.log10(pixel_max / np.sqrt(mse))

        return psnr

    @to_json
    def calc_spatial_variability(self, **kwargs):
        """
        Calculates the ratio between the spatial variability of differental operator
        with order 1 (or 2) forecast and
        reference data using the calc_geo_spatial-method.
        :param kwargs: 'order' to control the order of spatial differential operator
                       'non_spatial_avg_dims' to add averaging in addition to spatial
                       averaging performed with calc_geo_spatial
        :return: the ratio between spatial variabilty in the forecast and reference data field
        """
        order = kwargs.get("order", 1)
        avg_dims = kwargs.get("non_spatial_avg_dims")

        fcst_grad = self.calc_geo_spatial_diff(self.prediction, order=order)
        ref_grd = self.calc_geo_spatial_diff(self.ground_truth, order=order)

        ratio_spat_variability = fcst_grad / ref_grd
        if avg_dims is not None:
            ratio_spat_variability = ratio_spat_variability.mean(dim=avg_dims)

        return ratio_spat_variability

    @to_json
    def calc_seeps(
        self, seeps_weights: xr.DataArray, t1: xr.DataArray, t3: xr.DataArray, spatial_dims: list
    ):
        """
        Calculates stable equitable error in probabiliyt space (SEEPS), see Rodwell et al., 2011
        :param seeps_weights: SEEPS-parameter matrix to weight contingency table elements
        :param t1: threshold for light precipitation events
        :param t3: threshold for strong precipitation events
        :param spatial_dims: list/name of spatial dimensions of the data
        :return seeps skill score (i.e. 1-SEEPS)
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

        if self.prediction.ndim == 3:
            assert len(spatial_dims) == 2, (
                "Provide two spatial dimensions for three-dimensional data."
            )
            prediction, ground_truth = (
                self.prediction.stack({"xy": spatial_dims}),
                self.ground_truth.stack({"xy": spatial_dims}),
            )
            seeps_weights = seeps_weights.stack({"xy": spatial_dims})
            t3 = t3.stack({"xy": spatial_dims})
            lstack = True
        elif self.prediction.ndim == 2:
            prediction, ground_truth = self.prediction, self.ground_truth
            lstack = False
        else:
            raise ValueError("Data must be a two-or-three-dimensional array.")

        # check dimensioning of data
        assert prediction.ndim <= 2, (
            f"Data must be one- or two-dimensional, but has {prediction.ndim} dimensions."
            + "Check if stacking with spatial_dims may help."
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
            seeps_values = seeps_values_all.mean(dim=self.avg_dims)
        else:
            seeps_values = seeps_values_all

        return seeps_values

    ### Probablistic scores
    @to_json
    def calc_spread(self, **kwargs):
        """
        Calculate the spread of the forecast ensemble
        :return: spread averaged over the provided dimensions
        """
        if kwargs:
            print("Passed keyword arguments to calc_spread are without effect.")

        ens_std = self.prediction.std(dim=self.ens_dim)
        return np.sqrt((ens_std**2).mean(dim=self.avg_dims))

    @to_json
    def calc_ssr(self, **kwargs):
        """
        Calculate the Spread-Skill Ratio (SSR) of the forecast ensemble data w.r.t. reference data
        :return: the SSR averaged over provided dimensions
        """
        # ens_std = data_ens.std(dim = "ensemble")
        # spread = np.sqrt((ens_std**2).mean(dim = avg_dims))

        # spread = self.calc_spread(**kwargs)
        # mse = np.square(self.prediction - self.ground_truth).mean(dim = self.avg_dims)
        # rmse = np.sqrt(mse)
        return self.calc_spread(**kwargs) / self.calc_rmse(**kwargs)  # spread/rmse

    @to_json
    def calc_crps(self, method: str = "ensemble", **kwargs):
        """
        Wrapper around CRPS-methods provided by xskillscore-package.
        See https://xskillscore.readthedocs.io/en/stable/api
        :param method: Method to calculate CRPS. Supported methods: ["ensemble", "gaussian"]
        :param kwargs: Other keyword parameters supported by respective CRPS-method
        :return: calculated CRPS
        """
        crps_methods = ["ensemble", "gaussian"]

        assert self.ens_dim in self.prediction.dims, (
            "Forecast data array must have an 'ens'-dimension."
        )

        if method == "ensemble":
            func_kwargs = {
                "forecasts": self.prediction,
                "member_dim": self.ens_dim,
                "dim": self.avg_dims,
                **kwargs,
            }
            crps_func = xskillscore.crps_ensemble
        elif method == "gaussian":
            func_kwargs = {
                "mu": self.prediction.mean(dim=self.ens_dim),
                "sig": self.prediction.std(dim=self.ens_dim),
                "dim": self.avg_dims,
                **kwargs,
            }
            crps_func = xskillscore.crps_gaussian
        else:
            (
                f"Unsupported CRPS-calculation method {method} chosen."
                + f"Supported methods: {', '.join(crps_methods)}"
            )

        crps = crps_func(self.ground_truth, **func_kwargs)

        return crps

    def calc_rank_histogram(self, norm: bool = True, add_noise: bool = True, noise_fac=1.0e-03):
        """
        :param norm: Flag if normalized counts should be returned
        a:param add_noise: Add unsignificant amount of random noise to data for fair
                           computations, cf. Sec. 4.2.2 in Harris et al. 2022
        :param noise_fac: magnitude of random noise (only relevant if add_noise == True)
        """

        # unstack stacked time-dimension beforehand if required
        # (time may be stacked for forecast data)
        ground_truth = self.ground_truth
        if "time" in self.ground_truth.indexes:
            if isinstance(self.ground_truth.indexes["time"], pd.MultiIndex):
                ground_truth = self.ground_truth.reset_index("time")

        prediction = self.prediction
        if "time" in self.prediction.indexes:
            if isinstance(self.prediction.indexes["time"], pd.MultiIndex):
                prediction = self.prediction.reset_index("time")

        # perform the stacking
        obs_stacked = ground_truth.stack({"npoints": self.avg_dims})
        fcst_stacked = prediction.stack({"npoints": self.avg_dims})

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

    def calc_rank_histogram_xskillscore(self, **kwargs):
        """
        Wrapper around rank_histogram-method by xskillscore-package.
        See https://xskillscore.readthedocs.io/en/stable/api
        Note: this version is found to be very slow. Use calc_rank_histogram alternatively.
        """
        if kwargs:
            print("Passed keyword arguments to calc_rank_historam are without effect.")

        rank_hist = xskillscore.rank_histogram(
            self.ground_truth, self.prediction, member_dim=self.ens_dim, dim=self.avg_dims
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
        :return: the amplitude of the gradient/laplacian at each grid point
                 or over the whole domain (see avg_dom)
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

            return i, coord_names_expected[i]  # just take the first value

        lat_ind, lat_name = check_for_coords(dims, lat_dims)
        lon_ind, lon_name = check_for_coords(dims, lon_dims)

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
