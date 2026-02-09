import numpy as np
import xarray as xr

from weathergen.common.io import ZarrIO
from weathergen.evaluate.scores.score import Scores
from weathergen.verif.verif_config import Variable
from weathergen.verif.verif_interpolator import Verif_interpolator


class Processer:
    unit_conversion = {"kg/m^2": 1.0, "Pa": 1.0, "K": 1.0, "m/s": 1.0, "m": 1000.0}

    def __init__(
        self,
        zarrio: ZarrIO,
        obs: xr.DataArray,
        stream: str,
        interpolator: Verif_interpolator,
        dataset: str,
    ):
        self.zarrio = zarrio
        self.obs = obs
        self.stream = stream
        self.interpolator = interpolator
        self.dataset = dataset

        item = zarrio.get_data(sample=0, stream=stream, forecast_step=1)
        if self.dataset == "prediction":
            self.xdata = item.prediction.as_xarray()
        else:
            self.xdata = item.target.as_xarray()

        self.obs_dt = self.obs.time.values[1] - self.obs.time.values[0]
        self.obs_dt = self.obs_dt.astype("timedelta64[h]")

        self.zarr_dt = (
            self.xdata.source_interval_end.values[0] - self.xdata.source_interval_start.values[0]
        )
        self.zarr_dt = self.zarr_dt.astype("timedelta64[h]")

    def get_data(self, v: Variable, fcstdata, obsdata):
        for sample in range(len(self.zarrio.samples)):
            for step in range(len(self.zarrio.forecast_steps)):
                item = self.zarrio.get_data(
                    sample=sample, stream=self.stream, forecast_step=step + 1
                )

                if self.dataset == "prediction":
                    ydata = Scores.sort_by_coords(item.prediction.as_xarray(), self.xdata)
                else:
                    ydata = Scores.sort_by_coords(item.target.as_xarray(), self.xdata)

                obsdata[sample, step, :] = self.get_obsdata(
                    self.obs, v.obs_name, ydata.valid_time.values[0]
                )

                fcstdata[sample, step, :] = self.get_fcstdata(ydata, v, sample, step + 1)

    def get_obsdata(self, obs: xr.DataArray, name: str, time: np.datetime64) -> np.ndarray:
        return obs.data_vars[name].sel(time=time)

    def get_fcstdata(self, ydata: xr.DataArray, v: Variable, sample: int, step: int) -> np.ndarray:
        return (
            self.interpolator.interpolate(
                ydata.sel(
                    sample=sample,
                    stream=self.stream,
                    forecast_step=step,
                    channel=v.zarr_names,
                    ens=0,
                ).values
            )
            * self.unit_conversion[v.zarr_units]
        )


class MSLP_processer(Processer):
    def get_obsdata(self, obs: xr.DataArray, name: str, time: np.datetime64) -> np.ndarray:
        return self.compute_mslp(obs, time)

    def compute_mslp(self, obs: xr.DataArray, time: np.datetime64) -> np.ndarray:
        # g = 9.80665  # Gravitational acceleration (m/s**2)
        # R = 8.31447  # Universal gas constant (J/mol*K)

        # a = 0.0065  # Temperature lapse rate (K/m)
        # Ch = 0.0012  # (K/Pa)

        A = 17.625
        B = 243.03
        C = 6.1094

        P = obs.data_vars["surface_air_pressure"].sel(time=time)
        T = obs.data_vars["air_temperature"].sel(time=time)
        rh = obs.data_vars["relative_humidity"].sel(time=time)

        altitude = obs.altitude

        e = rh * 6.11 * np.power(10.0, ((7.5 * (T - 273.15)) / (T - 38.85)))

        dewpoint = np.where(~np.isnan(e), B * np.log(e / C) / (A - np.log(e / C)), T - 276.15)

        e = np.where(np.isnan(e), 0, e)

        Tv = T / (
            1.0 - 0.379 * (6.11 * np.power(10.0, ((7.5 * dewpoint) / (237.7 + dewpoint))) / P)
        )

        #        mslp = np.where(altitude >= 50.,
        #                        P * np.exp((g * altitude / R) / (T + 0.5 * a * altitude + e * Ch)),
        #                        P + P * altitude / (29.27 * Tv))

        mslp = P + P * altitude / (29.27 * Tv)

        return mslp


class Wind_processer(Processer):
    def get_fcstdata(self, ydata: xr.DataArray, v: Variable, sample: int, step: int) -> np.ndarray:
        if isinstance(v.zarr_names, str):
            return super().get_fcstdata(ydata, v, sample, step)

        else:
            u = self.interpolator.interpolate(
                ydata.sel(
                    sample=sample,
                    stream=self.stream,
                    forecast_step=step,
                    channel=v.zarr_names[0],
                    ens=0,
                ).values
            )

            v = self.interpolator.interpolate(
                ydata.sel(
                    sample=sample,
                    stream=self.stream,
                    forecast_step=step,
                    channel=v.zarr_names[1],
                    ens=0,
                ).values
            )

        return np.sqrt(np.square(u) + np.square(v))


class Precipitation_processer(Processer):
    def get_obsdata(self, obs: xr.DataArray, name: str, time: np.datetime64) -> np.ndarray:
        if self.obs_dt >= self.zarr_dt:
            return super().get_obsdata(obs, name, time)
        else:
            accumulate = np.zeros(self.obs.location.shape[0])
            int_factor = int(self.zarr_dt / self.obs_dt)

            for i in range(int_factor):
                back_time = time - self.zarr_dt + (i + 1) * self.obs_dt
                accumulate += super().get_obsdata(obs, name, back_time)

            return accumulate


class Processer_factory:
    def __init__(
        self,
        zarrio: ZarrIO,
        obs: xr.DataArray,
        stream: str,
        interpolator: Verif_interpolator,
        dataset: str,
    ):
        self.zarrio = zarrio
        self.obs = obs
        self.stream = stream
        self.interpolator = interpolator
        self.dataset = dataset

    def get_processer(self, name: str) -> Processer:
        if name == "mslp":
            return MSLP_processer(
                self.zarrio, self.obs, self.stream, self.interpolator, self.dataset
            )
        elif name == "wind":
            return Wind_processer(
                self.zarrio, self.obs, self.stream, self.interpolator, self.dataset
            )
        elif name == "tp":
            return Precipitation_processer(
                self.zarrio, self.obs, self.stream, self.interpolator, self.dataset
            )
        else:
            return Processer(self.zarrio, self.obs, self.stream, self.interpolator, self.dataset)
