import logging
import numpy as np
import xarray as xr


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

"""
Extra helper functions to preprocess data
e.g. for verif applications
"""


def compute_mslp(obs: xr.DataArray, time: np.datetime64) -> np.ndarray:
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

def compute_precip(obs_data, zarr_dt, frt):

    obs_dt = obs_data.time.values[1] - obs_data.time.values[0]
    obs_dt = obs_dt.astype("timedelta64[h]")

    if obs_dt >= zarr_dt:
        return obs_data["precipitation_amount_1h"].values
    else:
        accumulate = np.zeros(obs_data.location.shape[0])
        int_factor = int(zarr_dt / obs_dt)

        for i in range(int_factor):
            back_time = frt - zarr_dt + (i + 1) * obs_dt
            accumulate += obs_data.data_vars["precipitation_amount_1h"].sel(time=back_time)
        return accumulate
