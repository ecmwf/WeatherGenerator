import numpy as np
import xarray as xr

"""
Extra helper functions to preprocess data
e.g. for verif applications
"""


def compute_mslp(obs: xr.DataArray, time: np.datetime64) -> np.typing.NDArray:
    """
    Compute mean sea level pressure (MSLP) from surface air pressure,
    air temperature, and relative humidity.
    Parameters
    ----------
        obs : xarray DataArray
            Input data containing surface air pressure, air temperature, and relative humidity.
        time : np.datetime64
            Time over which to compute mean for the MSLP.
    Returns
    -------
        np.ndarray
            Computed mean sea level pressure values.
    """
    # g = 9.80665  # Gravitational acceleration (m/s**2)
    # R = 8.31447  # Universal gas constant (J/mol*K)
    # a = 0.0065  # Temperature lapse rate (K/m)
    # Ch = 0.0012  # (K/Pa)

    a = 17.625
    b = 243.03
    c = 6.1094

    p = obs.data_vars["surface_air_pressure"].sel(time=time)
    t = obs.data_vars["air_temperature"].sel(time=time)
    rh = obs.data_vars["relative_humidity"].sel(time=time)

    altitude = obs.altitude

    e = rh * 6.11 * np.power(10.0, ((7.5 * (t - 273.15)) / (t - 38.85)))

    dewpoint = np.where(~np.isnan(e), b * np.log(e / c) / (a - np.log(e / c)), t - 276.15)

    e = np.where(np.isnan(e), 0, e)

    tv = t / (1.0 - 0.379 * (6.11 * np.power(10.0, ((7.5 * dewpoint) / (237.7 + dewpoint))) / p))

    #        mslp = np.where(altitude >= 50.,
    #                        p * np.exp((g * altitude / R) / (t + 0.5 * a * altitude + e * Ch)),
    #                        p + p * altitude / (29.27 * tv))

    mslp = p + p * altitude / (29.27 * tv)

    return mslp


def compute_precip(
    obs_data: xr.Dataset, zarr_dt: np.timedelta64, frt: np.datetime64
) -> np.typing.NDArray:
    """
    Compute accumulated precipitation over the forecast time step.
    Parameters
    ----------
    obs_data : xarray Dataset
        Input data containing precipitation observations.
    zarr_dt : np.timedelta64
        Time difference between forecast steps in hours.
    frt : np.datetime64
        Forecast reference time for which to compute accumulated precipitation.
    Returns
    -------
    np.ndarray
        Accumulated precipitation values for the forecast time step."""
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
