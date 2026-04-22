import logging
import re
from dataclasses import dataclass

import numpy as np
import xarray as xr

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


def is_derivable_channel(name: str) -> bool:
    """Return ``True`` if *name* matches a known derivable-channel pattern (e.g. ``10ff``)."""
    return re.fullmatch(r"\d+ff", name) is not None


@dataclass
class DeriveChannels:
    def __init__(
        self,
        available_channels: np.array,
        channels: list,
        stream_cfg: dict,
    ):
        """
        Initializes the DeriveChannels class with necessary configurations for channel derivation.

        Args:
            available_channels (np.array): an array of all available channel names
            in the datasets (target or pred).
            channels (list): A list of channels of interest to be evaluated and/or plotted.
            stream_cfg (dict): A dictionary containing the stream configuration settings for
            evaluation and plottings.

        Returns:
            None
        """
        self.available_channels = available_channels
        self.channels = channels
        self.stream_cfg = stream_cfg

    def calc_xxff_channel(self, da: xr.DataArray, level: str) -> xr.DataArray | None:
        """
        Calculate wind speed at xx level ('xxff') from wind components or directly.
        Args:
            da: xarray DataArray with data
        Returns:
            xarray: Calculated xxff value, or None if calculation is not possible
        """

        channels = da.channel.values

        if f"{level}si" not in channels:
            for suffix in ["u", "v"]:
                for name in [
                    f"{level}{suffix}",
                    f"{suffix}_{level}",
                    f"obsvalue_{suffix}{level}m_0",
                ]:
                    component = da.sel(channel=name) if name in channels else None
                    if component is not None:
                        break
                if suffix == "u":
                    u_component = component if component is not None else None
                else:
                    v_component = component if component is not None else None
            if not (u_component is None or v_component is None):
                ff = np.sqrt(u_component**2 + v_component**2)
                return ff
            else:
                _logger.debug(
                    f"u or v not found for level {level} - skipping {level}ff calculation"
                )
                return None
        elif f"{level}si" in channels:
            ff = da.sel(channel=f"{level}si")
            return ff
        else:
            _logger.debug(f"Skipping {level}ff calculation - unsupported data format")
            return None

    def get_channel(self, data_tars, data_preds, tag, level, calc_func) -> None:
        """
        Add a new channel data to both target and prediction datasets.

        This method computes new channel values using given calculations methods
        and appends them as a new channel to both self.data_tars and self.data_preds.
        If the calculation returns None, the original datasets are preserved unchanged.

        The method updates:
        - data_tars: Target dataset with added 10ff channel
        - data_preds: Prediction dataset with added 10ff channel
        - self.channels: Channel list with '10ff' added

        Returns:
            None
        """

        data_updated = []

        for data in [data_tars, data_preds]:
            new_channel = calc_func(data, level)

            if new_channel is not None:
                conc = xr.concat(
                    [
                        data,
                        new_channel.expand_dims("channel").assign_coords(channel=[tag]),
                    ],
                    dim="channel",
                )

                data_updated.append(conc)

                self.channels = self.channels + ([tag] if tag not in self.channels else [])

            else:
                data_updated.append(data)

        data_tars, data_preds = data_updated
        return data_tars, data_preds

    def get_derived_channels(
        self,
        data_tars: xr.DataArray,
        data_preds: xr.DataArray,
    ) -> tuple[xr.DataArray, xr.DataArray, list]:
        """
        Derive channels from available channels in the data.

        Channels to derive are collected from two sources:

        1. The ``derive_channels`` key in the stream config (explicit).
        2. Any channel in ``self.channels`` that is absent from
           ``self.available_channels`` and whose name matches a known
           derivable pattern (e.g. ``10ff`` — wind speed from u/v).

        Parameters
        ----------
        data_tars : xr.DataArray
            Target dataset.
        data_preds : xr.DataArray
            Prediction dataset.

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray, list]
            Updated targets, predictions and the (possibly extended) channel list.
        """
        # Collect explicit tags from config …
        tags_to_derive: list[str] = list(self.stream_cfg.get("derive_channels", []))

        # … and auto-detect derivable channels requested by the user that are
        # not already present in the data.
        for ch in self.channels:
            if ch not in self.available_channels and ch not in tags_to_derive:
                if is_derivable_channel(ch):
                    tags_to_derive.append(ch)

        if not tags_to_derive:
            return data_tars, data_preds, self.channels

        for tag in tags_to_derive:
            if tag not in self.available_channels:
                match = re.search(r"(\d+)", tag)
                level = match.group() if match else None
                if tag == f"{level}ff":
                    data_tars, data_preds = self.get_channel(
                        data_tars, data_preds, tag, level, self.calc_xxff_channel
                    )
            else:
                _logger.debug(
                    f"Calculation of {tag} is skipped because it is included "
                    "in the available channels..."
                )
        return data_tars, data_preds, self.channels


def scale_z_channels(data: xr.DataArray, stream: str) -> xr.DataArray:
    """
    Scale geopotential (z_*) channels from m²/s² to geopotential height (m).

    Parameters
    ----------
    data :
        Input DataArray.
    stream :
        Stream name.  Scaling is only applied to ERA5-family streams.

    Returns
    -------
        DataArray with z_* channels divided by *g* (9.80665 m/s²).
    """
    if stream is None or not str(stream).startswith("ERA5"):
        return data

    channels_z = [ch for ch in np.atleast_1d(data.channel.values) if str(ch).startswith("z_")]
    factor = 9.80665

    if channels_z:
        channels = data.channel.astype(str)
        mask = channels.str.startswith("z_")
        data = data.where(~mask, data / factor)
    return data


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
            accumulate += (
                obs_data.data_vars["precipitation_amount_1h"].sel(time=back_time).squeeze()
            )
        return accumulate
