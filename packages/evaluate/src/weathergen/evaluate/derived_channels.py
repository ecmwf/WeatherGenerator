import logging
from dataclasses import dataclass

import numpy as np
import xarray as xr

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


@dataclass
class DeriveChannels:
    def __init__(
        self,
        available_channels: np.array,
        channels: list,
        stream_cfg: dict,
    ):
        self.available_channels = available_channels
        self.channels = channels
        self.stream_cfg = stream_cfg

    def calc_10ff_channel(self, da: xr.DataArray) -> xr.DataArray | None:
        """
        Calculate 10m wind speed ('10ff') from wind components or directly.
        Args:
            da: xarray DataArray with data
        Returns:
            xarray: Calculated 10ff value, or None if calculation is not possible
        """

        stream = da.stream.values
        channels = da.channel.values

        if stream == "ERA5" and "10si" not in channels:
            if "10u" in channels and "10v" in channels:
                u_component = da.sel(channel="10u")
                v_component = da.sel(channel="10v")
                ff = np.sqrt(u_component**2 + v_component**2)
                return ff
            else:
                _logger.debug("10u or 10v not found - skipping 10ff calculation")
                return None

        elif stream in ["CERRA", "ERA5"] and "10si" in channels:
            ff = da.sel(channel="10si")
            return ff

        else:
            _logger.debug("Skipping 10ff calculation - unsupported data format")
            return None

    def get_channel(self, tag, calc_func) -> None:
        """
        Add a new channel data to both target and prediction datasets.

        This method computes new channel values using given calculations methods
        and appends them as a new channel to both self.data_tars and self.data_preds.
        If the calculation returns None, the original datasets are preserved unchanged.

        The method updates:
        - self.data_tars: Target dataset with added 10ff channel
        - self.data_preds: Prediction dataset with added 10ff channel
        - self.channels: Channel list with '10ff' added

        Returns:
            None
        """

        data_updated = []

        for data in [self.data_tars, self.data_preds]:
            new_channel = calc_func(data)

            if new_channel is not None:
                conc = xr.concat(
                    [
                        data,
                        new_channel.expand_dims("channel").assign_coords(channel=[tag]),
                    ],
                    dim="channel",
                )

                data_updated.append(conc)

                self.channels = self.channels + (
                    [tag] if tag not in self.channels else []
                )

            else:
                data_updated.append(data)

        self.data_tars, self.data_preds = data_updated

    def get_derived_channels(
        self,
        data_tars: xr.DataArray,
        data_preds: xr.DataArray,
    ) -> tuple[xr.DataArray, xr.DataArray, list]:
        """
        Function to derive channels from available channels in the data

        Parameters:
        -----------
        - data_tars: Target dataset
        - data_preds: Prediction dataset

        Returns:
        --------
        - data_tars: Updated target dataset (if channel can be added)
        - data_preds:  Updated prediction dataset (if channel can be added)
        - self.channels: all the channels of interest

        """

        self.data_tars = data_tars
        self.data_preds = data_preds

        if "derive_channels" not in self.stream_cfg:
            return self.data_tars, self.data_preds, self.channels

        for tag in self.stream_cfg["derive_channels"]:
            if tag not in self.available_channels:
                if tag == "10ff":
                    self.get_channel(tag, self.calc_10ff_channel)
            else:
                _logger.debug(
                    f"Calculation of {tag} is skipped because it is included in the available channels..."
                )
        return self.data_tars, self.data_preds, self.channels
