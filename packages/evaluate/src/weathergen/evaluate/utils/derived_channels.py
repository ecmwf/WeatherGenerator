import logging
import re
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

        if "derive_channels" not in self.stream_cfg:
            return data_tars, data_preds, self.channels

        for tag in self.stream_cfg["derive_channels"]:
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
