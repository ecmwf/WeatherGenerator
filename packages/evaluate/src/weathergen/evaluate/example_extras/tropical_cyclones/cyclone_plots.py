import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cyclone_finder import track_error


def track_eval_plot(matched_tracks):
    """
    A four panel plot showing
    * the target and predicted track on a map
    * the track error in km
    * the pressure at the cyclone core for target and prediction
    * the maximum wind speed near the cyclone center
    """
    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(10, 6))
    fig.delaxes(axs[0, 0])
    axs[0, 0] = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
    axs[0, 0].coastlines()
    axs[0, 0].set_title("storm tracks")
    axs[0, 1].set_title("track error in km")
    axs[1, 0].set_title("core pressure in Pa")
    axs[1, 1].set_title("max wind speed in m/s")
    track_error(*matched_tracks.values()).plot(ax=axs[0, 1])
    for lab, track in matched_tracks.items():
        track.plot(x="lon", y="lat", ax=axs[0, 0], label=lab)
        track.plot(y="pressure", ax=axs[1, 0], label=lab)
        track.plot(y="wind", ax=axs[1, 1], label=lab)
    return fig, axs


def bounding_box(matched_tracks, pad=2):
    """
    Compute a lon/lat box containing the matched cyclone tracks.
    """
    all_lons = pd.concat([matched_tracks["target"]["lon"], matched_tracks["prediction"]["lon"]])
    all_lats = pd.concat([matched_tracks["target"]["lat"], matched_tracks["prediction"]["lat"]])
    lon_min = all_lons.min() - pad
    lon_max = all_lons.max() + pad
    lat_min = all_lats.min() - pad
    lat_max = all_lats.max() + pad
    bbox = (lon_min, lon_max, lat_min, lat_max)
    return bbox


def track_snapshots(matched_tracks, datasets, skip=5):
    """
    A plot with two rows showing the spatial distribution of windspeeds
    in prediction and target, with crosses marking the cyclone centers found
    by the tracker. The time difference between snapshots is controlled by
    skip.
    """
    bbox = bounding_box(matched_tracks)
    all_steps = matched_tracks["target"].index.union(matched_tracks["prediction"].index)
    selsteps = np.arange(0, len(all_steps), skip)
    plotdat = xr.concat(datasets.values(), dim=datasets.keys()).isel(valid_time=selsteps)
    plotdat = plotdat.sel(longitude=slice(bbox[0], bbox[1]), latitude=slice(bbox[2], bbox[3]))
    speed = np.sqrt(plotdat.u10**2 + plotdat.v10**2)
    p = speed.plot(
        row="concat_dim", col="valid_time", subplot_kws=dict(projection=ccrs.PlateCarree())
    )
    for ax in p.axs.flatten():
        ax.coastlines()
        ax.set_extent(bbox)
    for i, s in enumerate(all_steps[selsteps]):
        leadtime = plotdat.forecast_period[i].values / np.timedelta64(1, "h")
        p.axs[0, i].set_title(s)
        p.axs[1, i].set_title(f"{leadtime}h forecast")
        for j, tr in enumerate(matched_tracks.values()):
            if s in tr.index:
                tr.loc[[s]].plot.scatter(
                    x="lon", y="lat", ax=p.axs[j, i], color="tab:red", marker="x", s=100
                )
    return p
