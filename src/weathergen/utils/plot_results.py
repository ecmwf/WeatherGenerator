import os
from pathlib import Path
import argparse
import code

# code.interact(local=locals())

import sys
import pdb
import traceback

import zarr

import pandas as pd
import cartopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import time, os, re
import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


from weathergen.utils.config import Config

dpi = 300

idx_lats = 6
idx_lons = 7


def define_regional_mask(region, targets_coords):
    lats = targets_coords[..., idx_lats].flatten()
    if region:
        if region == "nh":
            mask = lats >= 20
        elif region == "tr":
            mask = (lats > -20) & (lats < 20)
        elif region == "sh":
            mask = lats <= 20
        elif region == "global":
            mask = np.ones_like(lats, dtype=bool)
    else:
        mask = np.ones_like(lats, dtype=bool)

    return mask


####################################################################################################
def plot_maps(
    cf, reportypes, targets_coords, values, masks_nan, title, cmap="bwr", root_err=False
):

    for obs_idx in range(len(values)):
        for ch in range(values[obs_idx].shape[-1]):

            vs = values[obs_idx][..., ch][masks_nan[obs_idx][..., ch]].flatten()
            tcs = np.array(targets_coords[obs_idx])[masks_nan[obs_idx][..., ch]]
            ts = tcs.shape
            t_coords = np.reshape(tcs, [-1, ts[-1]])

            lats = np.clip(
                np.floor((90.0 - t_coords[..., idx_lats])).astype(np.int32), 0, 179
            ).flatten()
            lons = np.clip(
                np.floor((180.0 + t_coords[..., idx_lons])).astype(np.int32), 0, 359
            ).flatten()

            # rasterize map
            map = np.zeros((180, 360))
            map_norm = np.ones((180, 360))
            for v, lat, lon in zip(vs, lats, lons):
                map[lat, lon] += v
                map_norm[lat, lon] += 1
            map = map / map_norm
            map = np.sqrt(map) if root_err else map

            vmin, vmax = map.min(), map.max()
            # divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)

            # translate channel label
            chp = ch + 1  # if cf.loss_chs is None else cf.loss_chs[obs_idx][ch]+1

            fig_title = f"average {title} for reportype={reportypes[obs_idx][1]} for channel={chp}"
            rtstr = (
                "{}".format(reportypes[obs_idx][1])
                .replace(" ", "_")
                .replace("-", "_")
                .replace(",", "")
            )
            fname = fig_path + "/{}/map_{}_ch{}_{}_{:05d}_{}.png".format(
                rtstr, rtstr, chp, cf.run_id, epoch, title
            )

            fig = plt.figure(figsize=(10, 5), dpi=dpi)
            ax = plt.axes(projection=cartopy.crs.Robinson())
            ax.add_feature(cartopy.feature.COASTLINE)
            ax.set_global()
            ax.set_title(fig_title)
            im = ax.imshow(
                map,
                cmap=cmap,
                transform=cartopy.crs.PlateCarree(),
                vmin=vmin,
                vmax=vmax,
            )
            # im = ax.imshow( map, cmap=cmap, transform=cartopy.crs.PlateCarree(), norm=divnorm)
            axins = inset_axes(
                ax, width="80%", height="5%", loc="lower center", borderpad=-2
            )
            clb = fig.colorbar(im, cax=axins, orientation="horizontal")
            clb.ax.yaxis.set_label_position("right")
            clb.ax.set_ylabel("[K]", rotation=90)

            plt.savefig(fname)
            plt.close()


####################################################################################################
def plotExampleMap(
    reportype,
    fstep,
    step,
    fig_name,
    date,
    map_data,
    map_coords,
    ch,
    chp,
    err,
    vmin=None,
    vmax=None,
    cmap="bwr",
    cmap_symmetric=False,
):

    if map_data.shape[0] == 0:
        return

    tc = map_coords
    data = map_data[..., ch].flatten()
    lats = tc[..., idx_lats].flatten()
    lons = tc[..., idx_lons].flatten()
    err = "{:3.3f}".format(err)
    # title = f'{fig_name} : {reportype}::{chp} err={err} ({np.datetime_as_string( date, unit='m')})'
    title = f"{fig_name} : {reportype}::{chp} err={err}"  # ({np.datetime_as_string( date, unit='m')})'
    fname = fig_path + "/example/{}/{}_{:05d}_{}_{}_{}_{:03d}_{:03d}.png".format(
        reportype, cf.run_id, epoch, reportype, chp, fig_name, fstep, step
    )
    print(fname)
    plotScatter(fname, title, lons, lats, data, cmap, vmin, vmax, cmap_symmetric)


##########################################################################################
def plotScatterMPL(
    filename,
    title,
    lats,
    lons,
    values,
    cmap="RdYlBu_r",
    vmin=None,
    vmax=None,
    cmap_symmetric=False,
):

    fig = plt.figure(figsize=(10, 5), dpi=dpi)
    ax = plt.axes(projection=cartopy.crs.Robinson())
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.set_global()
    ax.set_title(title)

    # TwoSlopeNorm allows for asymetric scaling with center at 0 but requires two color maps
    # divnorm = colors.TwoSlopeNorm(vmin=values.min(), vcenter=0., vmax=values.max())
    #                                                                 if cmap_symmetric else None
    divnorm = colors.CenteredNorm() if cmap_symmetric else None

    im = plt.scatter(
        x=lons,
        y=lats,
        c=values,
        cmap=cmap,
        s=0.25,
        alpha=1.0,
        vmin=vmin,
        vmax=vmax,
        norm=divnorm,
        transform=cartopy.crs.PlateCarree(),
    )
    axins = inset_axes(ax, width="80%", height="5%", loc="lower center", borderpad=-2)
    clb = fig.colorbar(im, cax=axins, orientation="horizontal")
    # clb.ax.yaxis.set_label_position('right')
    # clb.ax.set_ylabel('[K]', rotation=90 )

    plt.savefig(filename)
    plt.close()


##########################################################################################
def plotScatterDSH(
    filename,
    title,
    lats,
    lons,
    values,
    cmap="RdYlBu_r",
    vmin=None,
    vmax=None,
    cmap_symmetric=False,
):

    import datashader as dsh
    from datashader.mpl_ext import dsshow

    # set projection
    projection = ccrs.Robinson()

    # transform radians to geodetic coordinates
    coords = projection.transform_points(ccrs.Geodetic(), lons, lats)

    # create data frame of the variable values and the geodetic coordinates.
    df = pd.DataFrame({"val": values, "x": coords[:, 0], "y": coords[:, 1]})

    # create the plot
    fig, ax = plt.subplots(
        figsize=(30, 30), facecolor="white", subplot_kw={"projection": projection}
    )
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.set_global()
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        color="black",
        linewidth=0.5,
        alpha=0.7,
        x_inline=False,
    )
    gl.xlocator = mticker.FixedLocator(range(-180, 180 + 1, 60))
    gl.xformatter = LONGITUDE_FORMATTER

    artist = dsshow(
        df,
        dsh.Point("x", "y"),
        dsh.mean("val"),
        vmin=vmin if vmin is not None else values.min(),
        vmax=vmax if vmax is not None else values.max(),
        cmap=cmap,
        plot_width=400,  # #1600,
        plot_height=350,  # 1400,
        ax=ax,
    )

    fig.colorbar(artist, label=" ", shrink=0.3, pad=0.02)
    ax.set_global()

    plt.title(title, fontsize=18)
    plt.figtext(0.75, 0.33, "© 2024 ECMWF", ha="right", fontsize=10)
    plt.savefig(filename, bbox_inches="tight", dpi=dpi)
    plt.close()


####################################################################################################
def plotExampleMaps(
    cf,
    fstep,
    reportypes_active,
    cols_all,
    sources_all,
    targets_all,
    preds_all,
    targets_coords_all,
    targets_idxs_all,
    targets_lens_all,
    bidx=0,
    with_ens=False,
):

    # extract info from bidx-th batch
    sources, sources_coords, targets, preds, targets_coords, targets_idxs = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for k, (rt_idx, rt_name) in enumerate(reportypes_active):

        name = rt_name.replace(" ", "_").replace("-", "_").replace(",", "")
        print(f"example name: {name}")
        sources = sources_all[k][bidx][..., -len(cols_all[k]) :]
        sources_coords = sources_all[k][bidx][..., : -len(cols_all[k])]
        targets = targets_all[k][: targets_lens_all[k][bidx]]
        preds = preds_all[k][: targets_lens_all[k][bidx]]
        targets_coords = targets_coords_all[k][: targets_lens_all[k][bidx]]

        # for GEOs filter out a single time step/slice (matching year, day-of-the-year, and minute)
        if (
            "METEOSAT" in cf.streams[rt_idx]["name"]
            or "Surface" in cf.streams[rt_idx]["name"]
        ):

            # use only last step from source
            # idx = 0
            # mask = np.logical_and( np.logical_and( sources_coords[...,1] == sources_coords[...,1],
            #                                        sources_coords[...,2] == sources_coords[...,2]),
            #                                        sources_coords[...,3] == sources_coords[...,3])
            # sources = sources[mask]
            # sources_coords = sources_coords[mask]

            # dates = targets_coords[:,[1,2,3,3]] * np.array([1.,1.,1./60.,1./60.])
            # dates[:,2] = np.floor( dates[:,2])
            # dates[:,3] = np.remainder( dates[:,3], 1) * 60.
            # dates_str = ['{:0.0f}-{:0.0f}-{:0.0f}-{:0.0f}'.format( *aa) for aa in dates]
            # dates = pd.to_datetime( dates_str, format='%Y-%j-%H-%M')
            # dates_unique, idxs_unique = np.unique( dates, return_index=True)
            dates_unique, idxs_unique = np.unique(
                targets_coords[:, 1], return_index=True
            )

            # if len(idxs_unique) > cf.len_hrs :
            #   idxs = np.argsort( idxs_unique)
            #   # TODO: account for number of measurements in h
            #   idxs_unique = idxs_unique[ idxs[:cf.len_hrs] ]
            #   dates_unique = idxs_unique[ idxs[:cf.len_hrs] ]

            vmin = np.array(
                [np.nanmin(targets[:, i]) for i in range(targets.shape[-1])]
            )
            vmax = np.array(
                [np.nanmax(targets[:, i]) for i in range(targets.shape[-1])]
            )
            errs = []

            for i, (date, idx) in enumerate(zip(dates_unique, idxs_unique)):

                mask = np.logical_and(
                    np.logical_and(
                        targets_coords[idx, 1] == targets_coords[:, 1],
                        targets_coords[idx, 2] == targets_coords[:, 2],
                    ),
                    targets_coords[idx, 3] == targets_coords[:, 3],
                )

                errs_t = plotExampleMapsStream(
                    name,
                    fstep,
                    i,
                    cols_all[k],
                    sources,
                    sources_coords,
                    targets[mask],
                    preds[mask],
                    targets_coords[mask],
                    date,
                    vmin,
                    vmax,
                )
                errs += [errs_t]

                # only plot source in first step
                sources = np.array([])

        else:

            # dates = targets_coords[:,[1,2,3,3]] * np.array([1.,1.,1./60.,1./60.])
            # dates[:,2] = np.floor( dates[:,2])
            # dates[:,3] = np.remainder( dates[:,3], 1) * 60.
            # dates_str = ['{:0.0f}-{:0.0f}-{:0.0f}-{:0.0f}'.format( *aa) for aa in dates]
            # dates = pd.to_datetime( dates_str, format='%Y-%j-%H-%M')

            plotExampleMapsStream(
                name,
                fstep,
                0,
                cols_all[k],
                sources,
                sources_coords,
                targets,
                preds,
                targets_coords,
                None,
            )
            #  targets, preds, targets_coords, dates[0].to_numpy())

            # # compute error as a function of lead time
            # start = dates.min()
            # end = dates.min() + pd.Timedelta(minutes=60)
            errs = []
            # for i in range( cf.len_hrs) :
            #   mask = np.logical_and( dates >= start, dates <= end)
            #   w = np.where( mask)[0]
            #   errs_t = []
            #   for i_ch in range(targets.shape[-1]) :
            #     errs_t += [ np.sqrt( np.nanmean( np.square( targets[w,i_ch] - preds[w,0,i_ch]), axis=0))]
            #   errs += [ errs_t ]
            #   start = end
            #   end = start + pd.Timedelta(minutes=60)
            # errs += [0]

        if len(errs) > 0:
            # plot error as a function of lead time
            errs = np.array(errs)
            # remove "outlier" channels to better plot trends (log plot is another option but the
            # the trends are less clear there)
            idxs = np.array(
                [
                    i
                    for i in range(errs.shape[-1])
                    if np.nanmax(errs[:, i]) / 1000.0 <= np.nanmin(errs)
                ]
            )
            fig = plt.figure(figsize=(10, 7.5), dpi=dpi)
            plt.plot(errs[:, idxs], "-x")
            plt.legend(np.array(cols_all[k])[idxs].tolist(), loc="upper left")
            plt.title(f"{name} : avg error for lead time")
            plt.grid(True, which="both", ls="-")
            fig_name = fig_path + "/example/{}/{}_{}_error_lead_time.png".format(
                name, cf.run_id, name
            )
            plt.savefig(fig_name)
            plt.close()


####################################################################################################
def plotExampleMapsStream(
    name,
    fstep,
    step,
    cols_all,
    sources,
    sources_coords,
    targets,
    preds,
    targets_coords,
    str_date="",
    vmins=None,
    vmaxs=None,
):

    # str_dt= '{:0.0f}-{:0.0f}-{:0.0f}'.format( *targets_coords[0][i][[1,2,4]]*np.array([1.,1.,1./60.]))
    # date_start = pd.to_datetime( str_dt, format='%Y-%j-%H')
    # date_end = date_start + pd.Timedelta( cf.len_hrs, "h")
    # str_date = f'{date_start} - {date_end}'

    print(f"plotExampleMapsStream {name}")
    if sources.shape[0] < 1 and targets.shape[0] < 1:
        return

    errs = []
    for ch in range(len(cols_all)):

        # if '10u' != cols_all[ch] :
        #   continue

        # translate channel label
        chp = cols_all[ch]
        cmap = "RdYlBu_r"

        n_chs = targets.shape[-1]
        print(
            "Processing {}, fstep={}, step={} :: ch={} ({}/{}) :: {} / {} / {}.".format(
                name,
                fstep,
                step,
                chp,
                ch,
                n_chs,
                sources.shape,
                targets.shape,
                preds.shape,
            )
        )

        mask_nan = ~np.isnan(targets[..., ch])
        if 0 == targets[mask_nan, ch].shape[0]:
            return
        vmin = np.min(targets[mask_nan, ch]) if vmins is None else vmins[ch]
        vmax = np.max(targets[mask_nan, ch]) if vmaxs is None else vmaxs[ch]
        err = np.sqrt(
            np.nanmean(np.square(targets[:, ch] - preds.mean(1)[:, ch]), axis=0)
        )

        if sources.shape[0] > 0:
            plotExampleMap(
                name,
                fstep,
                step,
                "source",
                str_date,
                sources,
                sources_coords,
                ch,
                chp,
                err,
                vmin,
                vmax,
                cmap,
            )

        plotExampleMap(
            name,
            fstep,
            step,
            "target",
            str_date,
            targets,
            targets_coords,
            ch,
            chp,
            err,
            vmin,
            vmax,
            cmap,
        )

        # targets_nan = targets[~mask_nan]
        # targets_nan[...] = 1.
        # plotExampleMap( name, step, 'target_nan', str_date, targets_nan,
        #                 targets_coords[~mask_nan], ch, chp, vmin, vmax, cmap)

        plotExampleMap(
            name,
            fstep,
            step,
            "prediction",
            str_date,
            preds.mean(1),
            targets_coords,
            ch,
            chp,
            err,
            vmin,
            vmax,
            cmap,
        )

        if with_ens:
            for i_ens in range(preds.shape[1]):
                plotExampleMap(
                    name,
                    fstep,
                    step,
                    "prediction_ens{:02d}".format(i_ens),
                    str_date,
                    preds[:, i_ens],
                    targets_coords,
                    ch,
                    chp,
                    vmin,
                    vmax,
                    cmap,
                )

        diff = targets - preds.mean(1)
        plotExampleMap(
            name,
            fstep,
            step,
            "difference",
            str_date,
            diff,
            targets_coords,
            ch,
            chp,
            err,
            None,
            None,
            cmap="bwr",
            cmap_symmetric=True,
        )

        errs += [err]

    return errs


####################################################################################################
def compute_mean_space_time(preds, targets, targets_coords_all, mask_nan, ch=0):

    # example below:
    # compute lon \in [0,10] and all time steps for diurnal cycle example

    preds = np.mean(preds, axis=1)
    targets = np.array(targets)

    hours_of_day = targets_coords_all[-1][:, 2]
    hours_of_day_unique = np.unique(hours_of_day)

    lons = targets_coords_all[-1][:, 7]
    mask = np.logical_and(lons > 0.0, lons < 10.0)

    masks_lon_time = []
    for tt in hours_of_day_unique:
        mask_t = hours_of_day == tt
        masks_lon_time += [np.logical_and(mask, mask_t)]

    preds_t_mean, targets_t_mean, rmse_t = [], [], []
    for mask in masks_lon_time:
        p = preds[mask, ch][mask_nan[mask, ch]]
        t = targets[mask, ch][mask_nan[mask, ch]]
        preds_t_mean += [p.mean()]
        targets_t_mean += [t.mean()]
        rmse_t += [np.square(t - p).mean()]
    preds_t_mean = np.array(preds_t_mean)
    targets_t_mean = np.array(targets_t_mean)
    rmse_t = np.sqrt(np.array(rmse_t))

    print(f"compute_mean_space_time : ch={ch}")
    print(f"hours_of_day : {hours_of_day_unique // 60}")
    print(f"preds mean: {preds_t_mean}")
    print(f"targets mean: {targets_t_mean}")
    print(f"rmse: {rmse_t}")
    print("", flush=True)


####################################################################################################
def plot_scanlines(
    streams,
    reportypes_active,
    targets_all,
    targets_coords_all,
    num_samples=512,
    cmap="bwr",
):

    streams = [streams] if type(streams) is not list else streams

    for stream in streams:

        idx = [i for i, c in enumerate(reportypes_active) if stream in c][0]

        vmin = targets_all[idx][:num_samples, 0].min()
        vmax = targets_all[idx][:num_samples, 0].max()
        for i in range(num_samples):

            fig = plt.figure(figsize=(10, 5), dpi=dpi)
            ax = plt.axes(projection=cartopy.crs.Robinson())
            ax.add_feature(cartopy.feature.COASTLINE)
            ax.set_global()

            lats = targets_coords_all[idx][:i, idx_lats]
            lons = targets_coords_all[idx][:i, idx_lons]
            values = targets_all[idx][:i, 0]

            im = plt.scatter(
                x=lons,
                y=lats,
                c=values,
                cmap=cmap,
                s=1,
                alpha=1.0,
                vmin=vmin,
                vmax=vmax,
                transform=cartopy.crs.PlateCarree(),
            )

            str_stream = stream.replace(",", "").replace("/", "_").replace(" ", "_")
            filename = f"./plots/{str_stream}" + "_{0:05d}.png".format(i)
            plt.savefig(filename)
            plt.close()

        print(f"Finished {stream}.", flush=True)


####################################################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-id", "--run_id", required=True)
    parser.add_argument("-e", "--epoch", default="0", type=int)
    parser.add_argument("--channel_errors", default=False, type=bool)
    parser.add_argument("--avg_plots", default=False, type=bool)
    parser.add_argument("-s", "--example_samples", nargs="+", default=[])
    parser.add_argument("--ens", default=False, type=bool)
    parser.add_argument("-rt", "--reportypes", nargs="+", default=[])
    parser.add_argument(
        "-fs",
        "--forecast_steps",
        nargs="+",
        default=[],
        help="forecast steps as list or range",
    )
    parser.add_argument("--region", default="global")
    args = parser.parse_args()

    # plotScatter = plotScatterDSH
    plotScatter = plotScatterMPL

    run_id = args.run_id
    cf = Config.load(run_id)
    # cf.print()
    names = [s["name"] for s in cf.streams]
    base_path = "/work/ab0995/a270225/WeatherGenerator2/results/{}/".format(run_id)

    epoch = args.epoch
    with_channel_errors = args.channel_errors
    with_avg_plots = args.avg_plots
    with_ens = args.ens
    example_samples = [int(i) for i in args.example_samples]
    reportypes = args.reportypes
    region = args.region
    if with_avg_plots:
        with_channel_errors = True

    if len(args.forecast_steps) == 1 and "-" in args.forecast_steps[0]:
        fsteps = np.arange(
            int(args.forecast_steps[0].split("-")[0]),
            int(args.forecast_steps[0].split("-")[1]) + 1,
        )
    else:
        # fsteps = cf.forecacast_steps if len(args.forecast_steps)==0 else args.forecast_steps
        fsteps = [0] if len(args.forecast_steps) == 0 else args.forecast_steps
        fsteps = np.array(fsteps) if type(fsteps) == 0 else fsteps

    fsteps = [0]  # list( np.arange(4*6))

    # run_id, epoch = 'abcd', 0
    # with_plots = True
    print(f"Processing run_id={run_id} at epoch={epoch}")

    for fstep in fsteps:

        print(f"Processing fstep={fstep}.")

        # read data
        targets_all, preds_all, targets_coords_all, sources_all, targets_idxs_all = (
            [],
            [],
            [],
            [],
            [],
        )
        sources_coords_all, sources_lens_all, targets_lens_all = [], [], []
        cols_all = []
        rank = 0
        fname = base_path + "validation_epoch{:05d}_rank{:04d}.zarr".format(epoch, rank)
        print(fname)
        store = zarr.DirectoryStore(fname)
        ds = zarr.group(store=store)
        reportypes_active = []

        for ii, n in enumerate(names):
            if len(reportypes):
                if not np.array([r in n for r in reportypes]).any():
                    continue
            print(f"Using reportype: {n}")
            reportype = n.replace(" ", "_").replace("-", "_").replace(",", "")
            reportypes_active.append((ii, n))
            cols_all.append(ds[f"{reportype}/{fstep}"].attrs["cols"])
            sources_all.append(ds[f"{reportype}/{fstep}/sources"])
            sources_lens_all.append(ds[f"{reportype}/{fstep}/sources_lens"])
            preds_all.append(ds[f"{reportype}/{fstep}/preds"])
            targets_all.append(ds[f"{reportype}/{fstep}/targets"])
            targets_coords_all.append(ds[f"{reportype}/{fstep}/targets_coords"])
            targets_lens_all.append(ds[f"{reportype}/{fstep}/targets_lens"])

        masks_nan = [~np.isnan(target_data) for target_data in targets_all]

        fig_path = base_path + "figures/" + "epoch{:05d}".format(epoch) + "/"
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            os.makedirs(fig_path + "example")
            for i in range(len(names)):
                rt = names[i].replace(" ", "_").replace("-", "_").replace(",", "")
                os.makedirs(fig_path + f"/{rt}")
                os.makedirs(fig_path + f"/example/{rt}")

        if with_channel_errors:

            # std-dev and mean of ensemble
            preds_mean_all, preds_std_all = [], []
            for i, preds in enumerate(preds_all):
                preds_mean_all += [np.mean(preds, axis=1)]
                preds_std_all += [np.std(preds, axis=1)]
                print(f"Finished mean/std for {i+1} / {len(preds_all)}.", flush=True)

            # print basic error statistics
            print("reportype::channel : mse / mae / rmse / ens-spread / rel. mse:")
            for obs_idx, (ii, reportype) in enumerate(reportypes_active):

                fname = base_path + "/analysis_{}_{}_rmse".format(run_id, epoch)
                fname += "_{}.csv".format(
                    reportypes_active[obs_idx][1]
                    .replace(" ", "_")
                    .replace("-", "_")
                    .replace(",", "")
                )
                file_results = open(fname, "w" if fstep == fsteps[0] else "a")
                # TODO: add header

                cols = cols_all[obs_idx]
                cols = (
                    [c[9:] for c in cols if c[:9] == "obsvalue_"]
                    if cf.streams[obs_idx]["type"] == "obs"
                    else cols
                )

                p = preds_mean_all[obs_idx]
                var = preds_std_all[obs_idx]
                t = targets_all[obs_idx]
                mask_nan = masks_nan[obs_idx]
                mask_regional = define_regional_mask(
                    region, targets_coords_all[obs_idx]
                )

                chs_rmse = np.zeros(p.shape[-1])
                for ch in range(p.shape[-1]):
                    chp = ch + 1  # if cf.loss_chs is None else cf.loss_chs[obs_idx][ch]
                    if not np.all(~mask_nan[..., ch]):
                        m = np.logical_and(mask_nan[..., ch], mask_regional)
                        err_mse = np.square(p[..., ch][m] - t[..., ch][m]).mean()
                        err_mae = np.abs(p[..., ch][m] - t[..., ch][m]).mean()
                        err_rmse = np.sqrt(err_mse)
                        std_dev = var[..., ch][m].mean()
                        err_mse_rel = err_mse / np.square(t[..., ch][m]).mean()
                        print(
                            "{} : ch={} : \t {:4E} / {:4E} / {:4E} / {:4E} / {:4E}".format(
                                reportype,
                                cols[ch],
                                err_mse,
                                err_mae,
                                err_rmse,
                                std_dev,
                                err_mse_rel,
                            )
                        )
                        chs_rmse[ch] = err_rmse
                    else:
                        print("{} : ch={} : \t - / - / - / -".format(reportype, chp))
                        chs_rmse[ch] = np.nan
                np.savetxt(file_results, chs_rmse)

                if "Surface" in reportype:
                    m = np.logical_and(mask_nan[..., 2], mask_nan[..., 3])
                    m = np.logical_and(m, mask_regional)
                    l1 = np.sqrt(np.square(p[..., 2][m]) + np.square(p[..., 3][m]))
                    l2 = np.sqrt(np.square(t[..., 2][m]) + np.square(t[..., 3][m]))
                    err_mse = np.square(l1 - l2).mean()
                    err_mae = np.abs(l1 - l2).mean()
                    err_rmse = np.sqrt(np.square(l1 - l2).mean())
                    print(
                        "{} : ch=10ff : \t {:4E} / {:4E} / {:4E}".format(
                            reportype, err_mse, err_mae, err_rmse
                        )
                    )

            print("", flush=True)
            file_results.close()

        if with_avg_plots:
            print("Plotting average maps.", flush=True)

            # std-dev maps
            plot_maps(
                cf,
                reportypes_active,
                targets_coords_all,
                preds_std_all,
                masks_nan,
                "ens-std",
            )
            print("Finished stddev maps.", flush=True)

            # mse maps
            errs_all = []
            for i in range(len(targets_all)):
                shape = preds_mean_all[i].shape
                p = preds_mean_all[i]
                t = targets_all[i]
                errs_chs = []
                for ch in range(p.shape[-1]):
                    errs_chs += [
                        np.expand_dims(
                            np.square(p[..., ch] - t[..., ch]), axis=len(p.shape) - 1
                        )
                    ]
                errs_all.append(np.concatenate(errs_chs, len(p.shape) - 1))
            plot_maps(
                cf,
                reportypes_active,
                targets_coords_all,
                errs_all,
                masks_nan,
                "rmse",
                "bwr",
                True,
            )
            # plot_maps( reportypes, targets_coords_all, errs_all, 'rme', 'bwr', True)
            # plot_maps( reportypes, targets_coords_all, errs_all, 'me', 'bwr', True)
            print("Finished error maps.", flush=True)

        if len(example_samples) > 0:
            # split source back to batch items
            # for large #samples, reading and splitting sources_all[i] can be expensive so support the
            # common special case that can be dealt with much more efficiently
            if example_samples == [0]:
                for i in range(len(sources_all)):
                    # TODO: there seems to be an issue in zarr Array that causes np.array(sources_lens_all[i])
                    #       to create a deprecation warning
                    lens = np.array(sources_lens_all[i]).cumsum(0)
                    sources_all[i] = [sources_all[i][: lens[0]]]
            else:
                for i in range(len(sources_all)):
                    sources_all[i] = np.split(
                        sources_all[i], np.array(sources_lens_all[i]).cumsum(0)[:-1]
                    )

        for ex_idx in example_samples:

            try:
                print("Plotting example maps")
                # one batch example maps
                plotExampleMaps(
                    cf,
                    fstep,
                    reportypes_active,
                    cols_all,
                    sources_all,
                    targets_all,
                    preds_all,
                    targets_coords_all,
                    targets_idxs_all,
                    targets_lens_all,
                    bidx=ex_idx,
                    with_ens=with_ens,
                )

            except:
                extype, value, tb = sys.exc_info()
                traceback.print_exc()
                pdb.post_mortem(tb)

    print(f"Finished plotting results for run_id {run_id}")
