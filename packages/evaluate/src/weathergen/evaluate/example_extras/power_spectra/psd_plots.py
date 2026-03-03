"""
Adapted from Martin Willet's code for zonal power spectra for use with the WeatherGenerator model.
It takes 1D Fourier transforms along the longitude dimension of the data, using an
    upper and lower longitude as regional bounds.
It produces log-log plots of the spectra and semilogx plots of the ratio of the spectra
    to a reference (e.g., targets or other model predictions).
The script is designed to be flexible and can be used to plot spectra for different
    diagnostics, regions, and forecast times.
DISCLAIMER: It is NOT suitable for bounded box regions
"""

import logging
import warnings

import iris
import iris.cube
import matplotlib.pyplot as plt
import numpy as np
from psd_calc import calcposfreq, cubepsd

_logger = logging.getLogger(__name__)
warnings.simplefilter(action="ignore", category=FutureWarning)

# A couple of physical constants
g = 9.81  # Acceleration due to gravity. USed to convert GP to GPH.
re = 6.37e6  # earth radius.
re2 = re * re  # earth radius squared

# Define dictionary of regions of interest
regions = {
    "FullGlobe": dict(label="FullGlobe", lonW=0.0, lonE=360.0, latS=-90.0, latN=90.0),
    "ShortGlobe": dict(label="ShortGlobe", lonW=0.0, lonE=360.0, latS=-60.0, latN=60.0),
    "N-Mid-Lats": dict(label="N-Mid-Lats", lonW=0.0, lonE=360.0, latS=30.0, latN=60.0),
    "S-Mid-Lats": dict(label="S-Mid-Lats", lonW=0.0, lonE=360.0, latS=-60.0, latN=-30.0),
    "Tropics": dict(label="Tropics", lonW=0.0, lonE=360.0, latS=-30.0, latN=30.0),
    "Deep_Tropics": dict(label="Deep_Tropics", lonW=0.0, lonE=360.0, latS=-10.0, latN=10.0),
    "NN-Sub-Tropics": dict(label="NN-Sub-Tropics", lonW=0.0, lonE=360.0, latS=30.0, latN=40.0),
    "SS-Sub-Tropics": dict(label="SS-Sub-Tropics", lonW=0.0, lonE=360.0, latS=-40.0, latN=-30.0),
    "NN-Mid-Lats": dict(label="NN-Mid-Lats", lonW=0.0, lonE=360.0, latS=45.0, latN=75.0),
    "SS-Mid-Lats": dict(label="SS-Mid-Lats", lonW=0.0, lonE=360.0, latS=-75.0, latN=-45.0),
    "N-Polar": dict(label="N-Polar", lonW=0.0, lonE=360.0, latS=75.0, latN=90.0),
    "S-Polar": dict(label="S-Polar", lonW=0.0, lonE=360.0, latS=-90.0, latN=-75.0),
}

# Define dictionary of potential diagnostics to plot
diags = {
    "q": {
        "ncvar": "q",
        "ncname": "specific_humidity_at_pressure_levels",
        "std": "specific_humidity",
        "units": "kg kg-1",
        "levtype": "pressure",
        "scale": 100 * g,
        "slope": -5,
        "yscale": 0.1,
    },
    "t": {
        "ncvar": "t",
        "ncname": "temperature_at_pressure_levels",
        "std": "air_temperature",
        "units": "K",
        "levtype": "pressure",
        "scale": 1.0,
        "slope": -3,
        "yscale": 0.1,
    },
    "u": {
        "ncvar": "u",
        "ncname": "u_wind_at_pressure_levels",
        "std": "x_wind",
        "units": "m s-1",
        "levtype": "pressure",
        "scale": 1.0,
        "slope": -3,
        "yscale": 1.0,
    },
    "v": {
        "ncvar": "v",
        "ncname": "v_wind_at_pressure_levels",
        "std": "y_wind",
        "units": "m s-1",
        "levtype": "pressure",
        "scale": 1.0,
        "slope": -3,
        "yscale": 1.0,
    },
    "z": {
        "ncvar": "z",
        "ncname": "geopotential_at_pressure_levels",
        "std": "geopotential",
        "units": "m2 s-2",
        "levtype": "pressure",
        "scale": 0.1 / g,
        "slope": -5,
        "yscale": 1.0,
    },
    "u10": {
        "ncvar": "10u",
        "ncname": "u_wind_at_10m",
        "std": "x_wind",
        "units": "m s-1",
        "levtype": "surface",
        "scale": 1.0,
        "slope": -3,
        "yscale": 1.0,
    },
    "v10": {
        "ncvar": "v10",
        "ncname": "v_wind_at_10m",
        "std": "y_wind",
        "units": "m s-1",
        "levtype": "surface",
        "scale": 1.0,
        "slope": -3,
        "yscale": 1.0,
    },
    "d2m": {
        "ncvar": "d2m",
        "ncname": "dew_point_temperature_at_screen_level",
        "std": "dew_point_temperature",
        "units": "K",
        "levtype": "surface",
        "scale": 1.0,
        "slope": -3,
        "yscale": 1.0,
    },
    "t2m": {
        "ncvar": "t2m",
        "ncname": "temperature_at_screen_level",
        "std": "air_temperature",
        "units": "K",
        "levtype": "surface",
        "scale": 1.0,
        "slope": -3,
        "yscale": 1.0,
    },
    "msl": {
        "ncvar": "msl",
        "ncname": "mean_sea_level_pressure",
        "std": "air_pressure_at_mean_sea_level",
        "units": "Pa",
        "levtype": "surface",
        "scale": 0.01,
        "slope": -5,
        "yscale": 1.0,
    },
    "skt": {
        "ncvar": "skt",
        "ncname": "skin_temperature",
        "std": "sea_surface_skin_temperature",
        "units": "K",
        "levtype": "surface",
        "scale": 1.0,
        "slope": -3,
        "yscale": 1.0,
    },
    "sp": {
        "ncvar": "sp",
        "ncname": "surface_pressure",
        "std": "surface_air_pressure",
        "units": "Pa",
        "levtype": "surface",
        "scale": 0.01,
        "slope": -5,
        "yscale": 1.0,
    },
}


def add_levels(weathergen_diags: dict, plevels: list) -> None:
    """
    Add levels to the diagnostics dictionary.
    Based on whether they are pressure level of surface variables"""
    for var_dict in weathergen_diags:
        if weathergen_diags[var_dict]["levtype"] == "pressure":
            weathergen_diags[var_dict]["levels"] = plevels
        elif weathergen_diags[var_dict]["levtype"] == "surface":
            weathergen_diags[var_dict]["levels"] = [0]


def addwvns(axes: plt.Axes) -> None:
    """
    Adds lines of equal wavenumber to plots
    """
    yscale = axes.yaxis.get_scale()
    ylims = axes.get_ylim()

    if yscale == "log":
        ytxt = 10.0 ** (0.85 * (np.log10(ylims[1] / ylims[0])) + np.log10(ylims[0]))
    else:
        ytxt = 0.85 * (ylims[1] - ylims[0]) + ylims[0]

    wvns = [1, 2, 4, 8, 16, 24, 48, 96, 144, 216, 320, 640, 1280, 2560]
    for wvn in wvns:
        axes.plot(
            np.array([wvn / 360.0, wvn / 360.0]),
            np.array(ylims),
            color="black",
            lw=1.0,
            scalex=False,
            scaley=False,
        )
        axes.text(wvn / 360.0, ytxt, f"n{wvn:3.0f}", rotation="vertical")


def addlengths(axes: plt.Axes, region: dict) -> None:
    """
    Adds lines of equal spatial scale in km to plots
    """

    re = 6.37e6  # earth radius. Used to plot phyical lengths on plots.
    yscale = axes.yaxis.get_scale()
    ylims = axes.get_ylim()

    if yscale == "log":
        ytxt = 10.0 ** (0.05 * (np.log10(ylims[1] / ylims[0])) + np.log10(ylims[0]))
    else:
        ytxt = 0.05 * (ylims[1] - ylims[0]) + ylims[0]

    lengths = np.array([1.0e4, 3.0e3, 1.0e3, 3.0e2, 1.0e2, 3e1, 1e1])

    flengths = (
        2.0
        * np.pi
        * re
        * np.cos((region["latN"] + region["latS"]) / 360.0 * np.pi)
        / (1000.0 * lengths * 360.0)
    )

    for ilength in range(len(lengths)):
        axes.plot(
            np.array([flengths[ilength], flengths[ilength]]),
            np.array(ylims),
            color="black",
            linestyle="dashed",
            lw=1.0,
            scalex=False,
            scaley=False,
        )
        axes.text(flengths[ilength], ytxt, f"{lengths[ilength]:5.0f}km", rotation="vertical")


def addidealslope(
    axes: plt.Axes, slope: float, defxs: list | None = None, defy0: float = 10.0
) -> None:
    """
    Adds an idealised slope to a log-log spectra plot
    """
    if defxs is None:
        defxs = [0.01, 0.1]
    slopexs = np.array(defxs)
    slopeys = defy0 * np.array([1.0, (slopexs[1] / slopexs[0]) ** slope])
    xtxt = np.sqrt(np.prod(slopexs))
    ytxt = np.sqrt(np.prod(slopeys))

    axes.plot(slopexs, slopeys, color="black", lw=2.0, scalex=False, scaley=False)
    axes.text(xtxt, ytxt, "$k^{" + str(slope) + "}$", fontsize="xx-large", weight="bold")


def region_constraint(region: dict) -> tuple[iris.Constraint, iris.Constraint]:
    """
    Given a region definition, returns a longitude and latitude constraint
    """
    # Setup iris constraint to extract data for this region:
    lat_constraint = iris.Constraint(
        latitude=lambda lat: lat >= region["latS"] and lat <= region["latN"]
    )
    # Case where region straddles the Greenwich Meridian:
    if region["lonW"] > region["lonE"]:
        lon_constraint = iris.Constraint(
            longitude=lambda lon: lon >= region["lonW"] or lon <= region["lonE"]
        )
    # Normal case
    else:
        lon_constraint = iris.Constraint(
            longitude=lambda lon: lon >= region["lonW"] and lon <= region["lonE"]
        )
    # end if

    return lat_constraint, lon_constraint


def tidy_plot(axes: plt.Axes, plttitle: str, ylabel: str, ylims: list, region: dict) -> None:
    """
    Add plots stuff common to all plots
    """
    axes.set_title(plttitle)
    axes.set_xlabel("Frequency (1/deg long)")
    axes.set_ylabel(ylabel)
    axes.grid(True, which="major", linewidth=1.0)
    axes.grid(True, which="minor", linewidth=0.5)
    axes.set_xlim(1.0e-3, 1.0e1)
    axes.set_ylim(ylims[0], ylims[1])
    addwvns(axes)
    addlengths(axes, region)


def setuppage():
    plt.rc("figure", figsize=(8.27, 11.69))
    plt.subplots_adjust(hspace=0.3)
    # plt.rcParams['font.size']=11
    plt.rcParams["font.size"] = 13


def plot_psds(
    comparison_dict: dict,
    regkeys: list,
    diagkeys: list,
    usencname: bool = False,
    fc_times: list | None = None,
    fname: str | None = None,
    outdir: str | None = None,
    plevels: list | None = None,
):
    """
    Calculates and plots power spectra
    comparison_dict containing
        testnames  - a list of the names of test
        testfiles  - a list of the filenames - wildcards can be used. One for each test.
    It is assumed that the first of the tests is to be used as the reference.
    regkeys  - a list of keys specifying which regions to produce plots for.
    diagkeys  - a list of keys specifying which diagnostics are required.
    usencname - if True the diagnostic contraint will use the netcdf name.
                If False (default) stash code will be used.
    fctimes - an optional 2d-array containing the forecast-times for each plot.
    fname   - optional prefix for plot filename.
    outdir  - optional output directory.
    """

    failed_string = ""

    if fname is None:
        fname = ""

    if fc_times is None:
        n_fc_times = 1
    else:
        n_fc_times = len(fc_times[:])

    # Setup some standard settings.
    loglog_ylims = np.array([1.0e-5, 1.0e2])
    # loglog_ylims   = np.array([1.e-5, 1.e3])
    # semilogx_ylims = [0.0, 2.0]
    semilogx_ylims = [0.0, 3.0]
    colors = ["b", "r", "m", "c", "g", "orange"]

    # prep diag_keys
    if plevels is None:
        plevels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    add_levels(diags, plevels)

    if regions is None:
        regkeys = ["ShortGlobe", "N-Mid-Lats", "S-Mid-Lats", "Tropics"]

    if diagkeys is None:
        diagkeys = ["q", "t", "u", "v", "z", "t2m", "msl", "u10", "v10", "d2m", "skt", "sp"]

    for diagkey in diagkeys:
        # For each diagnostic...
        diag = diags[diagkey]
        if usencname:
            if diag["ncname"] is not None:
                field_constraint = diag["ncname"]
            else:
                field_constraint = None

        _logger.debug("field_constraint is:", field_constraint)
        scale = diag["scale"]
        levels = diag["levels"]
        n_levels = len(levels)

        for regkey in regkeys:
            # For each region...
            region = regions[regkey]

            # Calc lat and lon contraint from region
            lat_constraint, lon_constraint = region_constraint(region)

            # for level in np.nditer(levels):
            for i_level in range(n_levels):
                level = levels[i_level]
                if diag["levtype"] == "pressure":
                    lev_constraint = iris.Constraint(pressure=level)
                elif diag["levtype"] == "mlevels":
                    lev_constraint = iris.Constraint(model_level_number=level)

                for i_fc_time in range(n_fc_times):
                    plttitle = diag["ncname"] + ": "
                    figname = fname + diag["ncname"] + "_"

                    if diag["levtype"] == "pressure":
                        plttitle += str(level) + "hPa: "
                        figname += str(level) + "_"
                    elif diag["levtype"] == "mlevels":
                        plttitle += "ML" + str(level) + ": "
                        figname += str(level) + "_"

                    plttitle += region["label"]
                    figname += region["label"]

                    if fc_times is not None:
                        fc_time = fc_times[i_fc_time]
                        desired_time = iris.Constraint(forecast_period=fc_time)
                        plttitle += "T" + str(fc_time)
                        figname += "T" + str(fc_time)
                    else:
                        fc_time = None

                    figname += "_spectra.png"
                    _logger.info("Creating " + plttitle)

                    # Initialise plot page
                    setuppage()

                    n_test = len(comparison_dict.keys())
                    # for testkey in testkeys:
                    for i_test, comp_key in enumerate(comparison_dict):
                        testname = comp_key
                        testfiles = comparison_dict[comp_key]
                        ############## Read data ##########################
                        _logger.debug("About to read field")

                        num_failed_runs = 0

                        try:
                            psds = []
                            for testfile in testfiles:
                                field = iris.load(testfile, field_constraint)

                                _logger.debug(field)

                                coord_names = [coord.name() for coord in field[0].coords()]
                                _logger.debug(coord_names)

                                tot_constraint = lat_constraint & lon_constraint

                                if "air_pressure" in coord_names:
                                    for cube in field:
                                        cube.coord("air_pressure").rename("pressure")

                                if diag["levtype"] == "pressure" or diag["levtype"] == "mlevels":
                                    tot_constraint = tot_constraint & lev_constraint

                                if "forecast_period" in coord_names and fc_time is not None:
                                    _logger.debug(field[0].coord("forecast_period"))
                                    for cube in field:
                                        cube.coord("forecast_period").convert_units("hours")
                                    _logger.debug(
                                        "forecast_period:", field[0].coord("forecast_period")
                                    )

                                if "forecast_period" in coord_names and fc_time is not None:
                                    # create the constraint
                                    tot_constraint = tot_constraint & desired_time
                                _logger.debug("Tot_constraint:", tot_constraint)
                                field = field.extract(tot_constraint)
                                _logger.debug("Completed read field")

                                ############## PSD calcs ##########################
                                # Create frequencies
                                posfreq = calcposfreq(field[0])

                                # Calculate PSD
                                field_psd = scale * scale * cubepsd(field, dimension="longitude")
                                psds.append(field_psd)

                            _logger.debug("Averaging PSDs over all samples for one forecast_time")
                            field_psd = np.mean(psds, axis=0)

                            if i_test == 0:
                                # Take a copy of the data
                                ref_psd = np.copy(field_psd)

                            _logger.debug("Completed calc PSDs")

                            ############## Plotting ##########################
                            plt.subplot(2, 1, 1)
                            plt.loglog(posfreq, field_psd, color=colors[i_test], label=testname)

                            if i_test == n_test - 1:
                                # last test. Add plt stuff
                                _logger.debug(plttitle)
                                tidy_plot(
                                    plt.gca(),
                                    plttitle + ": zonal spectra",
                                    "Power ((" + diag["units"] + ")^2 deg)",
                                    diag["yscale"] * loglog_ylims,
                                    region,
                                )
                                plt.legend(loc="lower left")

                                # Add idealised slopes
                                addidealslope(
                                    plt.gca(), float(diag["slope"]), defy0=10.0 * diag["yscale"]
                                )

                            plt.subplot(2, 1, 2)
                            plt.semilogx(
                                posfreq, field_psd / ref_psd, color=colors[i_test], label=testname
                            )

                            if i_test == n_test - 1:
                                tidy_plot(
                                    plt.gca(),
                                    plttitle + ": ratio of zonal spectra",
                                    "Power ratio",
                                    semilogx_ylims,
                                    region,
                                )
                                plt.legend()
                        except Exception as e:
                            _logger.error(e)
                            _logger.error(f"Plotting power spectra failed for {testfile}")
                            failed_string += f"{testfile}"
                            num_failed_runs += 1
                    plt.savefig(outdir / figname)
                    plt.close()

    _logger.info(f"{num_failed_runs} runs failed: {failed_string}")
