import datetime
import logging
import os
import warnings
from pathlib import Path

import cartopy
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import omegaconf as oc
import xarray as xr
from astropy_healpix import HEALPix as HEALPixGrid
from cartopy.io import DownloadWarning
from matplotlib.collections import LineCollection

from weathergen.common.config import _load_private_conf
from weathergen.evaluate.plotting.plot_utils import DefaultMarkerSize, format_datetime
from weathergen.evaluate.utils.regions import RegionBoundingBox

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

work_dir = Path(_load_private_conf(None)["path_shared_working_dir"]) / "assets/cartopy"

cartopy.config["data_dir"] = str(work_dir)
cartopy.config["pre_existing_data_dir"] = str(work_dir)
os.environ["CARTOPY_DATA_DIR"] = str(work_dir)

# Route Cartopy DownloadWarnings through the logging system so they are visible in logs.
logging.captureWarnings(True)
warnings.filterwarnings("always", category=DownloadWarning)


def _download_cartopy_off(enabled: bool) -> None:
    """Enable/disable blocking Cartopy downloads by elevating DownloadWarning to error."""
    if enabled:
        warnings.filterwarnings("error", category=DownloadWarning)
        _logger.info(
            "Auto-downloads are blocked for cartopy; only local cartopy data will be used."
        )
    else:
        warnings.filterwarnings("default", category=DownloadWarning)


np.seterr(divide="ignore", invalid="ignore")

logging.getLogger("matplotlib.category").setLevel(logging.ERROR)

_logger.debug(f"Taking cartopy paths from {work_dir}")


class Plotter:
    """
    Contains all basic plotting functions.
    """

    def __init__(self, plotter_cfg: dict, output_basedir: str | Path, stream: str | None = None):
        """
        Initialize the Plotter class.

        Parameters
        ----------
        plotter_cfg:
            Configuration dictionary containing basic information for plotting.
            Expected keys are:
                - image_format: Format of the saved images (e.g., 'png', 'pdf', etc.)
                - dpi_val: DPI value for the saved images
                - fig_size: Size of the figure (width, height) in inches
                - tokenize_spacetime: If True, all valid times will be plotted in one plot
        output_basedir:
            Base directory under which the plots will be saved.
            Expected scheme `<results_base_dir>/<run_id>`.
        stream:
            Stream identifier for which the plots will be created.
            It can also be set later via update_data_selection.
        """

        _logger.info(f"Taking cartopy paths from {work_dir}")

        self.image_format = plotter_cfg.get("image_format")
        self.animation_format = plotter_cfg.get("animation_format")
        self.dpi_val = plotter_cfg.get("dpi_val")
        self.fig_size = plotter_cfg.get("fig_size")
        self.fps = plotter_cfg.get("fps")
        self.regions = plotter_cfg.get("regions")
        _download_cartopy_off(enabled=True)
        self.plot_subtimesteps = plotter_cfg.get(
            "plot_subtimesteps", False
        )  # True if plots are created for each valid time separately
        self.run_id = output_basedir.name

        self.out_plot_basedir = Path(output_basedir) / "plots"

        if not os.path.exists(self.out_plot_basedir):
            _logger.info(f"Creating dir {self.out_plot_basedir}")
            os.makedirs(self.out_plot_basedir, exist_ok=True)

        self.sample = None
        self.stream = stream
        self.fstep = None
        self.select = {}

    def update_data_selection(self, select: dict):
        """
        Set the selection for the plots. This will be used to filter the data for plotting.

        Parameters
        ----------
        select:
            Dictionary containing the selection criteria. Expected keys are:
                - "sample": Sample identifier
                - "stream": Stream identifier
                - "forecast_step": Forecast step identifier
        """
        self.select = select

        if "sample" not in select:
            _logger.warning("No sample in the selection. Might lead to unexpected results.")
        else:
            self.sample = select["sample"]

        if "stream" not in select:
            _logger.warning("No stream in the selection. Might lead to unexpected results.")
        else:
            self.stream = select["stream"]

        if "forecast_step" not in select:
            _logger.warning("No forecast_step in the selection. Might lead to unexpected results.")
        else:
            self.fstep = select["forecast_step"]

        return self

    def clean_data_selection(self):
        """
        Clean the data selection by resetting all selected values.
        """
        self.sample = None
        self.stream = None
        self.fstep = None

        self.select = {}
        return self

    def select_from_da(self, da: xr.DataArray, selection: dict) -> xr.DataArray:
        """
        Select data from an xarray DataArray based on given selectors.

        Parameters
        ----------
        da:
            xarray DataArray to select data from.
        selection:
            Dictionary of selectors where keys are coordinate names and values are the values to
            select.

        Returns
        -------
            xarray DataArray with selected data.
        """
        for key, value in selection.items():
            if key not in da.coords and key not in da.dims:
                # Key is not a coordinate or dimension of this DataArray
                # (e.g. "stream" is used for file-naming only). Skip it.
                continue
            elif key in da.coords and key not in da.dims:
                # Coordinate like 'sample' aligned to another dim
                da = da.where(da[key] == value, drop=True)
            else:
                # Scalar coord or dim coord (e.g., 'forecast_step', 'channel')
                da = da.sel({key: value})
        return da

    def create_histograms_per_sample(
        self,
        target: xr.DataArray,
        preds: xr.DataArray,
        variables: list,
        select: dict,
        tag: str = "",
    ) -> list[str]:
        """
        Plot histogram of target vs predictions for each variable and valid time in the DataArray.

        Parameters
        ----------
        target: xr.DataArray
            Target sample for a specific (stream, sample, fstep)
        preds: xr.DataArray
            Predictions sample for a specific (stream, sample, fstep)
        variables: list
            List of variables to be plotted
        select: dict
            Selection to be applied to the DataArray
        tag: str
            Any tag you want to add to the plot

        Returns
        -------
            List of plot names for the saved histograms.
        """
        plot_names = []

        self.update_data_selection(select)

        # Basic map output directory for this stream
        hist_output_dir = self.out_plot_basedir / self.stream / "histograms"

        if not os.path.exists(hist_output_dir):
            _logger.info(f"Creating dir {hist_output_dir}")
            os.makedirs(hist_output_dir, exist_ok=True)

        for var in variables:
            select_var = self.select | {"channel": var}

            targ, prd = (
                self.select_from_da(target, select_var),
                self.select_from_da(preds, select_var),
            )

            # Remove NaNs
            targ = targ.dropna(dim="ipoint")
            prd = prd.dropna(dim="ipoint")
            assert targ.size > 0, "Data array must not be empty or contain only NAs"
            assert prd.size > 0, "Data array must not be empty or contain only NAs"

            if self.plot_subtimesteps:
                ntimes_unique = len(np.unique(targ.valid_time))
                _logger.info(
                    f"Creating histograms for {ntimes_unique} valid times of variable {var}."
                )

                groups = zip(targ.groupby("valid_time"), prd.groupby("valid_time"), strict=False)
            else:
                _logger.info(f"Plotting histogram for all valid times of {var}")

                groups = [((None, targ), (None, prd))]  # wrap once with dummy valid_time

            for (valid_time, targ_t), (_, prd_t) in groups:
                if valid_time is not None:
                    _logger.debug(f"Plotting histogram for {var} at valid_time {valid_time}")
                name = self.plot_histogram(targ_t, prd_t, hist_output_dir, var, tag=tag)
                plot_names.append(name)

        self.clean_data_selection()

        return plot_names

    def plot_histogram(
        self,
        target_data: xr.DataArray,
        pred_data: xr.DataArray,
        hist_output_dir: Path,
        varname: str,
        tag: str = "",
    ) -> str:
        """
        Plot a histogram comparing target and prediction data for a specific variable.

        Parameters
        ----------
        target_data: xr.DataArray
            DataArray containing the target data for the variable.
        pred_data: xr.DataArray
            DataArray containing the prediction data for the variable.
        hist_output_dir: Path
            Directory where the histogram will be saved.
        varname: str
            Name of the variable to be plotted.
        tag: str
            Any tag you want to add to the plot.

        Returns
        -------
            Name of the saved plot file.
        """

        # Get common bin edges
        vals = np.concatenate([target_data, pred_data])
        bins = np.histogram_bin_edges(vals, bins=50)

        # Plot histograms
        plt.hist(target_data, bins=bins, alpha=0.7, label="Target")
        plt.hist(pred_data, bins=bins, alpha=0.7, label="Prediction")

        # set labels and title
        plt.xlabel(f"Variable: {varname}")
        plt.ylabel("Frequency")
        plt.title(
            f"Histogram of Target and Prediction: {self.stream}, {varname} : "
            f"fstep = {self.fstep:03}"
        )
        plt.legend(frameon=False)

        valid_time = (
            target_data["valid_time"][0]
            .values.astype("datetime64[m]")
            .astype(datetime.datetime)
            .strftime("%Y-%m-%dT%H%M")
        )

        # TODO: make this nicer
        parts = [
            "histogram",
            self.run_id,
            tag,
            str(self.sample),
            valid_time,
            self.stream,
            varname,
            str(self.fstep).zfill(3),
        ]
        name = "_".join(filter(None, parts))

        fname = hist_output_dir / f"{name}.{self.image_format}"
        _logger.debug(f"Saving histogram to {fname}")
        plt.savefig(fname)
        plt.close()

        return name

    def create_maps_per_sample(
        self,
        data: xr.DataArray,
        variables: list,
        select: dict,
        tag: str = "",
        map_kwargs: dict | None = None,
    ) -> list[str]:
        """
        Plot 2D map for each variable and valid time in the DataArray.

        Parameters
        ----------
        data: xr.DataArray
            DataArray for a specific (stream, sample, fstep)
        variables: list
            List of variables to be plotted
        label: str
            Any tag you want to add to the plot
        select: dict
            Selection to be applied to the DataArray
        tag: str
            Any tag you want to add to the plot. Note: This is added to the plot directory.
        map_kwargs: dict
            Additional keyword arguments for the map.
            Known keys are:
                - marker_size: base size of the marker (default is 1)
                - scale_marker_size: if True, the marker size will be scaled based on latitude
                  (default is False)
                - marker: marker style (default is 'o')
            Unknown keys will be passed to the scatter plot function.

        Returns
        -------
            List of plot names for the saved maps.
        """
        self.update_data_selection(select)

        # copy global plotting options, not specific to any variable
        map_kwargs_global = {
            key: value
            for key, value in (map_kwargs or {}).items()
            if not isinstance(value, oc.DictConfig)
        }

        # Basic map output directory for this stream
        map_output_dir = self.get_map_output_dir(tag)

        if not os.path.exists(map_output_dir):
            _logger.info(f"Creating dir {map_output_dir}")
            os.makedirs(map_output_dir, exist_ok=True)

        for region in self.regions:
            if region != "global":
                bbox = RegionBoundingBox.from_region_name(region)
                reg_data = bbox.apply_mask(data)
            else:
                reg_data = data

            plot_names = []
            for var in variables:
                select_var = self.select | {"channel": var}
                da = self.select_from_da(reg_data, select_var).compute()

                if self.plot_subtimesteps:
                    ntimes_unique = len(np.unique(da.valid_time))
                    _logger.debug(f"Creating maps for variable {var} - {tag}")
                    if ntimes_unique == 0:
                        _logger.warning(
                            f"No valid times found for variable {var} - {tag}. Skipping."
                        )
                        continue
                    groups = da.groupby("valid_time")
                else:
                    _logger.debug(f"Creating maps for variable {var} - {tag}")
                    groups = [(None, da)]  # single dummy group

                for valid_time, da_t in groups:
                    if valid_time is not None:
                        _logger.debug(f"Plotting map for {var} at valid_time {valid_time}")

                    da_t = da_t.dropna(dim="ipoint")
                    if da_t.size == 0:
                        _logger.warning(
                            f"Data array for {var} at valid_time {valid_time} is empty after "
                            f"dropping NAs. Skipping this plot."
                        )
                        continue

                    name = self.scatter_plot(
                        da_t,
                        map_output_dir,
                        var,
                        region,
                        tag=tag,
                        map_kwargs=dict(map_kwargs.get(var, {})) | map_kwargs_global,
                        title=self.get_map_title(var, valid_time, da_t),
                    )
                    plot_names.append(name)

        self.clean_data_selection()

        return plot_names

    def scatter_plot(
        self,
        data: xr.DataArray,
        map_output_dir: Path,
        varname: str,
        regionname: str | None,
        tag: str = "",
        map_kwargs: dict | None = None,
        title: str | None = None,
    ):
        """
        Plot a 2D map for a data array using scatter plot.

        Parameters
        ----------
        data: xr.DataArray
            DataArray to be plotted
        map_output_dir: Path
            Directory where the map will be saved
        varname: str
            Name of the variable to be plotted
        regionname: str
            Name of the region to be plotted
        tag: str
            Any tag you want to add to the plot
        map_kwargs: dict | None
            Additional keyword arguments for the map.
        title: str | None
            Title for the plot.

        Returns
        -------
            Name of the saved plot file.
        """
        # check for known keys in map_kwargs
        map_kwargs_save = map_kwargs.copy() if map_kwargs is not None else {}
        marker_size_base = map_kwargs_save.pop(
            "marker_size", DefaultMarkerSize.get_marker_size(self.stream)
        )
        scale_marker_size = map_kwargs_save.pop("scale_marker_size", False)
        marker = map_kwargs_save.pop("marker", "o")
        vmin = map_kwargs_save.pop("vmin", None)
        vmax = map_kwargs_save.pop("vmax", None)
        cmap = plt.get_cmap(map_kwargs_save.pop("colormap", "coolwarm"))

        # Healpix grid configuration
        add_healpix_grid = map_kwargs_save.pop("add_healpix_grid", False)
        healpix_nside = map_kwargs_save.pop("healpix_nside", 4)
        healpix_color = map_kwargs_save.pop("healpix_color", "black")
        healpix_linewidth = map_kwargs_save.pop("healpix_linewidth", 0.2)
        healpix_step = map_kwargs_save.pop("healpix_step", 64)
        healpix_linestyle = map_kwargs_save.pop("healpix_linestyle", "-")

        if isinstance(map_kwargs_save.get("levels", False), oc.listconfig.ListConfig):
            norm = mpl.colors.BoundaryNorm(
                map_kwargs_save.pop("levels", None), cmap.N, extend="both"
            )
        else:
            norm = mpl.colors.Normalize(
                vmin=vmin,
                vmax=vmax,
                clip=False,
            )

        # scale marker size
        marker_size = marker_size_base
        if scale_marker_size:
            marker_size = np.clip(
                marker_size / np.cos(np.radians(data["lat"])) ** 2,
                a_max=marker_size * 10.0,
                a_min=marker_size,
            )

        # Create figure and axis objects
        fig = plt.figure(dpi=self.dpi_val)

        proj = ccrs.PlateCarree()
        if regionname == "global":
            proj = ccrs.Robinson()

        ax = fig.add_subplot(1, 1, 1, projection=proj)
        try:
            ax.coastlines()
        except Exception:
            _logger.warning("Could not add coastlines to plot; continuing without them.")

        data = data.squeeze()

        assert data["lon"].shape == data["lat"].shape == data.shape, (
            f"Scatter plot:: Data shape do not match. Shapes: "
            f"lon {data['lon'].shape}, lat {data['lat'].shape}, data {data.shape}."
        )

        scatter_plt = ax.scatter(
            data["lon"],
            data["lat"],
            c=data,
            norm=norm,
            cmap=cmap,
            s=marker_size,
            marker=marker,
            transform=ccrs.PlateCarree(),
            linewidths=0.0,  # only markers, avoids aliasing for very small markers
            **map_kwargs_save,
        )

        # Add Healpix grid (optimized with LineCollection)
        if add_healpix_grid:
            lc = self.healpixlines(
                healpix_nside, healpix_color, healpix_linewidth, healpix_step, healpix_linestyle
            )
            ax.add_collection(lc)
        else:
            ax.gridlines(draw_labels=False, linestyle="--", color="black", linewidth=0.2)

        plt.colorbar(scatter_plt, ax=ax, orientation="horizontal", label=f"Variable: {varname}")
        plt.title(title, fontsize=9.5)
        if regionname == "global":
            ax.set_global()
        else:
            region_extent = [
                data["lon"].min().item(),
                data["lon"].max().item(),
                data["lat"].min().item(),
                data["lat"].max().item(),
            ]
            ax.set_extent(region_extent, crs=ccrs.PlateCarree())

        # TODO: make this nicer
        parts = ["map", self.run_id, tag]

        if self.sample is not None:
            parts.append(str(self.sample))

        if "valid_time" in data.coords:
            valid_time = data["valid_time"][0].values
            if ~np.isnat(valid_time):
                valid_time = (
                    valid_time.astype("datetime64[m]")
                    .astype(datetime.datetime)
                    .strftime("%Y-%m-%dT%H%M")
                )

                parts.append(valid_time)

        if self.stream:
            parts.append(self.stream)

        parts.append(regionname)
        parts.append(varname)

        if self.fstep is not None:
            parts.extend(["fstep", f"{self.fstep:03d}"])

        name = "_".join(filter(None, parts))
        fname = f"{map_output_dir.joinpath(name)}.{self.image_format}"

        _logger.debug(f"Saving map to {fname}")
        plt.savefig(fname)
        plt.close()

        return name

    def healpixlines(
        self, healpix_nside, healpix_color, healpix_linewidth, healpix_step, healpix_linestyle
    ):
        hp_grid = HEALPixGrid(nside=healpix_nside, order="ring")
        lon_all, lat_all = hp_grid.boundaries_lonlat(np.arange(hp_grid.npix), step=healpix_step)
        # Ensure closure of polygons
        lon_closed = np.concatenate([lon_all.deg, lon_all.deg[:, 0:1]], axis=1)
        lat_closed = np.concatenate([lat_all.deg, lat_all.deg[:, 0:1]], axis=1)
        # Stack as (N_polys, N_points, 2)
        segments = np.stack([lon_closed, lat_closed], axis=-1)
        # (cartopy handles transform for LineCollection via set_transform)
        lc = LineCollection(
            segments,
            colors=healpix_color,
            linewidths=healpix_linewidth,
            linestyles=healpix_linestyle,
            alpha=0.5,
            zorder=10,
        )
        lc.set_transform(ccrs.PlateCarree())
        return lc

    def get_map_output_dir(self, tag):
        return self.out_plot_basedir / self.stream / "maps" / tag

    def get_map_title(self, var, valid_time, data):
        title = f"{self.stream}, {var} : fstep = {self.fstep:03}"
        if valid_time is not None:
            title += f" ({format_datetime(valid_time)})"
        elif "valid_time" in data.coords:
            valid_time_start = data["valid_time"].values.min()
            valid_time_end = data["valid_time"].values.max()
            if valid_time_start != valid_time_end:
                title += (
                    f" ({format_datetime(valid_time_start)} - {format_datetime(valid_time_end)})"
                )
            else:
                title += f" ({format_datetime(valid_time_start)})"

        return title
