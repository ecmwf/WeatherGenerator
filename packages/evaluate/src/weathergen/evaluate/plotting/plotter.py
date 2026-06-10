# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
import os
import warnings
from dataclasses import dataclass
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
from scipy.stats import skew
from scipy.stats import wasserstein_distance as wd

try:
    import datashader as ds
    import pandas as pd

    HAS_DATASHADER = True
except ImportError:
    HAS_DATASHADER = False

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
        _logger.debug(
            "Auto-downloads are blocked for cartopy; only local cartopy data will be used."
        )
    else:
        warnings.filterwarnings("default", category=DownloadWarning)


np.seterr(divide="ignore", invalid="ignore")

logging.getLogger("matplotlib.category").setLevel(logging.ERROR)

_logger.debug(f"Taking cartopy paths from {work_dir}")


@dataclass
class DistStats:
    """Summary statistics for a 1-D distribution."""

    count: int
    min: float
    max: float
    mean: float
    median: float
    std: float
    skewness: float

    @classmethod
    def from_array(cls, v: np.typing.NDArray) -> "DistStats":
        v = np.asarray(v).ravel()
        return cls(
            count=len(v),
            min=float(np.min(v)),
            max=float(np.max(v)),
            mean=float(np.mean(v)),
            median=float(np.median(v)),
            std=float(np.std(v)),
            skewness=float(skew(v, nan_policy="omit")),
        )

    def summary(self, label: str) -> str:
        return (
            f"{label:8s} N={self.count}  min={self.min:.3g}  max={self.max:.3g}  "
            f"mean={self.mean:.3g}  med={self.median:.3g}  "
            f"std={self.std:.3g}  skew={self.skewness:.3g}"
        )


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

        _logger.debug(f"Taking cartopy paths from {work_dir}")

        self.image_format = plotter_cfg.get("image_format")
        self.animation_format = plotter_cfg.get("animation_format")
        self.dpi_val = plotter_cfg.get("dpi_val")
        self.fig_size = plotter_cfg.get("fig_size")
        self.fps = plotter_cfg.get("fps")
        self.log_colorbar = plotter_cfg.get("log_colorbar", False)
        self.regions = plotter_cfg.get("regions")
        self.log_x = plotter_cfg.get("log_x", False)
        self.log_y = plotter_cfg.get("log_y", False)
        self.n_bins = plotter_cfg.get("n_bins", 50)
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
            # "all_samples" is a proxy for across-samples aggregation;
            # remove it from self.select so it won't be used in .sel()
            if select["sample"] == "all_samples":
                self.select.pop("sample")

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

    def create_histograms(
        self,
        target: xr.DataArray,
        preds: xr.DataArray,
        variables: list,
        select: dict,
        tag: str = "",
        ranges: dict | None = None,
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

        # Basic histogram output directory for this stream
        hist_output_dir = self.get_hist_output_dir()

        if not os.path.exists(hist_output_dir):
            _logger.info(f"Creating dir {hist_output_dir}")
            os.makedirs(hist_output_dir, exist_ok=True)

        for region in self.regions:
            if region != "global":
                bbox = RegionBoundingBox.from_region_name(region)
                reg_target = bbox.apply_mask(target)
                reg_preds = bbox.apply_mask(preds)
            else:
                reg_target = target
                reg_preds = preds

            for var in variables:
                select_var = self.select | {"channel": var}

                targ, prd = (
                    self.select_from_da(reg_target, select_var),
                    self.select_from_da(reg_preds, select_var),
                )

                # Remove NaNs
                targ = targ.dropna(dim="ipoint")
                prd = prd.dropna(dim="ipoint")
                assert targ.size > 0, "Data array must not be empty or contain only NAs"
                assert prd.size > 0, "Data array must not be empty or contain only NAs"

                if self.plot_subtimesteps and str(self.sample) != "all_samples":
                    ntimes_unique = len(np.unique(targ.valid_time))
                    _logger.debug(
                        f"Creating histograms for {ntimes_unique} valid times of variable {var}."
                    )

                    groups = zip(
                        targ.groupby("valid_time"), prd.groupby("valid_time"), strict=False
                    )
                else:
                    _logger.debug(f"Plotting histogram for all valid times of {var}")

                    groups = [((None, targ), (None, prd))]  # wrap once with dummy valid_time

                for (valid_time, targ_t), (_, prd_t) in groups:
                    if valid_time is not None:
                        _logger.debug(f"Plotting histogram for {var} at valid_time {valid_time}")
                    var_range = ranges.get(var, {}) if ranges else {}
                    name = self.plot_histogram(
                        targ_t,
                        prd_t,
                        hist_output_dir,
                        var,
                        tag=tag,
                        region=region,
                        xlim=(var_range.get("vmin"), var_range.get("vmax")),
                    )
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
        region: str = "",
        xlim: tuple | None = None,
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

        tar_vals = np.asarray(target_data).ravel()
        prd_vals = np.asarray(pred_data).ravel()

        # Get common bin edges — use fixed xlim range if provided for consistency
        xmin, xmax = xlim if xlim else (None, None)
        # Fall back to data-derived bounds if either limit is missing
        if xmin is None or xmax is None:
            vals = np.concatenate([tar_vals, prd_vals])
            if xmin is None:
                xmin = float(np.nanmin(vals))
            if xmax is None:
                xmax = float(np.nanmax(vals))
        # Add 5% margin on each side so tails are clearly visible
        margin = (xmax - xmin) * 0.05
        xmin -= margin
        xmax += margin
        bins = np.linspace(xmin, xmax, self.n_bins + 1)

        # Compute histograms
        target_counts, _ = np.histogram(tar_vals, bins=bins)
        pred_counts, _ = np.histogram(prd_vals, bins=bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        color_tar = "black"
        color_pred = "#00897B"  # teal / green-blue

        # Create figure with two subplots: histogram + ratio
        fig, (ax_hist, ax_ratio) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=self.fig_size or (8, 6),
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        )

        # Upper panel: histogram curves
        ax_hist.plot(
            bin_centers, target_counts, alpha=0.7, label="Target", linewidth=1.5, color=color_tar
        )
        ax_hist.plot(
            bin_centers, pred_counts, alpha=0.7, label="Prediction", linewidth=1.5, color=color_pred
        )
        ax_hist.set_ylabel("Frequency")
        ax_hist.set_title(f"{self.stream}, {varname} : fstep = {self.fstep:03}")
        ax_hist.legend(frameon=False)
        if self.log_y:
            ax_hist.set_yscale("log")
        ax_hist.grid(True, linestyle="--", alpha=0.5)

        # Lower panel: ratio (prediction / target)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(target_counts > 0, pred_counts / target_counts, np.nan)
        ax_ratio.plot(bin_centers, ratio, linewidth=1.2, color=color_pred)
        ax_ratio.axhline(1.0, linestyle="--", color="gray", linewidth=0.8)
        ax_ratio.set_ylabel("Pred / Target")
        ax_ratio.set_xlabel(f"Variable: {varname}")
        ax_ratio.set_ylim(0, 2)
        ax_ratio.grid(True, linestyle="--", alpha=0.5)

        if self.log_x:
            ax_hist.set_xscale("log")
            ax_ratio.set_xscale("log")
        ax_ratio.set_xlim(xmin, xmax)

        t_s = DistStats.from_array(tar_vals)
        p_s = DistStats.from_array(prd_vals)

        # Wasserstein distance
        w_dist = wd(tar_vals, prd_vals)

        stat_text = (
            f"Wasserstein distance: {w_dist:.4g}\n{t_s.summary('Target:')}\n{p_s.summary('Pred:')}"
        )

        fig.text(
            0.5,
            -0.02,
            stat_text,
            ha="center",
            va="top",
            fontsize=7,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
        )

        # For "all_samples" (across-samples) histograms, omit the valid_time from the name
        is_global = str(self.sample) == "all_samples"

        if is_global:
            valid_time = None
        else:
            valid_time = (
                target_data["valid_time"][0]
                .values.astype("datetime64[m]")
                .astype(datetime.datetime)
                .strftime("%Y-%m-%dT%H%M")
            )

        parts = [
            "histogram",
            str(self.run_id),
            str(tag) if tag else "",
            str(self.sample),
            valid_time,
            str(self.stream),
            region if region else "",
            varname,
            f"{self.fstep:03d}",
        ]
        name = "_".join(filter(None, parts))

        fname = hist_output_dir / f"{name}.{self.image_format}"
        _logger.debug(f"Saving histogram to {fname}")
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)

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

    # map_kwargs parsing
    @staticmethod
    def _parse_map_kwargs(map_kwargs: dict | None, stream: str | None) -> dict:
        """Extract known plotting keys from *map_kwargs*, returning a structured dict.

        Unknown keys are kept under ``"extra"`` and forwarded to ``ax.scatter``.

        Parameters
        ----------
        map_kwargs : dict or None
            Raw keyword arguments from the caller. Known keys (``marker_size``,
            ``scale_marker_size``, ``marker``, ``vmin``, ``vmax``, ``colormap``,
            ``use_datashader``, ``levels``, and HEALPix-related keys) are extracted;
            remaining keys are collected under ``"extra"``.
        stream : str or None
            Stream name used to look up the default marker size when
            ``marker_size`` is not provided in *map_kwargs*.

        Returns
        -------
        dict
            Structured dictionary with the following keys:
                - marker_size_base (float)
                - scale_marker_size (bool)
                - marker (str)
                - vmin, vmax (float or None)
                - cmap (matplotlib.colors.Colormap)
                - use_datashader (bool)
                - norm (matplotlib.colors.Normalize or BoundaryNorm)
                - add_healpix_grid (bool) and related healpix_* keys
                - extra (dict) – leftover kwargs for ``ax.scatter``
        """
        kw = map_kwargs.copy() if map_kwargs is not None else {}

        parsed = {
            "marker_size_base": kw.pop(
                "marker_size", DefaultMarkerSize.get_marker_size(stream) if stream else 0.5
            ),
            "scale_marker_size": kw.pop("scale_marker_size", False),
            "marker": kw.pop("marker", "o"),
            "vmin": kw.pop("vmin", None),
            "vmax": kw.pop("vmax", None),
            "cmap": plt.get_cmap(kw.pop("colormap", "coolwarm")),
            "use_datashader": kw.pop("use_datashader", False),
            "levels": kw.pop("levels", None),
            # HEALPix grid
            "add_healpix_grid": kw.pop("add_healpix_grid", False),
            "healpix_nside": kw.pop("healpix_nside", 4),
            "healpix_color": kw.pop("healpix_color", "black"),
            "healpix_linewidth": kw.pop("healpix_linewidth", 0.2),
            "healpix_step": kw.pop("healpix_step", 64),
            "healpix_linestyle": kw.pop("healpix_linestyle", "-"),
        }

        parsed["extra"] = kw  # remaining kwargs forwarded to scatter
        return parsed

    # rendering backends
    @staticmethod
    def _render_datashader(ax, proj, data, norm, cmap, marker_size_base):
        """Rasterise points with datashader and display via imshow.

        Bypasses ``dsshow`` because it calls ``get_xlim()``/``get_ylim()``
        which return infinity on cartopy GeoAxes with non-PlateCarree
        projections. Instead rasterises manually with ``ds.Canvas``.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            GeoAxes on which to render the rasterised image.
        proj : cartopy.crs.Projection
            The map projection used by *ax*.
        data : xr.DataArray
            DataArray with ``lon``, ``lat`` coordinates and scalar values.
        norm : matplotlib.colors.Normalize
            Colour normalisation instance.
        cmap : matplotlib.colors.Colormap
            Colourmap for the rendered image.
        marker_size_base : float
            Base marker size (reserved for future tuning).

        Returns
        -------
        matplotlib.image.AxesImage
            The artist returned by ``ax.imshow``, suitable for ``plt.colorbar``.
        """
        projected = proj.transform_points(
            ccrs.PlateCarree(),
            np.asarray(data["lon"], dtype=np.float64),
            np.asarray(data["lat"], dtype=np.float64),
        )
        df = pd.DataFrame(
            {
                "x": projected[:, 0],
                "y": projected[:, 1],
                "val": np.asarray(data.values, dtype=np.float64),
            }
        )
        df = df.dropna(subset=["x", "y"])

        x_range = (float(df["x"].min()), float(df["x"].max()))
        y_range = (float(df["y"].min()), float(df["y"].max()))

        # Determine raster resolution from the figure size + dpi
        fig = ax.get_figure()
        bbox = ax.get_position()
        plot_width = int(fig.get_figwidth() * fig.dpi * bbox.width)
        plot_height = int(fig.get_figheight() * fig.dpi * bbox.height)

        # Adapt resolution to point density so every pixel is filled.
        n_pts = len(df)
        effective_side = max(int(0.8 * np.sqrt(n_pts)), 100)

        # The canvas should not exceed the effective grid resolution —
        # going higher creates empty pixels that show as white bands.
        plot_width = min(plot_width, effective_side * 2)
        plot_height = min(plot_height, effective_side)

        cvs = ds.Canvas(
            plot_width=plot_width,
            plot_height=plot_height,
            x_range=x_range,
            y_range=y_range,
        )
        agg = cvs.points(df, "x", "y", agg=ds.mean("val"))

        # Fill small NaN gaps (isolated empty pixels between data rows)
        # with nearest-neighbour interpolation so no white bands remain.

        raw = agg.values.astype(np.float64)
        mask = np.isnan(raw)
        if mask.any() and not mask.all():
            from scipy.interpolate import NearestNDInterpolator

            yy, xx = np.where(~mask)
            interp = NearestNDInterpolator(list(zip(xx, yy, strict=False)), raw[~mask])
            yy_m, xx_m = np.where(mask)
            # Only fill pixels whose nearest valid neighbour is within 2 pixels
            filled = raw.copy()
            filled[mask] = interp(xx_m, yy_m)
            # Limit fill distance: re-mask pixels far from any data
            from scipy.ndimage import binary_dilation

            data_mask = ~mask
            dilated = binary_dilation(data_mask, iterations=2)
            filled[~dilated] = np.nan
            raw = filled

        vals = np.flipud(raw)
        masked = np.ma.masked_invalid(vals)

        artist = ax.imshow(
            masked,
            extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
            origin="upper",
            aspect="auto",
            transform=proj,
            interpolation="bilinear",
            norm=norm,
            cmap=cmap,
        )
        return artist

    @staticmethod
    def _render_scatter(ax, data, norm, cmap, marker_size, marker, extra_kwargs):
        """Render points with matplotlib scatter and return the artist.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            GeoAxes on which to render the scatter points.
        data : xr.DataArray
            DataArray with ``lon``, ``lat`` coordinates and scalar values.
        norm : matplotlib.colors.Normalize
            Colour normalisation instance.
        cmap : matplotlib.colors.Colormap
            Colourmap for the scatter points.
        marker_size : float or np.ndarray
            Marker size(s) in matplotlib scatter units (pt²).
        marker : str
            Marker style (e.g. ``'o'``, ``'s'``).
        extra_kwargs : dict
            Additional keyword arguments forwarded to ``ax.scatter``.

        Returns
        -------
        matplotlib.collections.PathCollection
            The scatter artist, suitable for ``plt.colorbar``.
        """
        return ax.scatter(
            data["lon"],
            data["lat"],
            c=data,
            norm=norm,
            cmap=cmap,
            s=marker_size,
            marker=marker,
            transform=ccrs.PlateCarree(),
            linewidths=0.0,
            rasterized=True,
            **extra_kwargs,
        )

    # filename builder
    def _build_map_filename(self, varname: str, regionname: str, tag: str, data: xr.DataArray):
        """Build the canonical filename parts list for a map plot.

        Parameters
        ----------
        varname : str
            Name of the variable being plotted.
        regionname : str
            Name of the geographical region.
        tag : str
            Additional tag inserted into the filename.
        data : xr.DataArray
            DataArray used to extract ``valid_time`` for the filename.

        Returns
        -------
        str
            Joined filename string (without extension).
        """
        parts = ["map", self.run_id, tag]

        if self.sample is not None:
            parts.append(str(self.sample))

        if "valid_time" in data.coords:
            valid_time = data["valid_time"][0].values
            if ~np.isnat(valid_time):
                parts.append(
                    valid_time.astype("datetime64[m]")
                    .astype(datetime.datetime)
                    .strftime("%Y-%m-%dT%H%M")
                )

        if self.stream:
            parts.append(self.stream)

        parts.append(regionname)
        parts.append(varname)

        if self.fstep is not None:
            parts.append(f"{self.fstep:03d}")

        return "_".join(filter(None, parts))

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
        """Plot a 2D map for a data array using scatter (or datashader).

        Parameters
        ----------
        data : xr.DataArray
            DataArray to be plotted.
        map_output_dir : Path
            Directory where the map will be saved.
        varname : str
            Name of the variable to be plotted.
        regionname : str
            Name of the region to be plotted.
        tag : str
            Any tag you want to add to the plot.
        map_kwargs : dict | None
            Additional keyword arguments for the map.
        title : str | None
            Title for the plot.

        Returns
        -------
        str
            Name of the saved plot file (without directory or extension).
        """
        # parse kwargs
        opts = self._parse_map_kwargs(map_kwargs, self.stream)

        data = data.squeeze()
        assert data["lon"].shape == data["lat"].shape == data.shape, (
            f"Scatter plot:: Data shape do not match. Shapes: "
            f"lon {data['lon'].shape}, lat {data['lat'].shape}, data {data.shape}."
        )

        # Pick figure size: for dense grids (>=200 K points) use a larger
        # canvas so fine structure is visible; sparse grids use the default.
        figsize = self.fig_size
        if figsize is None and data.size >= 200_000:
            figsize = (15, 7)

        proj = ccrs.PlateCarree()
        if regionname:
            try:
                # This uses the method already available in RegionBoundingBox
                bbox = RegionBoundingBox.from_region_name(regionname)
                proj = bbox.projection
            except ValueError:
                # If regionname isn't in the library, fall back to PlateCarree
                _logger.warning(f"Region '{regionname}' not found in library, using PlateCarree.")
                proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=figsize, dpi=self.dpi_val)
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        try:
            ax.coastlines(linewidth=0.3)
        except Exception:
            _logger.warning("Could not add coastlines to plot; continuing without them.")

        for spine in ax.spines.values():
            spine.set_linewidth(0.3)

        if opts["vmin"] is None or opts["vmax"] is None:
            valid_vals = data.values[np.isfinite(data.values)]
            if valid_vals.size > 0:
                p_lo, p_hi = np.percentile(valid_vals, [5, 95])
                if opts["vmin"] is None:
                    opts["vmin"] = float(p_lo)
                if opts["vmax"] is None:
                    opts["vmax"] = float(p_hi)

        if isinstance(opts["levels"], oc.listconfig.ListConfig):
            opts["norm"] = mpl.colors.BoundaryNorm(opts["levels"], opts["cmap"].N, extend="both")
        elif self.log_colorbar and opts["vmin"] is not None and opts["vmin"] > 0:
            opts["norm"] = mpl.colors.LogNorm(vmin=opts["vmin"], vmax=opts["vmax"])
        else:
            if self.log_colorbar:
                _logger.warning(
                    "log_colorbar=True but vmin=%.3g <= 0; falling back to linear norm.",
                    opts["vmin"],
                )
            opts["norm"] = mpl.colors.Normalize(vmin=opts["vmin"], vmax=opts["vmax"], clip=False)

        if regionname == "global":
            ax.set_global()
        else:
            ax.set_extent(
                [
                    data["lon"].min().item(),
                    data["lon"].max().item(),
                    data["lat"].min().item(),
                    data["lat"].max().item(),
                ],
                crs=ccrs.PlateCarree(),
            )

        # render points
        if opts["use_datashader"] and HAS_DATASHADER:
            artist = self._render_datashader(
                ax, proj, data, opts["norm"], opts["cmap"], opts["marker_size_base"]
            )
        else:
            if opts["use_datashader"] and not HAS_DATASHADER:
                _logger.warning(
                    "use_datashader=True but datashader is not installed. "
                    "Falling back to scatter. Install with: pip install datashader"
                )
            marker_size = DefaultMarkerSize.auto_marker_size(
                n_points=data.size,
                fig_width_in=fig.get_figwidth(),
                fig_height_in=fig.get_figheight(),
                stream_default=opts["marker_size_base"],
                scale=opts["scale_marker_size"],
                lat=data["lat"],
            )

            artist = self._render_scatter(
                ax, data, opts["norm"], opts["cmap"], marker_size, opts["marker"], opts["extra"]
            )

        # overlays
        if opts["add_healpix_grid"]:
            lc = self.healpixlines(
                opts["healpix_nside"],
                opts["healpix_color"],
                opts["healpix_linewidth"],
                opts["healpix_step"],
                opts["healpix_linestyle"],
            )
            ax.add_collection(lc)
        else:
            ax.gridlines(draw_labels=False, linestyle="--", color="gray", linewidth=0.6, alpha=0.7)

        cbar = plt.colorbar(
            artist,
            ax=ax,
            fraction=0.03,
            pad=0.02,
            shrink=0.6,
            orientation="horizontal",
        )
        cbar.set_label(f"Variable: {varname}", fontsize=7)
        cbar.ax.tick_params(labelsize=6)
        cbar.outline.set_linewidth(0.3)
        plt.title(title, fontsize=8)

        # save
        name = self._build_map_filename(varname, regionname, tag, data)
        fname = f"{map_output_dir.joinpath(name)}.{self.image_format}"
        _logger.debug(f"Saving map to {fname}")
        plt.savefig(fname, bbox_inches="tight")
        plt.close()

        return name

    def healpixlines(
        self, healpix_nside, healpix_color, healpix_linewidth, healpix_step, healpix_linestyle
    ):
        """Create a LineCollection of HEALPix pixel boundaries for overlay on a map.

        Parameters
        ----------
        healpix_nside : int
            HEALPix ``nside`` parameter controlling grid resolution.
        healpix_color : str
            Colour of the grid lines.
        healpix_linewidth : float
            Width of the grid lines in points.
        healpix_step : int
            Number of interpolation points per boundary edge.
        healpix_linestyle : str
            Line style (e.g. ``'-'``, ``'--'``).

        Returns
        -------
        matplotlib.collections.LineCollection
            Collection of HEALPix boundary polygons, ready to be added to a
            cartopy GeoAxes via ``ax.add_collection``.
        """
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
        """Return the output directory path for map plots.

        Parameters
        ----------
        tag : str
            Sub-directory tag (e.g. ``'target'``, ``'prediction'``).

        Returns
        -------
        Path
            Resolved directory path: ``<out_plot_basedir>/<stream>/maps/<tag>``.
        """
        return self.out_plot_basedir / self.stream / "maps" / tag

    def get_hist_output_dir(self):
        """Return the output directory path for histogram plots.

        Returns
        -------
        Path
            Resolved directory path: ``<out_plot_basedir>/<stream>/histograms``.
        """
        return self.out_plot_basedir / self.stream / "histograms"

    def get_map_title(self, var, valid_time, data):
        """Build the title string for a map plot.

        Parameters
        ----------
        var : str
            Variable name to include in the title.
        valid_time : numpy.datetime64 or None
            Single valid time for the plot. If ``None``, the range is
            extracted from *data*.
        data : xr.DataArray
            DataArray from which to extract ``valid_time`` range when
            *valid_time* is ``None``.

        Returns
        -------
        str
            Formatted title string.
        """
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
