import datetime
import glob
import logging
import os
from pathlib import Path

import cartopy
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import omegaconf as oc
import seaborn as sns
import xarray as xr
from matplotlib.lines import Line2D
from PIL import Image
from scipy.stats import wilcoxon

from weathergen.common.config import _load_private_conf
from weathergen.evaluate.plotting.plot_utils import (
    DefaultMarkerSize,
)
from weathergen.evaluate.utils.regions import RegionBoundingBox

work_dir = Path(_load_private_conf(None)["path_shared_working_dir"]) / "assets/cartopy"

cartopy.config["data_dir"] = str(work_dir)
cartopy.config["pre_existing_data_dir"] = str(work_dir)
os.environ["CARTOPY_DATA_DIR"] = str(work_dir)

np.seterr(divide="ignore", invalid="ignore")

logging.getLogger("matplotlib.category").setLevel(logging.ERROR)

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

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
        self.dpi_val = plotter_cfg.get("dpi_val")
        self.fig_size = plotter_cfg.get("fig_size")
        self.fps = plotter_cfg.get("fps")
        self.regions = plotter_cfg.get("regions")
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
            if key in da.coords and key not in da.dims:
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
            os.makedirs(hist_output_dir)

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
            os.makedirs(map_output_dir)

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
                    _logger.info(
                        f"Creating maps for {ntimes_unique} valid times of variable {var} - {tag}"
                    )
                    if ntimes_unique == 0:
                        _logger.warning(
                            f"No valid times found for variable {var} - {tag}. Skipping."
                        )
                        continue
                    groups = da.groupby("valid_time")
                else:
                    _logger.info(f"Creating maps for all valid times of {var} - {tag}")
                    groups = [(None, da)]  # single dummy group

                for valid_time, da_t in groups:
                    if valid_time is not None:
                        _logger.debug(f"Plotting map for {var} at valid_time {valid_time}")

                    da_t = da_t.dropna(dim="ipoint")
                    assert da_t.size > 0, "Data array must not be empty or contain only NAs"

                    name = self.scatter_plot(
                        da_t,
                        map_output_dir,
                        var,
                        region,
                        tag=tag,
                        map_kwargs=dict(map_kwargs.get(var, {})) | map_kwargs_global,
                        title=f"{self.stream}, {var} : fstep = {self.fstep:03} ({valid_time})",
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
        ax.coastlines()

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

        plt.colorbar(scatter_plt, ax=ax, orientation="horizontal", label=f"Variable: {varname}")
        plt.title(title)
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
        ax.gridlines(draw_labels=False, linestyle="--", color="black", linewidth=1)

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

    def animation(self, samples, fsteps, variables, select, tag) -> list[str]:
        """
        Plot 2D animations for a dataset

        Parameters
        ----------
        samples: list
            List of the samples to be plotted
        fsteps: list
            List of the forecast steps to be plotted
        variables: list
            List of variables to be plotted
        select: dict
            Selection to be applied to the DataArray
        tag: str
            Any tag you want to add to the plot

        Returns
        -------
            List of plot names for the saved animations.

        """

        self.update_data_selection(select)
        map_output_dir = self.get_map_output_dir(tag)

        # Convert FPS to duration in milliseconds
        duration_ms = int(1000 / self.fps) if self.fps > 0 else 400

        for region in self.regions:
            for _, sa in enumerate(samples):
                for _, var in enumerate(variables):
                    _logger.info(f"Creating animation for {var} sample: {sa} - {tag}")
                    image_paths = []
                    for _, fstep in enumerate(fsteps):
                        # breakpoint()
                        # TODO: refactor to avoid code duplication with scatter_plot
                        parts = [
                            "map",
                            self.run_id,
                            tag,
                            str(sa),
                            "*",
                            self.stream,
                            region,
                            var,
                            "fstep",
                            str(fstep).zfill(3),
                        ]

                        name = "_".join(filter(None, parts))
                        fname = f"{map_output_dir.joinpath(name)}.{self.image_format}"

                        names = glob.glob(fname)
                        image_paths += names

                    if image_paths:
                        images = [Image.open(path) for path in image_paths]
                        images[0].save(
                            f"{map_output_dir}/animation_{self.run_id}_{tag}_{sa}_{self.stream}_{region}_{var}.gif",
                            save_all=True,
                            append_images=images[1:],
                            duration=duration_ms,
                            loop=0,
                        )

                    else:
                        _logger.warning(f"No images found for animation {var} sample {sa}")

        return image_paths

    def get_map_output_dir(self, tag):
        return self.out_plot_basedir / self.stream / "maps" / tag


class LinePlots:
    def __init__(self, plotter_cfg: dict, output_basedir: str | Path):
        """
        Initialize the LinePlots class.

        Parameters
        ----------
        plotter_cfg:
            Configuration dictionary containing basic information for plotting.
            Expected keys are:
                - image_format: Format of the saved images (e.g., 'png', 'pdf', etc.)
                - dpi_val: DPI value for the saved images
                - fig_size: Size of the figure (width, height) in inches
                -  plot_ensemble:
                    If True, plot ensemble spread if 'ens' dimension is present. Options are:
                        - False: do not plot ensemble spread
                        - "std": plot mean +/- standard deviation
                        - "minmax": plot min-max range
                        - "members": plot individual ensemble members
        output_basedir:
            Base directory under which the plots will be saved.
            Expected scheme `<results_base_dir>/<run_id>`.
        """

        self.image_format = plotter_cfg.get("image_format")
        self.dpi_val = plotter_cfg.get("dpi_val")
        self.fig_size = plotter_cfg.get("fig_size")
        self.log_scale = plotter_cfg.get("log_scale")
        self.add_grid = plotter_cfg.get("add_grid")
        self.plot_ensemble = plotter_cfg.get("plot_ensemble", False)
        self.baseline = plotter_cfg.get("baseline")
        self.out_plot_dir = Path(output_basedir) / "line_plots"
        if not os.path.exists(self.out_plot_dir):
            _logger.info(f"Creating dir {self.out_plot_dir}")
            os.makedirs(self.out_plot_dir, exist_ok=True)

        _logger.info(f"Saving summary plots to: {self.out_plot_dir}")

    def _check_lengths(self, data: xr.DataArray | list, labels: str | list) -> tuple[list, list]:
        """
        Check if the lengths of data and labels match.

        Parameters
        ----------
        data:
            DataArray or list of DataArrays to be plotted
        labels:
            Label or list of labels for each dataset

        Returns
        -------
            data_list, label_list - lists of data and labels
        """
        assert isinstance(data, xr.DataArray | list), (
            "Compare::plot - Data should be of type xr.DataArray or list"
        )
        assert isinstance(labels, str | list), (
            "Compare::plot - Labels should be of type str or list"
        )

        # convert to lists

        data_list = [data] if isinstance(data, xr.DataArray) else data
        label_list = [labels] if isinstance(labels, str) else labels

        assert len(data_list) == len(label_list), "Compare::plot - Data and Labels do not match"

        return data_list, label_list

    def print_all_points_from_graph(self, fig: plt.Figure) -> None:
        for ax in fig.get_axes():
            for line in ax.get_lines():
                ydata = line.get_ydata()
                xdata = line.get_xdata()
                label = line.get_label()
                _logger.info(f"Summary for {label} plot:")
                for xi, yi in zip(xdata, ydata, strict=False):
                    _logger.info(f"  x: {xi:.3f}, y: {yi:.3f}")
                _logger.info("--------------------------")
        return

    def _plot_ensemble(self, data: xr.DataArray, x_dim: str, label: str) -> None:
        """
        Plot ensemble spread for a data array.

        Parameters
        ----------
        data: xr.xArray
            DataArray to be plotted
        x_dim: str
            Dimension to be used for the x-axis.
        label: str
            Label for the dataset
        Returns
        -------
            None
        """
        averaged = data.mean(dim=[dim for dim in data.dims if dim != x_dim], skipna=True).sortby(
            x_dim
        )

        lines = plt.plot(
            averaged[x_dim],
            averaged.values,
            label=label,
            marker="o",
            linestyle="-",
        )
        line = lines[0]
        color = line.get_color()

        ens = data.mean(
            dim=[dim for dim in data.dims if dim not in [x_dim, "ens"]], skipna=True
        ).sortby(x_dim)

        if self.plot_ensemble == "std":
            std_dev = ens.std(dim="ens", skipna=True).sortby(x_dim)
            plt.fill_between(
                averaged[x_dim],
                (averaged - std_dev).values,
                (averaged + std_dev).values,
                label=f"{label} - std dev",
                color=color,
                alpha=0.2,
            )

        elif self.plot_ensemble == "minmax":
            ens_min = ens.min(dim="ens", skipna=True).sortby(x_dim)
            ens_max = ens.max(dim="ens", skipna=True).sortby(x_dim)

            plt.fill_between(
                averaged[x_dim],
                ens_min.values,
                ens_max.values,
                label=f"{label} - min max",
                color=color,
                alpha=0.2,
            )

        elif self.plot_ensemble == "members":
            for j in range(ens.ens.size):
                plt.plot(
                    ens[x_dim],
                    ens.isel(ens=j).values,
                    color=color,
                    alpha=0.2,
                )
        else:
            _logger.warning(
                f"LinePlot:: Unknown option for plot_ensemble: {self.plot_ensemble}. "
                "Skipping ensemble plotting."
            )

    def _preprocess_data(
        self, data: xr.DataArray, x_dim: str | list[str], verbose: bool = True
    ) -> xr.DataArray:
        """
        Average all dimensions except x_dim (which may be a string or list)
        and then sort the result.

        Parameters
        ----------
        data : xr.DataArray
            DataArray to be preprocessed.
        x_dim : str or list of str
            Dimension(s) to be preserved for the x-axis.
        verbose : bool
            Log information about averaging.

        Returns
        -------
        xr.DataArray
            Preprocessed DataArray.
        """

        x_dims = [x_dim] if isinstance(x_dim, str) else list(x_dim)

        non_x_dims = [dim for dim in data.dims if dim not in x_dims]

        if any(data.sizes.get(dim, 1) > 1 for dim in non_x_dims) and verbose:
            logging.info(f"Averaging over dimensions: {non_x_dims}")

        out = data.mean(dim=non_x_dims, skipna=True)

        for xd in x_dims:
            out = out.sortby(xd)

        return out

    def plot(
        self,
        data: xr.DataArray | list,
        labels: str | list,
        tag: str = "",
        x_dim: str = "forecast_step",
        y_dim: str = "value",
        print_summary: bool = False,
        plot_ensemble: str | bool = False,
    ) -> None:
        """
        Plot a line graph comparing multiple datasets.

        Parameters
        ----------
        data:
            DataArray or list of DataArrays to be plotted
        labels:
            Label or list of labels for each dataset
        tag:
            Tag to be added to the plot title and filename
        x_dim:
            Dimension to be used for the x-axis. The code will average over all other dimensions.
        y_dim:
            Name of the dimension to be used for the y-axis.
        print_summary:
            If True, print a summary of the values from the graph.
        Returns
        -------
            None
        """

        data_list, label_list = self._check_lengths(data, labels)

        assert x_dim in data_list[0].dims, (
            "x dimension '{x_dim}' not found in data dimensions {data_list[0].dims}"
        )

        fig = plt.figure(figsize=(12, 6), dpi=self.dpi_val)

        for i, data in enumerate(data_list):
            non_zero_dims = [dim for dim in data.dims if dim != x_dim and data[dim].shape[0] > 1]

            if self.plot_ensemble and "ens" in non_zero_dims:
                _logger.info(f"LinePlot:: Plotting ensemble with option {self.plot_ensemble}.")
                self._plot_ensemble(data, x_dim, label_list[i])
            else:
                averaged = self._preprocess_data(data, x_dim)

                plt.plot(
                    averaged[x_dim],
                    averaged.values,
                    label=label_list[i],
                    marker="o",
                    linestyle="-",
                )

        parts = ["compare", tag]
        name = "_".join(filter(None, parts))
        self._plot_base(fig, name, x_dim, y_dim, print_summary)

    def _plot_base(
        self,
        fig: plt.Figure,
        name: str,
        x_dim: str,
        y_dim: str,
        print_summary: bool = False,
        line: float | None = None,
        title: str | None = None,
    ) -> None:
        """
        Apply labels, title, legend, save and optionally print summary.
        Parameters
        ----------
        fig:
            Matplotlib figure to be finalized
        name:
            Name of the plot file
        x_dim:
            Label for the x-axis
        y_dim:
            Label for the y-axis
        print_summary:
            If True, print a summary of the values from the graph.
        line:
            If provided, draw a horizontal line at the given y-value.
        title:
            Title for the plot.
        Returns
        -------
            None
        """
        plt.xlabel("".join(c if c.isalnum() else " " for c in x_dim))
        plt.ylabel("".join(c if c.isalnum() else " " for c in y_dim))
        plt.title(title if title is not None else " ".join(c if c.isalnum() else " " for c in name))
        plt.legend(frameon=False)

        if self.add_grid:
            plt.grid(True, linestyle="--", color="gray", alpha=0.5)

        if self.log_scale:
            plt.yscale("log")

        if print_summary:
            _logger.info(f"Summary values for {name}")
            self.print_all_points_from_graph(fig)

        if line:
            plt.axhline(y=line, color="black", linestyle="--", linewidth=1)

        plt.tight_layout()
        plt.savefig(f"{self.out_plot_dir.joinpath(name)}.{self.image_format}")
        plt.close()

    def ratio_plot(
        self,
        data: xr.DataArray | list,
        run_ids: list[str],
        labels: str | list,
        tag: str = "",
        x_dim: str = "forecast_step",
        y_dim: str = "value",
        print_summary: bool = False,
    ) -> None:
        """
        Plot a ratio plot comparing multiple datasets to the first dataset.
        Parameters
        ----------
        data:
            DataArray or list of DataArrays to be plotted
        run_ids:
            List of run IDs corresponding to each dataset
        labels:
            Label or list of labels for each dataset
        tag:
            Tag to be added to the plot title and filename
        x_dim:
            Dimension to be used for the x-axis. The code will average over all other dimensions.
        y_dim:
            Name of the dimension to be used for the y-axis.
        print_summary:
            If True, print a summary of the values from the graph.
        Returns
        -------
            None
        """

        data_list, label_list = self._check_lengths(data, labels)

        if len(data_list) < 2:
            _logger.warning("Ratio plot requires at least two datasets to compare. Skipping.")
            return

        baseline_name = self.baseline
        baseline_idx = run_ids.index(self.baseline) if self.baseline in run_ids else None
        if baseline_idx is not None:
            _logger.info(f"Using baseline run ID '{self.baseline}' for ratio plot.")
            baseline = data_list[baseline_idx]

        else:
            baseline_name = run_ids[0]
            baseline = data_list[0]

        ref_raw = self._preprocess_data(baseline, x_dim, verbose=False)

        channel_names = set(ref_raw.channel.values)
        # Merge channels from remaining datasets
        for data in data_list[1:]:
            channel_names.update(data.channel.values)  # add new channels

        # Sort the merged list
        ref_channel_names = sorted(channel_names)

        ref = align_labels(ref_raw, ref_channel_names, x_dim).reindex(channel=ref_channel_names)

        fig = plt.figure(figsize=(max(12, len(ref_channel_names) * 0.25), 6))

        for data, run_id, lbl in zip(data_list, run_ids, label_list, strict=False):
            if run_id == baseline_name:
                continue  # skip baseline

            num_raw = self._preprocess_data(data, x_dim, verbose=False)
            num = align_labels(num_raw, ref_channel_names, x_dim).reindex(channel=ref_channel_names)

            ratio = num.sel(channel=ref_channel_names) / ref.sel(channel=ref_channel_names)

            plt.plot(
                ref_channel_names,
                ratio.values,
                label=lbl,
                marker="o",
                linestyle="-",
            )

        parts = ["ratio_plot", tag]
        name = "_".join(filter(None, parts))
        plt.xticks(rotation=90, ha="right")
        plt.grid(True, linestyle="--", color="gray", alpha=0.5)
        title = f"Ratio plot {tag.split('_')[0]} - {tag.split('_')[-1]} (baseline: {baseline_name})"
        self._plot_base(fig, name, x_dim, y_dim, print_summary, line=1.0, title=title)

    def heat_map(
        self,
        data: xr.DataArray | list,
        labels: str | list,
        metric: str,
        tag: str = "",
    ) -> None:
        """
        Plot a heat map comparing multiple datasets.
        Parameters
        ----------
        data:
            DataArray or list of DataArrays to be plotted
        labels:
            Label or list of labels for each dataset
        metric:
            Metric for which we are plotting
        tag:
            Tag to be added to the plot title and filename
        Returns
        -------
            None
        """

        data_list, label_list = self._check_lengths(data, labels)

        fsteps = sorted(data_list[0].forecast_step.values)
        n_runs = len(data_list)

        x_ticks_names = set()
        for data in data_list:
            da = data.sel(forecast_step=fsteps[0])
            x_ticks_names.update(map(str, da.channel.values))

        ref_ticks_names = sorted(x_ticks_names)

        fig, axes = plt.subplots(
            1, n_runs, figsize=(8 * n_runs, max(12, len(ref_ticks_names) * 0.25)), squeeze=False
        )

        global_min = float("inf")
        global_max = float("-inf")

        for ax, data, label in zip(axes[0], data_list, labels, strict=False):
            ref = self._preprocess_data(
                data.sel(channel=ref_ticks_names, forecast_step=fsteps[0]), "channel", verbose=False
            )

            num = self._preprocess_data(data, ["forecast_step", "channel"], verbose=False)
            num = num.sel(channel=ref_ticks_names, forecast_step=fsteps)

            heatmap_data = num / ref

            cmap = plt.get_cmap("magma_r") if lower_is_better(metric) else plt.get_cmap("magma")
            global_min = min(global_min, float(heatmap_data.min()))
            global_max = max(global_max, float(heatmap_data.max()))

            last_hm = sns.heatmap(
                heatmap_data.values.T,
                ax=ax,
                cmap=cmap,
                vmin=global_min,
                vmax=global_max,
                xticklabels=fsteps,
                yticklabels=ref_ticks_names,
                annot=False,
                fmt=".2f",
                cbar=False,
            )
            ax.set_title(f"Heatmap {metric} – {label}")
            ax.set_xlabel("Forecast Step (h)")
            ax.set_ylabel("Variable")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        cbar = fig.colorbar(
            last_hm.collections[0], ax=axes.ravel().tolist(), shrink=0.6, location="right", pad=0.02
        )
        cbar.set_label(f"{metric} - fstep[0]/fstep[x]")
        parts = ["heat_map", metric, tag]
        name = "_".join(filter(None, parts))
        plt.savefig(f"{self.out_plot_dir.joinpath(name)}.{self.image_format}")


class ScoreCards:
    """
    Initialize the ScoreCards class.

    Parameters
    ----------
    plotter_cfg:
        Configuration dictionary containing basic information for plotting.
        Expected keys are:
            - image_format: Format of the saved images (e.g., 'png', 'pdf', etc.)
            - improvement: Size of the figure (width, height) in inches
    output_basedir:
        Base directory under which the score cards will be saved.
    """

    def __init__(self, plotter_cfg: dict, output_basedir: str | Path) -> None:
        self.image_format = plotter_cfg.get("image_format")
        self.dpi_val = plotter_cfg.get("dpi_val")
        self.improvement = plotter_cfg.get("improvement_scale", 0.2)
        self.out_plot_dir = Path(output_basedir) / "score_cards"
        self.baseline = plotter_cfg.get("baseline")
        if not os.path.exists(self.out_plot_dir):
            _logger.info(f"Creating dir {self.out_plot_dir}")
            os.makedirs(self.out_plot_dir, exist_ok=True)

    def plot(
        self,
        data: list[xr.DataArray],
        runs: list[str],
        metric: str,
        channels: list[str],
        tag: str,
    ) -> None:
        """
        Plot score cards comparing performance between run_ids against a baseline over channels
        of interest.

        Parameters
        ----------
        data:
            List of (xarray) DataArrays with the scores (stream, region and metric specific)
        runs:
            List containing runs (in str format) to be compared (provided in the config)
        metric:
            Metric for which we are plotting
        channels:
            List containing channels (in str format) of interest (provided in the config)
        tag:
            Tag to be added to the plot title and filename
        """
        n_runs = len(runs)

        if self.baseline and self.baseline in runs:
            baseline_idx = runs.index(self.baseline)
            runs = [runs[baseline_idx]] + runs[:baseline_idx] + runs[baseline_idx + 1 :]
            data = [data[baseline_idx]] + data[:baseline_idx] + data[baseline_idx + 1 :]

        common_channels, n_common_channels = self.extract_common_channels(data, channels, n_runs)

        fig, ax = plt.subplots(figsize=(2 * n_runs, 1.2 * n_common_channels))

        baseline = data[0]
        skill_models = []
        for run_index in range(1, n_runs):
            skill_model = 0.0
            for var_index, var in enumerate(common_channels):
                if var not in data[0].channel.values or var not in data[run_index].channel.values:
                    continue
                diff, avg_diff, avg_skill = self.compare_models(
                    data, baseline, run_index, var, metric
                )
                skill_model += avg_skill.values

                # Get symbols based on difference and performance as well as coordinates
                # for the position of the triangles.

                x, y, alt, color, triangle, size = self.get_plot_symbols(
                    run_index, var_index, avg_skill, avg_diff, metric
                )

                ax.scatter(x, y, marker=triangle, color=color, s=size.values, zorder=3)

                # Perform Wilcoxon test
                if diff["forecast_step"].item() > 1.0:
                    stat, p = wilcoxon(diff, alternative=alt)

                    # Draw rectangle border for significance
                    if p < 0.05:
                        lw = 2 if p < 0.01 else 1
                        rect_color = color
                        rect = plt.Rectangle(
                            (x - 0.25, y - 0.25),
                            0.5,
                            0.5,
                            fill=False,
                            edgecolor=rect_color,
                            linewidth=lw,
                            zorder=2,
                        )
                        ax.add_patch(rect)

            skill_models.append(skill_model / n_common_channels)

        # Set axis labels
        ylabels = [
            f"{var}\n({baseline.coords['metric'].item().upper()}={baseline.sel(channel=var).mean().values.squeeze():.3f})"
            for var in common_channels
        ]
        xlabels = [
            f"{model_name}\nSkill: {skill_models[i]:.3f}" for i, model_name in enumerate(runs[1::])
        ]
        ax.set_xticks(np.arange(1, n_runs))
        ax.set_xticklabels(xlabels, fontsize=10)
        ax.set_yticks(np.arange(n_common_channels) + 0.5)
        ax.set_yticklabels(ylabels, fontsize=10)
        for label in ax.get_yticklabels():
            label.set_horizontalalignment("center")
            label.set_x(-0.17)
        ax.set_ylabel("Variable", fontsize=14)
        ax.set_title(
            f"Model Scorecard vs. Baseline '{runs[0]}'",
            fontsize=16,
            pad=20,
        )
        for x in np.arange(0.5, n_runs - 1, 1):
            ax.axvline(x, color="gray", linestyle="--", linewidth=0.5, zorder=0, alpha=0.5)
        ax.set_xlim(0.5, n_runs - 0.5)
        ax.set_ylim(0, n_common_channels)

        legend = [
            Line2D(
                [0],
                [0],
                marker="^",
                color="white",
                label=f"{self.improvement * 100:.0f}% improvement",
                markerfacecolor="blue",
                markersize=np.sqrt(200),
            )
        ]
        plt.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.02, 1.0))

        _logger.info(f"Saving scorecards to: {self.out_plot_dir}")

        parts = ["score_card", tag] + runs
        name = "_".join(filter(None, parts))
        plt.savefig(
            f"{self.out_plot_dir.joinpath(name)}.{self.image_format}",
            bbox_inches="tight",
            dpi=self.dpi_val,
        )
        plt.close(fig)

    def extract_common_channels(self, data, channels, n_runs):
        common_channels = []
        for run_index in range(1, n_runs):
            for var in channels:
                if var not in data[0].channel.values or var not in data[run_index].channel.values:
                    continue
                common_channels.append(var)
        common_channels = list(set(common_channels))
        n_vars = len(common_channels)
        return common_channels, n_vars

    def compare_models(
        self,
        data: list[xr.DataArray],
        baseline: xr.DataArray,
        run_index: int,
        var: str,
        metric: str,
        x_dim="forecast_step",
    ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """
        Compare a model with a baseline model and calculate skill scores.

        Parameters
        ----------
        data: list[xr.DataArray]
            List of all scores in xarray format for each model.

        baseline: xarray DataArray
            The baseline scores in xarrays format.

        run_index: int
            The order index over the run_ids.

        var: str
            The specified channel over which we compare.

        xdim: str
            The dimension for which an average will not be calculated.

        Returns
        ----------
        diff: xr.DataArray
            Difference in scores between baseline and model.

        diff.mean(dim="forecast_step"): xr.DataArray
            Average difference in scores over all forecast steps between baseline and model .

        skill.mean(dim="forecast_step"): xr.DataArray
            Average skill scores over all forecast steps between baseline and model .

        """
        baseline_var = baseline.sel({"channel": var})
        data_var = data[run_index].sel({"channel": var})

        baseline_score, model_score = calculate_average_over_dim(x_dim, baseline_var, data_var)
        diff = baseline_score - model_score

        skill = self.get_skill_score(model_score, baseline_score, metric)
        return diff, diff.mean(dim=x_dim), skill.mean(dim=x_dim)

    def get_skill_score(
        self, score_model: xr.DataArray, score_ref: xr.DataArray, metric: str
    ) -> xr.DataArray:
        """
        Calculate skill score comparing a model against a baseline.

        Skill score is defined as: (model_score - baseline_score) / (perfect_score - baseline_score)

        Parameters
        ----------
        score_model : xr.DataArray
            The scores of the model being evaluated
        score_ref : xr.DataArray
            The scores of the reference/baseline model
        metric : str
            The metric name for which to calculate skill score

        Returns
        -------
        xr.DataArray
            Skill scores comparing model to baseline
        """
        perf_score = self.get_perf_score(metric)
        skill_score = (score_model - score_ref) / (perf_score - score_ref)
        return skill_score

    def get_perf_score(self, metric: str) -> float:
        """
        Get the perfect score for a given metric.

        Perfect scores represent ideal performance:
        - Error metrics: 0 (lower is better)
        - Skill/score metrics: 1 (higher is better)
        - PSNR: 100 (higher is better)

        Parameters
        ----------
        metric : str
            Metric name

        Returns
        -------
        float
            Perfect score for the specified metric
        """
        # Metrics where lower values indicate better performance (error metrics)
        if lower_is_better(metric):
            return 0.0

        # Metrics where higher values indicate better performance (with specific perfect score)
        elif metric in ["psnr"]:
            return 100.0

        # Metrics where higher values indicate better performance (default perfect score)
        else:
            return 1.0

    def get_plot_symbols(
        self,
        run_index: int,
        var_index: int,
        avg_skill: xr.DataArray,
        avg_diff: xr.DataArray,
        metric: str,
    ) -> tuple[int, float, str, str, str, xr.DataArray]:
        """
        Determine plot symbol properties based on performance difference.

        Parameters
        ----------
        run_index : int
            Index of the model.
        var_index : int
            Index of the variable/channel.
        avg_skill : xr.DataArray
            Average skill score of the model.
        avg_diff : xr.DataArray
            Average difference between baseline and model.
        metric : str
            Metric used for interpretation.

        Returns
        -------
        Tuple[int, float, str, str, str, xr.DataArray]
            x, y coordinates, alternative hypothesis, color, triangle symbol, size.
        """
        # Conservative choice
        alt = "two-sided"
        modus = "different"
        color = "gray"

        # Determine if diff_mean indicates improvement
        is_improvement = (avg_diff > 0 and lower_is_better(metric)) or (
            avg_diff < 0 and not lower_is_better(metric)
        )

        if is_improvement:
            alt = "greater"
            modus = "better"
            color = "blue"
        elif not is_improvement and avg_diff != 0:
            alt = "less"
            modus = "worse"
            color = "red"
        else:
            alt = "two-sided"
            modus = "different"

        triangle = "^" if modus == "better" else "v"

        # Triangle coordinates
        x = run_index
        # First row is model 1 vs model 0
        y = var_index + 0.5

        size = 200 * (1 - (1 / (1 + abs(avg_skill) / self.improvement)))  # Add base size to all

        return x, y, alt, color, triangle, size


class BarPlots:
    """
    Initialize the BarPlots class.

    Parameters
    ----------
    plotter_cfg:
        Configuration dictionary containing basic information for plotting.
        Expected keys are:
            - image_format: Format of the saved images (e.g., 'png', 'pdf', etc.)
            - improvement: Size of the figure (width, height) in inches
    output_basedir:
        Base directory under which the score cards will be saved.
    """

    def __init__(self, plotter_cfg: dict, output_basedir: str | Path) -> None:
        self.image_format = plotter_cfg.get("image_format")
        self.dpi_val = plotter_cfg.get("dpi_val")
        self.cmap = plotter_cfg.get("cmap", "bwr")
        self.out_plot_dir = Path(output_basedir) / "bar_plots"
        self.baseline = plotter_cfg.get("baseline")
        _logger.info(f"Saving bar plots to: {self.out_plot_dir}")
        if not os.path.exists(self.out_plot_dir):
            _logger.info(f"Creating dir {self.out_plot_dir}")
            os.makedirs(self.out_plot_dir, exist_ok=True)

    def plot(
        self,
        data: list[xr.DataArray],
        runs: list[str],
        metric: str,
        channels: list[str],
        tag: str,
    ) -> None:
        """
        Plot (ratio) bar plots comparing performance between different run_ids over channels of
        interest.

        Parameters
        ----------
        data:
            List of (xarray) DataArrays with the scores (stream, region and metric specific)
        runs:
            List containing runs (in str format) to be compared (provided in the config)
        metric:
            Metric name
        channels:
            List containing channels (in str format) of interest (provided in the config)
        tag:
            Tag to be added to the plot title and filename
        """

        fig, ax = plt.subplots(
            1,
            len(runs) - 1,
            figsize=(5 * len(runs), 2 * len(channels)),
            dpi=self.dpi_val,
            squeeze=False,
        )
        ax = ax.flatten()

        if self.baseline and self.baseline in runs:
            baseline_idx = runs.index(self.baseline)
            runs = [runs[baseline_idx]] + runs[:baseline_idx] + runs[baseline_idx + 1 :]
            data = [data[baseline_idx]] + data[:baseline_idx] + data[baseline_idx + 1 :]

        for run_index in range(1, len(runs)):
            ratio_score, channels_per_comparison = self.calc_ratio_per_run_id(
                data, channels, run_index
            )
            if len(ratio_score) > 0:
                ax[run_index - 1].barh(
                    np.arange(len(ratio_score)),
                    ratio_score,
                    color=self.colors(ratio_score, metric),
                    align="center",
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax[run_index - 1].set_yticks(
                    np.arange(len(ratio_score)), labels=channels_per_comparison
                )
                ax[run_index - 1].invert_yaxis()
                ax[run_index - 1].set_xlabel(
                    f"Relative {data[0].coords['metric'].item().upper()}: "
                    f"Target Model ({runs[run_index]}) / Reference Model ({runs[0]})"
                )
            else:
                ax[run_index - 1].set_visible(False)  # or annotate as missing
                # Or show a message:
                ax[run_index - 1].text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    transform=ax[run_index - 1].transAxes,
                )

        _logger.info(f"Saving bar plots to: {self.out_plot_dir}")
        parts = ["bar_plot_compare", tag] + runs
        name = "_".join(filter(None, parts))
        plt.savefig(
            f"{self.out_plot_dir.joinpath(name)}.{self.image_format}",
            bbox_inches="tight",
            dpi=self.dpi_val,
        )
        plt.close(fig)

    def calc_ratio_per_run_id(
        self,
        data: list[xr.DataArray],
        channels: list[str],
        run_index: int,
        x_dim="channel",
    ) -> tuple[np.array, str]:
        """
        This function calculates the ratio per comparison model for each channel.

        Parameters
        ----------
        data: list[xr.DataArray]
            List of all scores for each model in xarrays format.
        channels: list[str]
            All the available channels.
        run_index: int
            The order index over the run_ids.
        xdim: str
            The dimension for which an average will not be calculated.

        Returns
        ----------
        ratio_score: np.array
            The (ratio) skill over each channel for a specific model
        channels_per_comparison: str
            The common channels over which the baseline and the other model will be compared.

        """
        ratio_score = []
        channels_per_comparison = []
        for _, var in enumerate(channels):
            if var not in data[0].channel.values or var not in data[run_index].channel.values:
                continue
            baseline_var = data[0].sel({"channel": var})
            data_var = data[run_index].sel({"channel": var})
            channels_per_comparison.append(var)

            baseline_score, model_score = calculate_average_over_dim(x_dim, baseline_var, data_var)

            ratio_score.append(model_score / baseline_score)

        ratio_score = np.array(ratio_score) - 1
        return ratio_score, channels_per_comparison

    def colors(self, ratio_score: np.array, metric: str) -> list[tuple]:
        """
        This function calculates colormaps based on the skill scores. From negative value blue
        color variations should be given otherwise red color variations should be given.

        Parameters
        ----------
        ratio_score: np.array
            The (ratio) skill for a specific model
        metric: str
            The metric of interest
        Returns
        ----------
        colors: list[tuple]
            The color magnitude (blue to red) of the bars in the plots
        """
        max_val = np.abs(ratio_score).max()
        if lower_is_better(metric):
            cmap = plt.get_cmap("bwr")
        else:
            cmap = plt.get_cmap("bwr_r")
        colors = [cmap(0.5 + v / (2 * max_val)) for v in ratio_score]
        return colors


def calculate_average_over_dim(
    x_dim: str, baseline_var: xr.DataArray, data_var: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate average over xarray dimensions that are larger than 1. Those might be the
    forecast-steps or the samples.

    Parameters
    ----------
    xdim: str
        The dimension for which an average will not be calculated.
    baseline_var: xr.DataArray
        xarray DataArray with the scores of the baseline model for a specific channel/variable
    data_var: xr.DataArray
        xarray DataArray with the scores of the comparison model for a specific channel/variable

    Returns
    -------
    baseline_score: xarray DataArray
        The baseline average scores over the dimensions not specified by xdim
    model_score: xarray DataArray
        The model average scores over the dimensions not specified by xdim
    """
    non_zero_dims = [
        dim for dim in baseline_var.dims if dim != x_dim and baseline_var[dim].shape[0] > 1
    ]

    if non_zero_dims:
        _logger.info(f"Found multiple entries for dimensions: {non_zero_dims}. Averaging...")

    baseline_score = baseline_var.mean(
        dim=[dim for dim in baseline_var.dims if dim != x_dim], skipna=True
    )
    model_score = data_var.mean(dim=[dim for dim in data_var.dims if dim != x_dim], skipna=True)

    return baseline_score, model_score


def lower_is_better(metric: str) -> bool:
    # Determine whether lower or higher is better
    return metric in {"l1", "l2", "mae", "mse", "rmse", "vrmse", "bias", "crps", "spread"}


def compute_offsets(n, spacing=0.11):
    idx = np.arange(n)
    return (idx - (n - 1) / 2.0) * spacing


def align_labels(da: xr.DataArray, labels: list[str], x_dim: str) -> xr.DataArray:
    """
    Reindex a DataArray to include all labels in the canonical order.
    Missing variables are filled with NaN.
    """
    # Convert labels → index format expected by xarray
    labels = np.array(labels, dtype=object)

    # Reindex, inserting NaN for missing labels
    return da.reindex({x_dim: labels})
