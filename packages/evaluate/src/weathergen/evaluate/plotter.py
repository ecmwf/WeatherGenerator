import glob
import logging
import os
from pathlib import Path

import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from PIL import Image

from weathergen.utils.config import _load_private_conf

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

    def __init__(self, plotter_cfg: dict, output_basedir: str | Path):
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
        """

        _logger.info(f"Taking cartopy paths from {work_dir}")

        self.image_format = plotter_cfg.get("image_format")
        self.dpi_val = plotter_cfg.get("dpi_val")
        self.fig_size = plotter_cfg.get("fig_size")
        self.plot_subtimesteps = plotter_cfg.get(
            "plot_subtimesteps", False
        )  # True if plots are created for each valid time separately
        self.run_id = output_basedir.name

        self.out_plot_basedir = Path(output_basedir) / "plots"

        if not os.path.exists(self.out_plot_basedir):
            _logger.info(f"Creating dir {self.out_plot_basedir}")
            os.makedirs(self.out_plot_basedir, exist_ok=True)

        self.sample = None
        self.stream = None
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
            _logger.warning(
                "No sample in the selection. Might lead to unexpected results."
            )
        else:
            self.sample = select["sample"]

        if "stream" not in select:
            _logger.warning(
                "No stream in the selection. Might lead to unexpected results."
            )
        else:
            self.stream = select["stream"]

        if "forecast_step" not in select:
            _logger.warning(
                "No forecast_step in the selection. Might lead to unexpected results."
            )
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
            Dictionary of selectors where keys are coordinate names and values are the values to select.

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

            if self.plot_subtimesteps:
                ntimes_unique = len(np.unique(targ.valid_time))
                _logger.info(
                    f"Creating histograms for {ntimes_unique} valid times of variable {var}."
                )

                groups = zip(
                    targ.groupby("valid_time"), prd.groupby("valid_time"), strict=False
                )
            else:
                _logger.info(f"Plotting histogram for all valid times of {var}")

                groups = [
                    ((None, targ), (None, prd))
                ]  # wrap once with dummy valid_time

            for (valid_time, targ_t), (_, prd_t) in groups:
                if valid_time is not None:
                    _logger.debug(
                        f"Plotting histogram for {var} at valid_time {valid_time}"
                    )

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
            f"Histogram of Target and Prediction: {self.stream}, {varname} : fstep = {self.fstep:03}"
        )
        plt.legend(frameon=False)

        valid_time = str(target_data["valid_time"][0].values.astype("datetime64[m]"))

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
        logging.debug(f"Saving histogram to {fname}")
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
                - scale_marker_size: if True, the marker size will be scaled based on latitude (default is False)
                - marker: marker style (default is 'o')
            Unknown keys will be passed to the scatter plot function.

        Returns
        -------
            List of plot names for the saved maps.
        """
        self.update_data_selection(select)

        map_kwargs_save = (
            {
                key: value
                for key, value in map_kwargs.copy().items()
                if key not in variables
            }
            if map_kwargs is not None
            else {}
        )

        if not hasattr(map_kwargs_save, "marker_size"):
            # Set default marker size if not specified in maps_config
            map_kwargs_save["marker_size"] = DefaultMarkerSize.get_marker_size(
                self.stream
            )

        # Basic map output directory for this stream
        map_output_dir = self.get_map_output_dir(tag)

        if not os.path.exists(map_output_dir):
            _logger.info(f"Creating dir {map_output_dir}")
            os.makedirs(map_output_dir)

        plot_names = []
        for var in variables:
            select_var = self.select | {"channel": var}
            da = self.select_from_da(data, select_var).compute()

            if self.plot_subtimesteps:
                ntimes_unique = len(np.unique(da.valid_time))
                _logger.info(
                    f"Creating maps for {ntimes_unique} valid times of variable {var} - {tag}"
                )

                groups = da.groupby("valid_time")
            else:
                _logger.info(f"Creating maps for all valid times of {var} - {tag}")
                groups = [(None, da)]  # single dummy group

            for valid_time, da_t in groups:
                if valid_time is not None:
                    _logger.debug(f"Plotting map for {var} at valid_time {valid_time}")

                name = self.scatter_plot(
                    da_t,
                    map_output_dir,
                    var,
                    tag=tag,
                    map_kwargs=map_kwargs[var],
                )
                plot_names.append(name)

        self.clean_data_selection()

        return plot_names

    def scatter_plot(
        self,
        data: xr.DataArray,
        map_output_dir: Path,
        varname: str,
        tag: str = "",
        map_kwargs: dict | None = None,
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
        tag: str
            Any tag you want to add to the plot
        map_kwargs: dict | None
            Additional keyword arguments for the map.

        Returns
        -------
            Name of the saved plot file.
        """
        map_kwargs_save = map_kwargs.copy() if map_kwargs is not None else {}
        # check for known keys in map_kwargs
        marker_size_base = map_kwargs_save.pop("marker_size", 1)
        scale_marker_size = map_kwargs_save.pop("scale_marker_size", False)
        marker = map_kwargs_save.pop("marker", "o")
        vmin = map_kwargs_save.pop("vmin", None)
        vmax = map_kwargs_save.pop("vmax", None)

        # Create figure and axis objects
        fig = plt.figure(dpi=self.dpi_val)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        ax.coastlines()

        marker_size = marker_size_base
        if scale_marker_size:
            marker_size = np.clip(
                marker_size / np.cos(np.radians(data["lat"])) ** 2,
                a_max=marker_size * 10.0,
                a_min=marker_size,
            )

        valid_time = str(data["valid_time"][0].values.astype("datetime64[m]"))

        scatter_plt = ax.scatter(
            data["lon"],
            data["lat"],
            c=data,
            cmap="coolwarm",
            s=marker_size,
            marker=marker,
            transform=ccrs.PlateCarree(),
            vmin=vmin,
            vmax=vmax,
            linewidths=0.0,  # only markers, avoids aliasing for very small markers
            **map_kwargs_save,
        )

        plt.colorbar(
            scatter_plt, ax=ax, orientation="horizontal", label=f"Variable: {varname}"
        )
        plt.title(f"{self.stream}, {varname} : fstep = {self.fstep:03} ({valid_time})")
        ax.set_global()
        ax.gridlines(draw_labels=False, linestyle="--", color="black", linewidth=1)

        # TODO: make this nicer
        parts = [
            "map",
            self.run_id,
            tag,
            str(self.sample),
            valid_time,
            self.stream,
            varname,
            str(self.fstep).zfill(3),
        ]

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

        for _, sa in enumerate(samples):
            for _, var in enumerate(variables):
                _logger.info(f"Creating animation for {var} sample: {sa} - {tag}")
                image_paths = []
                for _, fstep in enumerate(fsteps):
                    # TODO: refactor to avoid code duplication with scatter_plot
                    parts = [
                        "map",
                        self.run_id,
                        tag,
                        str(sa),
                        "*",
                        self.stream,
                        var,
                        str(fstep).zfill(3),
                    ]

                    name = "_".join(filter(None, parts))
                    fname = f"{map_output_dir.joinpath(name)}.{self.image_format}"

                    names = glob.glob(fname)
                    image_paths += names

                images = [Image.open(path) for path in image_paths]
                images[0].save(
                    f"{map_output_dir}/animation_{self.run_id}_{tag}_{sa}_{self.stream}_{var}.gif",
                    save_all=True,
                    append_images=images[1:],
                    duration=500,
                    loop=0,
                )

        return image_paths

    def get_map_output_dir(self, tag):
        return self.out_plot_basedir / self.stream / "maps" / tag


class LinePlots:
    def __init__(self, cfg: dict, output_basedir: str | Path):
        self.cfg = cfg
        self.image_format = cfg.image_format
        self.dpi_val = cfg.get("dpi_val")
        self.fig_size = cfg.get("fig_size", (8, 10))
        self.log_scale = cfg.evaluation.get("log_scale", False)
        self.add_grid = cfg.evaluation.get("add_grid", False)
        self.out_plot_dir = Path(output_basedir) / "line_plots"
        if not os.path.exists(self.out_plot_dir):
            _logger.info(f"Creating dir {self.out_plot_dir}")
            os.makedirs(self.out_plot_dir, exist_ok=True)

        _logger.info(f"Saving summary plots to: {self.out_plot_dir}")

    def _check_lengths(
        self, data: xr.DataArray | list, labels: str | list
    ) -> tuple[list, list]:
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
        assert type(data) == xr.DataArray or type(data) == list, (
            "Compare::plot - Data should be of type xr.DataArray or list"
        )
        assert type(labels) == str or type(labels) == list, (
            "Compare::plot - Labels should be of type str or list"
        )

        # convert to lists

        data_list = [data] if type(data) == xr.DataArray else data
        label_list = [labels] if type(labels) == str else labels

        assert len(data_list) == len(label_list), (
            "Compare::plot - Data and Labels do not match"
        )

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

    def plot(
        self,
        data: xr.DataArray | list,
        labels: str | list,
        tag: str = "",
        x_dim: str = "forecast_step",
        y_dim: str = "value",
        print_summary: bool = False,
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
        """

        data_list, label_list = self._check_lengths(data, labels)

        assert x_dim in data_list[0].dims, (
            "x dimension '{x_dim}' not found in data dimensions {data_list[0].dims}"
        )

        fig = plt.figure(figsize=(12, 6), dpi=self.dpi_val)

        for i, data in enumerate(data_list):
            non_zero_dims = [
                dim for dim in data.dims if dim != x_dim and data[dim].shape[0] > 1
            ]
            if non_zero_dims:
                logging.info(
                    f"LinePlot:: Found multiple entries for dimensions: {non_zero_dims}. Averaging..."
                )
            averaged = data.mean(
                dim=[dim for dim in data.dims if dim != x_dim], skipna=True
            ).sortby(x_dim)

            plt.plot(
                averaged[x_dim],
                averaged.values,
                label=label_list[i],
                marker="o",
                linestyle="-",
            )

        xlabel = "".join(c if c.isalnum() else " " for c in x_dim)
        plt.xlabel(xlabel)

        ylabel = "".join(c if c.isalnum() else " " for c in y_dim)
        plt.ylabel(ylabel)

        title = "".join(c if c.isalnum() else " " for c in tag)
        plt.title(title)
        plt.legend(frameon=False)

        if self.add_grid:
            plt.grid(True, linestyle="--", color="gray", alpha=0.5)

        if self.log_scale:
            plt.yscale("log")

        if print_summary:
            _logger.info(f"Summary values for {tag}")
            self.print_all_points_from_graph(fig)

        parts = ["compare", tag]
        name = "_".join(filter(None, parts))
        plt.savefig(f"{self.out_plot_dir.joinpath(name)}.{self.image_format}")
        plt.close()


class DefaultMarkerSize:
    """
    Utility class for managing default configuration values, such as marker sizes
    for various data streams.
    """

    _marker_size_stream = {
        "era5": 2.5,
        "imerg": 0.25,
        "cerra": 0.1,
    }

    _default_marker_size = 0.5

    @classmethod
    def get_marker_size(cls, stream_name: str) -> float:
        """
        Get the default marker size for a given stream name.

        Parameters
        ----------
        stream_name : str
            The name of the stream.

        Returns
        -------
        float
            The default marker size for the stream.
        """
        return cls._marker_size_stream.get(
            stream_name.lower(), cls._default_marker_size
        )

    @classmethod
    def list_streams(cls):
        """
        List all streams with defined marker sizes.

        Returns
        -------
        list[str]
            List of stream names.
        """
        return list(cls._marker_size_stream.keys())
