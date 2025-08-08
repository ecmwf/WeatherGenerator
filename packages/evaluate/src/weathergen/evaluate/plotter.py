import logging
import os
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from PIL import Image

np.seterr(divide="ignore", invalid="ignore")

logging.getLogger("matplotlib.category").setLevel(logging.ERROR)

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class Plotter:
    """
    Contains all basic plotting functions.
    """

    def __init__(self, cfg: dict, model_id: str = ""):
        """
        Initialize the Plotter class.

        Parameters
        ----------
        cfg:
            Configuration dictionary containing all information for the plotting.
        model_id:
            If a model_id is given, the output will be saved in a folder called as the model_id.
        """

        self.cfg = cfg

        out_plot_dir = Path(cfg.output_plotting_dir)
        self.image_format = cfg.image_format
        self.dpi_val = cfg.get("dpi_val")
        self.fig_size = cfg.get("fig_size", (8, 10))

        self.out_plot_dir = out_plot_dir.joinpath(self.image_format).joinpath(model_id)

        if not os.path.exists(self.out_plot_dir):
            _logger.info(f"Creating dir {self.out_plot_dir}")
            os.makedirs(self.out_plot_dir, exist_ok=True)

        self.sample = None
        self.stream = None
        self.fstep = None
        self.model_id = model_id
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

    def histogram(
        self,
        target: xr.DataArray,
        preds: xr.DataArray,
        variables: list,
        select: dict,
        tag: str = "",
    ) -> list[str]:
        """
        Plot histogram of target vs predictions for a set of variables.

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

        for var in variables:
            select_var = self.select | {"channel": var}

            # get common bin edges
            targ, prd = (
                self.select_from_da(target, select_var),
                self.select_from_da(preds, select_var),
            )
            vals = np.concatenate([targ, prd])
            bins = np.histogram_bin_edges(vals, bins=50)
            plt.hist(targ, bins=bins, alpha=0.7, label="Target")
            plt.hist(prd, bins=bins, alpha=0.7, label="Prediction")

            # set labels and title
            plt.xlabel(f"Variable: {var}")
            plt.ylabel("Frequency")
            plt.title(
                f"Histogram of Target and Prediction: {self.stream}, {var} : fstep = {self.fstep:03}"
            )
            plt.legend(frameon=False)

            # TODO: make this nicer
            parts = [
                "histogram",
                self.model_id,
                tag,
                str(self.sample),
                self.stream,
                var,
                str(self.fstep).zfill(3),
            ]
            name = "_".join(filter(None, parts))
            plt.savefig(f"{self.out_plot_dir.joinpath(name)}.{self.image_format}")
            plt.close()
            plot_names.append(name)

        self.clean_data_selection()

        return plot_names

    def map(
        self,
        data: xr.DataArray,
        variables: list,
        select: dict,
        tag: str = "",
        map_kwargs: dict | None = None,
    ) -> list[str]:
        """
        Plot 2D map for a dataset

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
            Any tag you want to add to the plot
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
        map_kwargs_save = map_kwargs.copy() if map_kwargs is not None else {}
        # check for known keys in map_kwargs
        marker_size_base = map_kwargs_save.pop("marker_size", 1)
        scale_marker_size = map_kwargs_save.pop("scale_marker_size", False)
        marker = map_kwargs_save.pop("marker", "o")

        self.update_data_selection(select)

        plot_names = []
        for var in variables:
            select_var = self.select | {"channel": var}
            fig = plt.figure(dpi=self.dpi_val)
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
            ax.coastlines()
            da = self.select_from_da(data, select_var).compute()

            marker_size = marker_size_base
            if scale_marker_size:
                marker_size = (marker_size + 1.0) * np.cos(np.radians(da["lat"]))

            scatter_plt = ax.scatter(
                da["lon"],
                da["lat"],
                c=da,
                cmap="coolwarm",
                vmin=da["min"].item(),
                vmax=da["max"].item(),
                s=marker_size,
                marker=marker,
                transform=ccrs.PlateCarree(),
                linewidths=0.0,  # only markers, avoids aliasing for very small markers
                **map_kwargs_save,
            )
            plt.colorbar(
                scatter_plt, ax=ax, orientation="horizontal", label=f"Variable: {var}"
            )
            plt.title(
                f"{self.stream}, {var} : fstep = {self.fstep:03} ({da['valid_time'][0].values.astype('datetime64[s]')})"
            )
            ax.set_global()
            ax.gridlines(draw_labels=False, linestyle="--", color="black", linewidth=1)

            # TODO: make this nicer
            parts = [
                "map",
                self.model_id,
                tag,
                str(self.sample),
                self.stream,
                var,
                str(self.fstep).zfill(3),
            ]
            name = "_".join(filter(None, parts))
            fname = f"{self.out_plot_dir.joinpath(name)}.{self.image_format}"
            _logger.debug(f"Saving map to {fname}")
            plt.savefig(fname)
            plt.close()
            plot_names.append(name)

        self.clean_data_selection()

        return plot_names

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

        for _, sa in enumerate(samples):
            for _, var in enumerate(variables):
                image_paths = []
                for _, fstep in enumerate(fsteps):
                    image_paths.append(
                        f"{self.out_plot_dir}/map_{self.model_id}_{tag}_{sa}_{self.stream}_{var}_{fstep:03d}.png"
                    )

                images = [Image.open(path) for path in image_paths]
                images[0].save(
                    f"{self.out_plot_dir}/animation_{self.model_id}_{tag}_{sa}_{self.stream}_{var}.gif",
                    save_all=True,
                    append_images=images[1:],
                    duration=500,
                    loop=0,
                )

        return image_paths


class LinePlots:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        out_plot_dir = Path(cfg.output_plotting_dir)
        self.image_format = cfg.image_format
        self.dpi_val = cfg.get("dpi_val")
        self.fig_size = cfg.get("fig_size", (8, 10))

        self.out_plot_dir = out_plot_dir.joinpath(self.image_format).joinpath(
            "line_plots"
        )
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
