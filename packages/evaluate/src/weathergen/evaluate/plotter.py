import os
import json
from typing import List
import zarr
from random import randint
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from omegaconf import OmegaConf

import xarray as xr
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import shutil
import logging
logging.getLogger("matplotlib.category").setLevel(logging.ERROR)

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


import cartopy
import cartopy.crs as ccrs  #https://scitools.org.uk/cartopy/docs/latest/installing.html
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature


class Plotter(object):
    """
    Contains all basic plotting functions.
    """
    def __init__ (self, cfg: dict, model_id: str = ""):
        """
        Initialize the Plotter class.
        :param cfg: config from the yaml file
        :param model_id: if a model_id is given, the output will be saved in a folder called as the model_id
        """
        
        self.cfg = cfg
        
        out_plot_dir = Path(cfg.output_plotting_dir)
        self.image_format = cfg.image_format
        self.create_html = cfg.create_html
        self.dpi_val = cfg.get("dpi_val", None)
        self.fig_size = cfg.get("fig_size", (8, 10)) 

        self.out_plot_dir = out_plot_dir.joinpath(self.image_format).joinpath(model_id) 

        os.makedirs(self.out_plot_dir, exist_ok=True)

        self.sample   = None
        self.stream   = None
        self.fstep    = None
        self.model_id = model_id
        self.select   = {}


    def update_data_selection(self, sample: str, stream: str, fstep:str):
        """
        Set the selection for the plots. This will be used to filter the data for plotting.
        :param sample: sample name
        :param stream: stream name
        :param fstep: forecasting step
        """
        self.sample = sample
        self.stream = stream 
        self.fstep  = fstep
        self.select = { "sample" : self.sample, 
                        "stream" : self.stream, 
                        "forecast_step"  : self.fstep }
        return self
    
    def update_data_selection(self, select: dict):
        """
        Set the selection for the plots. This will be used to filter the data for plotting.
        :param select: dictionary containing the selection parameters
        """
        self.select = select
       
        if not "sample" in select:
            _logger.warning("No sample in the selection. Might lead to unexpected results.")
        else:
            self.sample = select["sample"]

        if not "stream" in select:
            _logger.warning("No stream in the selection. Might lead to unexpected results.")
        else:
            self.stream = select["stream"]  
      
        if not "forecast_step" in select:
            _logger.warning("No forecast_step in the selection. Might lead to unexpected results.")
        else:
            self.fstep = select["forecast_step"]    

        return self
    
    def clean_data_selection(self):
        """
        :param sample: sample name
        :param stream: stream name
        :param fstep: forecasting step
        """
        self.sample = None
        self.stream = None
        self.fstep  = None
        self.select = {}
        return self
    

    def select_from_da(self, da: xr.DataArray, selection: dict) -> xr.DataArray:
        """
        Select data from an xarray DataArray based on given selectors.
        :param da: xarray DataArray to select data from.
        :param selection: Dictionary of selectors where keys are coordinate names and values are the values to select.
        :return: xarray DataArray with selected data.
        """
        for key, value in selection.items():
            if key in da.coords and key not in da.dims:
                # Coordinate like 'sample' aligned to another dim
                da = da.where(da[key] == value, drop=True)
            else:
                # Scalar coord or dim coord (e.g., 'forecast_step', 'channel')
                da = da.sel({key: value})
        return da


    def histogram(self, target: xr.DataArray, preds: xr.DataArray, variables: list, select: dict, tag = ""):
        """
        Plot histogram of target vs predictions for a set of variables. 

        :param target: target sample for a specific (stream, sample, fstep)
        :param preds: predictions sample for a specific (stream, sample, fstep)
        :param variables: list of variables to be plotted
        :param label: any tag you want to add to the plot
        """
        plot_names = []
        
        self.update_data_selection(select)
        
        for var in variables:
            select_var = self.select | {"channel" : var}
            fig = plt.figure(figsize=self.fig_size, dpi=self.dpi_val)

            #get common bin edges
            targ, prd = self.select_from_da(target, select_var), self.select_from_da(preds, select_var)
            vals = np.concatenate([targ, prd])
            bins = np.histogram_bin_edges(vals, bins=50)
            plt.hist(targ, bins=bins, alpha=0.7, label='Target')
            plt.hist(prd, bins=bins, alpha=0.7, label='Prediction')

            #set labels and title
            plt.xlabel(f"Variable: {var}")
            plt.ylabel("Frequency")
            plt.title(f"Histogram of Target and Prediction: {var}")
            plt.legend(frameon=False)

            #TODO: make this nicer
            parts = ["histogram", self.model_id, tag, str(self.sample), self.stream, str(self.fstep), var]
            name = "_".join(filter(None,parts))
            plt.savefig(f"{self.out_plot_dir.joinpath(name)}.{self.image_format}")
            plt.close()
            plot_names.append(name)

        self.clean_data_selection()

        return plot_names

    def map(self, data: xr.DataArray, variables: list, select: dict, tag: str = ""):
        """
        Plot 2D map for a dataset

        :param data: DataArray for a specific (stream, sample, fstep)
        :param variables: list of variables to be plotted
        :param label: any tag you want to add to the plot
        :param select: selection to be applied to the DataArray
        """
        
        self.update_data_selection(select)

        plot_names = []
        for var in variables:
            select_var = self.select | {"channel" : var}
            fig = plt.figure(dpi=self.dpi_val)
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
            ax.coastlines()   
            da = self.select_from_da(data, select_var).compute()
            scatter_plt = ax.scatter(da["lon"], da["lat"], c=da,
                                        cmap='coolwarm', s=1, transform=ccrs.PlateCarree())
            plt.colorbar(scatter_plt, ax=ax, orientation='horizontal', label=f"Variable: {var}")
            ax.set_global()
            ax.gridlines(draw_labels=False, linestyle="--", color="black", linewidth=1)

            #TODO: make this nicer
            parts = ["map",self.model_id, tag, str(self.sample), self.stream, str(self.fstep), var]
            name = "_".join(filter(None,parts))
            plt.savefig(f"{self.out_plot_dir.joinpath(name)}.{self.image_format}")
            plt.close()
            plot_names.append(name)
        
        self.clean_data_selection()

        return plot_names



class LinePlots(object):

    def __init__(self, cfg: dict):
        self.cfg = cfg
        out_plot_dir = Path(cfg.output_plotting_dir)
        self.image_format = cfg.image_format
        self.create_html = cfg.create_html
        self.dpi_val = cfg.get("dpi_val", None)
        self.fig_size = cfg.get("fig_size", (8, 10))

        self.out_plot_dir = out_plot_dir.joinpath(self.image_format).joinpath("line_plots")
        os.makedirs(self.out_plot_dir,exist_ok=True)

    def _check_lengths(self, data: xr.DataArray | list, labels: str | list):
        """
        Check if the lengths of data and labels match.
        :param data: DataArray or list of DataArrays to be plotted
        :param labels: Label or list of labels for each dataset
        :return: data_list, label_list - lists of data and labels
        """
        assert type(data) == xr.DataArray or type(data) == list, "Compare::plot - Data should be of type xr.DataArray or list"
        assert type(labels) == str or type(labels) == list, "Compare::plot - Labels should be of type str or list"

        # convert to lists
        
        data_list = [data] if type(data) == xr.DataArray else data
        label_list = [labels] if type(labels) == str else labels

        assert len(data_list) == len(label_list), "Compare::plot - Data and Labels do not match"
        
        return data_list, label_list

    def clean_string(self, string: str):
        """
        Clean a string by removing non-alphanumeric characters and replacing spaces with underscores.
        :param string: String to be cleaned
        :return: Cleaned string
        """
        return ''.join(c if c.isalnum() else ' ' for c in string)   


    def plot(self, data: xr.DataArray | list, labels: str | list, tag: str = "", x_dim = "forecast_step", y_dim = "value"):
        """
        Plot a line graph comparing multiple datasets.
        :param data: DataArray or list of DataArrays to be plotted
        :param labels: Label or list of labels for each dataset
        :param tag: Tag to be added to the plot title and filename
        :param x_dim: Dimension to be used for the x-axis. The code will average over all other dimensions. (default is "forecast_step")
        :param y_dim: Name of the dimension to be used for the y-axis (default is "value")
        :return: None
        """

        data_list, label_list = self._check_lengths(data, labels)
        
        assert x_dim in data_list[0].dims, "x dimension '{x_dim}' not found in data dimensions {data_list[0].dims}"
        
        fig = plt.figure(figsize=self.fig_size, dpi=self.dpi_val)
        for i, data in enumerate(data_list):
            non_zero_dims = [dim for dim in data.dims if dim != x_dim and data[dim].shape[0] > 1]
            if non_zero_dims:
                logging.warning(f"LinePlot:: Found multiple entries for dimensions: {non_zero_dims}. Averaging...")
            averaged = data.mean(dim=[dim for dim in data.dims if dim != x_dim], skipna=True) 
            plt.plot(averaged[x_dim], averaged.values, label = label_list[i], marker = "o", linestyle = "-")

        xlabel = self.clean_string(x_dim)
        plt.xlabel(xlabel)

        ylabel = self.clean_string(y_dim)
        plt.ylabel(ylabel)
        
        title = self.clean_string(tag)
        plt.title(title)
        plt.legend(frameon=False)

        parts = ["compare", tag]
        name = "_".join(filter(None,parts))
        plt.savefig(f"{self.out_plot_dir.joinpath(name)}.{self.image_format}")
        plt.close()

