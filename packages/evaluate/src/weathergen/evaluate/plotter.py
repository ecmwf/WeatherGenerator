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
        
        out_dir = Path(cfg.output_dir)
        self.out_format = cfg.save_mode
        self.create_html = cfg.create_html
        self.dpi_val = OmegaConf.select(cfg, "dpi_val") or None
        self.fig_size = OmegaConf.select(cfg, "fig_size") or (8, 10)

        self.out_dir = out_dir.joinpath(self.out_format).joinpath(model_id) 
        overwrite = OmegaConf.select(cfg, "overwrite") or False
        print(self.out_dir)
        if overwrite:
            if os.path.exists(self.out_dir):
                shutil.rmtree(self.out_dir)  # Remove the existing directory
            os.makedirs(self.out_dir)
        else:
            os.makedirs(self.out_dir)

        self.sample   = None
        self.stream   = None
        self.fstep    = None
        self.model_id = model_id
        self.select   = {}
        
    def selection(self, sample: str, stream: str, fstep:str):
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

    def clean_selection(self):
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
    
    def update_selection(self, select: dict):
        self.select = select
        return self

    def histogram(self, target: xr.DataArray, preds: xr.DataArray, variables: list, tag = "", select: dict = {}):
        """
        Plot histogram of target vs predictions for a set of variables. 

        :param target: target sample for a specific (stream, sample, fstep)
        :param preds: predictions sample for a specific (stream, sample, fstep)
        :param variables: list of variables to be plotted
        :param label: any tag you want to add to the plot
        """
        for var in variables:
            select_var = self.select | select | {"channel" : var}
            fig = plt.figure(figsize=self.fig_size, dpi=self.dpi_val)

            #get common bin edges
            targ = target.sel(select_var)
            prds = preds.sel(select_var)
            vals = np.concatenate([targ, prds])
            bins = np.histogram_bin_edges(vals, bins=50)
            plt.hist(targ, bins=bins, alpha=0.7, label='Target')
            plt.hist(prds, bins=bins, alpha=0.7, label='Prediction')

            #set labels and title
            plt.xlabel(f"Variable: {var}")
            plt.ylabel('Frequency')
            plt.title(f"Histogram of Target and Prediction: {var}")
            plt.legend(frameon=False)

            #TODO: make this nicer
            parts = ["histogram", self.model_id, tag, str(self.sample), self.stream, str(self.fstep), var]
            name = "_".join(filter(None,parts))
            plt.savefig(f"{self.out_dir.joinpath(name)}.{self.out_format}")
            plt.close()
        #TODO: change the return when adding the dashboard support
        return

    def map(self, data: xr.DataArray, variables: list, tag: str = "", select: dict = {}):
        """
        Plot 2D map for a dataset

        :param data: DataArray for a specific (stream, sample, fstep)
        :param variables: list of variables to be plotted
        :param label: any tag you want to add to the plot
        :param select: selection to be applied to the DataArray
        """

        for var in variables:
            select_var = self.select | select | {"channel" : var}
            fig = plt.figure(dpi=self.dpi_val)
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
            ax.coastlines()   
            ds = data.sel(select_var)
            scatter_plt = ax.scatter(ds["lon"], ds["lat"], c=ds,
                                        cmap='coolwarm', s=10, transform=ccrs.PlateCarree())
            plt.colorbar(scatter_plt, ax=ax, orientation='horizontal', label=f"Variable: {var}")
            ax.set_global()
            
            #TODO: make this nicer
            parts = ["map",self.model_id, tag, str(self.sample), self.stream, str(self.fstep), var]
            name = "_".join(filter(None,parts))
            plt.savefig(f"{self.out_dir.joinpath(name)}.{self.out_format}")
            plt.close()
        #TODO: change the return when adding the dashboard support
        return



class LinePlots(object):

    def __init__(self, cfg: dict):
        self.cfg = cfg
        out_dir = Path(cfg.output_dir)
        self.out_format = cfg.save_mode
        self.create_html = cfg.create_html
        self.dpi_val = OmegaConf.select(cfg, "dpi_val") or None
        self.fig_size = OmegaConf.select(cfg, "fig_size") or (8, 10)

        self.out_dir = out_dir.joinpath(self.out_format).joinpath("line_plots")
        overwrite = OmegaConf.select(cfg, "overwrite") or False

        if overwrite:
            if os.path.exists(self.out_dir):
                shutil.rmtree(self.out_dir)  # Remove the existing directory
            os.makedirs(self.out_dir)
        else:
            os.makedirs(self.out_dir,exist_ok=True)

    def _check_lengths(self, data, labels):
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


    def plot(self, data: xr.DataArray | list, labels: str | list, tag: str = "", x = "forecast_step", y = "value"):
        """
        Plot a line graph comparing multiple datasets.
        :param data: DataArray or list of DataArrays to be plotted
        :param labels: Label or list of labels for each dataset
        :param tag: Tag to be added to the plot title and filename
        :param x: Dimension to be used for the x-axis. The code will average over all other dimensions. (default is "forecast_step")
        :param y: Name of the dimension to be used for the y-axis (default is "value")
        :return: None
        """

        data_list, label_list = self._check_lengths(data, labels)
        
        assert x in data_list[0].dims, f"Compare::plot - x dimension '{x}' not found in data dimensions {data_list[0].dims}"
        
        fig = plt.figure(figsize=self.fig_size, dpi=self.dpi_val)
        for i, data in enumerate(data_list):
            averaged = data.mean(dim=[dim for dim in data.dims if dim != x])
            plt.plot(averaged[x], averaged.values, label = label_list[i], marker = "o", linestyle = "-")

        xlabel = self.clean_string(x)
        plt.xlabel(xlabel)

        ylabel = self.clean_string(y)
        plt.ylabel(ylabel)
        
        title = self.clean_string(tag)
        plt.title(title)
        plt.legend(frameon=False)

        parts = ["compare", tag]
        name = "_".join(filter(None,parts))
        plt.savefig(f"{self.out_dir.joinpath(name)}.{self.out_format}")
        plt.close()

