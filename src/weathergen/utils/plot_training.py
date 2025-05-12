# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import argparse
import json as js
import logging
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import weathergen.utils.config as config
from weathergen.utils.train_logger import Metrics, TrainLogger


####################################################################################################
def clean_plot_folder(plot_dir: Path = "./plots/"):
    """
    Clean the plot folder by removing all png-files in it.

    Parameters
    ----------
    plot_dir : Path
        Path to the plot directory
    """
    for image in plot_dir.glob("*.png"):
        image.unlink()


####################################################################################################
def get_stream_names(run_id: str, model_path: Path | None = "./model"):
    """
    Get the stream names from the model configuration file.

    Parameters
    ----------
    run_id : str
        ID of the training run
    model_path : Path
        Path to the model directory
    Returns
    -------
    -------
    list
        List of stream names
    """
    # return col names from training (should be identical to validation)
    cf = config.load_model_config(run_id, -1, model_path=model_path)
    return [si["name"].replace(",", "").replace("/", "_").replace(" ", "_") for si in cf.streams]


####################################################################################################
def plot_lr(
    runs_ids: dict[str, list],
    runs_data: list[Metrics],
    runs_active: list[bool],
    x_axis: str = "samples",
    plot_dir: Path = Path("./plots"),
):
    """
    Plot learning rate curves of training runs.

    Parameters
    ----------
    runs_ids : dict
        dictionary with run ids as keys and list of SLURM job ids and descriptions as values
    runs_data : list
        list of Metrics objects containing the training data
    runs_active : list
        list of booleans indicating whether the run is still active
    x_axis : str
        x-axis strings used in the column names (options: "samples", "dtime")
    plot_dir : Path
        directory to save the plots
    """
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"] + ["r", "g", "b", "k", "y", "m"]
    _fig = plt.figure(figsize=(10, 7), dpi=300)

    linestyle = "-"

    legend_str = []
    for j, run_data in enumerate(runs_data):
        if run_data.train.is_empty():
            continue
        run_id = run_data.run_id
        x_col = next(filter(lambda c: x_axis in c, run_data.train.columns))
        data_cols = list(filter(lambda c: "learning_rate" in c, run_data.train.columns))

        plt.plot(
            run_data.train[x_col],
            run_data.train[data_cols],
            linestyle,
            color=colors[j % len(colors)],
        )
        legend_str += [
            ("R" if runs_active[j] else "X") + " : " + run_id + " : " + runs_ids[run_id][1]
        ]

    if len(legend_str) < 1:
        return

    plt.legend(legend_str)
    plt.grid(True, which="both", ls="-")
    plt.yscale("log")
    plt.title("learning rate")
    plt.ylabel("lr")
    plt.xlabel(x_axis)
    plt.tight_layout()
    rstr = "".join([f"{r}_" for r in runs_ids])
    plt.savefig(plot_dir / f"{rstr}lr.png")
    plt.close()


####################################################################################################
def plot_utilization(
    runs_ids: dict[str, list],
    runs_data: list[Metrics],
    runs_active: list[bool],
    x_axis: str = "samples",
    plot_dir: Path = Path("./plots"),
):
    """
    Plot compute utilization of training runs.

    Parameters
    ----------
    runs_ids : dict
        dictionary with run ids as keys and list of SLURM job ids and descriptions as values
    runs_data : list
        list of Metrics objects containing the training data
    runs_active : list
        list of booleans indicating whether the run is still active
    x_axis : str
        x-axis strings used in the column names (options: "samples", "dtime")
    plot_dir : Path
        directory to save the plots
    """
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"] + ["r", "g", "b", "k", "y", "m"]
    _fig = plt.figure(figsize=(10, 7), dpi=300)

    linestyles = ["-", "--", ".-"]

    legend_str = []
    for j, (run_id, run_data) in enumerate(zip(runs_ids, runs_data, strict=False)):
        if run_data.train.is_empty():
            continue

        x_col = next(filter(lambda c: x_axis in c, run_data.train.columns))
        data_cols = run_data.system.columns[1:]

        for ii, col in enumerate(data_cols):
            plt.plot(
                run_data.train[x_col],
                run_data.system[col],
                linestyles[ii],
                color=colors[j % len(colors)],
            )
            legend_str += [
                ("R" if runs_active[j] else "X")
                + " : "
                + run_id
                + ", "
                + col
                + " : "
                + runs_ids[run_id][1]
            ]

    if len(legend_str) < 1:
        return

    plt.legend(legend_str)
    plt.grid(True, which="both", ls="-")
    # plt.yscale( 'log')
    plt.title("utilization")
    plt.ylabel("percentage utilization")
    plt.xlabel(x_axis)
    plt.tight_layout()
    rstr = "".join([f"{r}_" for r in runs_ids])
    plt.savefig(plot_dir / f"{rstr}utilization.png")
    plt.close()


####################################################################################################
def plot_loss_per_stream(
    modes: list[str],
    runs_ids: dict[str, list],
    runs_data: list[Metrics],
    runs_active: list[bool],
    stream_names: list[str],
    errs: list[str] = ["mse"],
    x_axis: str = "samples",
    x_type: str = "step",
    x_scale_log: bool = False,
    plot_dir: Path = Path("./plots"),
):
    """
    Plot each stream in stream_names (using matching to data columns) for all run_ids

    Parameters
    ----------
    modes : list
        list of modes for which losses are plotted (e.g. train, val)
    runs_ids : dict
        dictionary with run ids as keys and list of SLURM job ids and descriptions as values
    runs_data : list
        list of Metrics objects containing the training data
    runs_active : list
        list of booleans indicating whether the run is still active
    stream_names : list
        list of stream names to plot
    errs : list
        list of errors to plot (e.g. mse, stddev)
    x_axis : str
        x-axis strings used in the column names (options: "samples", "dtime")
    x_type : str
        x-axis type (options: "step", "reltime")
    x_scale_log : bool
        whether to use log scale for x-axis
    plot_dir : Path
        directory to save the plots
    """

    modes = [modes] if type(modes) is not list else modes
    # repeat colors when train and val is plotted simultaneously
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"] + ["r", "g", "b", "k", "m", "y"]

    for stream_name in stream_names:
        _fig = plt.figure(figsize=(10, 7), dpi=300)

        legend_strs = []
        min_val = np.finfo(np.float32).max
        max_val = 0.0
        for mode in modes:
            legend_strs += [[]]
            for err in errs:
                linestyle = "-" if mode == "train" else ("--x" if len(modes) > 1 else "-x")
                linestyle = ":" if "stddev" in err else linestyle
                alpha = 1.0
                if "train" in modes and "val" in modes:
                    alpha = 0.35 if "train" in mode else alpha

                for j, run_data in enumerate(runs_data):
                    run_data_mode = run_data.by_mode(mode)
                    if run_data_mode.is_empty():
                        continue
                    # find the col of the request x-axis (e.g. samples)
                    x_col = next(filter(lambda c: x_axis in c, run_data_mode.columns))
                    # find the cols of the requested metric (e.g. mse) for all streams
                    # TODO: fix captialization
                    data_cols = filter(
                        lambda c: err in c and stream_name in c, run_data_mode.columns
                    )

                    for col in data_cols:
                        x_vals = np.array(run_data_mode[x_col])
                        y_data = np.array(run_data_mode[col])

                        plt.plot(
                            x_vals,
                            y_data,
                            linestyle,
                            color=colors[j % len(colors)],
                            alpha=alpha,
                        )
                        legend_strs[-1] += [
                            ("R" if runs_active[j] else "X")
                            + " : "
                            + run_data.run_id
                            + " : "
                            + runs_ids[run_data.run_id][1]
                            + ": "
                            + col
                        ]

                        min_val = np.min([min_val, np.nanmin(y_data)])
                        max_val = np.max([max_val, np.nanmax(y_data)])

        # TODO: ensure that legend is plotted with full opacity
        legend_str = legend_strs[0]
        if len(legend_str) < 1:
            plt.close()
            continue

        legend = plt.legend(legend_str, loc="upper right" if not x_scale_log else "lower left")
        for line in legend.get_lines():
            line.set(alpha=1.0)
        plt.grid(True, which="both", ls="-")
        plt.yscale("log")
        # cap at 1.0 in case of divergence of run (through normalziation, max should be around 1.0)
        plt.ylim([0.95 * min_val, (None if max_val < 2.0 else min(1.1, 1.025 * max_val))])
        if x_scale_log:
            plt.xscale("log")
        plt.title(stream_name)
        plt.ylabel("loss")
        plt.xlabel(x_axis if x_type == "step" else "rel. time [h]")
        plt.tight_layout()
        rstr = "".join([f"{r}_" for r in runs_ids])
        plt.savefig(
            plot_dir / "{}{}{}.png".format(rstr, "".join([f"{m}_" for m in modes]), stream_name)
        )
        plt.close()


####################################################################################################
def plot_loss_per_run(
    modes: list[str],
    run_id: str,
    run_desc: str,
    run_data: Metrics,
    stream_names: list[str],
    errs: list[str] = ["mse"],
    x_axis: str = "samples",
    x_scale_log: bool = False,
    plot_dir: Path = Path("./plots"),
):
    """
    Plot all stream_names (using matching to data columns) for given run_id

    Parameters
    ----------
    modes : list
        list of modes for which losses are plotted (e.g. train, val)
    run_id : str
        ID of the training run to plot
    run_desc : List[str]
        Description of the training run
    run_data : Metrics
        Metrics object containing the training data
    stream_names : list
        list of stream names to plot
    errs : list
        list of errors to plot (e.g. mse, stddev)
    x_axis : str
        x-axis strings used in the column names (options: "samples", "dtime")
    x_scale_log : bool
        whether to use log scale for x-axis
    plot_dir : Path
        directory to save the plots
    """
    plot_dir = Path(plot_dir)

    modes = [modes] if type(modes) is not list else modes
    # repeat colors when train and val is plotted simultaneously
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"] + ["r", "g", "b", "k", "y", "m"]

    _fig = plt.figure(figsize=(10, 7), dpi=300)

    legend_strs = []
    for mode in modes:
        legend_strs += [[]]
        for err in errs:
            linestyle = "-" if mode == "train" else ("--x" if len(modes) > 1 else "-x")
            linestyle = ":" if "stddev" in err else linestyle
            alpha = 1.0
            if "train" in modes and "val" in modes:
                alpha = 0.35 if "train" in mode else alpha
            run_data_mode = run_data.by_mode(mode)

            x_col = [c for _, c in enumerate(run_data_mode.columns) if x_axis in c][0]
            # find the cols of the requested metric (e.g. mse) for all streams
            data_cols = [c for _, c in enumerate(run_data_mode.columns) if err in c]

            for _, col in enumerate(data_cols):
                for j, stream_name in enumerate(stream_names):
                    if stream_name in col:
                        # skip when no data is available
                        if run_data_mode[col].shape[0] == 0:
                            continue

                        x_vals = np.array(run_data_mode[x_col])
                        y_data = np.array(run_data_mode[col])

                        plt.plot(
                            x_vals,
                            y_data,
                            linestyle,
                            color=colors[j % len(colors)],
                            alpha=alpha,
                        )
                        legend_strs[-1] += [col]

    legend_str = legend_strs[0]
    if len(legend_str) < 1:
        plt.close()
        return

    plt.title(run_id + " : " + run_desc[1])
    legend = plt.legend(legend_str, loc="lower left")
    for line in legend.get_lines():
        line.set(alpha=1.0)
    plt.yscale("log")
    if x_scale_log:
        plt.xscale("log")
    plt.grid(True, which="both", ls="-")
    plt.ylabel("loss")
    plt.xlabel("samples")
    plt.tight_layout()
    sstr = "".join(
        [f"{r}_".replace(",", "").replace("/", "_").replace(" ", "_") for r in legend_str]
    )
    plt.savefig(plot_dir / "{}_{}{}.png".format(run_id, "".join([f"{m}_" for m in modes]), sstr))
    plt.close()


####################################################################################################
if __name__ == "__main__":
    # Example usage:
    # python plot_training.py -ids '{"qlz6n9eg": [12341234, "My experiments"]}' -m ./trained_models -o ./training_plots

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o", "--output_dir", default="./plots/", type=str, help="Directory where plots are saved"
    )
    parser.add_argument(
        "-m",
        "--model_base_dir",
        default="./models/",
        type=str,
        help="Base-directory where models are saved",
    )
    parser.add_argument(
        "-d",
        "--delete",
        default=False,
        action="store_true",
        help="Delete all plots in the output directory before plotting",
    )
    parser.add_argument(
        "-ids",
        "--runs_ids",
        type=js.loads,
        required=True,
        help="JSON string with run ids as keys and list of SLURM job ids and descriptions as values",
    )

    args = parser.parse_args()

    model_base_dir = Path(args.model_base_dir)
    out_dir = Path(args.output_dir)
    runs_ids = args.runs_ids

    if args.delete == "True":
        clean_plot_folder(out_dir)

    # read logged data

    runs_data = [TrainLogger.read(run_id, model_path=model_base_dir) for run_id in runs_ids]

    # determine which runs are still alive (as a process, though they might hang internally)
    ret = subprocess.run(["squeue"], capture_output=True)
    lines = str(ret.stdout).split("\\n")
    runs_active = [np.array([str(v[0]) in l for l in lines[1:]]).any() for v in runs_ids.values()]

    x_scale_log = False
    x_type = ("step",)  #'reltime'
    # x_type = "step"

    # plot learning rate
    plot_lr(runs_ids, runs_data, runs_active, plot_dir=out_dir)

    # plot performance
    plot_utilization(runs_ids, runs_data, runs_active, plot_dir=out_dir)

    # compare different runs
    plot_loss_per_stream(
        ["train", "val"],
        runs_ids,
        runs_data,
        runs_active,
        ["era5", "METEOSAT", "NPP"],
        x_type=x_type,
        x_scale_log=x_scale_log,
        plot_dir=out_dir,
    )
    plot_loss_per_stream(
        ["val"],
        runs_ids,
        runs_data,
        runs_active,
        ["era5", "METEOSAT", "NPP"],
        x_type=x_type,
        x_scale_log=x_scale_log,
        plot_dir=out_dir,
    )
    plot_loss_per_stream(
        ["train"],
        runs_ids,
        runs_data,
        runs_active,
        ["ERA5", "METEOSAT", "NPP"],
        x_type=x_type,
        x_scale_log=x_scale_log,
        plot_dir=out_dir,
    )

    # plot all cols for all run_ids
    for run_id, run_data in zip(runs_ids, runs_data, strict=False):
        plot_loss_per_run(
            ["train", "val"],
            run_id,
            runs_ids[run_id],
            run_data,
            get_stream_names(run_id, model_path=model_base_dir),
            plot_dir=out_dir,
        )
    plot_loss_per_run(
        ["val"],
        run_id,
        runs_ids[run_id],
        run_data,
        get_stream_names(run_id, model_path=model_base_dir),
        plot_dir=out_dir,
    )
