# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

import weathergen.common.config as config
from weathergen.utils.train_logger import Metrics, TrainLogger

_logger = logging.getLogger(__name__)

DEFAULT_RUN_FILE = Path("./config/runs_plot_train.yml")


####################################################################################################
def _ensure_list(value):
    """
    Ensure that the input value is a list. If it is not a list, convert it to a list.
    Parameters
    ----------
    value : any
        Input value to check.
    Returns
    -------
    list
        A list containing the input value if it was not a list,
          or the input value itself if it was already a list.
    """
    return value if isinstance(value, list) else [value]


####################################################################################################
def _check_run_id_dict(run_id_dict: dict) -> bool:
    """
    Check if the run_id_dict is valid.

    Parameters
    ----------
    run_id_dict : dict
        Dictionary to check.
    Returns
    -------
    """
    if not isinstance(run_id_dict, dict):
        return False

    for k, v in run_id_dict.items():
        if not isinstance(k, str) or not isinstance(v, list) or len(v) != 2:
            raise argparse.ArgumentTypeError(
                (
                    "Each key must be a string and",
                    f" each value must be a list of [job_id, experiment_name], but got: {k}: {v}",
                )
            )


####################################################################################################
def _read_str_config(yaml_str: str) -> dict:
    """
    Read a dictionary-like string to get a configuration dictionary.

    Parameters
    ----------
    yaml_str : str
        Dictionary-like string to read.
    Returns
    -------
    dict
        The content of the string as a dictionary.
    """
    config_dict = yaml.safe_load(yaml_str)

    # Validate the structure: {run_id: [job_id, experiment_name]}
    _check_run_id_dict(config_dict)

    return config_dict


####################################################################################################
def _read_yaml_config(yaml_file_path):
    """
    Read a YAML file to get a configuration dictionary for plotting training diagnostics.
    Expected structure in the YAML file:
    train:
        plot:
            run_id:
                slurm_id : SLURM_JOB (specify 0 if not available)
                description: job description
            run_id:
                slurm_id : SLURM_JOB (specify 0 if not available)
                description : job description
            ...

    Parameters
    ----------
    yaml_file_path : str or Path
        Path to the YAML file containing the configuration.
    Returns
    -------
    dict
        A dictionary with run IDs as keys and a list of [job ID, experiment name] as values.
    """
    with open(yaml_file_path) as f:
        data = yaml.safe_load(f)

    # Extract configuration for plotting training diagnostics
    config_dict_temp = data.get("train", {}).get("plot", {})

    # sanity checks
    assert len(config_dict_temp) > 0, "At least one run must be specified."

    # convert to legacy format
    config_dict = {}
    for k, v in config_dict_temp.items():
        assert isinstance(v["slurm_id"], int), "slurm_id has to be int."
        assert isinstance(v["description"], str), "description has to be str."
        config_dict[k] = [v["slurm_id"], v["description"]]

    # Validate the structure: {run_id: [job_id, experiment_name]}
    _check_run_id_dict(config_dict)

    return config_dict


####################################################################################################
def clean_plot_folder(plot_dir: Path):
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
    cf = config.load_run_config(run_id, -1, model_path=model_path)
    return [si["name"].replace(",", "").replace("/", "_").replace(" ", "_") for si in cf.streams]


####################################################################################################
def plot_lr(
    runs_ids: dict[str, list],
    runs_data: list[Metrics],
    runs_active: list[bool],
    plot_dir: Path,
    x_axis: str = "samples",
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
    plot_dir : Path
        directory to save the plots
    x_axis : str
        x-axis strings used in the column names (options: "samples", "dtime")
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
        _logger.warning(
            "Could not find any data for plotting the learning rates of the runs: ", runs_ids
        )
        return

    plt.legend(legend_str)
    plt.grid(True, which="both", ls="-")
    plt.yscale("log")
    plt.title("learning rate")
    plt.ylabel("lr")
    plt.xlabel(x_axis)
    plt.tight_layout()
    rstr = "".join([f"{r}_" for r in runs_ids])

    # save the plot
    plt_fname = plot_dir / f"{rstr}lr.png"
    _logger.info(f"Saving learning rate plot to '{plt_fname}'")
    plt.savefig(plt_fname)
    plt.close()


####################################################################################################
def plot_utilization(
    runs_ids: dict[str, list],
    runs_data: list[Metrics],
    runs_active: list[bool],
    plot_dir: Path,
    x_axis: str = "samples",
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
    plot_dir : Path
        directory to save the plots
    x_axis : str
        x-axis strings used in the column names (options: "samples", "dtime")
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
        _logger.warning("Could not find any data for utilization plot")
        return

    plt.legend(legend_str)
    plt.grid(True, which="both", ls="-")
    # plt.yscale( 'log')
    plt.title("utilization")
    plt.ylabel("percentage utilization")
    plt.xlabel(x_axis)
    plt.tight_layout()
    rstr = "".join([f"{r}_" for r in runs_ids])

    # save the plot
    plt_fname = plot_dir / f"{rstr}utilization.png"
    _logger.info(f"Saving utilization plot to '{plt_fname}'")
    plt.savefig(plt_fname)
    plt.close()


####################################################################################################
def plot_loss_per_stream(
    modes: list[str],
    runs_ids: dict[str, list],
    runs_data: list[Metrics],
    runs_active: list[bool],
    stream_names: list[str],
    plot_dir: Path,
    errs: list[str] | None = None,
    x_axis: str = "samples",
    x_type: str = "step",
    x_scale_log: bool = False,
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
    plot_dir : Path
        directory to save the plots
    errs : list
        list of errors to plot (e.g. mse, stddev)
    x_axis : str
        x-axis strings used in the column names (options: "samples", "dtime")
    x_type : str
        x-axis type (options: "step", "reltime")
    x_scale_log : bool
        whether to use log scale for x-axis
    """

    if errs is None:
        errs = ["loss_mse"]

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
                        lambda c: err in c and stream_name.lower() in c.lower(),
                        run_data_mode.columns,
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

                        # skip all-nan slices
                        if (~np.isnan(y_data)).sum() > 0:
                            min_val = np.min([min_val, np.nanmin(y_data)])
                            max_val = np.max([max_val, np.nanmax(y_data)])

        # TODO: ensure that legend is plotted with full opacity
        legend_str = legend_strs[0]
        if len(legend_str) < 1:
            plt.close()
            _logger.warning(f"Could not find any data for stream: {stream_name}")
            continue

        # no valid data found
        if (min_val >= max_val) or np.isnan(min_val) or np.isnan(max_val):
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

        # save the plot
        plt_fname = plot_dir / "{}{}{}.png".format(
            rstr, "".join([f"{m}_" for m in modes]), stream_name
        )
        _logger.info(f"Saving loss per stream plot to '{plt_fname}'")
        plt.savefig(plt_fname)
        plt.close()


####################################################################################################
def plot_loss_per_run(
    modes: list[str],
    run_id: str,
    run_desc: str,
    run_data: Metrics,
    stream_names: list[str],
    plot_dir: Path,
    errs: list[str] | None = None,
    x_axis: str = "samples",
    x_scale_log: bool = False,
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
    plot_dir : Path
        directory to save the plots
    errs : list
        list of errors to plot (e.g. mse, stddev)
    x_axis : str
        x-axis strings used in the column names (options: "samples", "dtime")
    x_scale_log : bool
        whether to use log scale for x-axis
    """
    if errs is None:
        errs = ["mse"]

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

            data_cols = list(data_cols)

            for _, col in enumerate(data_cols):
                for j, stream_name in enumerate(stream_names):
                    if stream_name.lower() in col.lower():
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
        _logger.warning(f"Could not find any data for run: {run_id}")
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

    # save the plot
    plt_fname = plot_dir / "{}_{}{}.png".format(run_id, "".join([f"{m}_" for m in modes]), sstr)
    _logger.info(f"Saving loss plot for {run_id}-run to '{plt_fname}'")
    plt.savefig(plt_fname)
    plt.close()


def plot_train(args=None):
    # Example usage:
    # When providing a YAML for configuring the run IDs:
    # python plot_training.py -rf eval_run.yml -m ./trained_models -o ./training_plots
    # When providing a string for configuring the run IDs:
    # python plot_training.py -rs "{run_id: [job_id, experiment_name]}"
    #    -m ./trained_models -o ./training_plots

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(
        description="""Plot training diagnostics from logged data during training.
                       An example YAML file looks like this:
                            train:
                                plot:
                                    run_id:
                                        slurm_id : SLURM_JOB (specify 0 if not available)
                                        description: job description
                                    run_id:
                                        slurm_id : SLURM_JOB (specify 0 if not available)
                                        description : job description
                                            ...

                        A dictionary-string can also be specified on the command line, e.g.:
                            "{'abcde': ['123456', 'experiment1'],
                            'fghij': ['654321', 'experiment2']}"
                            """
    )

    parser.add_argument(
        "-o", "--output_dir", default="./plots/", type=Path, help="Directory where plots are saved"
    )
    parser.add_argument(
        "-m",
        "--model_base_dir",
        default=None,
        type=Path,
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
        "--streams",
        "-s",
        dest="streams",
        default=["ERA5"],
        type=str,
        nargs="+",
        help="List of streams to plot",
    )
    parser.add_argument(
        "--x_type",
        "-x",
        dest="x_type",
        default="step",
        type=str,
        choices=["step", "reltime"],
        help="Type of x-axis used in plots. Options: 'step' or 'reltime'",
    )

    run_id_group = parser.add_mutually_exclusive_group()
    run_id_group.add_argument(
        "-fd",
        "--from_dict",
        type=_read_str_config,
        dest="fd",
        help="Dictionary-string of form '{run_id: [job_id, experiment_name]}'"
        + "for training runs to plot",
    )

    run_id_group.add_argument(
        "-fy",
        "--from_yaml",
        dest="fy",
        type=_read_yaml_config,
        help="YAML file configuring the training run ids to plot",
    )

    # parse the command line arguments
    args = parser.parse_args(args)

    model_base_dir = Path(args.model_base_dir) if args.model_base_dir else None
    out_dir = Path(args.output_dir)
    streams = list(args.streams)
    x_types_valid = ["step"]  # TODO: add "reltime" support when fix available
    if args.x_type not in x_types_valid:
        raise ValueError(f"x_type must be one of {x_types_valid}, but got {args.x_type}")

    # Post-processing default logic for config from YAML-file
    if args.fd is None and args.fy is None:
        if DEFAULT_RUN_FILE.exists():
            args.fy = _read_yaml_config(DEFAULT_RUN_FILE)
        else:
            raise ValueError(
                f"Please provide a run_id dictionary or a YAML file with run_ids, "
                f"or create a default file at {DEFAULT_RUN_FILE}."
            )

    runs_ids = args.fd if args.fd is not None else args.fy

    if args.delete == "True":
        clean_plot_folder(out_dir)

    # read logged data

    runs_data = [TrainLogger.read(run_id, model_path=model_base_dir) for run_id in runs_ids]

    # determine which runs are still alive (as a process, though they might hang internally)
    ret = subprocess.run(["squeue"], capture_output=True)
    lines = str(ret.stdout).split("\\n")
    runs_active = [
        np.array([str(v[0]) in line for line in lines[1:]]).any() for v in runs_ids.values()
    ]

    x_scale_log = False

    # plot learning rate
    plot_lr(runs_ids, runs_data, runs_active, plot_dir=out_dir)

    # # plot performance
    # plot_utilization(runs_ids, runs_data, runs_active, plot_dir=out_dir)

    # compare different runs
    plot_loss_per_stream(
        ["train", "val"],
        runs_ids,
        runs_data,
        runs_active,
        streams,
        x_type=args.x_type,
        x_scale_log=x_scale_log,
        plot_dir=out_dir,
    )
    plot_loss_per_stream(
        ["val"],
        runs_ids,
        runs_data,
        runs_active,
        streams,
        x_type=args.x_type,
        x_scale_log=x_scale_log,
        plot_dir=out_dir,
    )
    plot_loss_per_stream(
        ["train"],
        runs_ids,
        runs_data,
        runs_active,
        streams,
        x_type=args.x_type,
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
            get_stream_names(run_id, model_path=model_base_dir),  # limit to available streams
            plot_dir=out_dir,
        )
    plot_loss_per_run(
        ["val"],
        run_id,
        runs_ids[run_id],
        run_data,
        get_stream_names(run_id, model_path=model_base_dir),  # limit to available streams
        plot_dir=out_dir,
    )


if __name__ == "__main__":
    args = sys.argv[1:]  # get CLI args

    plot_train(args)
