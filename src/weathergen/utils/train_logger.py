# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import json
import logging
import math
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from torch import Tensor

import weathergen.common.config as config
from weathergen.utils.metrics import read_metrics_file

_weathergen_timestamp = "weathergen.timestamp"
_weathergen_reltime = "weathergen.reltime"
_weathergen_time = "weathergen.time"
_performance_gpu = "perf.gpu"
_performance_memory = "perf.memory"

_logger = logging.getLogger(__name__)

Stage = Literal["train", "val"]
RunId = str

# All the stages currently implemented:
TRAIN: Stage = "train"
VAL: Stage = "val"


# Helper functions for metric keys (implied from usage across multiple methods)
def _key_loss(stream_name: str, loss_fn_name: str) -> str:
    return f"loss.{stream_name}.{loss_fn_name}"


def _key_loss_chn(stream_name: str, loss_fn_name: str, channel_name: str) -> str:
    return f"loss.{stream_name}.{loss_fn_name}.{channel_name}"


def _key_stddev(stream_name: str) -> str:
    return f"stddev.{stream_name}"


@dataclass
class Metrics:
    run_id: RunId
    stage: Stage
    train: pl.DataFrame
    val: pl.DataFrame
    system: pl.DataFrame

    def by_mode(self, s: str) -> pl.DataFrame:
        match s:
            case "train":
                return self.train
            case "val":
                return self.val
            case "system":
                return self.system
            case _:
                raise ValueError(f"Unknown mode {s}. Use 'train', 'val' or 'system'.")


class TrainLogger:
    # Define relative sub-paths as constants
    METRICS_FILE_NAME = "metrics.jsonl"
    TRAIN_LOG_FILE_NAME = "train_log.txt"
    VAL_LOG_FILE_NAME = "val_log.txt"
    PERF_LOG_FILE_NAME = "perf_log.txt"

    #######################################
    def __init__(self, cf, base_run_path: Path) -> None:
        """
        Initializes the TrainLogger with a configuration object and the base run directory path.
        All log files will be created within this base_run_path.
        """
        self.cf = cf
        self._run_dir_path = base_run_path
        self._run_dir_path.mkdir(parents=True, exist_ok=True)  # Ensure the run directory exists

        # Pre-derive full paths for all log files
        self._metrics_path = self._run_dir_path / self.METRICS_FILE_NAME
        self._train_log_path = self._run_dir_path / self.TRAIN_LOG_FILE_NAME
        self._val_log_path = self._run_dir_path / self.VAL_LOG_FILE_NAME
        self._perf_log_path = self._run_dir_path / self.PERF_LOG_FILE_NAME

    def log_metrics(self, stage: Stage, metrics: dict[str, float]) -> None:
        """
        Log scalar metrics to a JSONL file.
        """
        clean_metrics = {
            _weathergen_timestamp: time.time_ns() // 1_000_000,
            _weathergen_time: int(datetime.datetime.now().strftime("%Y%m%d%H%M%S")),
            "stage": stage,
        }
        for key, value in metrics.items():
            v = float(value)
            if math.isnan(v) or math.isinf(v):
                v = str(v)
            clean_metrics[key] = v

        # Open in text mode 'a' for JSON lines
        with open(self._metrics_path, "a") as f:
            s = json.dumps(clean_metrics) + "\n"
            f.write(s)

    #######################################
    def add_train(
        self,
        samples: int,
        lr: float,
        avg_loss: Tensor,
        losses_all: dict[str, Tensor],
        stddev_all: dict[str, Tensor],
        perf_gpu: float = 0.0,
        perf_mem: float = 0.0,
    ) -> None:
        """
        Log training data including losses, learning rate, and performance metrics.
        """
        metrics: dict[str, float] = dict(num_samples=samples)

        log_vals: list[float] = [int(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))]
        log_vals += [samples]

        metrics["loss_avg_mean"] = avg_loss.nanmean().item()
        metrics["learning_rate"] = lr
        metrics["num_samples"] = int(samples)
        log_vals += [avg_loss.nanmean().item()]
        log_vals += [lr]

        for st in self.cf.streams:
            loss = losses_all[st["name"]]
            stddev = stddev_all[st["name"]]

            for j, (lf_name, _) in enumerate(self.cf.loss_fcts):
                metrics[_key_loss(st["name"], lf_name)] = loss[:, :, j].nanmean().item()

                for k, ch_n in enumerate(st.train_target_channels):
                    metrics[_key_loss_chn(st["name"], lf_name, ch_n)] = (
                        loss[:, k, j].nanmean().item()
                    )
                log_vals += [loss[:, :, j].nanmean().item()]

            metrics[_key_stddev(st["name"])] = stddev.nanmean().item()

            log_vals += [stddev.nanmean().item()]

        # Fix: Open in text mode 'a' and pass 2D array to np.savetxt for single-row logging
        with open(self._train_log_path, "a") as f:
            np.savetxt(f, np.array(log_vals).reshape(1, -1), fmt='%.6f', delimiter=',')

        log_vals = []
        log_vals += [perf_gpu]
        log_vals += [perf_mem]
        metrics[_performance_gpu] = perf_gpu
        metrics[_performance_memory] = perf_mem
        self.log_metrics("train", metrics)
        # Fix: Open in text mode 'a' and pass 2D array to np.savetxt for single-row logging
        with open(self._perf_log_path, "a") as f:
            np.savetxt(f, np.array(log_vals).reshape(1, -1), fmt='%.6f', delimiter=',')

    #######################################
    def add_val(
        self, samples: int, losses_all: dict[str, Tensor], stddev_all: dict[str, Tensor]
    ) -> None:
        """
        Log validation data
        """

        metrics: dict[str, float] = dict(num_samples=int(samples))

        log_vals: list[float] = [int(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))]
        log_vals += [samples]

        for st in self.cf.streams:
            loss = losses_all[st["name"]]
            stddev = stddev_all[st["name"]]
            for j, (lf_name, _) in enumerate(self.cf.loss_fcts_val):
                metrics[_key_loss(st["name"], lf_name)] = loss[:, :, j].nanmean().item()
                for k, ch_n in enumerate(st.val_target_channels):
                    metrics[_key_loss_chn(st["name"], lf_name, ch_n)] = (
                        loss[:, k, j].nanmean().item()
                    )
                log_vals += [loss[:, :, j].nanmean().item()]

            metrics[_key_stddev(st["name"])] = stddev.nanmean().item()
            log_vals += [stddev.nanmean().item()]

        self.log_metrics("val", metrics)
        # Fix: Open in text mode 'a' and pass 2D array to np.savetxt for single-row logging
        with open(self._val_log_path, "a") as f:
            np.savetxt(f, np.array(log_vals).reshape(1, -1), fmt='%.6f', delimiter=',')

    #######################################
    @staticmethod
    def read(run_id: str, model_path: str = None, epoch: int = -1) -> Metrics:
        """
        Read data for run_id, populating the Metrics dataclass from the generated log files.
        """
        # Load config from given model_path if provided, otherwise use path from private config
        if model_path:
            cf = config.load_model_config(run_id=run_id, epoch=epoch, model_path=model_path)
        else:
            cf = config.load_config(private_home=None, from_run_id=run_id, epoch=epoch)
        run_id_from_config = cf.run_id  # Use a new variable name to avoid confusion with parameter

        # The base_run_path is the directory where all log files for this run are stored.
        # This is derived using the refactored config path retrieval.
        base_run_path = config.get_path_run(cf)

        # Paths for metrics.jsonl (from log_metrics)
        fname_metrics_jsonl = base_run_path / TrainLogger.METRICS_FILE_NAME

        # Paths for savetxt files (from add_train, add_val)
        fname_perf_txt = base_run_path / TrainLogger.PERF_LOG_FILE_NAME

        # Read metrics from JSONL file (train and val data for Metrics dataclass)
        all_metrics_df = pl.DataFrame()
        try:
            all_metrics_df = read_metrics_file(fname_metrics_jsonl)
        except (FileNotFoundError, PermissionError, OSError) as e:
            _logger.error(
                (
                    f"Error: no general metrics loaded for run_id={run_id_from_config}",
                    "File system error occurred while handling the metrics.jsonl file.",
                    f"Due to specific error: {e}",
                )
            )
        except Exception:
            _logger.error(
                (
                    f"Error: no general metrics loaded for run_id={run_id_from_config}",
                    f"Due to exception with trace:\n{traceback.format_exc()}",
                )
            )

        train_metrics_df = pl.DataFrame()
        if not all_metrics_df.is_empty():
            train_metrics_df = all_metrics_df.filter(pl.col("stage") == TRAIN)

        val_metrics_df = pl.DataFrame()
        if not all_metrics_df.is_empty():
            val_metrics_df = all_metrics_df.filter(pl.col("stage") == VAL)

        # Read performance log data (system data for Metrics dataclass)
        log_perf = np.array([])
        try:
            # Open in text mode 'r' and specify delimiter
            with open(fname_perf_txt, "r") as f:
                log_perf = np.loadtxt(f, delimiter=',')
            # If log_perf is 1D (e.g., a single entry), reshape it to 2D for consistent DataFrame creation
            if log_perf.ndim == 1:
                log_perf = log_perf.reshape(-1, len([_performance_gpu, _performance_memory]))
        except (
            TypeError,
            AttributeError,
            IndexError,
            ZeroDivisionError,
            ValueError,
        ) as e:
            _logger.warning(
                (
                    f"Warning: no performance data loaded for run_id={run_id_from_config}",
                    "Data loading or reshaping failed — "
                    "possible format, dimension, or logic issue.",
                    f"Due to specific error: {e}",
                )
            )
        except (FileNotFoundError, PermissionError, OSError) as e:
            _logger.error(
                (
                    f"Error: no performance data loaded for run_id={run_id_from_config}",
                    "File system error occurred while handling the perf log file.",
                    f"Due to specific error: {e}",
                )
            )
        except Exception:
            _logger.error(
                (
                    f"Error: no performance data loaded for run_id={run_id_from_config}",
                    f"Due to exception with trace:\n{traceback.format_exc()}",
                )
            )

        system_df = pl.DataFrame(
            log_perf, schema=[_performance_gpu, _performance_memory]
        ) if log_perf.size > 0 else pl.DataFrame(schema=[_performance_gpu, _performance_memory])


        # The original code also read _train_log.txt and _val_log.txt using np.loadtxt,
        # but these results were not used to populate the returned Metrics dataclass.
        # Therefore, these redundant loadtxt calls are removed.

        return Metrics(
            run_id=run_id_from_config,
            stage=TRAIN,  # Keeping this as original, even with separate train/val DFs
            train=train_metrics_df,
            val=val_metrics_df,
            system=system_df,
        )