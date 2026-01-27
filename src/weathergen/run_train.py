# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
The entry point for training and inference weathergen-atmo
"""

import logging
import os
import pdb
import sys
import time
import traceback
from pathlib import Path

import weathergen.common.config as config
import weathergen.utils.cli as cli
from weathergen.common.logger import init_loggers
from weathergen.train.trainer import Trainer

logger = logging.getLogger(__name__)


def inference():
    # By default, arguments from the command line are read.
    inference_from_args(sys.argv[1:])


def inference_from_args(argl: list[str]):
    """
    Inference function for WeatherGenerator model.
    Entry point for calling the inference code from the command line.

    When running integration tests, the arguments are directly provided.
    """
    parser = cli.get_inference_parser()
    args = parser.parse_args(argl)

    inference_overwrite = {
        "test_config": dict(
            shuffle=False,
            start_date=args.start_date,
            end_date=args.end_date,
            samples_per_mini_epoch=args.samples,
            output=dict(num_samples=args.samples if args.save_samples else 0),
            streams_output=args.streams_output,
        )
    }

    cli_overwrite = config.from_cli_arglist(args.options)
    cf = config.load_merge_configs(
        args.private_config,
        args.from_run_id,
        args.mini_epoch,
        args.base_config,
        *args.config,
        inference_overwrite,
        cli_overwrite,
    )
    cf = config.set_run_id(cf, args.run_id, args.reuse_run_id)

    devices = Trainer.init_torch()
    cf = Trainer.init_ddp(cf)

    init_loggers(cf.general.run_id)

    logger.info(f"DDP initialization: rank={cf.rank}, world_size={cf.world_size}")

    cf.general.run_history += [(args.from_run_id, cf.general.istep)]

    trainer = Trainer(cf.train_log_freq)
    try:
        trainer.inference(cf, devices, args.from_run_id, args.mini_epoch)
    except Exception:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        if cf.world_size == 1:
            pdb.post_mortem(tb)


####################################################################################################
def train_continue() -> None:
    """
    Function to continue training for WeatherGenerator model.
    Entry point for calling train_continue from the command line.
    Configurations are set in the function body.

    Args:
      from_run_id (str): Run/model id of pretrained WeatherGenerator model to
        continue training. Defaults to None.
    Note: All model configurations are set in the function body.
    """
    train_continue_from_args(sys.argv[1:])


def train_continue_from_args(argl: list[str]):
    parser = cli.get_continue_parser()
    args = parser.parse_args(argl)

    cli_overwrite = config.from_cli_arglist(args.options)
    cf = config.load_merge_configs(
        args.private_config,
        args.from_run_id,
        args.mini_epoch,
        args.base_config,
        *args.config,
        {},
        cli_overwrite,
    )
    cf = config.set_run_id(cf, args.run_id, args.reuse_run_id)

    mp_method = cf.general.get("multiprocessing_method", "fork")
    devices = Trainer.init_torch(multiprocessing_method=mp_method)
    cf = Trainer.init_ddp(cf)

    init_loggers(cf.general.run_id)

    # track history of run to ensure traceability of results
    cf.general.run_history += [(args.from_run_id, cf.general.istep)]

    trainer = Trainer(cf.train_log_freq)

    try:
        trainer.run(cf, devices, args.from_run_id, args.mini_epoch)
    except Exception:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        if cf.world_size == 1:
            pdb.post_mortem(tb)


####################################################################################################
def train() -> None:
    """
    Training function for WeatherGenerator model.
    Entry point for calling the training code from the command line.
    Configurations are set in the function body.

    Args:
      run_id (str, optional): Run/model id of pretrained WeatherGenerator model to
        continue training. Defaults to None.
    Note: All model configurations are set in the function body.
    """
    train_with_args(sys.argv[1:], None)


def train_with_args(argl: list[str], stream_dir: str | None):
    """
    Training function for WeatherGenerator model."""
    parser = cli.get_train_parser()
    args = parser.parse_args(argl)

    cli_overwrite = config.from_cli_arglist(args.options)

    cf = config.load_merge_configs(
        args.private_config, None, None, args.base_config, *args.config, cli_overwrite
    )
    cf = config.set_run_id(cf, args.run_id, False)

    cf.data_loading.rng_seed = int(time.time())
    mp_method = cf.general.get("multiprocessing_method", "fork")
    devices = Trainer.init_torch(multiprocessing_method=mp_method)
    cf = Trainer.init_ddp(cf)

    # this line should probably come after the processes have been sorted out else we get lots
    # of duplication due to multiple process in the multiGPU case
    init_loggers(cf.general.run_id)

    logger.info(f"DDP initialization: rank={cf.rank}, world_size={cf.world_size}")

    cf.streams = config.load_streams(Path(cf.streams_directory))

    if cf.with_flash_attention:
        assert cf.with_mixed_precision

    trainer = Trainer(cf.train_log_freq)

    try:
        trainer.run(cf, devices)
    except Exception:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        if cf.world_size == 1:
            pdb.post_mortem(tb)


if __name__ == "__main__":
    try:
        stage = os.environ.get("WEATHERGEN_STAGE")
    except KeyError as e:
        msg = "missing environment variable 'WEATHERGEN_STAGE'"
        raise ValueError(msg) from e

    if stage == "train":
        # Entry point for slurm script.
        # Check whether --from-run-id passed as argument.
        if any("--from-run-id" in arg for arg in sys.argv):
            train_continue()
        else:
            train()
    elif stage == "inference":
        inference()
    else:
        logger.error("No stage was found.")
