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


def train() -> None:
    """Entry point for calling the training code from the command line."""
    main([cli.Stage.train] + sys.argv[1:])


def train_continue() -> None:
    """Entry point for calling train_continue from the command line."""
    main([cli.Stage.train_continue] + sys.argv[1:])


def inference():
    """Entry point for calling the inference code from the command line."""
    main([cli.Stage.inference] + sys.argv[1:])


def main(argl: list[str]):
    try:
        argl = _fix_argl(argl)
    except ValueError as e:
        logger.error(str(e))

    parser = cli.get_main_parser()
    args = parser.parse_args(argl)
    match args.stage:
        case cli.Stage.train:
            run_train(args)
        case cli.Stage.train_continue:
            run_continue(args)
        case cli.Stage.inference:
            run_inference(args)
        case _:
            logger.error("No stage was found.")


def _fix_argl(argl):  # TODO remove this fix after grace period
    """Ensure `stage` positional argument is in arglist."""
    if argl[0] not in cli.Stage:
        try:
            stage = os.environ.get("WEATHERGEN_STAGE")
        except KeyError as e:
            msg = (
                "`stage` postional argument and environment variable 'WEATHERGEN_STAGE' missing.",
                "Provide either one or the other.",
            )
            raise ValueError(msg) from e

        argl = [stage] + argl

    return argl


def run_inference(args):
    """
    Inference function for WeatherGenerator model.

    Note: Additional configuration for inference (`test_config`) is set in the function.
    """
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


def run_continue(args):
    """
    Function to continue training for WeatherGenerator model.

    Note: All model configurations are set in the function body.
    """

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


def run_train(args):
    """
    Training function for WeatherGenerator model.

    Note: All model configurations are set in the function body.
    """

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
    main(sys.argv[1:])
