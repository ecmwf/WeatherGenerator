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
import pdb
import sys
import time
import traceback
from pathlib import Path

import weathergen.common.config as config
import weathergen.utils.cli as cli
from weathergen.train.trainer import Trainer
from weathergen.utils.logger import init_loggers

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

    inference_overwrite = dict(
        shuffle=False,
        start_date_val=args.start_date,
        end_date_val=args.end_date,
        samples_per_validation=args.samples,
        log_validation=args.samples if args.save_samples else 0,
        analysis_streams_output=args.analysis_streams_output,
    )

    cli_overwrite = config.from_cli_arglist(args.options)
    cf = config.load_config(
        args.private_config,
        args.from_run_id,
        args.epoch,
        *args.config,
        inference_overwrite,
        cli_overwrite,
    )
    cf = config.set_run_id(cf, args.run_id, args.reuse_run_id)

    devices = Trainer.init_torch()
    cf = Trainer.init_ddp(cf)

    init_loggers(cf.run_id)

    logger.info(f"DDP initialization: rank={cf.rank}, world_size={cf.world_size}")

    cf.run_history += [(args.from_run_id, cf.istep)]

    trainer = Trainer(cf.train_log_freq)
    trainer.inference(cf, devices, args.from_run_id, args.epoch)


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

    if args.finetune_forecast:
        finetune_overwrite = dict(
            training_mode="forecast",
            forecast_delta_hrs=6,  # 12
            forecast_steps= 4,  # [j for j in range(1,9) for i in range(4)]
            forecast_policy="fixed",  # 'sequential_random' # 'fixed' #'sequential' #_random'
            forecast_att_dense_rate=1.0,  # 0.25
            fe_num_blocks=8,
            fe_num_heads=16,
            fe_dropout_rate=0.1,
            fe_with_qk_lnorm=True,
            lr_start=0.000001,
            lr_max=0.00003,
            lr_final_decay=0.00003,
            lr_final=0.0,
            lr_steps_warmup=1024,
            lr_steps_cooldown=4096,
            lr_policy_warmup="cosine",
            lr_policy_decay="linear",
            lr_policy_cooldown="linear",
            num_epochs=12,  # len(cf.forecast_steps) + 4
            istep=0,
        )
    else:
        finetune_overwrite = dict()

    cli_overwrite = config.from_cli_arglist(args.options)
    cf = config.load_config(
        args.private_config,
        args.from_run_id,
        args.epoch,
        finetune_overwrite,
        *args.config,
        cli_overwrite,
    )
    cf = config.set_run_id(cf, args.run_id, args.reuse_run_id)

    devices = Trainer.init_torch()
    cf = Trainer.init_ddp(cf)

    init_loggers(cf.run_id)

    # track history of run to ensure traceability of results
    cf.run_history += [(args.from_run_id, cf.istep)]

    trainer = Trainer(cf.train_log_freq)
    trainer.run(cf, devices, args.from_run_id, args.epoch)


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

    cf = config.load_config(args.private_config, None, None, *args.config, cli_overwrite)
    cf = config.set_run_id(cf, args.run_id, False)

    cf.data_loader_rng_seed = int(time.time())
    devices = Trainer.init_torch()
    cf = Trainer.init_ddp(cf)

    # if cf.rank == 0:
    # this line should probably come after the processes have been sorted out else we get lots
    # of duplication due to multiple process in the multiGPU case
    init_loggers(cf.run_id)

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
        pdb.post_mortem(tb)


if __name__ == "__main__":
    # Entry point for slurm script.
    # Check whether --from_run_id passed as argument.
    if next((True for arg in sys.argv if "--from_run_id" in arg), False):
        train_continue()
    else:
        train()
