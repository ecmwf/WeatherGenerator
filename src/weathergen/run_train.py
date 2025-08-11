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

import weathergen.utils.cli as cli
import weathergen.utils.config as config
from weathergen.train.trainer import Trainer
from weathergen.utils.logger import init_loggers


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

    init_loggers(
        logging_level=logging.DEBUG, debug_output_streams=f"./logs/debug_log_{cf.run_id}.txt"
    )

    cf.run_history += [(args.from_run_id, cf.istep)]
    cf = config.set_paths(cf)

    trainer = Trainer()
    trainer.inference(cf, args.from_run_id, args.epoch)


####################################################################################################
def train_continue() -> None:
    parser = cli.get_continue_parser()
    args = parser.parse_args()

    init_loggers()

    if args.finetune_forecast:
        finetune_overwrite = dict(
            training_mode="forecast",
            forecast_delta_hrs=0,  # 12
            forecast_steps=1,  # [j for j in range(1,9) for i in range(4)]
            forecast_policy="fixed",  # 'sequential_random' # 'fixed' #'sequential' #_random'
            forecast_freeze_model=True,
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

    # track history of run to ensure traceability of results
    cf.run_history += [(args.from_run_id, cf.istep)]
    cf = config.set_paths(cf)

    if args.finetune_forecast:
        if cf.forecast_freeze_model:
            cf.with_fsdp = False
            import torch

            torch._dynamo.config.optimize_ddp = False
    trainer = Trainer()
    trainer.run(cf, args.from_run_id, args.epoch)


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

    init_loggers()

    cli_overwrite = config.from_cli_arglist(args.options)
    cf = config.load_config(args.private_config, None, None, *args.config, cli_overwrite)
    cf = config.set_run_id(cf, args.run_id, False)

    cf.streams = config.load_streams(Path(cf.streams_directory))
    cf = config.set_paths(cf)

    if cf.with_flash_attention:
        assert cf.with_mixed_precision
    cf.data_loader_rng_seed = int(time.time())

    trainer = Trainer(checkpoint_freq=250, print_freq=10)

    try:
        trainer.run(cf)
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
