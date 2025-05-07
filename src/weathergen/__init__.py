# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pdb
import sys
import time
import traceback

import pandas as pd

import weathergen.utils.cli as cli
import weathergen.utils.config as config
from weathergen.train.trainer import Trainer
from weathergen.utils.logger import init_loggers


def evaluate():
    # By default, arguments from the command line are read.
    evaluate_from_args(sys.argv[1:])


def evaluate_from_args(argl: list[str]):
    """
    Evaluation function for WeatherGenerator model.
    Entry point for calling the evaluation code from the command line.

    When running integration tests, the arguments are directly provided.
    """
    parser = cli.get_evaluate_parser()
    args = parser.parse_args(argl)

    # TODO: move somewhere else
    init_loggers()

    cf = config.load_config(args.private_config, args.run_id, args.epoch, args.config)

    cf.run_history += [(cf.run_id, cf.istep)]

    cf.samples_per_validation = args.samples
    cf.log_validation = args.samples if args.save_samples else 0
    start_date, end_date = pd.to_datetime(args.start_date), pd.to_datetime(args.end_date)

    cf.start_date_val = start_date.strftime("%Y%m%d%H%M")
    cf.end_date_val = end_date.strftime("%Y%m%d%H%M")

    cf.shuffle = args.shuffle

    cf.forecast_steps = args.forecast_steps if args.forecast_steps else cf.forecast_steps
    # cf.forecast_policy = 'fixed'

    # cf.analysis_streams_output = ['Surface', 'Air', 'METEOSAT', 'ATMS', 'IASI', 'AMSR2']
    cf.analysis_streams_output = args.analysis_streams_output

    # make sure number of loaders does not exceed requested samples
    cf.loader_num_workers = min(cf.loader_num_workers, args.samples)

    trainer = Trainer()
    trainer.evaluate(cf, args.run_id, args.epoch, run_id_new=args.eval_run_id)


####################################################################################################
def train_continue() -> None:
    parser = cli.get_continue_parser()
    args = parser.parse_args()

    if args.finetune_forecast:
        finetune_overwrite = dict(
            forecast_delta_hrs = 0,  # 12
            forecast_steps = 1,  # [j for j in range(1,9) for i in range(4)]
            forecast_policy = "fixed",  # 'sequential_random' # 'fixed' #'sequential' #_random'
            forecast_freeze_model = True,
            forecast_att_dense_rate = 1.0,  # 0.25
            fe_num_blocks = 8,
            fe_num_heads = 16,
            fe_dropout_rate = 0.1,
            fe_with_qk_lnorm = True,
            lr_start = 0.000001,
            lr_max = 0.00003,
            lr_final_decay = 0.00003,
            lr_final = 0.0,
            lr_steps_warmup = 1024,
            lr_steps_cooldown = 4096,
            lr_policy_warmup = "cosine",
            lr_policy_decay = "linear",
            lr_policy_cooldown = "linear",
            num_epochs = 12,  # len(cf.forecast_steps) + 4
            istep = 0,
            training_mode = "forecast"
        )

    cf = config.load_config(args.private_config, args.run_id, args.epoch, args.config,finetune_overwrite)

    # track history of run to ensure traceability of results
    cf.run_history += [(cf.run_id, cf.istep)]

    if args.finetune_forecast:
        if cf.forecast_freeze_model:
            cf.with_fsdp = False
            import torch

            torch._dynamo.config.optimize_ddp = False
    trainer = Trainer()
    trainer.run(cf, args.run_id, args.epoch, args.run_id_new)


####################################################################################################
def train() -> None:
    """
    Training function for WeatherGenerator model.
    Entry point for calling the training code from the command line.
    Configurations are set in the function body.

    Args:
      run_id (str, optional): Run/model id of pretrained WeatherGenerator model to continue training. Defaults to None.

    Note: All model configurations are set in the function body.
    """
    train_with_args(sys.argv[1:], None)


def train_with_args(argl: list[str], stream_dir: str | None):
    """
    Training function for WeatherGenerator model."""
    parser = cli.get_train_parser()
    args = parser.parse_args(argl)

    # TODO: move somewhere else
    init_loggers()

    cf = config.load_config(args.private_config, None, None, args.config)

    if cf.with_flash_attention:
        assert cf.with_mixed_precision
    cf.data_loader_rng_seed = int(time.time())

    trainer = Trainer(checkpoint_freq=250, print_freq=10)

    try:
        trainer.run(cf, run_id_new=args.run_id)
    except Exception:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


if __name__ == "__main__":
    train()
    # train_continue()
