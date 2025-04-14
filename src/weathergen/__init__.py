# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import argparse
import pdb
import sys
import time
import traceback

import pandas as pd

from weathergen.train.trainer import Trainer
from weathergen.utils.config import Config, load_overwrite_conf, load_private_conf
from weathergen.utils.logger import init_loggers


####################################################################################################
def evaluate():
    """
    Evaluation function for WeatherGenerator model.
    Entry point for calling the evaluation code from the command line.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="Run/model id of pretrained WeatherGenerator model.",
    )
    parser.add_argument(
        "--start_date",
        "-start",
        type=str,
        required=False,
        default="2022-10-01",
        help="Start date for evaluation. Format must be parsable with pd.to_datetime.",
    )
    parser.add_argument(
        "--end_date",
        "-end",
        type=str,
        required=False,
        default="2022-12-01",
        help="End date for evaluation. Format must be parsable with pd.to_datetime.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch of pretrained WeatherGenerator model used for evaluation (Default None corresponds to the last checkpoint).",
    )
    parser.add_argument(
        "--forecast_steps",
        type=int,
        default=None,
        help="Number of forecast steps for evaluation. Uses attribute from config when None is set.",
    )
    parser.add_argument(
        "--samples", type=int, default=10000000, help="Number of evaluation samples."
    )
    parser.add_argument(
        "--shuffle", type=bool, default=False, help="Shuffle samples from evaluation."
    )
    parser.add_argument(
        "--save_samples", type=bool, default=True, help="Save samples from evaluation."
    )
    parser.add_argument(
        "--analysis_streams_output",
        type=list,
        default=["ERA5"],
        help="Analysis output streams during evaluation.",
    )
    parser.add_argument(
        "--private_config",
        type=str,
        default=None,
        help="Path to private configuration file for paths.",
    )

    args = parser.parse_args()

    # get the paths from the private config
    private_cf = load_private_conf(args.private_config)

    # TODO: move somewhere else
    init_loggers()

    # load config: if run_id is full path, it loads from there
    model_path = private_cf["model_path"] if "model_path" in private_cf.keys() else "./models"
    cf = Config.load(args.run_id, args.epoch, model_path)

    # add parameters from private (paths) config
    for k, v in private_cf.items():
        setattr(cf, k, v)

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
    trainer.evaluate(cf, args.run_id, args.epoch, True)


####################################################################################################
def train_continue() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-id",
        "--run_id",
        type=str,
        required=True,
        help="run id of to be continued",
    )
    parser.add_argument(
        "-e",
        "--epoch",
        type=int,
        required=False,
        default=-1,
        help="epoch where to continue run",
    )
    parser.add_argument(
        "-n",
        "--run_id_new",
        type=bool,
        required=False,
        default=False,
        help="create new run id for cont'd run",
    )
    parser.add_argument(
        "--private_config",
        type=str,
        default=None,
        help="Path to private configuration file for paths.",
    )
    parser.add_argument(
         "--finetune_forecast",
        action='store_true',
        help="Fine tune for forecasting. It overwrites some of the Config settings.",
    )

    args = parser.parse_args()
    # get the paths from the private config
    private_cf = load_private_conf(args.private_config)

    # load config if specified
    model_path = private_cf["model_path"] if "model_path" in private_cf.keys() else "./models"
    cf = Config.load(args.run_id, args.epoch, model_path)

    # track history of run to ensure traceability of results
    if "run_history" not in cf.__dict__:
        cf.run_history = []
    cf.run_history += [(cf.run_id, cf.istep)]

    #########################
    if args.finetune_forecast:
        cf.forecast_delta_hrs = 0  # 12
        cf.forecast_steps = 1  # [j for j in range(1,9) for i in range(4)]
        cf.forecast_policy = "fixed"  # 'sequential_random' # 'fixed' #'sequential' #_random'
        cf.forecast_freeze_model = True
        cf.forecast_att_dense_rate = 1.0  # 0.25

        if cf.forecast_freeze_model:
            cf.with_fsdp = False
            import torch

            torch._dynamo.config.optimize_ddp = False

        cf.fe_num_blocks = 8
        cf.fe_num_heads = 16
        cf.fe_dropout_rate = 0.1
        cf.fe_with_qk_lnorm = True

        cf.lr_start = 0.000001
        cf.lr_max = 0.00003
        cf.lr_final_decay = 0.00003
        cf.lr_final = 0.0
        cf.lr_steps_warmup = 1024
        cf.lr_steps_cooldown = 4096
        cf.lr_policy_warmup = "cosine"
        cf.lr_policy_decay = "linear"
        cf.lr_policy_cooldown = "linear"

        cf.num_epochs = 12  # len(cf.forecast_steps) + 4
        cf.istep = 0

    trainer = Trainer()
    trainer.run(cf, private_cf, args.run_id, args.epoch, args.run_id_new)


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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run id",
    )
    parser.add_argument(
        "--private_config",
        type=str,
        default=None,
        help="Path to private configuration file for paths",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration with parameters to overwrite",
    )

    args = parser.parse_args()

    # TODO: move somewhere else
    init_loggers()

    # get the non-default configs: private and overwrite
    private_cf = load_private_conf(args.private_config)
    overwrite_cf = load_overwrite_conf(args.config)

    cf = Config()

    # directory where input streams are specified
    # cf.streams_directory = './streams_large/'
    cf.streams_directory = "./config/streams/streams_anemoi/"
    # cf.streams_directory = "./config/streams/streams_mixed/"
    # cf.streams_directory = "./streams_mixed/"

    # embed_orientation : 'channels' or 'columns'
    # channels: embedding is per channel for a token (#tokens=num_channels)
    # columns:  embedding is per "column", all channels are embedded together (#tokens=token_size)
    # the per-stream embedding paramters, in particular dim_embed, have to be chosen accordingly
    cf.embed_orientation = "channels"
    cf.embed_local_coords = True
    # False since per cell coords are meaningless for cells
    cf.embed_centroids_local_coords = False
    cf.embed_size_centroids = 64
    cf.embed_unembed_mode = "block"

    cf.target_cell_local_prediction = True
    cf.target_coords_local = True

    # parameters for local assimilation engine
    cf.ae_local_dim_embed = 1024  # 2048 #1024
    cf.ae_local_num_blocks = 2
    cf.ae_local_num_heads = 16
    cf.ae_local_dropout_rate = 0.1
    cf.ae_local_with_qk_lnorm = True

    # assimilation engine local -> global adapter
    cf.ae_local_num_queries = 2
    cf.ae_local_queries_per_cell = False
    cf.ae_adapter_num_heads = 16
    cf.ae_adapter_embed = 128
    cf.ae_adapter_with_qk_lnorm = True
    cf.ae_adapter_with_residual = True
    cf.ae_adapter_dropout_rate = 0.1

    # parameters for global assimilation engine
    cf.ae_global_dim_embed = 2048
    cf.ae_global_num_blocks = 8
    cf.ae_global_num_heads = 32
    cf.ae_global_dropout_rate = 0.1
    cf.ae_global_with_qk_lnorm = True
    cf.ae_global_att_dense_rate = 0.2  # 0.25 : every 4-th block is dense attention
    cf.ae_global_block_factor = 64
    cf.ae_global_mlp_hidden_factor = 2

    cf.pred_adapter_kv = False
    cf.pred_self_attention = True
    cf.pred_dyadic_dims = False
    cf.pred_mlp_adaln = True

    # forecasting engine
    cf.forecast_delta_hrs = 0
    cf.forecast_steps = 0  # [j for j in range(1,11) for i in range(1)]
    cf.forecast_policy = None  #'fixed', 'sequential'
    cf.forecast_freeze_model = False  # False
    cf.forecast_att_dense_rate = 0.25

    cf.fe_num_blocks = 0
    cf.fe_num_heads = 16
    cf.fe_dropout_rate = 0.1
    cf.fe_with_qk_lnorm = True

    cf.healpix_level = 5

    # working precision
    cf.with_mixed_precision = True
    cf.with_flash_attention = True
    if cf.with_flash_attention:
        assert cf.with_mixed_precision
    # compile entire model
    cf.compile_model = False

    cf.with_fsdp = True

    cf.loss_fcts = [["mse", 1.0]]
    cf.loss_fcts_val = [["mse", 1.0]]
    cf.tendencies = False
    # cf.loss_fcts = [['mse', 0.5], ['stats', 0.5]]
    # cf.loss_fcts_val = [['mse', 0.5], ['stats', 0.5]]

    cf.batch_size = 1
    cf.batch_size_validation = 1

    # forecast
    cf.masking_mode = "forecast"
    cf.masking_rate = 0.0
    cf.masking_rate_sampling = True  # False
    cf.sampling_rate_target = 1.0

    cf.num_epochs = 24
    cf.samples_per_epoch = 4096
    cf.samples_per_validation = 512
    cf.shuffle = True

    cf.lr_scaling_policy = "sqrt"
    cf.lr_start = 0.000001
    cf.lr_max = 0.00003
    cf.lr_final_decay = 0.000001
    cf.lr_final = 0.0
    cf.lr_steps_warmup = 256
    cf.lr_steps_cooldown = 4096
    cf.lr_policy_warmup = "cosine"
    cf.lr_policy_decay = "linear"
    cf.lr_policy_cooldown = "linear"

    cf.grad_clip = 5.0
    cf.weight_decay = 0.1
    cf.norm_type = "LayerNorm"  #'LayerNorm' #'RMSNorm'
    cf.nn_module = "te"

    cf.start_date = 201301010000
    cf.end_date = 202012310000
    cf.start_date_val = 202101010000
    cf.end_date_val = 202201010000
    cf.len_hrs = 6
    cf.step_hrs = 6
    cf.input_window_steps = 1

    cf.val_initial = False

    cf.loader_num_workers = 0 #8
    cf.data_loader_rng_seed = int(time.time())
    cf.log_validation = 0

    cf.istep = 0
    cf.run_history = []

    cf.run_id = args.run_id
    cf.desc = ""

    # overwrite parameters from private config
    for k, v in private_cf.items():
        setattr(cf, k, v)
    cf.data_path = private_cf["data_path_anemoi"]  # for backward compatibility

    # overwrite parameters from overwrite config
    for k, v in overwrite_cf.items():
        setattr(cf, k, v)

    trainer = Trainer(log_freq=20, checkpoint_freq=250, print_freq=10)

    try:
        trainer.run(cf, private_cf)
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


if __name__ == "__main__":
    train()
    # train_continue()
