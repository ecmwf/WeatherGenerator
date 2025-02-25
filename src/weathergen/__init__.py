# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import time
import sys
import pdb
import traceback

from weathergen.utils.config import Config
from weathergen.train.trainer import Trainer
from weathergen.train.utils import get_run_id


####################################################################################################
def evaluate(
    run_id,
    epoch,
    masking_mode=None,
    forecacast_steps=None,
    samples=10000000,
    shuffle=False,
    save_samples=True,
    gridded_output_streams=[],
):
    # load config if specified
    cf = Config.load(run_id, epoch if epoch is not None else -1)

    cf.run_history += [(cf.run_id, cf.istep)]

    cf.samples_per_validation = samples
    cf.log_validation = samples if save_samples else 0

    if masking_mode is not None:
        cf.masking_mode = masking_mode

    # Oct-Nov 2022
    cf.start_date_val = 202210011600
    cf.end_date_val = 202212010400
    # # 2022
    # cf.start_date_val = 202201010400
    # cf.end_date_val = 202301010400

    cf.step_hrs = 12

    cf.shuffle = shuffle

    cf.forecast_steps = forecacast_steps if forecacast_steps else cf.forecast_steps
    # cf.forecast_policy = 'fixed'

    # cf.analysis_streams_output = ['Surface', 'Air', 'METEOSAT', 'ATMS', 'IASI', 'AMSR2']
    cf.analysis_streams_output = ["ERA5"]

    # make sure number of loaders does not exceed requested samples
    cf.loader_num_workers = min(cf.loader_num_workers, samples)

    trainer = Trainer()
    trainer.evaluate(cf, run_id, epoch, True)


####################################################################################################
def train(run_id=None) -> None:
    cf = Config()

    # directory where input streams are specified
    # cf.streams_directory = './streams_large/'
    # cf.streams_directory = './streams_anemoi/'
    cf.streams_directory = "./streams_mixed/"

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

    cf.data_path = "/home/mlx/ai-ml/datasets/stable/"
    # cf.data_path = '/lus/h2resw01/fws4/lb/project/ai-ml/observations/v1'
    # cf.data_path = '/leonardo_scratch/large/userexternal/clessig0/obs/v1'
    cf.start_date = 201301010000
    cf.end_date = 202012310000
    cf.start_date_val = 202101010000
    cf.end_date_val = 202201010000
    cf.len_hrs = 6
    cf.step_hrs = 6
    cf.input_window_steps = 1

    cf.val_initial = False

    cf.loader_num_workers = 8
    cf.data_loader_rng_seed = int(time.time())
    cf.log_validation = 0

    cf.istep = 0
    cf.run_history = []

    cf.run_id = run_id
    cf.desc = ""

    trainer = Trainer(log_freq=20, checkpoint_freq=250, print_freq=10)

    try:
        trainer.run(cf)
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


if __name__ == "__main__":
    train()
