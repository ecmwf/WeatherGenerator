import time

import pytest
import torch

import weathergen.train.loss as losses
import weathergen.utils.config as config
from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler
from weathergen.model.model import Model, ModelParams
from weathergen.train.loss_module import LossModule
from weathergen.train.trainer import Trainer
from weathergen.utils.train_logger import VAL


def prepare_trainer(cf):
    trainer = Trainer()
    trainer.init(cf)
    trainer.dataset_val = MultiStreamDataSampler(
        cf,
        cf.start_date_val,
        cf.end_date_val,
        cf.batch_size_validation,
        cf.samples_per_validation,
        train_logger=None,
        stage=VAL,
        shuffle=cf.shuffle,
    )
    '''
    batch = next(iter(trainer.dataset_val))
    batch = trainer.batch_to_device(batch)
    batch[0][0][0].target_times_raw
    breakpoint()
    '''

    sources_size = trainer.dataset_val.get_sources_size()
    targets_num_channels = trainer.dataset_val.get_targets_num_channels()
    targets_coords_size = trainer.dataset_val.get_targets_coords_size()
    trainer.devices = trainer.init_torch()
    trainer.model = Model(cf, sources_size, targets_num_channels, targets_coords_size).create()
    trainer.model = trainer.model.to(trainer.devices[0])
    trainer.model_params = ModelParams().create(cf).to(trainer.devices[0])
    trainer.mixed_precision_dtype = config.get_dtype(cf.attention_dtype)
    # get function handles for loss function terms
    trainer.loss_fcts = [[getattr(losses, name), w] for name, w in cf.loss_fcts]
    trainer.loss_fcts_val = [[getattr(losses, name), w] for name, w in cf.loss_fcts_val]

    return trainer


@pytest.fixture
def cf():
    cf = config.load_config(
        None,
        None,
        None,
    )
    cf = config.set_run_id(cf, None, False)
    cf.data_loader_rng_seed = int(time.time())

    return cf


def test_loss(cf):
    trainer = prepare_trainer(cf)

    batch = next(iter(trainer.dataset_val))
    batch = trainer.batch_to_device(batch)

    trainer.model.eval()
    # with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cf.with_mixed_precision):
    with torch.autocast(
        device_type="cuda", dtype=trainer.mixed_precision_dtype, enabled=cf.with_mixed_precision
    ):
        preds = trainer.model(trainer.model_params, batch, cf.forecast_offset, cf.forecast_steps)

    ### Original loss computation
    # TODO: remove dependency from old trainer.compute_loss() function
    loss1, losses_all1, stddev_all1, _ = trainer.compute_loss(
        trainer.loss_fcts_val,
        cf.forecast_offset,
        cf.forecast_steps,
        batch[0],
        preds,
        VAL,
    )

    ### Loss computation with new class
    loss_module_val = LossModule(
        cf=cf,
        stage=VAL,
        device=trainer.devices[0],
    )

    loss2, losses_all2, stddev_all2 = loss_module_val.compute_loss(
        preds=preds,
        streams_data=batch[0],
    )

    assert loss1 == loss2, "Loss computations return different values."


def test_logging_prep(cf):
    trainer = prepare_trainer(cf)

    batch = next(iter(trainer.dataset_val))
    batch = trainer.batch_to_device(batch)

    trainer.model.eval()
    # with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cf.with_mixed_precision):
    with torch.autocast(
        device_type="cuda", dtype=trainer.mixed_precision_dtype, enabled=cf.with_mixed_precision
    ):
        preds = trainer.model(trainer.model_params, batch, cf.forecast_offset, cf.forecast_steps)

    preds2 = [[preds[0][0].clone()]]

    ### Original logging preparation
    # TODO: remove dependency from old trainer.compute_loss() function
    _, _, _, logging_items = trainer.compute_loss(
        trainer.loss_fcts_val,
        cf.forecast_offset,
        cf.forecast_steps,
        batch[0],
        preds,
        VAL,
        log_data=True
    )
    preds_all, targets_all, targets_coords_raw, targets_times_raw, targets_lens = logging_items

    ### New logging preparation
    logging_items2 = trainer._prepare_logging(
        preds=preds2,
        forecast_offset=cf.forecast_offset,
        forecast_steps=cf.forecast_steps,
        streams_data=batch[0],
    )
    preds_all2, targets_all2, targets_coords_raw2, targets_times_raw2, targets_lens2 = logging_items2

    assert (preds_all[0][0][0] == preds_all2[0][0][0]).all()
    assert (targets_all[0][0][0] == targets_all2[0][0][0]).all()
    assert (targets_coords_raw[0][0] == targets_coords_raw2[0][0]).all()
    assert (targets_times_raw[0][0] == targets_times_raw2[0][0]).all()
    assert targets_lens == targets_lens2
