# ruff: noqa: B006

# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import itertools
import logging
import re
from pathlib import Path

import torch
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.tensor import distribute_tensor

from weathergen.common.config import Config
from weathergen.model.attention import (
    MultiCrossAttentionHeadVarlen,
    MultiCrossAttentionHeadVarlenSlicedQ,
    MultiSelfAttentionHead,
    MultiSelfAttentionHeadLocal,
    MultiSelfAttentionHeadVarlen,
)
from weathergen.model.ema import EMAModel
from weathergen.model.layers import MLP
from weathergen.model.model import Model, ModelParams
from weathergen.model.utils import freeze_weights
from weathergen.train.target_and_aux_module_base import PhysicalTargetAndAux
from weathergen.utils.distributed import is_root
from weathergen.utils.utils import apply_overrides_to_dict, get_dtype

logger = logging.getLogger(__name__)


# same as in config: student_teacher, forecasting, masking
type TrainingMode = str


def init_model_and_shard(
    cf, dataset, run_id_contd, mini_epoch_contd, training_mode, device, overrides={}
):
    model_creation_device = "meta" if cf.with_ddp and cf.with_fsdp else "cuda"
    with torch.device(model_creation_device):
        model = get_model(cf, training_mode, dataset, overrides)

    freeze_modules = cf.freeze_modules

    # freeze request model part
    for name, module in model.named_modules():
        name = module.name if hasattr(module, "name") else name
        # avoid the whole model element which has name ''
        if name == "":
            continue
        if re.fullmatch(freeze_modules, name) is not None:
            freeze_weights(module)

    if cf.with_ddp and not cf.with_fsdp:
        # create DDP model if running without FSDP
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            broadcast_buffers=True,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
            bucket_cap_mb=512,
        )

    elif cf.with_ddp and cf.with_fsdp:
        # with DDP *and() FSDP
        fsdp_kwargs = {
            "mp_policy": (
                MixedPrecisionPolicy(
                    param_dtype=get_dtype(cf.mixed_precision_dtype),
                    reduce_dtype=torch.float32,
                )
                if cf.with_mixed_precision
                else None
            ),
        }
        modules_to_shard = (
            MLP,
            MultiSelfAttentionHeadLocal,
            MultiSelfAttentionHead,
            MultiCrossAttentionHeadVarlen,
            MultiCrossAttentionHeadVarlenSlicedQ,
            MultiSelfAttentionHeadVarlen,
        )

        for module in model.encoder.ae_local_engine.ae_local_blocks.modules():
            if isinstance(module, modules_to_shard):
                fully_shard(module, **fsdp_kwargs)

        for module in model.encoder.ae_local_global_engine.ae_adapter.modules():
            if isinstance(module, modules_to_shard):
                fully_shard(module, **fsdp_kwargs)

        for module in model.encoder.ae_global_engine.ae_global_blocks.modules():
            if isinstance(module, modules_to_shard):
                fully_shard(module, **fsdp_kwargs)

        for module in model.forecast_engine.fe_blocks.modules():
            if isinstance(module, modules_to_shard):
                fully_shard(module, **fsdp_kwargs)

        full_precision_fsdp_kwargs = {
            "mp_policy": (
                MixedPrecisionPolicy(
                    param_dtype=torch.float32,
                    reduce_dtype=torch.float32,
                )
                if cf.with_mixed_precision
                else None
            ),
        }
        for module in model.pred_adapter_kv.modules():
            if isinstance(module, modules_to_shard):
                fully_shard(module, **full_precision_fsdp_kwargs)

        for module in model.target_token_engines.modules():
            if isinstance(module, modules_to_shard):
                fully_shard(module, **full_precision_fsdp_kwargs)

    if cf.with_ddp and cf.with_fsdp:
        fully_shard(model)
        for tensor in itertools.chain(model.parameters(), model.buffers()):
            assert tensor.device == torch.device("meta")

        # For reasons we do not yet fully understand, when using train continue in some
        # instances, FSDP2 does not register the forward_channels and forward_columns
        # functions in the embedding engine as forward functions. Thus, yielding a crash
        # because the input tensors are not converted to DTensors. This seems to primarily
        # occur during validation.
        for embed in model.encoder.embed_engine.embeds.values():
            torch.distributed.fsdp.register_fsdp_forward_method(embed, "forward_channels")
            torch.distributed.fsdp.register_fsdp_forward_method(embed, "forward_columns")

    # complete initalization and load model if inference/continuing a run
    if run_id_contd is None:
        if cf.with_ddp and cf.with_fsdp:
            model.to_empty(device="cuda")
            if cf.with_fsdp:
                model.reset_parameters()
    else:
        if is_root():
            logger.info(f"Continuing run with id={run_id_contd} at mini_epoch {mini_epoch_contd}.")
        model = load_model(cf, model, device, run_id_contd, mini_epoch_contd)

    # model params
    model_params = ModelParams(cf).create(cf)
    model_params.reset_parameters(cf)
    model_params = model_params.to(f"cuda:{cf.local_rank}")

    return model, model_params


def load_model(cf, model, device, run_id: str, mini_epoch=-1):
    """Loads model state from checkpoint and checks for missing and unused keys.
    Args:
        run_id : model_id of the trained model
        mini_epoch : The mini_epoch to load. Default (-1) is the latest mini_epoch
    """

    path_run = Path(cf.model_path) / run_id
    mini_epoch_id = (
        f"chkpt{mini_epoch:05d}" if mini_epoch != -1 and mini_epoch is not None else "latest"
    )
    filename = f"{run_id}_{mini_epoch_id}.chkpt"

    if not (path_run / filename).exists():
        mini_epoch_id = f"epoch{mini_epoch:05d}"
        filename = f"{run_id}_{mini_epoch_id}.chkpt"

    params = torch.load(
        path_run / filename, map_location=torch.device("cpu"), mmap=True, weights_only=True
    )

    is_model_sharded = cf.with_ddp and cf.with_fsdp
    if is_model_sharded:
        meta_sharded_sd = model.state_dict()
        maybe_sharded_sd = {}
        for param_name, full_tensor in params.items():
            sharded_meta_param = meta_sharded_sd.get(param_name)
            sharded_tensor = distribute_tensor(
                full_tensor,
                sharded_meta_param.device_mesh,
                sharded_meta_param.placements,
            )
            # maybe_sharded_sd[param_name.replace("module.", "")] = nn.Parameter(sharded_tensor)
            maybe_sharded_sd[param_name] = torch.nn.Parameter(sharded_tensor)
        # choose `assign=True` for sharded model since we cannot call `copy_` on meta tensor
        mkeys, ukeys = model.load_state_dict(maybe_sharded_sd, strict=False, assign=True)

        # new network parts (e.g. for fine-tuning)
        if mkeys:
            # Get the unique parent modules for the missing parameters
            new_modules_to_init = {key.rsplit(".", 1)[0] for key in mkeys}

            # Find the highest-level "root" new modules to avoid redundant initializations
            root_new_modules = set()
            for path in sorted(list(new_modules_to_init)):
                if not any(path.startswith(root + ".") for root in root_new_modules):
                    root_new_modules.add(path)

            # Get all modules for quick lookup and initialize the new ones
            all_modules = dict(model.named_modules())
            for path in root_new_modules:
                if is_root():
                    logger.info(f"Initializing new module not found in checkpoint: {path}")
                module_to_init = all_modules[path]
                module_to_init.to_empty(device="cuda")
                module_to_init.reset_parameters()

    else:
        if not cf.with_ddp:
            params_temp = {}
            for k in params.keys():
                params_temp[k.replace("module.", "")] = params[k]
            params = params_temp
        mkeys, ukeys = model.load_state_dict(params, strict=False)
        model = model.to(device)

    # warn about difference in checkpoint and model
    if len(mkeys) == 0 and len(ukeys) == 0:
        logger.info(f"Checkpoint {filename} loaded successfully with all weights matching.")
    if len(mkeys) > 0:
        logger.warning(f"Missing keys when loading model: {mkeys}")
    if len(ukeys) > 0:
        logger.warning(f"Unused keys when loading model: {mkeys}")

    return model


def get_model(cf: Config, training_mode: TrainingMode, dataset, overrides):
    """
    Create model

    cf :
    training_mode :
    dataset :
    """

    # TODO: how to avoid the dependence on dataset
    sources_size = dataset.get_sources_size()
    targets_num_channels = dataset.get_targets_num_channels()
    targets_coords_size = dataset.get_targets_coords_size()

    cf_with_overrides = apply_overrides_to_dict(cf, overrides)
    return Model(
        cf_with_overrides, sources_size, targets_num_channels, targets_coords_size
    ).create()


def get_target_aux_calculator(cf: Config, dataset, model, device, **kwargs):
    """
    Create target aux calculator
    """

    target_aux = None

    target_and_aux_calc = cf.get("target_and_aux_calc", "physical")
    if target_and_aux_calc == "physical":
        target_aux = PhysicalTargetAndAux(cf, model)

    elif target_and_aux_calc == "EMATeacher":
        # batch_size = get_batch_size(cf, cf.world_size_original)

        meta_ema_model, _ = init_model_and_shard(cf, dataset, None, None, "student", device)
        ema_model = EMAModel(
            model,
            meta_ema_model,
            halflife_steps=cf.get("ema_halflife_in_thousands", 1e-3),
            rampup_ratio=cf.get("ema_ramp_up_ratio", 0.09),
            is_model_sharded=(cf.with_ddp and cf.with_fsdp),
        )

        raise NotImplementedError(f"{target_and_aux_calc} is not implemented : {type(ema_model)}")

    else:
        raise NotImplementedError(f"{target_and_aux_calc} is not implemented")

    return target_aux
