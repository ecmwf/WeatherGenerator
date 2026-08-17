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

import torch
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.tensor import distribute_tensor

from weathergen.common.config import Config, get_path_model, merge_configs
from weathergen.model.attention import (
    MultiCrossAttentionHeadVarlen,
    MultiCrossAttentionHeadVarlenSlicedQ,
    MultiSelfAttentionHead,
    MultiSelfAttentionHeadLocal,
    MultiSelfAttentionHeadVarlen,
)
from weathergen.model.layers import MLP
from weathergen.model.model import Model, ModelParams
from weathergen.model.utils import (
    apply_fct_to_blocks,
    broadcast_matching_params,
    check_reset_not_frozen,
    freeze_weights,
    log_trainable_summary,
    reset_weights,
)
from weathergen.utils.distributed import is_root
from weathergen.utils.utils import get_dtype

logger = logging.getLogger(__name__)


# same as in config: student_teacher, forecasting, masking
type TrainingMode = str


def _has_trainable_params(module: torch.nn.Module) -> bool:
    """True if the module has at least one parameter with requires_grad=True.

    FSDP2 raises "RuntimeError: _chunk_cat expects non-empty tensor" in the
    backward reduce-scatter (foreach_reduce) when a fully_shard group contains
    only frozen parameters, since there are no gradients to reduce. This happens
    during fine-tuning (e.g. forecast fine-tuning freezes the encoder and
    latent_heads). Skipping fully_shard for fully-frozen modules leaves their
    parameters in the root FSDP group, which still has trainable parameters, so
    they remain sharded without triggering the empty-gradient reduce.
    """
    return any(p.requires_grad for p in module.parameters())


def init_model_and_shard(
    cf,
    dataset,
    run_id_contd,
    mini_epoch_contd,
    training_mode,
    device,
    with_ddp,
    with_fsdp,
    overrides={},
):
    model_creation_device = "meta" if with_ddp and with_fsdp else "cuda"
    with torch.device(model_creation_device):
        model = get_model(cf, training_mode, dataset, overrides)

    # freeze request model part
    apply_fct_to_blocks(model, cf.freeze_modules, freeze_weights)

    # TODO: this should be handled in the encoder to be close where q_cells is defined
    if "q_cells" in cf.freeze_modules:
        model.encoder.q_cells.requires_grad = False
    if "q_aux" in cf.freeze_modules:
        if model.encoder.q_aux is not None:
            model.encoder.q_aux.requires_grad = False

    if with_ddp and not with_fsdp:
        # create DDP model if running without FSDP
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            broadcast_buffers=True,
            find_unused_parameters=cf.get("ddp_find_unused_parameters", True),
            gradient_as_bucket_view=True,
            bucket_cap_mb=512,
            static_graph=cf.get("ddp_static_graph", False),
        )

    elif with_ddp and with_fsdp:
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
            if isinstance(module, modules_to_shard) and _has_trainable_params(module):
                fully_shard(module, **fsdp_kwargs)

        for module in model.encoder.ae_local_global_engine.ae_adapter.modules():
            if isinstance(module, modules_to_shard) and _has_trainable_params(module):
                fully_shard(module, **fsdp_kwargs)

        for module in model.encoder.ae_global_engine.ae_global_blocks.modules():
            if isinstance(module, modules_to_shard) and _has_trainable_params(module):
                fully_shard(module, **fsdp_kwargs)

        if cf.get("fe_diffusion_model", False):
            model_fe_blocks = model.forecast_engine.net.fe_blocks
        else:
            model_fe_blocks = model.forecast_engine.fe_blocks
        for module in model_fe_blocks.modules():
            if isinstance(module, modules_to_shard):
                fully_shard(module, **fsdp_kwargs)

        for module in model.latent_heads.modules():
            if isinstance(module, modules_to_shard):
                # reshard_after_forward=False keeps FE parameters unsharded
                # during the multi-step rollout loop.
                # Needed for pushforward trick.
                fully_shard(module, reshard_after_forward=False, **fsdp_kwargs)

        for module in model.latent_heads.modules():
            if isinstance(module, modules_to_shard) and _has_trainable_params(module):
                fully_shard(module, **fsdp_kwargs)

        if model.deep_ssl_fusion is not None:
            for module in model.deep_ssl_fusion.modules():
                if isinstance(module, modules_to_shard) and _has_trainable_params(module):
                    fully_shard(module, **fsdp_kwargs)

        if model.deep_ssl_level_projections is not None:
            for module in model.deep_ssl_level_projections.modules():
                if isinstance(module, modules_to_shard) and _has_trainable_params(module):
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

        for module in model.target_token_engines.modules():
            if isinstance(module, modules_to_shard) and _has_trainable_params(module):
                fully_shard(module, **full_precision_fsdp_kwargs)

    if with_ddp and with_fsdp:
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
    loaded_from_run_id = None
    if run_id_contd is not None:
        if is_root():
            logger.info(f"Continuing run with id={run_id_contd} at mini_epoch {mini_epoch_contd}.")
        model = load_model(cf, model, device, run_id_contd, with_ddp, with_fsdp, mini_epoch_contd)
        loaded_from_run_id = run_id_contd
    elif cf.get("load_chkpt", {}).get("run_id", None):
        run_id = cf.load_chkpt.run_id
        mini_epoch = cf.load_chkpt.get("mini_epoch", -1)
        if is_root():
            logger.info(f"Loading checkpoint from id={run_id} at mini_epoch {mini_epoch}.")
        model = load_model(cf, model, device, run_id, with_ddp, with_fsdp, mini_epoch)
        loaded_from_run_id = run_id
    else:
        if with_ddp and with_fsdp:
            model.to_empty(device="cuda")
            if with_fsdp:
                model.reset_parameters()

    # Reset specified modules when starting a new stage (e.g. pretrain -> finetune);
    # skip when resuming the same run.
    current_run_id = cf.general.run_id
    if loaded_from_run_id is not None and loaded_from_run_id != current_run_id:
        reset_modules = cf.get("reset_modules", "")
        if reset_modules:
            assert not with_fsdp, "reset_modules with FSDP-sharded parameters is not supported"
            # a parameter that is both reset and frozen would stay random forever
            check_reset_not_frozen(model, reset_modules)
            if is_root():
                logger.info(f"Resetting weights for modules matching: {reset_modules}")
            apply_fct_to_blocks(model, reset_modules, reset_weights)
            # each rank resets with its own RNG; sync to rank 0 like DDP does at wrap time
            broadcast_matching_params(model, reset_modules, src=0)

    if is_root():
        log_trainable_summary(model)

    # Optionally overlay the physical decoder from a separate checkpoint. This runs after the
    # primary load so it takes precedence for decoder weights while keeping the encoder /
    # forecast engine from the primary checkpoint (e.g. inference with a latent-diffusion run
    # that has no trained decoder, reusing a pretrained decoder from another run).
    decoder_run_id = cf.get("load_decoder_chkpt", {}).get("run_id", None)
    if decoder_run_id:
        decoder_mini_epoch = cf.load_decoder_chkpt.get("mini_epoch", -1)
        if is_root():
            logger.info(
                f"Loading decoder weights from id={decoder_run_id} "
                f"at mini_epoch {decoder_mini_epoch}."
            )
        model = load_decoder_from_checkpoint(
            cf, model, device, decoder_run_id, with_ddp, with_fsdp, decoder_mini_epoch
        )

    # model params
    model_params = ModelParams(cf).create(cf)
    model_params.reset_parameters(cf)
    model_params = model_params.to(f"cuda:{cf.local_rank}")

    return model, model_params


def load_model(cf, model, device, run_id: str, with_ddp: bool, with_fsdp: bool, mini_epoch: int):
    """Loads model state from checkpoint and checks for missing and unused keys.
    Args:
        run_id : model_id of the trained model
        mini_epoch : The mini_epoch to load.
    """

    path_run = get_path_model(run_id=run_id)
    mini_epoch_id = (
        f"chkpt{mini_epoch:05d}" if mini_epoch != -1 and mini_epoch is not None else "latest"
    )
    filename = f"{run_id}_{mini_epoch_id}.chkpt"

    params = torch.load(
        path_run / filename, map_location=torch.device("cpu"), mmap=True, weights_only=True
    )

    is_model_sharded = with_ddp and with_fsdp
    if is_model_sharded:
        model_has_prefix_module = list(model.state_dict().keys())[0].split(".")[0] == "module"
        params_has_prefix_module = list(params.keys())[0].split(".")[0] == "module"
        if model_has_prefix_module and not params_has_prefix_module:
            # add "module." prefix
            params_temp = {}
            for k in params.keys():
                params_temp["module." + k] = params[k]
            params = params_temp
        elif not model_has_prefix_module and params_has_prefix_module:
            # remove "module." prefix
            params_temp = {}
            for k in params.keys():
                params_temp[k.replace("module.", "")] = params[k]
            params = params_temp

        meta_sharded_sd = model.state_dict()
        maybe_sharded_sd = {}
        for param_name, full_tensor in params.items():
            sharded_meta_param = meta_sharded_sd.get(param_name)
            if (
                sharded_meta_param is None
                or type(sharded_meta_param) is not torch.distributed.tensor.DTensor
            ):
                logger.warning(f"Parameter {param_name} from checkpoint not found in model.")
                continue
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

            # Get all modules for quick lookup and initialize the new ones.
            # named_modules() deduplicates shared modules (remove_duplicate=True by default),
            # yielding only the first-encountered path. FSDP2's state_dict() uses
            # remove_duplicate=False, so mkeys may contain alias paths (e.g. both
            # `layers.0.embed_aux.2` and `lnorm.embed_aux.2` for the same AdaLayerNorm
            # inside MLP where lnorm is appended to self.layers). Alias paths are absent
            # from all_modules; skip them – the canonical path is processed separately.
            all_modules = dict(model.named_modules())
            for path in root_new_modules:
                if path not in all_modules:
                    if is_root():
                        logger.debug(
                            f"Skipping alias module path not in named_modules "
                            f"(duplicate FSDP2 state-dict entry): {path}"
                        )
                    continue
                if is_root():
                    logger.info(f"Initializing new module not found in checkpoint: {path}")
                module_to_init = all_modules[path]
                module_to_init.to_empty(device="cuda")
                module_to_init.reset_parameters()

    else:
        # fix mismatch between state_dict keys that can occur between interactive/non-interactive
        model_has_prefix_module = list(model.state_dict().keys())[0].split(".")[0] == "module"
        params_has_prefix_module = list(params.keys())[0].split(".")[0] == "module"
        if model_has_prefix_module and not params_has_prefix_module:
            # add "module." prefix
            params_temp = {}
            for k in params.keys():
                params_temp["module." + k] = params[k]
            params = params_temp
        elif not model_has_prefix_module and params_has_prefix_module:
            # remove "module." prefix
            params_temp = {}
            for k in params.keys():
                params_temp[k.replace("module.", "")] = params[k]
            params = params_temp
        # load checkpoint
        mkeys, ukeys = model.load_state_dict(params, strict=False)
        model = model.to(device)

    # warn about difference in checkpoint and model
    if len(mkeys) == 0 and len(ukeys) == 0:
        logger.info(f"Checkpoint {filename} loaded successfully with all weights matching.")
    if len(mkeys) > 0:
        logger.warning(f"Missing keys when loading model: {mkeys}")
    if len(ukeys) > 0:
        logger.warning(f"Unused keys when loading model: {ukeys}")

    return model


# top-level module prefixes that make up the physical decoder
_DECODER_PREFIXES = ("embed_target_coords", "target_token_engines", "pred_heads")


def load_decoder_from_checkpoint(
    cf, model, device, run_id: str, with_ddp: bool, with_fsdp: bool, mini_epoch=-1
):
    """Overlay only the physical decoder weights from a separate checkpoint.

    Filters the checkpoint to ``embed_target_coords.*``, ``target_token_engines.*`` and
    ``pred_heads.*`` and loads them into ``model`` with ``strict=False`` (all other model
    weights are left untouched). This allows e.g. reusing a pretrained decoder together with an
    encoder / forecast engine that was trained separately (see load_model for the primary load).

    Args:
        run_id : model_id of the run providing the decoder weights
        mini_epoch : The mini_epoch to load. Default (-1) is the latest mini_epoch
    """

    path_run = get_path_model(run_id=run_id)
    mini_epoch_id = (
        f"chkpt{mini_epoch:05d}" if mini_epoch != -1 and mini_epoch is not None else "latest"
    )
    filename = f"{run_id}_{mini_epoch_id}.chkpt"

    params = torch.load(
        path_run / filename, map_location=torch.device("cpu"), mmap=True, weights_only=True
    )

    def _strip(key: str) -> str:
        return key[len("module.") :] if key.startswith("module.") else key

    decoder_params = {k: v for k, v in params.items() if _strip(k).startswith(_DECODER_PREFIXES)}

    if not decoder_params:
        logger.warning(
            f"No decoder weights (matching {_DECODER_PREFIXES}) found in checkpoint {filename}."
        )
        return model

    is_model_sharded = with_ddp and with_fsdp
    if is_model_sharded:
        meta_sharded_sd = model.state_dict()
        maybe_sharded_sd = {}
        for param_name, full_tensor in decoder_params.items():
            sharded_meta_param = meta_sharded_sd.get(param_name)
            if (
                sharded_meta_param is None
                or type(sharded_meta_param) is not torch.distributed.tensor.DTensor
            ):
                logger.warning(
                    f"Decoder parameter {param_name} from checkpoint not found in model "
                    "or not sharded; skipping."
                )
                continue
            sharded_tensor = distribute_tensor(
                full_tensor,
                sharded_meta_param.device_mesh,
                sharded_meta_param.placements,
            )
            maybe_sharded_sd[param_name] = torch.nn.Parameter(sharded_tensor)
        _, ukeys = model.load_state_dict(maybe_sharded_sd, strict=False, assign=True)
        loaded = maybe_sharded_sd
    else:
        # align "module." prefix with the model's state dict key convention
        model_has_prefix_module = list(model.state_dict().keys())[0].split(".")[0] == "module"
        params_has_prefix_module = next(iter(decoder_params)).split(".")[0] == "module"
        if model_has_prefix_module and not params_has_prefix_module:
            decoder_params = {"module." + k: v for k, v in decoder_params.items()}
        elif not model_has_prefix_module and params_has_prefix_module:
            decoder_params = {_strip(k): v for k, v in decoder_params.items()}
        _, ukeys = model.load_state_dict(decoder_params, strict=False)
        model = model.to(device)
        loaded = decoder_params

    logger.info(
        f"Loaded {len(loaded)} decoder tensors from checkpoint {filename} (run_id={run_id})."
    )
    # ukeys = decoder keys that are absent in the model; missing keys are intentionally not
    # reported here since the primary checkpoint provides all non-decoder weights.
    if len(ukeys) > 0:
        logger.warning(f"Decoder keys from checkpoint not present in model: {ukeys}")

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

    cf_with_overrides = merge_configs(cf, overrides)
    return Model(
        cf_with_overrides, sources_size, targets_num_channels, targets_coords_size
    ).create()
