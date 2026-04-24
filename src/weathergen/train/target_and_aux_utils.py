import omegaconf

from weathergen.common.config import Config, merge_configs
from weathergen.model.ema import EMAModel
from weathergen.model.model_interface import init_model_and_shard
from weathergen.train.target_and_aux_module_base import PhysicalTargetAndAux
from weathergen.train.target_and_aux_ssl_teacher import EMATeacher, FrozenTeacher
from weathergen.train.teacher_utils import load_encoder_from_checkpoint, prepare_encoder_teacher


def get_target_aux_calculator(
    cf: Config, loss_cfg: omegaconf.OmegaConf, dataset, model, device, batch_size_per_gpu, **kwargs
):
    """
    Create target aux calculator
    """

    target_and_aux_calc_cfg = loss_cfg.get("target_and_aux_calc", "Physical")

    # parse target_and_aux_calc_cfg specification which can either be a string or config dict
    if type(target_and_aux_calc_cfg) is str:
        target_and_aux_calc = target_and_aux_calc_cfg
        target_and_aux_calc_params = {}
    elif type(target_and_aux_calc_cfg) is omegaconf.dictconfig.DictConfig:
        # single key is the target_and_aux_calc type
        target_and_aux_calc = list(target_and_aux_calc_cfg.keys())[0]
        # value is dict with the target_and_aux_calc parameters
        target_and_aux_calc_params = list(target_and_aux_calc_cfg.values())[0]
    else:
        assert False, "target_and_aux_calc needs either be name or config dict."

    # create target_and_aux_calc
    if target_and_aux_calc == "Physical":
        target_aux = PhysicalTargetAndAux(loss_cfg, model)

    elif target_and_aux_calc == "EMATeacher":
        # work around for problems with FSDP2
        assert not cf.with_fsdp, "EMATeacher not supported with FSDP(2) at the moment"

        meta_ema_model, _ = init_model_and_shard(
            cf,
            dataset,
            None,
            None,
            "student",
            device,
            with_ddp=False,
            with_fsdp=False,
            overrides=target_and_aux_calc_params.get("model_param_overrides", {}),
        )

        # Strip to encoder + create fresh heads
        cf_overridden = merge_configs(
            cf, target_and_aux_calc_params.get("model_param_overrides", {})
        )
        prepare_encoder_teacher(meta_ema_model, cf.training_config, cf_overridden)

        ema_model = EMAModel(
            model,
            meta_ema_model,
            halflife_steps=target_and_aux_calc_params.get("ema_halflife_in_thousands", 1e-3),
            rampup_ratio=target_and_aux_calc_params.get("ema_ramp_up_ratio", 0.09),
            is_model_sharded=(cf.with_ddp and cf.with_fsdp),
        )

        batch_size = cf.get("world_size_original", cf.get("world_size")) * batch_size_per_gpu
        target_aux = EMATeacher(model, ema_model, batch_size, cf.training_config)

        # Optional: warm start encoder from checkpoint
        teacher_run_id = target_and_aux_calc_params.get("teacher_run_id")
        if teacher_run_id is not None:
            teacher_mini_epoch = target_and_aux_calc_params.get("teacher_mini_epoch", -1)
            load_encoder_from_checkpoint(
                ema_model.ema_model, cf, teacher_run_id, teacher_mini_epoch, device
            )

    elif target_and_aux_calc == "FrozenTeacher":
        target_aux = FrozenTeacher.from_pretrained(cf, dataset, device, target_and_aux_calc_params)

    else:
        raise NotImplementedError(f"{target_and_aux_calc} is not implemented")

    return target_aux
