from weathergen.train.target_and_aux_module_base import TargetAndAuxModuleBase, IdentityTargetAndAux


class EMATeacher(TargetAndAuxModuleBase):
    def __init__(self):
        pass


# should be moved to its own file so as to prevent cyclical imports
def get_target_and_aux_calculator(config, model, rng, **kwargs):
    target_and_aux_calc = config.get("target_and_aux_calc", None)
    if target_and_aux_calc is None or target_and_aux_calc == "identity":
        return IdentityTargetAndAux(model, rng, config)
    elif target_and_aux_calc == "EMATeacher":
        return EMATeacher(model, rng, kwargs["ema_model"])
    else:
        raise NotImplemented(f"{target_and_aux_calc} is not implemented")
