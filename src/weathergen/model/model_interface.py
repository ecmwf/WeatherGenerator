# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from weathergen.common.config import Config
from weathergen.model.model import Model

# same as in config: student_teacher, forecasting, masking
type TrainingMode = str
# model or target_aux
type CalculatorType = str

# TODO: how to name "model" or "calculator"?

def get_model(
    calculator_type : CalculatorType,
    cf: Config,
    dataset,
    **kwargs,
):
  """
    Create model

  """

    # TODO: how to avoid the dependence on dataset
    sources_size = dataset.get_sources_size()
    targets_num_channels = dataset.get_targets_num_channels()
    targets_coords_size = dataset.get_targets_coords_size()

    training_mode = cf.get( "training_mode", None)
    if training_mode == "student_teacher":
        model = Model(cf, sources_size, targets_num_channels, targets_coords_size).create()
    
    elif training_model == "masking" :
        pass  

    elif training_mode == "forecasting" :
        pass

    else:
        raise NotImplementedError(
            f"The training mode {cf['training_mode']} is not implemented."
        )

    if calculator_type = "target_aux" :
      
        # 
        target_and_aux_calc = cf.get("target_and_aux_calc", None)
        if target_and_aux_calc is None or target_and_aux_calc == "identity":
            model = PhysicalTargetAndAux(model, rng, config=cf)

        elif target_and_aux_calc == "EMATeacher":
            raise NotImplementedError(f"{target_and_aux_calc} is not implemented")

        else:
            raise NotImplementedError(f"{target_and_aux_calc} is not implemented")

    return model