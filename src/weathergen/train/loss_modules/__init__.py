# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from .loss_module_base import LossModuleBase, LossValues
from .loss_module_physical import LossPhysical

class LossLatentSSLStudentTeacher(LossModuleBase):
    def __init__(
        self,
        cf,
        loss_fcts: list,
        stage,
        device: str,
    ):
        LossModuleBase.__init__(self)
    # a placeholder to test configs
    def compute_loss(
        self,
        preds: dict,
        targets: dict,
    ) -> LossValues:
        return LossValues(loss=0, losses_all={}, stddev_all={})

__all__ = [LossPhysical, LossLatentSSLStudentTeacher]
