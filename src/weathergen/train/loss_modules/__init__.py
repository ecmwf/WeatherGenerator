from .loss_module_latent import LossLatent
from .loss_module_physical import LossPhysical, LossPhysicalTwo
from .loss_module_ssl import LossStudentTeacher

__all__ = [LossLatent, LossPhysical, LossPhysicalTwo, LossStudentTeacher]
