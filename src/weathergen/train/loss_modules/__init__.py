from .loss_module_latent import LossLatentDiffusion
from .loss_module_physical import LossPhysical, LossPhysicalTwo
from .loss_module_ssl import LossStudentTeacher

__all__ = [LossLatentDiffusion, LossPhysical, LossPhysicalTwo, LossStudentTeacher]
