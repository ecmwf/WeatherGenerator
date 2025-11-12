# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch


class Preprocessor:
    """
    A preprocessor that is called before the forecast engine. It can be used to add noise to the
    input. If not specified, it falls back to an identity, effectively realizing no preprocessing.
    """

    def __init__(self, preprocessor_type: str = "identity", **kwargs):
        if preprocessor_type == "identity":
            self.preprocess = self.identity
        elif preprocessor_type == "impute_normal_noise":
            if kwargs["training"]:
                self.preprocess = self.impute_normal_noise
                self.noise_std = kwargs["noise_std"]
            else:
                self.preprocess = self.identity
        elif preprocessor_type == "diffusion_model_noise_adder":
            self.preprocess = self.diffusion_model_noise_adder
            self.p_mean = kwargs["p_mean"]
            self.p_std = kwargs["p_std"]
        else:
            raise ValueError(f"Provided preprocessor_type `{preprocessor_type}` is not supported")

    def identity(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    def impute_normal_noise(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * torch.norm(x) * self.noise_std

    def diffusion_model_noise_adder(self, x: torch.Tensor) -> [torch.Tensor, float]:
        noise = torch.randn(x.shape, device=x.device)
        sigma = (noise * self.p_std + self.p_mean).exp()
        n = torch.randn_like(x) * sigma
        return x + n, sigma
