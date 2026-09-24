# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# ----------------------------------------------------------------------------
# Flow matching / score matching forecast engine.
#
# Implements the Gaussian-probability-path framework from the MIT course
# "An Introduction to Flow Matching and Diffusion Models" (Holderrieth & Erives),
# chapters 3 (flow matching) and 4 (score matching). The engine operates in the
# encoder latent space, mirroring the data flow of DiffusionForecastEngine
# (diffusion.py) but WITHOUT importing from it: it is intentionally self-contained
# and structured around four swappable seams (Path / Parameterization /
# Preconditioner / Schedule) so an EDM formulation can be added later as one
# configuration rather than a rewrite.
#
# Conventions (course convention): a scalar t in [0, 1] indexes the Gaussian path,
# with t=0 -> pure noise N(0, I) (pinit) and t=1 -> data. A Gaussian path is
#     x_t = alpha_t * z + beta_t * eps,   eps ~ N(0, I).
# EDM is the special case alpha_t = 1, beta_t = sigma_t with a denoiser (x0)
# parameterization -- it drops into the same GaussianPath / Preconditioner seams.
# ----------------------------------------------------------------------------

import logging
import math

import torch

from weathergen.common.config import Config, get_path_run
from weathergen.datasets.batch import SampleMetaData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seam #1 (Path) + Seam #2 (Parameterization / conversions)
# ---------------------------------------------------------------------------
class GaussianPath:
    """Gaussian probability path ``x_s = alpha_s z + beta_s eps`` and the closed-form
    conversions between the velocity field, the score, the noise and the denoiser
    (course Prop. 1 / Remark 16).

    Schedulers return ``(alpha, beta, alpha_dot, beta_dot)`` as tensors broadcastable
    to the data, as functions of the path variable ``s``:

    - ``condot``: linear (conditional optimal transport) path, ``alpha=s, beta=1-s`` with
      the path *time* s=t in [0,1] ascending (t=0 noise, t=1 data).
    - ``ve`` (EDM / variance-exploding): ``alpha=1, beta=s`` with the *noise level* s=sigma
      itself as the path variable, descending sigma_max -> 0 during sampling. With a
      denoiser prediction, ``to_velocity`` reduces to ``(x - D)/sigma`` — exactly EDM's
      probability-flow ODE slope — so the generic sampler reproduces
      DiffusionForecastEngine._run_ode step for step.

    This is the single source of truth imported by the engine, the target encoder
    and the loss module so all three agree on the math.
    """

    VALID_PREDICTION_TYPES = ("velocity", "noise", "score", "denoiser")

    def __init__(self, kind: str = "condot"):
        self.kind = kind

    def coeffs(self, t: torch.Tensor):
        """Return ``(alpha, beta, alpha_dot, beta_dot)`` for scalar/broadcast path var ``t``."""
        if self.kind == "condot":
            alpha = t
            beta = 1.0 - t
            alpha_dot = torch.ones_like(t)
            beta_dot = -torch.ones_like(t)
            return alpha, beta, alpha_dot, beta_dot
        if self.kind == "ve":
            # EDM: x = z + sigma * eps, path variable is sigma itself.
            alpha = torch.ones_like(t)
            beta = t
            alpha_dot = torch.zeros_like(t)
            beta_dot = torch.ones_like(t)
            return alpha, beta, alpha_dot, beta_dot
        raise ValueError(f"Unknown Gaussian path kind: {self.kind!r}")

    # -- conditional training targets (given clean z, noise eps, time t) --------
    def conditional_target(
        self, z: torch.Tensor, eps: torch.Tensor, t: torch.Tensor, prediction_type: str
    ) -> torch.Tensor:
        """Analytic conditional regression target for the chosen parameterization.

        - velocity (CFM, Eq. 37): ``alpha_dot z + beta_dot eps``  (CondOT: ``z - eps``)
        - noise   (DDPM / DSM, Alg. 4):     ``eps``
        - score   (denoising score, Eq. 40): ``-eps / beta``
        - denoiser (EDM / x0-prediction):    ``z``  (what LossLatentDiffusion regresses)
        """
        alpha, beta, alpha_dot, beta_dot = self.coeffs(t)
        if prediction_type == "velocity":
            return alpha_dot * z + beta_dot * eps
        if prediction_type == "noise":
            return eps
        if prediction_type == "score":
            return -eps / beta
        if prediction_type == "denoiser":
            return z
        raise ValueError(f"Unknown prediction_type: {prediction_type!r}")

    # -- conversions from a raw network prediction to any quantity --------------
    # All quantities are linear reparameterizations of the (predicted) noise, so we
    # first recover the predicted noise eps_hat, then map it to whatever is needed.
    def _to_eps(
        self, pred: torch.Tensor, x: torch.Tensor, t: torch.Tensor, prediction_type: str
    ) -> torch.Tensor:
        alpha, beta, alpha_dot, beta_dot = self.coeffs(t)
        if prediction_type == "noise":
            return pred
        if prediction_type == "score":
            return -beta * pred
        if prediction_type == "denoiser":
            # pred is the clean-data estimate D; x = alpha*D + beta*eps => eps = (x - alpha*D)/beta
            return (x - alpha * pred) / beta
        if prediction_type == "velocity":
            # From u = alpha_dot z + beta_dot eps and x = alpha z + beta eps:
            #      z = (x - beta eps) / alpha
            #      u = alpha_dot (x - beta eps) / alpha + beta_dot eps
            #      u = (alpha_dot / alpha) x + (beta_dot - (alpha_dot beta / alpha)) eps
            #      eps = (alpha u - alpha_dot x) / (alpha beta_dot - alpha_dot beta) QED
            #   eps = (alpha u - alpha_dot x) / (alpha beta_dot - alpha_dot beta)
            denom = alpha * beta_dot - alpha_dot * beta
            return (alpha * pred - alpha_dot * x) / denom
        raise ValueError(f"Unknown prediction_type: {prediction_type!r}")

    def to_velocity(self, pred, x, t, prediction_type):
        if prediction_type == "velocity":
            return pred
        alpha, beta, alpha_dot, beta_dot = self.coeffs(t)
        eps = self._to_eps(pred, x, t, prediction_type)
        z = (x - beta * eps) / alpha
        return alpha_dot * z + beta_dot * eps

    def to_score(self, pred, x, t, prediction_type):
        if prediction_type == "score":
            return pred
        _, beta, _, _ = self.coeffs(t)
        eps = self._to_eps(pred, x, t, prediction_type)
        return -eps / beta

    def to_denoiser(self, pred, x, t, prediction_type):
        """Posterior-mean (x0 / clean-latent) estimate.

        For a velocity prediction we use the direct form (Remark 16 / Eq. 43)

            D_t(x) = (beta_t * u_t - beta_dot_t * x) / (alpha_dot_t * beta_t - alpha_t * beta_dot_t)

        whose denominator is the Wronskian-like term (= 1 for CondOT) and therefore involves **no
        division by alpha**: it stays well-conditioned all the way down to t -> 0. (Going via eps
        would divide by alpha=t and amplify float error ~1/t, even though the algebra cancels.)

        For noise/score predictions the 1/alpha is intrinsic — recovering x0 from an estimate of the
        noise genuinely requires dividing by alpha — so those remain ill-conditioned as t -> 0,
        which is expected: at the noise end a noise estimate carries almost no information about x0.
        """
        if prediction_type == "denoiser":
            return pred  # already the clean-data estimate
        alpha, beta, alpha_dot, beta_dot = self.coeffs(t)
        if prediction_type == "velocity":
            return (beta * pred - beta_dot * x) / (alpha_dot * beta - alpha * beta_dot)
        eps = self._to_eps(pred, x, t, prediction_type)
        return (x - beta * eps) / alpha


class FlowMatchingForecastEngine(torch.nn.Module):
    """Latent-space flow-matching / score-matching forecast engine.

    Training (``training_forward``): draws ``t`` (from ``noise_level_rn``) and noise
    ``eps``, forms ``x_t = alpha_t z + beta_t eps``, runs the (time-conditioned)
    forecasting backbone and returns the *raw* network prediction (velocity / noise /
    score per ``fe_flow_prediction_type``). It stashes ``eps`` and ``t`` into
    ``meta_info`` so the FlowMatchingTargetEncoder (which runs after the forward pass)
    can build the exact analytic regression target for LossFlowMatching.

    Inference (``inference_forward``): integrates from pure noise ``X_0 ~ N(0, I)`` to
    data with either the deterministic flow ODE (default) or the score-based SDE.
    Returns the same shapes as DiffusionForecastEngine so model.py is unchanged.
    """

    _FORECAST = "forecast"

    def __init__(self, cf: Config, num_healpix_cells: int, forecast_engine):
        super().__init__()
        self.cf = cf
        self.num_healpix_cells = num_healpix_cells
        self.net = forecast_engine

        # Time embedding (analogous to diffusion's NoiseEmbedder; self-contained).
        self.frequency_embedding_dim = cf.frequency_embedding_dim
        self.embedding_dim = cf.embedding_dim
        self.time_embedder = TimeEmbedder(
            embedding_dim=self.embedding_dim,
            frequency_embedding_dim=self.frequency_embedding_dim,
        )
        # Continuous t in (0,1) is scaled before the sinusoidal features so the
        # frequency band (max_period=1e4) is actually exercised (DiT/SiT convention).
        self.time_scale = cf.get("fm_time_scale", 1000.0)

        # Conditioning (reuse the diffusion conditioning keys so the shared backbone
        # is built consistently). Initial scope: "forecast" (previous latent state)
        # and unconditional. date_time conditioning can be ported later.
        self.conditioning = cf.get("fe_diffusion_model_conditioning", None)
        self.conditioning_type = cf.get("fe_diffusion_model_conditioning_type", None)
        assert self.conditioning in (None, self._FORECAST), (
            f"FlowMatchingForecastEngine currently supports conditioning in "
            f"{{None, 'forecast'}}, got {self.conditioning!r}"
        )

        # Optional expanded diffusion latent space (mirrors DiffusionForecastEngine).
        enc_dim = cf.ae_global_dim_embed
        lat_dim = cf.get("fe_diffusion_latent_dim", None) or enc_dim
        if lat_dim != enc_dim:
            self.latent_proj_up = torch.nn.Linear(enc_dim, lat_dim, bias=False)
            self.latent_proj_down = torch.nn.Linear(lat_dim, enc_dim, bias=False)
        else:
            self.latent_proj_up = None
            self.latent_proj_down = None
        if self.conditioning_type == "concatenate_hdMLP":
            self.concat_hd_proj = torch.nn.Linear(2 * lat_dim, lat_dim, bias=False)

        # -- Path (seam #1) + parameterization (seam #2) --
        self.path = GaussianPath(cf.get("fe_flow_path", "condot"))
        self.prediction_type = cf.get("fe_flow_prediction_type", "velocity")
        assert self.prediction_type in GaussianPath.VALID_PREDICTION_TYPES, (
            f"fe_flow_prediction_type must be one of {GaussianPath.VALID_PREDICTION_TYPES}"
        )

        # -- EDM / variance-exploding preconditioner + schedule params (seam #3/#4).
        # Same config keys as the diffusion configs so they stay portable. Inert for condot.
        self.sigma_data = cf.get("sigma_data", 1.0)
        self.sigma_min = cf.get("sigma_min", 0.002)
        self.sigma_max = cf.get("sigma_max", 80.0)
        self.rho = cf.get("rho", 7)
        self.p_mean = cf.get("p_mean", 1.5)
        self.p_std = cf.get("p_std", 1.2)
        self.sigma_min_quantile = cf.get("sigma_min_quantile", 0.05)
        # Opt-in scale on EDM's log(sigma)/4 embedder input (default 1.0 = spec-faithful).
        self.edm_noise_time_scale = cf.get("edm_noise_time_scale", 1.0)
        # EDM output preconditioning (ve path): default = original Karras c_skip/c_out. Set True to
        # force c_skip=0, c_out=1 (direct x0 prediction), reproducing diffusion.py's hardcoded case.
        self.no_skip_connection = cf.get("fe_diffusion_model_no_skip_connection", False)

        # -- Sampler (seam #4) --
        self.sampler = cf.get("fe_flow_sampler", "ode")
        self.t_eps = cf.get("fm_t_eps", 1e-3)
        self.num_steps_default = cf.get("fm_num_steps", 50)
        self.sde_sigma = cf.get("fm_sde_sigma", 0.0)
        if self.path.kind == "ve" and self.sampler == "sde" and self.sde_sigma > 0.0:
            logger.warning(
                "fe_flow_path=ve with the SDE sampler is mathematically valid but untested; "
                "EDM here reproduces the ODE (probability-flow) sampler."
            )

        # Validation uses a fixed t (set by the validation harness, mirroring diffusion).
        self._fixed_noise_level: float | None = None
        self.cur_token = None  # optional reference target for inference diagnostics

    # -----------------------------------------------------------------------
    # Routing (mirrors DiffusionForecastEngine.forward)
    # -----------------------------------------------------------------------
    def forward(
        self,
        tokens: torch.Tensor = None,
        fstep: int = None,
        meta_info: dict[str, SampleMetaData] = None,
        coords: torch.Tensor = None,
        num_steps: int = None,
    ):
        if self.training:
            if tokens is None or fstep is None or meta_info is None:
                raise ValueError("training_forward requires tokens, fstep, meta_info")
            return self.training_forward(tokens, fstep, meta_info, coords)

        # Eval mode: denoising analysis during train/val, generation during inference.
        if self.cf.stage in ("train", "train_continue"):
            return self.training_forward(tokens, fstep, meta_info, coords)
        elif self.cf.stage == "inference":
            if fstep is None:
                raise ValueError(f"inference requires fstep, got {fstep}")
            self.cur_token = tokens.detach() if tokens is not None else None
            return self.inference_forward(
                fstep=fstep,
                num_steps=num_steps or self.num_steps_default,
                meta_info=meta_info,
                coords=coords,
            )

    # -----------------------------------------------------------------------
    # Training: sample t and eps, form x_t, return raw prediction, stash eps/t
    # -----------------------------------------------------------------------
    def training_forward(
        self,
        tokens: torch.Tensor,
        fstep: int,
        meta_info: dict[str, SampleMetaData],
        coords: torch.Tensor = None,
    ) -> torch.Tensor:
        self.cur_token = tokens.detach()
        z = tokens  # clean latent target

        c = None
        #TODO: add date/time conditioning
        if self.conditioning == self._FORECAST:
            c = meta_info["ERA5"].params["conditioning_tokens"]

        if self.training:
            noise_level_rn = meta_info["ERA5"].params["noise_level_rn"]
        else:
            # Validation fixed level. Default when unset: condot -> t=0.5 (midpoint);
            # ve -> 0.0 read as log-sigma by the ve mapping => sigma=exp(0)=1 (diffusion.py:232).
            default_level = 0.0 if self.path.kind == "ve" else 0.5
            noise_level_rn = (
                self._fixed_noise_level if self._fixed_noise_level is not None else default_level
            )
        # t is kept in float32 (a scalar) so path coefficients are precise, matching how the
        # diffusion engine keeps sigma in float32; x_t then promotes to float32.
        t = self._t_from_noise_level(noise_level_rn, device=z.device)

        alpha, beta, _, _ = self.path.coeffs(t)
        eps = torch.randn_like(z)
        x_t = alpha * z + beta * eps

        # Stash eps, t and x_t so the loss (which runs after this forward pass) can build the
        # exact analytic target and the parameterization-invariant x0 diagnostic. Same channel
        # model.py uses for conditioning_tokens.
        meta_info["ERA5"].params["fm_eps"] = eps.detach()
        meta_info["ERA5"].params["fm_t"] = float(t.reshape(-1)[0].item())
        meta_info["ERA5"].params["fm_x_t"] = x_t.detach()

        return self._net_forward(x_t, t, c, fstep, coords)

    def _t_from_noise_level(self, noise_level_rn, device, dtype=torch.float32) -> torch.Tensor:
        """Map the per-sample ``noise_level_rn`` to the path variable ``s`` (a 1-element tensor):
        ``t`` for the condot path, ``sigma`` for the ve/EDM path.

        - **condot**: ``noise_distribution`` must be ``uniform`` — ``noise_level_rn`` already *is*
          ``t`` (drawn at the source in masking.py); clamp to ``[t_eps, 1-t_eps]``.
        - **ve (EDM)**: mirrors ``DiffusionForecastEngine.training_forward`` exactly —
          training + ``log_normal`` → ``sigma = exp(v*p_std + p_mean)``;
          training + ``log_uniform`` **or any eval/fixed level** → ``sigma = exp(v)``
          (diffusion.py:240-241 ``or not self.training``). No clamping (diffusion does not clamp).
        """
        dist = self.cf.get("noise_distribution", None)
        v = torch.as_tensor(noise_level_rn, device=device, dtype=dtype).reshape(1)

        if self.path.kind == "ve":
            assert dist in ("log_normal", "log_uniform"), (
                f"ve/EDM path expects noise_distribution log_normal|log_uniform, got {dist!r}"
            )
            if dist == "log_uniform" or not self.training:
                return v.exp()  # v is log-sigma (train log_uniform) or the fixed log-sigma (eval)
            return (v * self.p_std + self.p_mean).exp()  # v is eta ~ N(0,1)

        # condot: NB `==` not `is` — OmegaConf's string is not the interned literal.
        assert dist == "uniform", (
            f"condot path expects noise_distribution: uniform (t ~ U[0,1]), got {dist!r}"
        )
        return v.clamp(self.t_eps, 1.0 - self.t_eps)

    # -----------------------------------------------------------------------
    # Network call: preconditioning (seam #3) + conditioning handling + backbone
    # -----------------------------------------------------------------------
    def _net_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None,
        fstep: int,
        coords: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the (time-conditioned) forecasting backbone on ``x`` at path variable ``t``.

        Mirrors DiffusionForecastEngine.denoise. The preconditioner seam (``_c_in`` /
        ``_emb_input`` / ``_precondition_output``) is identity for condot and reproduces EDM's
        c_in / log(sigma)/4 / (c_skip=0, c_out=1) for the ve path. The conditioning-type branches
        match the shared backbone in engines.py.
        """
        time_emb = self.time_embedder(self._emb_input(t))
        net_input = self._c_in(t) * x

        # Project encoder tokens (and forecast conditioning) up into the diffusion latent space.
        if self.latent_proj_up is not None:
            net_input = self.latent_proj_up(net_input)
            if c is not None and self.conditioning_type not in {"ada_ln"}:
                c = self.latent_proj_up(c)

        if self.conditioning_type == "concatenate":
            combined = torch.cat([net_input, c], dim=1)
            crds = torch.cat([coords, coords], dim=1) if coords is not None else None
            raw = self.net(
                combined, fstep=fstep, coords=crds, noise_emb=time_emb, conditioning=None
            )
            raw = raw[:, : x.shape[1], :]
        elif self.conditioning_type == "concatenate_hiddendim":
            combined = torch.cat([net_input, c], dim=2)
            raw = self.net(
                combined, fstep=fstep, coords=coords, noise_emb=time_emb, conditioning=None
            )
        elif self.conditioning_type == "concatenate_hdMLP":
            combined = self.concat_hd_proj(torch.cat([net_input, c], dim=2))
            raw = self.net(
                combined, fstep=fstep, coords=coords, noise_emb=time_emb, conditioning=None
            )
        else:
            raw = self.net(
                net_input, fstep=fstep, coords=coords, noise_emb=time_emb, conditioning=c
            )

        if self.latent_proj_down is not None:
            raw = self.latent_proj_down(raw)
        return self._precondition_output(raw, x, t)

    # -----------------------------------------------------------------------
    # Preconditioner seam (#3). Dispatched on self.path.kind. For condot this is the
    # identity flow-matching path; for ve it reproduces DiffusionForecastEngine.denoise
    # (c_in = 1/sqrt(sigma^2+sigma_data^2), emb input log(sigma)/4) with the original EDM
    # c_skip/c_out output preconditioning by default (fe_diffusion_model_no_skip_connection
    # restores diffusion.py's hardcoded c_skip=0, c_out=1). Sampler/loss are untouched.
    # -----------------------------------------------------------------------
    def _c_in(self, s: torch.Tensor) -> torch.Tensor:
        """Input scaling applied before the network sees x."""
        if self.path.kind == "ve":
            return 1.0 / (s**2 + self.sigma_data**2).sqrt()
        return self.cf.get("fm_c_in", 1.0)

    def _emb_input(self, s: torch.Tensor) -> torch.Tensor:
        """Scalar fed to the (sinusoidal) noise/time embedder, as a 1-element tensor."""
        if self.path.kind == "ve":
            # EDM's c_noise = log(sigma)/4, times an opt-in scale (default 1.0 = spec-faithful;
            # see edm_noise_time_scale and the diffusion-branch noise-embedding A/B).
            return (s.reshape(1).log() / 4.0) * self.edm_noise_time_scale
            #TODO: Check why we are taking log/deviding by 4
        # condot: t in (0,1) scaled up into the frequency band the ladder was calibrated for.
        return s.reshape(1) * self.time_scale

    def _precondition_output(
        self, raw: torch.Tensor, x: torch.Tensor, s: torch.Tensor
    ) -> torch.Tensor:
        """Output preconditioning: the returned denoiser estimate ``D = c_skip*x + c_out*raw``.

        For the **ve/EDM** path this is the original Karras et al. (2022) preconditioning

            c_skip(sigma) = sigma_data^2 / (sigma^2 + sigma_data^2)
            c_out(sigma)  = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)

        which, together with the ``lambda(sigma)`` loss weight, makes the network's *effective*
        regression target unit-variance at every sigma (``lambda * c_out^2 = 1``) — the property
        diffusion.py forgoes by hardcoding c_skip=0, c_out=1. Set
        ``fe_diffusion_model_no_skip_connection: True`` to restore that direct-x0 prediction
        (c_skip=0, c_out=1), reproducing diffusion.py exactly.

        The returned value is the denoiser ``D`` either way, so the loss (regress ``D`` vs ``z``)
        and the sampler (``to_velocity(D) = (x-D)/sigma``) are unchanged. condot has no EDM
        preconditioning — it regresses the raw velocity/noise/score field directly (identity).
        """
        if self.path.kind == "ve" and not self.no_skip_connection:
            c_skip = self.sigma_data**2 / (s**2 + self.sigma_data**2)
            c_out = s * self.sigma_data / (s**2 + self.sigma_data**2).sqrt()
            return c_skip * x + c_out * raw
        # Identity (c_skip=0, c_out=1): condot always; ve when no_skip_connection is set.
        return raw

    # -----------------------------------------------------------------------
    # Inference: integrate noise -> data (ODE default, SDE optional)
    # -----------------------------------------------------------------------
    def inference_forward(
        self,
        fstep: int,
        num_steps: int = 50,
        meta_info: dict[str, SampleMetaData] = None,
        coords: torch.Tensor = None,
    ):
        c = None
        if self.conditioning == self._FORECAST:
            c = meta_info["ERA5"].params["conditioning_tokens"]

        num_members: int = self.cf.get("fe_flow_num_ensemble_members", 1)
        if num_members > 1:
            logger.info(f"Flow-matching ensemble mode: generating {num_members} members.")
            c_batched = c.expand(num_members, *c.shape[1:]) if c is not None else None
            final_x, _ = self._run_sampler(
                c=c_batched,
                fstep=fstep,
                num_steps=num_steps,
                coords=coords,
                batch_size=num_members,
                return_trajectory=False,
            )
            return final_x

        _, trajectory = self._run_sampler(
            c=c,
            fstep=fstep,
            num_steps=num_steps,
            coords=coords,
            return_trajectory=True,
        )
        return trajectory

    def _sampling_nodes(self, num_steps: int, device) -> torch.Tensor:
        """The ``num_steps + 1`` path-variable nodes ``s`` the sampler integrates over.

        - **condot**: ascending ``t`` linspace in ``[t_eps, 1 - t_eps]``.
        - **ve (EDM)**: descending, rho-spaced ``sigma`` between *training-aligned* bounds plus a
          terminal 0 — a faithful transcription of ``DiffusionForecastEngine._run_ode``
          (diffusion.py:442-476), so the generic sampler reproduces EDM's schedule exactly.
        """
        if self.path.kind != "ve":
            return torch.linspace(
                self.t_eps, 1.0 - self.t_eps, num_steps + 1, dtype=torch.float32, device=device
            )

        # Training-aligned sigma bounds (diffusion.py:442-457).
        sigma_max_train = math.exp(self.p_mean + 3.0 * self.p_std)
        sigma_max_eff = min(self.sigma_max, sigma_max_train)
        _z_scores = {0.01: -2.326, 0.025: -1.960, 0.05: -1.645, 0.10: -1.282}
        _z = _z_scores.get(self.sigma_min_quantile, -1.645)
        sigma_min_from_dist = math.exp(self.p_mean + _z * self.p_std)
        sigma_min_eff = max(self.sigma_min, sigma_min_from_dist, self.sigma_data * 0.01)
        logger.info(
            f"EDM sigma schedule: sigma_max_eff={sigma_max_eff:.4f} "
            f"(config={self.sigma_max}, train 3sigma={sigma_max_train:.4f}), "
            f"sigma_min_eff={sigma_min_eff:.4f} (config={self.sigma_min}, "
            f"dist q={self.sigma_min_quantile:.3f}/{sigma_min_from_dist:.4f}), "
            f"sigma_data={self.sigma_data}, rho={self.rho}, num_steps={num_steps}"
        )
        # rho-spacing (EDM Eq. 5) in float64, terminal sigma=0; cast per-step in the loop.
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        nodes = (
            sigma_max_eff ** (1 / self.rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min_eff ** (1 / self.rho) - sigma_max_eff ** (1 / self.rho))
        ) ** self.rho
        return torch.cat([nodes, torch.zeros_like(nodes[:1])])  # sigma_N = 0

    def _run_sampler(
        self,
        c: torch.Tensor | None,
        fstep: int,
        num_steps: int,
        coords: torch.Tensor | None,
        batch_size: int = 1,
        return_trajectory: bool = False,
    ):
        """Integrate the path from noise to data over the nodes from ``_sampling_nodes``.

        ODE (default): ``dX/ds = u_s(X)`` with Euler + Heun 2nd-order correction (Alg. 1). For the
        ve path with a denoiser this is exactly EDM's probability-flow ODE (``u = (x-D)/sigma``).
        SDE: Euler-Maruyama ``dX = [u + (sigma^2/2) score] ds + sigma dW`` (Thm. 17; condot only).
        """
        device = "cuda"
        dim = self.cf.ae_global_dim_embed
        t_steps = self._sampling_nodes(num_steps, device)

        # Initial noise scaled to the start-of-path marginal std. VE: X_0 ~ N(0, sigma_max^2)
        # (diffusion.py x*t_steps[0]); condot: pinit = N(0, I), std 1 (unchanged behaviour).
        x = torch.randn(batch_size, self.num_healpix_cells, dim, device=device)
        if self.path.kind == "ve":
            _, beta0, _, _ = self.path.coeffs(t_steps[0].float())
            x = x * beta0

        use_sde = self.sampler == "sde" and self.sde_sigma > 0.0
        logger.info(
            f"Flow-matching inference: sampler={self.sampler} "
            f"(sde_sigma={self.sde_sigma}), prediction_type={self.prediction_type}, "
            f"path={self.path.kind}, steps={num_steps}, "
            f"s: {t_steps[0].item():.4f} -> {t_steps[-2].item():.4f} -> {t_steps[-1].item():.4f}"
        )

        # Per-step diagnostics. At every t we compare the *current noisy state* x_t against the
        # *denoised estimate* x0_hat = D_t(x_t) predicted from it, in RMSE-to-target and in std.
        # RMSE (not raw L2 norm) so the magnitude is independent of latent size and directly
        # comparable to sigma_data / to std(z).
        track = {
            "t": [],
            "x_t_std": [],
            "x0_hat_std": [],
            "rmse_x_t": [],
            "rmse_x0_hat": [],
            "vel_norm": [],
        }
        trajectory: list[torch.Tensor] = [] if return_trajectory else None

        x_next = x
        for i in range(num_steps):
            # Cast per-step to float32 (nodes may be float64 for ve; diffusion.py does the same).
            # For ve, h = sigma_next - sigma_cur is NEGATIVE (descending) — Euler/Heun are
            # sign-agnostic and the SDE branch already uses h.abs().sqrt().
            t_cur = t_steps[i].float()
            t_next = t_steps[i + 1].float()
            h = t_next - t_cur
            x_cur = x_next

            raw = self._net_forward(x_cur, t_cur, c, fstep, coords)
            u = self.path.to_velocity(raw, x_cur, t_cur, self.prediction_type)

            if use_sde:
                score = self.path.to_score(raw, x_cur, t_cur, self.prediction_type)
                drift = u + 0.5 * self.sde_sigma**2 * score
                noise = self.sde_sigma * h.abs().sqrt() * torch.randn_like(x_cur)
                x_next = x_cur + h * drift + noise
            else:
                x_next = x_cur + h * u
                # Heun 2nd-order correction (skip on the last step).
                if i < num_steps - 1:
                    raw2 = self._net_forward(x_next, t_next, c, fstep, coords)
                    u2 = self.path.to_velocity(raw2, x_next, t_next, self.prediction_type)
                    x_next = x_cur + h * 0.5 * (u + u2)

            with torch.no_grad():
                # Denoised (clean-latent) estimate predicted from x_t at this t.
                x0_hat = self.path.to_denoiser(raw, x_cur, t_cur, self.prediction_type)
                track["t"].append(t_cur.item())
                track["x_t_std"].append(x_cur.std().item())
                track["x0_hat_std"].append(x0_hat.std().item())
                track["vel_norm"].append(u.norm().item())
                if self.cur_token is not None:
                    # RMSE = ||.|| / sqrt(numel), i.e. per-element error. Size-independent, so it
                    # is comparable across latent shapes and against sigma_data.
                    _rn = self.cur_token.numel() ** 0.5
                    track["rmse_x_t"].append((x_cur - self.cur_token).norm().item() / _rn)
                    track["rmse_x0_hat"].append((x0_hat - self.cur_token).norm().item() / _rn)

            if return_trajectory:
                trajectory.append(x_next)

        # Record the FINAL state x_next at the terminal node t_steps[num_steps]. The loop above only
        # tracks x_cur (the state *before* each update), so without this the actual returned sample
        # is never plotted and x_t appears to stall above x0_hat. For the ve path the last step is a
        # plain Euler step with h = -sigma_last and u = (x - D)/sigma, so x_next == D exactly: the
        # endpoint lands ON the x0_hat curve. x0_hat/velocity are not defined at the terminal node
        # (it would need another net forward, degenerate at sigma=0), so they get NaN and matplotlib
        # simply omits that marker.
        with torch.no_grad():
            track["t"].append(t_steps[num_steps].float().item())
            track["x_t_std"].append(x_next.std().item())
            track["x0_hat_std"].append(float("nan"))
            track["vel_norm"].append(float("nan"))
            if self.cur_token is not None:
                _rn = self.cur_token.numel() ** 0.5
                track["rmse_x_t"].append((x_next - self.cur_token).norm().item() / _rn)
                track["rmse_x0_hat"].append(float("nan"))

        #TODO: Make this optinal with log_diganostics flag
        self._plot_sampling_diagnostics(track, num_steps)
        return x_next, trajectory

    def _plot_sampling_diagnostics(self, track: dict, num_steps: int) -> None:
        """Diagnostic plot of the sampling trajectory.

        At every path time t we compare the current *noisy state* x_t against the *denoised
        estimate* x0_hat = D_t(x_t) predicted from it:
          1. RMSE to the target:  rmse(x_t, z)  vs  rmse(x0_hat, z)
          2. Standard deviation:  std(x_t)      vs  std(x0_hat)
        Panel 1 requires a reference target (``self.cur_token``) and is OMITTED when absent — which
        is the case in rollout mode (``forecast.num_steps>1``), where model.py sets tokens=None
        after the first step. Run with ``forecast.num_steps=1`` to get it.
        x0_hat should approach the target (and the target's std) much earlier than x_t does — x_t
        only gets there at t=1, whereas x0_hat is the model's best guess at the clean latent at
        every t.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Drop the first step(s) from the PLOT (the data is still tracked). At t ~ t_eps the
        # denoiser carries a 1/alpha factor for the noise/score parameterisations, so x0_hat and
        # the ODE drift blow up by ~1/t_eps there; left in, that single point dominates the log
        # y-scale and flattens everything else. Set fm_diag_skip_steps=0 to keep it.
        skip = int(self.cf.get("fm_diag_skip_steps", 1))
        ts_all = track["t"]
        skip = min(skip, max(len(ts_all) - 2, 0))  # never leave fewer than 2 points
        sl = slice(skip, None)

        ts = ts_all[sl]
        steps = list(range(skip, len(ts_all)))
        has_target = len(track["rmse_x_t"]) > 0
        n = 3 if has_target else 2
        fig, axes = plt.subplots(n, 1, figsize=(10, 3.2 * n), sharex=True)
        i = 0

        skipped = f"  [first {skip} step(s) omitted]" if skip else ""

        # 1) RMSE to target: noisy state vs denoised estimate
        if has_target:
            axes[i].semilogy(steps, track["rmse_x_t"][sl], "o-", ms=3, color="tab:blue",
                             label=r"rmse($x_t$, $z$)  (noisy state)")
            axes[i].semilogy(steps, track["rmse_x0_hat"][sl], "s-", ms=3, color="tab:red",
                             label=r"rmse($\hat{x}_0(x_t)$, $z$)  (denoised estimate)")
            axes[i].set_ylabel("RMSE to target")
            axes[i].set_title(
                f"Flow-matching sampling  |  path={self.path.kind}, "
                f"pred={self.prediction_type}, sampler={self.sampler}, steps={num_steps}{skipped}"
            )
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3, which="both")
            i += 1

        # 2) std: noisy state vs denoised estimate
        axes[i].plot(steps, track["x_t_std"][sl], "o-", ms=3, color="tab:blue",
                     label=r"std($x_t$)")
        axes[i].plot(steps, track["x0_hat_std"][sl], "s-", ms=3, color="tab:red",
                     label=r"std($\hat{x}_0(x_t)$)")
        if self.cur_token is not None:
            axes[i].axhline(self.cur_token.std().item(), color="grey", ls="--", lw=0.8,
                            label="target std")
        axes[i].set_ylabel("std")
        axes[i].set_yscale("log")
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3, which="both")
        i += 1

        # 3) velocity magnitude
        axes[i].semilogy(steps, track["vel_norm"][sl], "^-", ms=3, color="tab:orange")
        axes[i].set_ylabel("||velocity||")
        axes[i].grid(True, alpha=0.3, which="both")

        s_name = "sigma" if self.path.kind == "ve" else "t"
        axes[-1].set_xlabel(f"sampling step   ({s_name}: {ts[0]:.3f} -> {ts[-1]:.3f}){skipped}")
        fig.tight_layout()

        out_dir = get_path_run(self.cf) / "plots" / "validation" / "plots"
        out_dir.mkdir(exist_ok=True, parents=True)
        out_path = out_dir / "flow_sampling_diagnostics.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved flow sampling diagnostics to {out_path}")


class TimeEmbedder(torch.nn.Module):
    """Sinusoidal embedding of a scalar time ``t`` followed by an MLP.

    Self-contained equivalent of diffusion.py's NoiseEmbedder (feeds ``t`` rather than
    ``log(sigma)/4``); kept here to avoid coupling to the EDM engine.
    """

    def __init__(self, embedding_dim: int, frequency_embedding_dim: int, dtype=torch.bfloat16):
        super().__init__()
        self.dtype = dtype
        self.frequency_embedding_dim = frequency_embedding_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(frequency_embedding_dim, embedding_dim, bias=True),
            torch.nn.SiLU(),
            torch.nn.Linear(embedding_dim, embedding_dim, bias=True),
        )

    def timestep_embedding(self, t: torch.Tensor, max_period: int = 10000) -> torch.Tensor:
        if t.ndim == 0:
            t = t.view(1)
        half = self.frequency_embedding_dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=self.dtype) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t))
