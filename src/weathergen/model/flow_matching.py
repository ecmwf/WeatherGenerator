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
# encoder latent space and shares the conditioning, latent projections, registers,
# noise embedder and inference/decoding plumbing of DiffusionForecastEngineBase
# (diffusion.py) with its sibling EDMForecastEngine (edm.py). It is structured around
# four swappable seams
# (Path / Parameterization / Preconditioner / Schedule) so that the EDM formulation
# is one configuration rather than a rewrite.
#
# Conventions (course convention): a scalar t in [0, 1] indexes the Gaussian path,
# with t=0 -> pure noise N(0, I) (pinit) and t=1 -> data. A Gaussian path is
#     x_t = alpha_t * z + beta_t * eps,   eps ~ N(0, I).
# EDM is the special case alpha_t = 1, beta_t = sigma_t with a denoiser (x0)
# parameterization -- it drops into the same GaussianPath / Preconditioner seams.
#
# Configuration (flow-matching specific options, the shared ones are documented in
# diffusion.py):
#   diffusion:
#     type: flow_matching
#     flow_matching:
#       path: condot | ve                                     (seam #1)
#       prediction_type: velocity | noise | score | denoiser  (seam #2)
#       t_eps: 1.0e-3                clamp t to [t_eps, 1-t_eps] (condot)
#       c_in: 1.0                    input preconditioning (condot)
#       sigma_data: 1.0              (ve)
#       edm_preconditioning: True    EDM c_skip/c_out output preconditioning (ve + denoiser)
#       edm_noise_time_scale: 1.0    scale on EDM's log(sigma)/4 embedder input (ve)
#       sampler:                     (seam #4)
#         type: ode | sde
#         sde_sigma: 0.0             SDE diffusion coefficient (0 -> ODE)
#         rho, sigma_min, sigma_max, sigma_min_quantile   EDM sigma schedule (ve)
# The noise level drawn by the Masker (masking_strategy_config.noise_level) is the path
# variable: t ~ Uniform[t_min, t_max] (distribution: uniform) for condot, sigma
# (log_normal / log_uniform) for ve.
# ----------------------------------------------------------------------------

import logging

import torch
import tqdm

from weathergen.common.config import Config
from weathergen.datasets.masking import get_noise_level_cfg
from weathergen.model.diffusion import (
    DiffusionForecastEngineBase,
    get_inference_sigma_range,
    rho_sigma_steps,
)
from weathergen.model.engines import ForecastingEngine

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
      EDMForecastEngine._run_sampler step for step.

    This is the single source of truth imported by the engine and the loss module so both
    agree on the math.
    """

    VALID_KINDS = ("condot", "ve")
    VALID_PREDICTION_TYPES = ("velocity", "noise", "score", "denoiser")

    def __init__(self, kind: str = "condot", t_eps: float = 1e-3):
        assert kind in self.VALID_KINDS, f"Unknown Gaussian path kind: {kind!r}"
        self.kind = kind
        self.t_eps = t_eps

    @classmethod
    def from_config(cls, cf: Config) -> "GaussianPath":
        fm_cf = cf.diffusion.flow_matching
        return cls(fm_cf.get("path", "condot"), fm_cf.get("t_eps", 1e-3))

    def path_variable(self, noise_level, device) -> torch.Tensor:
        """Map the per-sample noise level drawn by the Masker to the path variable ``s``,
        shape (B, 1, 1) to broadcast over cells and channels: ``t`` for the condot path,
        clamped to ``[t_eps, 1-t_eps]``, and ``sigma`` itself for the ve/EDM path (no
        clamping, as in diffusion). Kept in float32 so path coefficients are precise, matching
        how the diffusion engine keeps sigma in float32.
        """
        s = torch.as_tensor(noise_level, dtype=torch.float32, device=device).reshape(-1, 1, 1)
        if self.kind == "ve":
            return s
        return s.clamp(self.t_eps, 1.0 - self.t_eps)

    def coeffs(self, t: torch.Tensor):
        """Return ``(alpha, beta, alpha_dot, beta_dot)`` for scalar/broadcast path var ``t``."""
        if self.kind == "condot":
            alpha = t
            beta = 1.0 - t
            alpha_dot = torch.ones_like(t)
            beta_dot = -torch.ones_like(t)
            return alpha, beta, alpha_dot, beta_dot
        # ve / EDM: x = z + sigma * eps, path variable is sigma itself.
        alpha = torch.ones_like(t)
        beta = t
        alpha_dot = torch.zeros_like(t)
        beta_dot = torch.ones_like(t)
        return alpha, beta, alpha_dot, beta_dot

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

    def denoiser_to_target_scale(self, t: torch.Tensor, prediction_type: str) -> torch.Tensor:
        """Factor ``a_t`` with ``pred - conditional_target = a_t * (D(pred) - z)`` at fixed x_t.

        Every parameterization is affine in the clean-latent estimate at fixed ``x_t``
        (``pred = a_t D + b_t x_t``), so a regression in parameterization space equals one in
        x0 space with the per-sample factor ``a_t``. This lets the engine always return the
        clean-latent estimate while LossFlowMatching still regresses in the configured space.

        - velocity: ``alpha_dot - beta_dot alpha / beta``  (CondOT: ``1 / (1 - t)``)
        - noise:    ``-alpha / beta``
        - score:    ``alpha / beta^2``
        - denoiser: ``1``
        """
        alpha, beta, alpha_dot, beta_dot = self.coeffs(t)
        if prediction_type == "velocity":
            return alpha_dot - beta_dot * alpha / beta
        if prediction_type == "noise":
            return -alpha / beta
        if prediction_type == "score":
            return alpha / beta**2
        if prediction_type == "denoiser":
            return torch.ones_like(t)
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


class FlowMatchingForecastEngine(DiffusionForecastEngineBase):
    """Latent-space flow-matching / score-matching forecast engine.

    Training (``training_forward``): reads ``t`` (the per-sample ``noise_level`` drawn by the
    Masker), draws noise ``eps``, forms ``x_t = alpha_t z + beta_t eps``, runs the
    (time-conditioned) forecasting backbone, which predicts velocity / noise / score / denoiser
    per ``diffusion.flow_matching.prediction_type``, and returns the corresponding clean-latent
    estimate ``D_t(x_t)``. Returning a latent state (rather than the raw velocity etc.) keeps
    the output meaningful for the decoders and the latent target; LossFlowMatching maps it back
    to the parameterization space (see ``GaussianPath.denoiser_to_target_scale``), which yields
    exactly the conditional regression target without passing ``eps`` to the loss.

    Inference (``_run_sampler``): integrates from pure noise ``X_0 ~ N(0, I)`` to data with either
    the deterministic flow ODE (default) or the score-based SDE. Returns the same shapes as
    EDMForecastEngine, so model.py and the inference/decoding plumbing are unchanged.
    """

    def __init__(self, cf: Config, num_healpix_cells: int, forecast_engine: ForecastingEngine):
        super().__init__(cf, num_healpix_cells, forecast_engine)
        fm_cf = self.diff_cf.flow_matching

        # -- Path (seam #1) + parameterization (seam #2) --
        self.path = GaussianPath.from_config(cf)
        self.prediction_type = fm_cf.get("prediction_type", "velocity")
        assert self.prediction_type in GaussianPath.VALID_PREDICTION_TYPES, (
            f"diffusion.flow_matching.prediction_type must be one of "
            f"{GaussianPath.VALID_PREDICTION_TYPES}"
        )
        noise_level_cfg = get_noise_level_cfg(cf.get("training_config", {}))
        if noise_level_cfg is not None:
            dist = noise_level_cfg.get("distribution", "log_normal")
            expected = ("log_normal", "log_uniform") if self.path.kind == "ve" else ("uniform",)
            assert dist in expected, (
                f"flow-matching path {self.path.kind!r} expects noise_level distribution in "
                f"{expected}, got {dist!r}"
            )

        # -- Preconditioner (seam #3). condot: input scaling c_in, identity output. ve: EDM's
        # c_in / log(sigma)/4, with the original Karras c_skip/c_out output preconditioning by
        # default for the denoiser (edm_preconditioning: False -> direct x0 prediction, as in
        # EDMForecastEngine's default).
        self.fm_c_in = fm_cf.get("c_in", 1.0)
        self.sigma_data = fm_cf.get("sigma_data", 1.0)
        self.edm_preconditioning = fm_cf.get(
            "edm_preconditioning", self.prediction_type == "denoiser"
        )
        # Opt-in scale on EDM's log(sigma)/4 embedder input (default 1.0 = spec-faithful).
        self.edm_noise_time_scale = fm_cf.get("edm_noise_time_scale", 1.0)
        assert (
            not (self.path.kind == "ve" and self.edm_preconditioning)
            or self.prediction_type == "denoiser"
        ), "flow_matching.edm_preconditioning with the ve path requires prediction_type 'denoiser'"

        # -- Sampler (seam #4) --
        sampler_cf = fm_cf.get("sampler", {})
        self.sampler = sampler_cf.get("type", "ode")
        assert self.sampler in ("ode", "sde"), f"Unknown flow-matching sampler {self.sampler!r}"
        self.sde_sigma = sampler_cf.get("sde_sigma", 0.0)
        if self.path.kind == "ve" and self.sampler == "sde" and self.sde_sigma > 0.0:
            logger.warning(
                "flow_matching.path=ve with the SDE sampler is mathematically valid but untested; "
                "EDM here reproduces the ODE (probability-flow) sampler."
            )
        # The sigma schedule only applies to the ve path; condot integrates t over
        # [t_eps, 1 - t_eps] (see _sampling_nodes).
        if self.path.kind == "ve":
            self.rho = sampler_cf.get("rho", 7.0)
            sigma_range = get_inference_sigma_range(
                cf, sampler_cf, self.sigma_data, "diffusion.flow_matching.sampler"
            )
            self.sigma_min, self.sigma_max = sigma_range.sigma_min, sigma_range.sigma_max

    # -----------------------------------------------------------------------
    # Training: sample eps, form x_t, return the clean-latent estimate
    # -----------------------------------------------------------------------
    def training_forward(
        self,
        tokens: torch.Tensor,
        fstep: int,
        meta_info: dict,
        coords: torch.Tensor = None,
    ) -> torch.Tensor:
        c = self._get_conditioning(meta_info)

        assert "noise_level" in meta_info, (
            "noise_level (sample meta_info, drawn by the Masker from "
            "masking_strategy_config.noise_level) is required for training"
        )
        t = self.path.path_variable(meta_info["noise_level"], device=tokens.device)
        assert t.shape[0] == tokens.shape[0], (
            f"got {t.shape[0]} noise levels for {tokens.shape[0]} samples"
        )

        # Clean latent target; only the healpix-cell tokens are noised.
        z = tokens[:, self.num_aux_tokens :]
        alpha, beta, _, _ = self.path.coeffs(t)
        eps = torch.randn_like(z)
        x_t = alpha * z + beta * eps

        pred = self._predict(x_t, t, c, fstep, coords)
        x0_hat = self.path.to_denoiser(pred, x_t, t, self.prediction_type)
        return self._prepend_aux_tokens(x0_hat)

    # -----------------------------------------------------------------------
    # Network call: preconditioning (seam #3) + shared conditioned backbone
    # -----------------------------------------------------------------------
    def _predict(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None,
        fstep: int,
        coords: torch.Tensor | None,
    ) -> torch.Tensor:
        """Prediction in the configured parameterization for ``x`` at path variable ``t``.

        Mirrors EDMForecastEngine.denoise. The preconditioner seam (``_c_in`` /
        ``_emb_input`` / ``_precondition_output``) is identity for condot and reproduces EDM's
        c_in / log(sigma)/4 / (c_skip, c_out) for the ve path. Conditioning, latent projections
        and registers are handled by the shared DiffusionForecastEngineBase._net_forward.
        """
        time_emb = self.noise_embedder(self._emb_input(t))
        raw = self._net_forward(
            self._c_in(t) * x, c=c, noise_emb=time_emb, fstep=fstep, coords=coords
        )
        return self._precondition_output(raw, x, t)

    # -----------------------------------------------------------------------
    # Preconditioner seam (#3). Dispatched on self.path.kind. For condot this is the
    # identity flow-matching path; for ve it reproduces EDMForecastEngine.denoise
    # (c_in = 1/sqrt(sigma^2+sigma_data^2), emb input log(sigma)/4, c_skip/c_out per
    # flow_matching.edm_preconditioning). Sampler/loss are untouched.
    # -----------------------------------------------------------------------
    def _c_in(self, s: torch.Tensor) -> torch.Tensor | float:
        """Input scaling applied before the network sees x."""
        if self.path.kind == "ve":
            return 1.0 / (s**2 + self.sigma_data**2).sqrt()
        return self.fm_c_in

    def _emb_input(self, s: torch.Tensor) -> torch.Tensor:
        """Scalar fed to the (sinusoidal) noise/time embedder, one per sample: (B,)."""
        if self.path.kind == "ve":
            # EDM's c_noise = log(sigma)/4, times an opt-in scale (default 1.0 = spec-faithful;
            # see edm_noise_time_scale and the diffusion-branch noise-embedding A/B).
            return (s.reshape(-1).log() / 4.0) * self.edm_noise_time_scale
        # condot: t in (0,1), embedded raw. The log_fourier band is stated directly as
        # [f_min, f_max] (0.5..8*pi by default, ~4 cycles across the range), so unlike the DiT
        # ladder it needs no pre-scaling -- the frequencies are fitted to the scalar's own
        # range, exactly as the diffusion engine does for c_noise = ln(sigma)/4.
        return s.reshape(-1)

    def _precondition_output(
        self, raw: torch.Tensor, x: torch.Tensor, s: torch.Tensor
    ) -> torch.Tensor:
        """Output preconditioning.

        For the **ve/EDM** path with ``flow_matching.edm_preconditioning`` this is the original
        Karras et al. (2022) preconditioning ``D = c_skip*x + c_out*raw`` with

            c_skip(sigma) = sigma_data^2 / (sigma^2 + sigma_data^2)
            c_out(sigma)  = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)

        which, together with the ``lambda(sigma)`` loss weight, makes the network's *effective*
        regression target unit-variance at every sigma (``lambda * c_out^2 = 1``). Without it,
        c_skip=0, c_out=1 (direct x0 prediction), reproducing EDMForecastEngine's default.

        condot has no EDM preconditioning — it regresses the raw velocity/noise/score field
        directly (identity).
        """
        if self.path.kind == "ve" and self.edm_preconditioning:
            c_skip = self.sigma_data**2 / (s**2 + self.sigma_data**2)
            c_out = s * self.sigma_data / (s**2 + self.sigma_data**2).sqrt()
            return c_skip * x + c_out * raw
        return raw

    # -----------------------------------------------------------------------
    # Inference: integrate noise -> data (ODE default, SDE optional)
    # -----------------------------------------------------------------------
    def _sampling_nodes(self, num_steps: int, device) -> torch.Tensor:
        """The ``num_steps + 1`` path-variable nodes ``s`` the sampler integrates over.

        - **condot**: ascending ``t`` linspace in ``[t_eps, 1 - t_eps]``.
        - **ve (EDM)**: descending, rho-spaced ``sigma`` between *training-aligned* bounds plus a
          terminal 0 — EDMForecastEngine's schedule (see get_inference_sigma_range), so
          the generic sampler reproduces EDM's schedule exactly for both ``log_normal`` and
          ``log_uniform`` training noise.
        """
        if self.path.kind == "ve":
            return rho_sigma_steps(self.sigma_min, self.sigma_max, self.rho, num_steps, device)
        return torch.linspace(
            self.path.t_eps,
            1.0 - self.path.t_eps,
            num_steps + 1,
            dtype=torch.float32,
            device=device,
        )

    def _run_sampler(
        self,
        c: torch.Tensor | None,
        fstep: int,
        num_steps: int,
        coords: torch.Tensor | None,
        batch_size: int = 1,
        return_diffusion_steps: bool = False,
    ) -> "tuple[torch.Tensor, list[torch.Tensor] | None]":
        """Integrate the path from noise to data over the nodes from ``_sampling_nodes``.

        ODE (default): ``dX/ds = u_s(X)`` with Euler + Heun 2nd-order correction (Alg. 1). For the
        ve path with a denoiser this is exactly EDM's probability-flow ODE (``u = (x-D)/sigma``).
        SDE: Euler-Maruyama ``dX = [u + (sigma^2/2) score] ds + sigma dW`` (Thm. 17; condot only).

        See DiffusionForecastEngineBase._run_sampler.
        """
        device = "cuda"
        t_steps = self._sampling_nodes(num_steps, device)

        # Initial noise scaled to the start-of-path marginal std. VE: X_0 ~ N(0, sigma_max^2)
        # (edm.py x*t_steps[0]); condot: pinit = N(0, I), std 1. Only the healpix-cell
        # tokens are noised.
        x = torch.randn(
            batch_size, self.num_healpix_cells, self.cf.ae_global_dim_embed, device=device
        )
        if self.path.kind == "ve":
            _, beta0, _, _ = self.path.coeffs(t_steps[0].float())
            x = x * beta0
        target_cells = (
            self.cur_token[:, self.num_aux_tokens :] if self.cur_token is not None else None
        )

        use_sde = self.sampler == "sde" and self.sde_sigma > 0.0
        logger.debug(
            f"Flow-matching inference: sampler={self.sampler} "
            f"(sde_sigma={self.sde_sigma}), prediction_type={self.prediction_type}, "
            f"path={self.path.kind}, steps={num_steps}, "
            f"s: {t_steps[0].item():.4f} -> {t_steps[-2].item():.4f} -> {t_steps[-1].item():.4f}"
        )

        # Per-step diagnostics. At every t we compare the *current noisy state* x_t against the
        # *denoised estimate* x0_hat = D_t(x_t) predicted from it, in RMSE-to-target and in std.
        # RMSE (not raw L2 norm) so the magnitude is independent of latent size and directly
        # comparable to sigma_data / to std(z). Plotted offline by
        # scripts/analytics/diffusion/plot_diffusion_sampling_diagnostics.py.
        track = {
            "t": [],
            "x_t_std": [],
            "x0_hat_std": [],
            "rmse_x_t": [],
            "rmse_x0_hat": [],
            "vel_norm": [],
            "target_std": [],
        }
        intermediate_x: list[torch.Tensor] | None = [] if return_diffusion_steps else None

        x_next = x
        for i in tqdm.tqdm(
            range(num_steps),
            desc="flow sampling step",
            position=1,
            leave=False,
            disable=self.cf.rank > 0,
        ):
            # Cast per-step to float32 (nodes may be float64 for ve; edm.py does the same).
            # For ve, h = sigma_next - sigma_cur is NEGATIVE (descending) — Euler/Heun are
            # sign-agnostic and the SDE branch already uses h.abs().sqrt().
            t_cur = t_steps[i].float().reshape(1, 1, 1)
            t_next = t_steps[i + 1].float().reshape(1, 1, 1)
            h = t_next - t_cur
            x_cur = x_next

            raw = self._predict(x_cur, t_cur, c, fstep, coords)
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
                    raw2 = self._predict(x_next, t_next, c, fstep, coords)
                    u2 = self.path.to_velocity(raw2, x_next, t_next, self.prediction_type)
                    x_next = x_cur + h * 0.5 * (u + u2)

            if self.log_analytics:
                with torch.no_grad():
                    # Denoised (clean-latent) estimate predicted from x_t at this t.
                    x0_hat = self.path.to_denoiser(raw, x_cur, t_cur, self.prediction_type)
                    track["t"].append(t_cur.item())
                    track["x_t_std"].append(x_cur.std().item())
                    track["x0_hat_std"].append(x0_hat.std().item())
                    track["vel_norm"].append(u.norm().item())
                    if target_cells is not None:
                        # RMSE = ||.|| / sqrt(numel), i.e. per-element error. Size-independent,
                        # so it is comparable across latent shapes and against sigma_data.
                        _rn = target_cells.numel() ** 0.5
                        track["rmse_x_t"].append((x_cur - target_cells).norm().item() / _rn)
                        track["rmse_x0_hat"].append((x0_hat - target_cells).norm().item() / _rn)

            if return_diffusion_steps:
                intermediate_x.append(self._prepend_aux_tokens(x_next))

        if self.log_analytics:
            # Record the FINAL state x_next at the terminal node t_steps[num_steps]. The loop above
            # only tracks x_cur (the state *before* each update), so without this the actual
            # returned sample is never plotted and x_t appears to stall above x0_hat. For the ve
            # path the last step is a plain Euler step with h = -sigma_last and u = (x - D)/sigma,
            # so x_next == D exactly: the endpoint lands ON the x0_hat curve. x0_hat/velocity are
            # not defined at the terminal node (it would need another net forward, degenerate at
            # sigma=0), so they get NaN and matplotlib simply omits that marker.
            with torch.no_grad():
                track["t"].append(t_steps[num_steps].float().item())
                track["x_t_std"].append(x_next.std().item())
                track["x0_hat_std"].append(float("nan"))
                track["vel_norm"].append(float("nan"))
                if target_cells is not None:
                    _rn = target_cells.numel() ** 0.5
                    track["rmse_x_t"].append((x_next - target_cells).norm().item() / _rn)
                    track["rmse_x0_hat"].append(float("nan"))
                    track["target_std"].append(target_cells.std().item())

        # self._diagnostics_writer is None outside inference.
        if self._diagnostics_writer is not None:
            self._diagnostics_writer.write(track)

        return self._prepend_aux_tokens(x_next), intermediate_x
