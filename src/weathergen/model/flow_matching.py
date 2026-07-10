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
    """Gaussian probability path ``x_t = alpha_t z + beta_t eps`` and the closed-form
    conversions between the velocity field, the score, the noise and the denoiser
    (course Prop. 1 / Remark 16).

    Schedulers return ``(alpha, beta, alpha_dot, beta_dot)`` as tensors broadcastable
    to the data. ``condot`` is the linear (conditional optimal transport) path
    ``alpha_t = t, beta_t = 1 - t``. Add a ``ve`` scheduler (``alpha=1, beta=sigma(t)``)
    later to obtain the EDM/variance-exploding family through the same interface.

    This is the single source of truth imported by the engine, the target encoder
    and the loss module so all three agree on the math.
    """

    VALID_PREDICTION_TYPES = ("velocity", "noise", "score")

    def __init__(self, kind: str = "condot"):
        self.kind = kind

    def coeffs(self, t: torch.Tensor):
        """Return ``(alpha_t, beta_t, alpha_dot_t, beta_dot_t)`` for scalar/broadcast ``t``."""
        if self.kind == "condot":
            alpha = t
            beta = 1.0 - t
            alpha_dot = torch.ones_like(t)
            beta_dot = -torch.ones_like(t)
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
        """
        alpha, beta, alpha_dot, beta_dot = self.coeffs(t)
        if prediction_type == "velocity":
            return alpha_dot * z + beta_dot * eps
        if prediction_type == "noise":
            return eps
        if prediction_type == "score":
            return -eps / beta
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
        if prediction_type == "velocity":
            # From u = alpha_dot z + beta_dot eps and x = alpha z + beta eps:
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
        """Posterior-mean (x0) estimate; ill-conditioned near t=0 (alpha->0)."""
        alpha, beta, _, _ = self.coeffs(t)
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

        # -- Sampler (seam #4) --
        self.sampler = cf.get("fe_flow_sampler", "ode")
        self.t_eps = cf.get("fm_t_eps", 1e-3)
        self.num_steps_default = cf.get("fm_num_steps", 50)
        self.sde_sigma = cf.get("fm_sde_sigma", 0.0)

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
        if self.conditioning == self._FORECAST:
            c = meta_info["ERA5"].params["conditioning_tokens"]

        if self.training:
            noise_level_rn = meta_info["ERA5"].params["noise_level_rn"]
        else:
            noise_level_rn = (
                self._fixed_noise_level if self._fixed_noise_level is not None else 0.5
            )
        # t is kept in float32 (a scalar) so path coefficients are precise, matching how the
        # diffusion engine keeps sigma in float32; x_t then promotes to float32.
        t = self._t_from_noise_level(noise_level_rn, device=z.device)

        alpha, beta, _, _ = self.path.coeffs(t)
        eps = torch.randn_like(z)
        x_t = alpha * z + beta * eps

        # Stash eps and t so the target encoder (runs after this forward pass) can build
        # the exact analytic target. Same channel model.py uses for conditioning_tokens.
        meta_info["ERA5"].params["fm_eps"] = eps.detach()
        meta_info["ERA5"].params["fm_t"] = float(t.reshape(-1)[0].item())

        return self._net_forward(x_t, t, c, fstep, coords)

    def _t_from_noise_level(self, noise_level_rn, device, dtype=torch.float32) -> torch.Tensor:
        """Map the per-sample ``noise_level_rn`` to ``t in [t_eps, 1 - t_eps]``.

        With ``noise_distribution: uniform`` (drawn at the source in masking.py),
        ``noise_level_rn`` already *is* ``t`` and we only clamp for numerical safety.
        Fallbacks are kept for reusing existing draws (see plan), but ``uniform`` is the
        primary path for flow matching.
        """
        dist = self.cf.get("noise_distribution", "uniform")
        v = torch.as_tensor(noise_level_rn, device=device, dtype=dtype).reshape(1)
        if dist == "uniform":
            t = v
        # TODO: Maybe close the branches below and instead assert uniform distribution
        elif dist == "log_normal":  # eta ~ N(0,1) -> Phi(eta) ~ Unif(0,1)
            t = 0.5 * (1.0 + torch.erf(v / math.sqrt(2.0)))
        elif dist == "log_uniform":  # log-sigma in [log smin, log smax] -> affine to [0,1]
            lo, hi = math.log(self.cf.sigma_min), math.log(self.cf.sigma_max)
            t = (v - lo) / (hi - lo)
        else:
            raise ValueError(f"Unsupported noise_distribution for flow matching: {dist!r}")
        return t.clamp(self.t_eps, 1.0 - self.t_eps)

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
        """Run the (time-conditioned) forecasting backbone on ``x`` at time ``t``.

        Mirrors DiffusionForecastEngine.denoise but with an identity preconditioner
        (no EDM c_skip/c_out/c_in) and a time embedding instead of a sigma embedding.
        The conditioning-type branches match the shared backbone in engines.py.
        """
        time_emb = self.time_embedder(t.reshape(1) * self.time_scale)
        net_input = self._precondition_input(x, t)

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
        return raw

    def _precondition_input(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Preconditioner seam (#3). Identity for flow matching; the EDM implant would
        return ``c_in(sigma) * x`` here (and pair it with c_skip/c_out on the output)."""
        c_in = self.cf.get("fm_c_in", 1.0)
        return c_in * x

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

    def _run_sampler(
        self,
        c: torch.Tensor | None,
        fstep: int,
        num_steps: int,
        coords: torch.Tensor | None,
        batch_size: int = 1,
        return_trajectory: bool = False,
    ):
        """Integrate the probability path from ``t_eps`` (noise) to ``1 - t_eps`` (data).

        ODE (default): ``dX/dt = u_t(X)`` with Euler + Heun 2nd-order correction (Alg. 1).
        SDE: Euler-Maruyama ``dX = [u + (sigma_t^2/2) score] dt + sigma_t dW`` (Thm. 17).
        """
        device = "cuda"
        dim = self.cf.ae_global_dim_embed
        x = torch.randn(batch_size, self.num_healpix_cells, dim, device=device)  # X_0 ~ N(0, I)

        t_steps = torch.linspace(
            self.t_eps, 1.0 - self.t_eps, num_steps + 1, dtype=torch.float32, device=device
        )
        use_sde = self.sampler == "sde" and self.sde_sigma > 0.0
        logger.info(
            f"Flow-matching inference: sampler={self.sampler} "
            f"(sde_sigma={self.sde_sigma}), prediction_type={self.prediction_type}, "
            f"path={self.path.kind}, steps={num_steps}, t in [{self.t_eps}, {1 - self.t_eps}]"
        )

        track = {"t": [], "x_std": [], "vel_norm": [], "l2_to_target": [], "x": [x.cpu()]}
        trajectory: list[torch.Tensor] = [] if return_trajectory else None

        x_next = x
        for i in range(num_steps):
            t_cur = t_steps[i]
            t_next = t_steps[i + 1]
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
                track["t"].append(t_cur.item())
                track["x_std"].append(x_next.std().item())
                track["vel_norm"].append(u.norm().item())
                track["x"].append(x_next.cpu())
                if self.cur_token is not None:
                    track["l2_to_target"].append((x_next - self.cur_token).norm().item())

            if return_trajectory:
                trajectory.append(x_next)

        self._plot_sampling_diagnostics(track, num_steps)
        return x_next, trajectory

    def _plot_sampling_diagnostics(self, track: dict, num_steps: int) -> None:
        """Compact diagnostic plot of the sampling trajectory (mirrors diffusion's)."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = list(range(len(track["t"])))
        has_target = len(track["l2_to_target"]) > 0
        n = 3 if has_target else 2
        fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)

        axes[0].plot(steps, track["x_std"], "o-", ms=3, label="x std")
        if self.cur_token is not None:
            axes[0].axhline(
                self.cur_token.std().item(), color="grey", ls="--", lw=0.8, label="target std"
            )
        axes[0].set_ylabel("std")
        axes[0].set_title(f"Flow-matching sampling  |  path={self.path.kind}, steps={num_steps}")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].semilogy(steps, track["vel_norm"], "s-", ms=3, color="tab:orange")
        axes[1].set_ylabel("||velocity||")
        axes[1].grid(True, alpha=0.3)

        if has_target:
            axes[2].plot(steps, track["l2_to_target"], "o-", ms=3, color="tab:red")
            axes[2].set_ylabel("L2 to target")
            axes[2].grid(True, alpha=0.3)

        axes[-1].set_xlabel("sampling step")
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
