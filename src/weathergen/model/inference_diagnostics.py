# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Per-ODE-step maps and spectra for the diffusion sampler.

At every step of the sampler we hold three latent states: the noisy state ``x_t``, the denoised
estimate ``x0_hat = D_t(x_t)`` predicted from it, and the clean target ``z``.  This module decodes
them to physical space and plots

1. maps of ``x_t`` / ``x0_hat`` / ``decode(z)`` / ground truth, and
2. angular power spectra of the same, in latent *and* physical space,

so the over-smoothing failure mode -- ``x0_hat`` converging in RMSE while missing high-wavenumber
power -- is visible.  ``decode(z)`` vs ground truth separates the sampler error from the
autoencoder's own reconstruction error.

Only active for single-step forecasting: in rollout mode ``model.py`` sets ``tokens=None`` after
the first step, so the forecast engine's ``cur_token`` is ``None`` and there is no reference.

Collection happens inside the sampler; **rendering is deferred** to
:meth:`ODEDiagnostics.render`, which the trainer calls after the forward pass -- the ground
truth, the per-point coordinates and the ``idxs_inv`` permutation only exist in the target/aux
output.  That also keeps ~150 matplotlib figures out of the model forward.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from weathergen.model.inference_spectra import healpix_sht_psd, physical_psd, white_noise_reference

logger = logging.getLogger(__name__)

# Curve/panel styling, shared by the map and spectrum figures.
_FIELDS = ("x_t", "x0_hat", "decode_z", "truth")
_LABELS = {
    "x_t": r"$x_t$ (noisy state)",
    "x0_hat": r"$\hat{x}_0(x_t)$ (denoised estimate)",
    "decode_z": r"decode($z$) (latent target)",
    "truth": "truth (data)",
}
_COLORS = {"x_t": "tab:blue", "x0_hat": "tab:red", "decode_z": "tab:green", "truth": "black"}
_STYLES = {"x_t": "-", "x0_hat": "-", "decode_z": "--", "truth": ":"}
# At the terminal node x_t IS the decoded sample (sigma=0), not a noisy intermediate state.
_FINAL_LABEL = r"$x_{t=0}$ (final decoded output)"


class ODEDiagnostics:
    """Collects decoded fields and latent spectra along the ODE, then renders them.

    Lifecycle, per sampled batch::

        set_batch(bidx)                  # trainer: self-disables for bidx > 0
        bind_decoder(fn)                 # model.py: fn(tokens) -> {stream: (pred, ...)}
        begin(z)                         # sampler: reference target
        on_step(i, t, x_t, x0_hat)       # sampler: once per ODE step
        render(target_aux_physical)      # trainer: writes the figures
    """

    def __init__(
        self,
        out_dir: Path,
        stream: str,
        channels: list[str],
        channel_names: list[str],
        nside: int,
        denormalize: Callable[[str, torch.Tensor], torch.Tensor],
        num_aux_tokens: int = 0,
        every_n_steps: int = 1,
        latent_channels: int = 128,
        image_format: str = "png",
    ) -> None:
        self.out_dir = Path(out_dir)
        self.stream = stream
        self.nside = nside
        self.denormalize = denormalize
        self.num_aux_tokens = num_aux_tokens
        self.every_n_steps = max(1, int(every_n_steps))
        self.latent_channels = int(latent_channels)
        self.image_format = image_format

        missing = [c for c in channels if c not in channel_names]
        if missing:
            msg = f"Diagnostic channels {missing} not in stream {stream!r}: {channel_names}"
            raise ValueError(msg)
        self.channels = list(channels)
        self.channel_idxs = [channel_names.index(c) for c in channels]

        self.enabled = False
        self._decode: Callable[[torch.Tensor], dict] | None = None
        self._reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _reset(self) -> None:
        self.steps: list[int] = []
        self.times: list[float] = []
        self.latent_psd: dict[str, list[np.typing.NDArray]] = {"x_t": [], "x0_hat": []}
        self.phys: dict[str, list[np.typing.NDArray]] = {"x_t": [], "x0_hat": []}
        self.latent_psd_z: np.typing.NDArray | None = None
        self.phys_z: np.typing.NDArray | None = None
        self.wavenumbers: np.typing.NDArray | None = None
        self._z_var: float = 1.0

    def set_batch(self, batch_idx: int) -> None:
        """Enable for the first batch only; each new batch starts from a clean slate."""
        self.enabled = batch_idx == 0
        self._reset()

    def bind_decoder(self, decode: Callable[[torch.Tensor], dict]) -> None:
        self._decode = decode

    # ------------------------------------------------------------------
    # Collection (called from the sampler)
    # ------------------------------------------------------------------
    def active(self) -> bool:
        return self.enabled and self._decode is not None

    def begin(self, z: torch.Tensor) -> None:
        """Record the clean latent target and its decoding."""
        if not self.active():
            return
        wavenumbers, psd = self._latent_psd(z)
        self.wavenumbers = wavenumbers
        self.latent_psd_z = psd
        self._z_var = float(self._latent_maps(z).var())
        self.phys_z = self._decode_channels(z)

    def on_step(
        self,
        step: int,
        t: float,
        x_t: torch.Tensor,
        x0_hat: torch.Tensor | None,
        force: bool = False,
    ) -> None:
        """Record one ODE step. Costs two decoder passes, hence ``every_n_steps``.

        The **terminal** state (``x0_hat=None``, ``force=True``) is the actual sample handed to
        the decoder — the sampler loop otherwise only sees ``x_cur``, the state *before* each
        update, so without this the returned output at ``sigma=0`` is never plotted and the last
        ``x_t`` frame stalls at ``sigma_min`` (still visibly noisy). The denoiser is undefined
        there (it would need another net forward at ``sigma=0``), so ``x0_hat`` is dropped.
        """
        if not self.active() or (not force and step % self.every_n_steps):
            return
        self.steps.append(step)
        self.times.append(float(t))
        self.latent_psd["x_t"].append(self._latent_psd(x_t)[1])
        self.phys["x_t"].append(self._decode_channels(x_t))
        if x0_hat is None:
            self.latent_psd["x0_hat"].append(None)
            self.phys["x0_hat"].append(None)
        else:
            self.latent_psd["x0_hat"].append(self._latent_psd(x0_hat)[1])
            self.phys["x0_hat"].append(self._decode_channels(x0_hat))

    # ------------------------------------------------------------------
    # Latent helpers
    # ------------------------------------------------------------------
    def _latent_maps(self, tokens: torch.Tensor) -> np.typing.NDArray:
        """``[1, n_tokens, dim]`` tokens -> ``[n_channels, npix]`` HEALPix maps (nested).

        Auxiliary (class/register) tokens are stripped exactly as ``predict_decoders`` does, and
        the per-cell query axis is folded into the channel axis so ``ae_local_num_queries > 1``
        works without special casing.
        """
        x = tokens.detach()[:, self.num_aux_tokens :].float().cpu().numpy()
        npix = 12 * self.nside**2
        n_cells = x.shape[1]
        if n_cells % npix:
            msg = f"Token count {n_cells} is not a multiple of npix={npix} (nside={self.nside})"
            raise ValueError(msg)
        # (batch, cells * queries, dim) -> (cells, queries * dim) -> (channels, cells)
        return x[0].reshape(npix, -1).T

    def _latent_psd(self, tokens: torch.Tensor) -> tuple[np.typing.NDArray, np.typing.NDArray]:
        maps = self._latent_maps(tokens)
        if 0 < self.latent_channels < maps.shape[0]:
            # Fixed subset across all steps so the curves are comparable; the mean over a random
            # subset is an unbiased estimate of the mean over all channels.
            idx = np.random.default_rng(0).choice(maps.shape[0], self.latent_channels, False)
            maps = maps[idx]
        return healpix_sht_psd(maps, self.nside)

    # ------------------------------------------------------------------
    # Physical helpers
    # ------------------------------------------------------------------
    def _decode_channels(self, tokens: torch.Tensor) -> np.typing.NDArray:
        """Decode latents and keep only the diagnostic channels, denormalized.

        Slicing *after* denormalization but *before* storing matters: the full channel set for
        every field at every step would be gigabytes, the two channels are ~40 k floats.
        """
        preds = self._decode(tokens)
        pred = preds[self.stream][0]  # first batch item; (ensemble, n_points, n_channels)
        pred = pred[0] if pred.ndim == 3 else pred
        pred = self.denormalize(self.stream, pred.to(torch.float32))
        return pred[:, self.channel_idxs].detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Rendering (called from the trainer, after the forward pass)
    # ------------------------------------------------------------------
    def render(self, target_aux_physical: dict) -> None:
        """Write all figures.

        ``target_aux_physical`` is ``target_aux_out.physical[fstep][stream]`` for the physical
        loss term -- the same structure ``write_output`` consumes.
        """
        if not self.enabled or not self.steps:
            return
        try:
            truth, lats, lons, order = self._truth_and_coords(target_aux_physical)
        except Exception:
            logger.exception("ODE diagnostics: could not extract truth/coords; skipping.")
            return

        n_points = truth.shape[0]
        # x0_hat holds None at the terminal frame (denoiser undefined at sigma=0); skip those.
        collected = [p for p in [*self.phys["x_t"], *self.phys["x0_hat"]] if p is not None]
        if any(p.shape[0] != n_points for p in collected):
            logger.warning(
                "ODE diagnostics: decoded fields do not match the %d target points; "
                "skipping (are target coords varying across the forward pass?).", n_points
            )
            return

        self.out_dir.mkdir(parents=True, exist_ok=True)
        for name in ("x_t", "x0_hat"):
            self.phys[name] = [None if p is None else p[order] for p in self.phys[name]]
        self.phys_z = self.phys_z[order] if self.phys_z is not None else None

        self._render_spectra(truth, lats, lons)
        self._render_maps(truth, lats, lons)
        logger.info(f"Saved ODE diagnostics for {len(self.steps)} steps to {self.out_dir}")

    def _truth_and_coords(self, target_aux_physical: dict):
        """Ground truth, coordinates and the permutation restoring dataset point order."""
        truth = target_aux_physical["target"][0]
        coords = target_aux_physical["target_coords"][0]
        idxs_inv = target_aux_physical["idxs_inv"][0]
        if idxs_inv is not None:
            truth = truth[idxs_inv]
            coords = coords[idxs_inv]
        truth = self.denormalize(self.stream, truth.to(torch.float32)).detach().cpu().numpy()
        coords = coords.detach().cpu().numpy()
        order = idxs_inv.detach().cpu().numpy() if idxs_inv is not None else slice(None)
        return truth[:, self.channel_idxs], coords[:, 0], coords[:, 1], order

    # -- spectra --
    def _render_spectra(
        self, truth: np.typing.NDArray, lats: np.typing.NDArray, lons: np.typing.NDArray
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        phys_psd = {name: [] for name in _FIELDS}
        wavenumbers_p = None
        for ch_pos in range(len(self.channels)):
            for name in ("x_t", "x0_hat"):
                curves = []
                for field in self.phys[name]:
                    res = None if field is None else physical_psd(field[:, ch_pos], lats, lons)
                    curves.append(None if res is None else res[1])
                    wavenumbers_p = wavenumbers_p if res is None else res[0]
                phys_psd[name].append(curves)
            for name, field in (("decode_z", self.phys_z), ("truth", truth)):
                res = None if field is None else physical_psd(field[:, ch_pos], lats, lons)
                phys_psd[name].append(None if res is None else res[1])
                wavenumbers_p = wavenumbers_p if res is None else res[0]

        out = self.out_dir / "spectra"
        out.mkdir(parents=True, exist_ok=True)

        # Per-step figure: latent + one panel per physical channel.
        n_panels = 1 + len(self.channels)
        for i, step in enumerate(self.steps):
            # Terminal frame: x0_hat is absent and x_t is the final decoded sample.
            is_final = self.latent_psd["x0_hat"][i] is None
            xt_label = _FINAL_LABEL if is_final else _LABELS["x_t"]
            fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.4))
            axes = np.atleast_1d(axes)
            self._spectrum_panel(
                axes[0],
                self.wavenumbers,
                {
                    "x_t": self.latent_psd["x_t"][i],
                    "x0_hat": self.latent_psd["x0_hat"][i],
                    "decode_z": self.latent_psd_z,
                },
                title="latent",
                noise_var=self._z_var,
                labels={"decode_z": r"$z$ (latent target)", "x_t": xt_label},
            )
            for c, channel in enumerate(self.channels):
                self._spectrum_panel(
                    axes[1 + c],
                    wavenumbers_p,
                    {
                        "x_t": phys_psd["x_t"][c][i],
                        "x0_hat": phys_psd["x0_hat"][c][i],
                        "decode_z": phys_psd["decode_z"][c],
                        "truth": phys_psd["truth"][c],
                    },
                    title=f"physical: {channel}",
                    labels={"x_t": xt_label},
                )
            tag = "  [final decoded output]" if is_final else ""
            fig.suptitle(f"ODE step {step}   (t = {self.times[i]:.4g}){tag}")
            fig.tight_layout()
            fig.savefig(out / f"step{step:03d}.{self.image_format}", dpi=130)
            plt.close(fig)

        # Evolution overlays: every step on one axis, colour-graded by step.
        self._render_evolution(out, "latent", self.wavenumbers, self.latent_psd["x0_hat"],
                               self.latent_psd_z, r"$z$")
        for c, channel in enumerate(self.channels):
            self._render_evolution(out, channel, wavenumbers_p, phys_psd["x0_hat"][c],
                                   phys_psd["truth"][c], "truth")

    @staticmethod
    def _spectrum_panel(ax, wavenumbers, curves: dict, title: str, noise_var=None, labels=None):
        labels = labels or {}
        if wavenumbers is None:
            ax.text(0.5, 0.5, "unavailable", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            return
        peak = 0.0
        for name, psd in curves.items():
            if psd is None:
                continue
            ax.loglog(wavenumbers[1:], psd[1:], _STYLES[name], color=_COLORS[name], lw=1.3,
                      label=labels.get(name, _LABELS[name]))
            finite = psd[1:][np.isfinite(psd[1:])]
            peak = max(peak, float(finite.max()) if finite.size else 0.0)
        if peak > 0:
            # Clamp to ~9 decades below the peak: a band-limited field's numerically-zero tail
            # would otherwise stretch the axis over 18 decades and flatten everything of interest.
            ax.set_ylim(peak * 1e-9, peak * 10)
        if noise_var is not None:
            ax.loglog(wavenumbers[1:], white_noise_reference(wavenumbers[1:], noise_var),
                      ":", color="grey", lw=1.0, label=r"white noise ($\propto 2\ell+1$)")
        ax.set_xlabel(r"total wavenumber $\ell$")
        ax.set_ylabel(r"PSD  $\sum_m |a_{\ell m}|^2$")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)

    def _render_evolution(self, out: Path, name: str, wavenumbers, curves, reference, ref_label):
        import matplotlib.pyplot as plt

        if wavenumbers is None or not curves or all(c is None for c in curves):
            return
        fig, ax = plt.subplots(figsize=(7, 5))
        cmap = plt.get_cmap("viridis")
        n = max(len(curves) - 1, 1)
        for i, psd in enumerate(curves):
            if psd is None:
                continue
            ax.loglog(wavenumbers[1:], psd[1:], color=cmap(i / n), lw=1.0)
        if reference is not None:
            ax.loglog(wavenumbers[1:], reference[1:], "k--", lw=1.6, label=ref_label)
            ax.legend(fontsize=8)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(self.steps[0], self.steps[-1]))
        fig.colorbar(sm, ax=ax, label="ODE step")
        ax.set_xlabel(r"total wavenumber $\ell$")
        ax.set_ylabel(r"PSD  $\sum_m |a_{\ell m}|^2$")
        ax.set_title(rf"{name}: $\hat{{x}}_0$ spectrum along the ODE")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"evolution_{name}.{self.image_format}", dpi=130)
        plt.close(fig)

    # -- maps --
    def _render_maps(
        self, truth: np.typing.NDArray, lats: np.typing.NDArray, lons: np.typing.NDArray
    ) -> None:
        try:
            plotter = _make_panel_plotter(self.out_dir, self.stream, self.image_format)
        except Exception:
            logger.exception("ODE diagnostics: map plotting unavailable; spectra kept.")
            return

        out = self.out_dir / "maps"
        out.mkdir(parents=True, exist_ok=True)
        for c, channel in enumerate(self.channels):
            # One colour scale for every panel and every step, taken from the truth, so the
            # frames can be compared (and flip-booked) directly.
            finite = truth[:, c][np.isfinite(truth[:, c])]
            vmin, vmax = np.percentile(finite, [2, 98])
            for i, step in enumerate(self.steps):
                x0_hat = self.phys["x0_hat"][i]  # None at the terminal frame
                is_final = x0_hat is None
                fields = {
                    "x_t": self.phys["x_t"][i][:, c],
                    "x0_hat": None if is_final else x0_hat[:, c],
                    "decode_z": None if self.phys_z is None else self.phys_z[:, c],
                    "truth": truth[:, c],
                }
                labels = {**_LABELS, "x_t": _FINAL_LABEL if is_final else _LABELS["x_t"]}
                panels = [(labels[k], v) for k, v in fields.items() if v is not None]
                tag = "  [final decoded output]" if is_final else ""
                plotter.create_map_panel(
                    panels,
                    lats,
                    lons,
                    varname=channel,
                    suptitle=f"{channel}  |  ODE step {step}  (t = {self.times[i]:.4g}){tag}",
                    out_path=out / f"step{step:03d}_{channel}.{self.image_format}",
                    map_kwargs={"vmin": float(vmin), "vmax": float(vmax)},
                )


def maybe_create(cf, model, denormalize: Callable[[str, torch.Tensor], torch.Tensor]):
    """Build the diagnostics and attach them to the forecast engine, or return ``None``.

    Off unless ``diag_ode_maps`` is set. Drives the diffusion engine's sampler, which exposes the
    ``self.diagnostics`` hook and the ``begin``/``on_step`` protocol. Only meaningful during
    inference, the only stage that runs a sampler.
    """
    if not cf.get("diag_ode_maps", False):
        return None
    if not cf.get("fe_diffusion_model", False):
        logger.warning("diag_ode_maps is set but this is not a diffusion run; "
                       "ignoring.")
        return None
    if cf.stage != "inference":
        logger.warning(f"diag_ode_maps is set but stage is {cf.stage!r}; ignoring.")
        return None
    if not hasattr(model.forecast_engine, "diagnostics"):
        logger.warning("diag_ode_maps is set but the forecast engine has no diagnostics hook "
                       f"({type(model.forecast_engine).__name__}); ignoring.")
        return None

    from weathergen.common.config import get_path_run

    stream_name = cf.get("diag_stream", "ERA5")
    streams = {s["name"]: s for s in cf.streams}
    if stream_name not in streams:
        logger.warning(f"diag_stream={stream_name!r} not in {list(streams)}; ignoring.")
        return None

    engine = model.forecast_engine
    diagnostics = ODEDiagnostics(
        out_dir=get_path_run(cf) / "plots" / "ode_diagnostics",
        stream=stream_name,
        channels=list(cf.get("diag_channels", ["2t", "q_850"])),
        channel_names=list(streams[stream_name].val_target_channels),
        nside=2**cf.healpix_level,
        denormalize=denormalize,
        num_aux_tokens=getattr(model, "num_aux_tokens", 0),
        every_n_steps=cf.get("diag_ode_every_n_steps", 1),
        latent_channels=cf.get("diag_latent_channels", 128),
    )
    engine.diagnostics = diagnostics
    logger.info(
        f"ODE diagnostics enabled: channels={diagnostics.channels}, "
        f"every_n_steps={diagnostics.every_n_steps} -> {diagnostics.out_dir}"
    )
    return diagnostics


def _make_panel_plotter(out_dir: Path, stream: str, image_format: str):
    """Build the map-panel plotter.

    Imported lazily: it pulls in cartopy and the evaluation package's private working-dir config,
    neither of which should be able to abort an inference run.
    """
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import xarray as xr

    from weathergen.evaluate.plotting.plot_utils import DefaultMarkerSize
    from weathergen.evaluate.plotting.plotter import Plotter

    class MapPanelPlotter(Plotter):
        """Multi-panel maps on one figure, reusing the evaluation package's rendering.

        ``Plotter.scatter_plot`` builds *and saves* a single-panel figure, so it cannot compose a
        row of panels.  Rather than modify the evaluation package (which is treated as read-only),
        this subclass adds the panel layout and delegates every rendering decision --
        option parsing, marker sizing, scatter/datashader, HEALPix overlay -- to the inherited
        methods, so panels match evaluation maps.  It leans on ``Plotter``'s underscore-prefixed
        helpers; an upstream rename breaks exactly this class.
        """

        def create_map_panel(self, panels, lats, lons, varname, suptitle, out_path, map_kwargs):
            opts = self._parse_map_kwargs(dict(map_kwargs or {}), self.stream)
            proj = ccrs.Robinson()
            fig, axes = plt.subplots(
                1, len(panels), figsize=(5.6 * len(panels), 3.6),
                subplot_kw={"projection": proj}, dpi=self.dpi_val,
            )
            axes = np.atleast_1d(axes)

            artist = None
            for ax, (title, values) in zip(axes, panels, strict=True):
                data = xr.DataArray(
                    np.asarray(values),
                    dims=("ipoint",),
                    coords={"lon": ("ipoint", np.asarray(lons)),
                            "lat": ("ipoint", np.asarray(lats))},
                )
                try:
                    ax.coastlines(linewidth=0.3)
                except Exception:
                    logger.warning("Could not add coastlines; continuing without them.")
                ax.set_global()
                marker_size = DefaultMarkerSize.auto_marker_size(
                    n_points=data.size,
                    fig_width_in=fig.get_figwidth() / len(panels),
                    fig_height_in=fig.get_figheight(),
                    stream_default=opts["marker_size_base"],
                    scale=opts["scale_marker_size"],
                    lat=data["lat"],
                )
                artist = self._render_scatter(
                    ax, data, opts["norm"], opts["cmap"], marker_size, opts["marker"],
                    opts["extra"],
                )
                ax.gridlines(draw_labels=False, linestyle="--", color="gray", linewidth=0.5,
                             alpha=0.6)
                ax.set_title(title, fontsize=9)

            cbar = fig.colorbar(artist, ax=axes.tolist(), fraction=0.02, pad=0.02, shrink=0.8,
                                orientation="horizontal")
            cbar.set_label(f"Variable: {varname}", fontsize=8)
            cbar.ax.tick_params(labelsize=7)
            fig.suptitle(suptitle, fontsize=10)
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)

    cfg = {"image_format": image_format, "dpi_val": 130, "fig_size": None, "regions": ["global"]}
    return MapPanelPlotter(cfg, out_dir, stream=stream)
