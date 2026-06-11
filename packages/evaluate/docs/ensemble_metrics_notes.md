# Ensemble spread-skill metrics: implementation notes & divergence from GenCast

This note documents the **current** behaviour of the ensemble spread / spread-skill metrics in
`packages/evaluate` and how they differ from the GenCast definitions. The metrics were enabled
(they were previously dead code, see below) **without changing their mathematical definitions** —
this document is the basis for a later decision on whether to align them with GenCast.

## What was enabled

The probabilistic metrics (`spread`, `ssr`, `crps`, `rank_histogram`) are registered in
`Scores.prob_metrics_dict` but were unreachable: `Scores.get_score` referenced an undefined
`self.ens_dim` and unconditionally `return None`ed before dispatching. The dispatch branch was
fixed to skip cleanly when the ensemble dim is absent and otherwise call the metric. No metric
definition was changed.

Enabled in `config/evaluate/eval_config_diffusion.yml`:
- `evaluation.metrics: [..., "spread", "ssr"]` → spread-skill line plots per `channel`
  (variable+level) vs `lead_time`.
- `evaluation.plot_score_maps: true` → ensemble spread **map** per `channel`/`forecast_step`
  (the score-map path computes each metric with `agg_dims="sample"`, keeping `ipoint`).

## Current definitions (in `scores/score.py`)

```python
def calc_spread(self, p):
    ens_std = p.std(dim="ens")          # xarray default ddof=0
    return self._mean(np.sqrt(ens_std**2))   # sqrt(std**2) == std; mean over agg_dims

def calc_ssr(self, p, gt):
    return self.calc_spread(p) / self.calc_rmse(p, gt)
```

`self._mean` is an **unweighted** `mean` over `agg_dims` (default `ipoint` in the summary path,
`sample` in the score-map path). `calc_rmse` is `sqrt(mean((p - gt)**2))`.

## GenCast reference (arXiv:2312.15796)

For forecast times `k`, grid points `i` with latitude weights `a_i`, ensemble members `m`,
ensemble mean `x̄`, truth `y`:

- Spread = `sqrt( mean_k [ Σ_i a_i · (1/(M-1)) Σ_m (x_{i,k}^m - x̄_{i,k})² ] )`
- Skill (ensemble-mean RMSE) = `sqrt( mean_k [ Σ_i a_i · (x̄_{i,k} - y_{i,k})² ] )`
- Spread/skill ratio = Spread / Skill; for a calibrated M-member ensemble this equals
  `sqrt((M+1)/M)`.

## Divergences (current → GenCast)

1. **Spread aggregation order.** Current returns `mean_i( std_m )` (spatial mean of the per-point
   std). GenCast uses `sqrt( mean_i( var_m ) )`. In general `mean(std) ≠ sqrt(mean(var))`.
2. **Unbiased variance.** Current `std`/`var` use xarray default `ddof=0`; GenCast uses `1/(M-1)`
   (`ddof=1`).
3. **Latitude weighting.** Current uses an unweighted mean over `ipoint`; GenCast applies
   `cos(lat)` weights `a_i`. (Deliberately kept unweighted for consistency with the existing
   `rmse`/`mae` in this repo.)
4. **sqrt vs forecast-time averaging order.** The summary path keeps the `sample` dimension and
   averages it at plot time (`LinePlots` averages over all non-x dims), i.e. *sqrt then average
   over forecast times*. GenCast averages over forecast times *inside* the sqrt.
5. **SSR denominator.** *(Aligned with GenCast.)* `calc_ssr` now divides by
   `calc_rmse(p.mean("ens"), gt)` — the RMSE of the ensemble **mean** (the skill), so SSR is a
   single value per variable-level-fstep with the standard calibration interpretation
   (under-/over-dispersion around the `sqrt((M+1)/M)`-corrected target). Previously it divided by
   the per-member RMSE (`calc_rmse(p, gt)` with the ensemble still present), which produced one
   ratio per ensemble member.
6. **No `sqrt((M+1)/M)` correction.** SSR is the raw spread/skill ratio, so the calibrated target
   line is `sqrt((M+1)/M)`, not 1.

## Spread map

The spread map is the same `calc_spread` evaluated with `agg_dims="sample"` (reduces `ens` and
`sample`, keeps `ipoint`), routed through `plot_score_maps_per_stream` →
`Plotter.scatter_plot`. Per point it is `mean_sample( std_m )` rather than
`sqrt( mean_sample( var_m ) )` — the same divergences (1)–(2) apply. No latitude weighting is
relevant here since it is a spatial field, not a spatial average.
