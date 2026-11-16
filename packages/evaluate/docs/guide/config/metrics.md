# Metrics Reference

Metrics are listed under `evaluation.metrics`. Each entry is either a plain string or a single-key dict:

```yaml
evaluation:
  metrics:
    - rmse
    - mae
    - psd:
        psd_method: "sht"
    - fbi:
        thresh: 280
```

## Deterministic metrics (require prediction and target)

| Name | Description |
|------|-------------|
| `mae` | Mean Absolute Error |
| `mse` | Mean Squared Error |
| `rmse` | Root Mean Squared Error |
| `vrmse` | Variance-normalized RMSE |
| `l1` | L1 error norm |
| `l2` | L2 error norm |
| `bias` | Mean bias (prediction - target) |
| `psnr` | Peak Signal-to-Noise Ratio |
| `nse` | Nash-Sutcliffe Efficiency |
| `ets` | Equitable Threat Score. Override threshold with `thresh`. |
| `pss` | Peirce Skill Score. Override threshold with `thresh`. |
| `fbi` | Frequency Bias Index. Override threshold with `thresh`. |
| `grad_amplitude` | Ratio of spatial variability between prediction and target (regular lat/lon required). |
| `qq_analysis` | Quantile-quantile analysis (Q-Q plots). |
| `psd` | Power Spectral Density (PSD plots). |

## Metrics requiring alignment between consecutive forecast steps

| Name | Description |
|------|-------------|
| `froct` | Forecast Rate of Change over Time |
| `troct` | Target Rate of Change over Time |

## Metrics requiring climatology

| Name | Description |
|------|-------------|
| `acc` | Anomaly Correlation Coefficient |
| `rps` | Ranked Probability Score |
| `rpss` | Ranked Probability Skill Score |
| `fact` | Forecast Activity |
| `tact` | Target Activity |
| `seeps` | Stable Equitable Error in Probability Space, reported as skill `1 - SEEPS_error` (higher is better). |

## Probabilistic metrics (require ensemble dimension)

| Name | Description |
|------|-------------|
| `ssr` | Spread-Skill Ratio |
| `crps` | Continuous Ranked Probability Score |
| `rank_histogram` | Rank Histogram (Talagrand diagram) |
| `spread` | Ensemble Spread |

## Special output metrics

### psd

| Parameter | Default | Description |
|-----------|---------|-------------|
| `psd_method` | `"sht"` | `"sht"` for spherical harmonics, `"fft"` for zonal FFT (regular grid required). |
| `psd_regrid_resolution` | `1.0` | Regrid resolution in degrees for FFT path. |

### qq_analysis

Distributional comparison plot between predictions and targets.

### rank_histogram

Ensemble calibration diagnostic (flat is ideal).

## Metric parameters

Any metric supporting thresholds can be configured as:

```yaml
evaluation:
  metrics:
    - fbi:
        thresh: 280
    - ets:
        thresh: 0.001
```

### crps parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | `"ecdf"` | `"ecdf"`, `"fair"`, `"tw_tail"`, or `"tw_interval"`. |
| `fair` | `false` | Shorthand for `method: "fair"`. |
| `threshold` | - | Required for `tw_tail`. |
| `tail` | `"upper"` | Tail for `tw_tail`: `"upper"` or `"lower"`. |
| `lower_threshold` | - | Lower bound for `tw_interval`. |
| `upper_threshold` | - | Upper bound for `tw_interval`. |

### seeps parameters

```yaml
evaluation:
  metrics:
    - seeps:
        minimum_dry_prob: 0.1
        maximum_dry_prob: 0.85
```

Values outside dry-probability bounds are excluded from SEEPS computation.
