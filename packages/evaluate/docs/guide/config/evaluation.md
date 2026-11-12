# Evaluation

Controls what to compute and how to visualize summary scores.

```yaml
evaluation:
  metrics: ["rmse", "mae"]
  regions: ["global", "nhem"]
  score_plots: [metric_plots, ratio]
  summary_dir: "./plots/"
  plot_ensemble: "members"
  print_summary: false
  log_scale: false
  add_grid: false
  baseline: "my_run_id"
  # agg_dims: ["ipoint"]
```

| Key | Type | Optional | Default | Description |
|-----|------|----------|---------|-------------|
| `metrics` | list | no | - | Metrics to compute. Each item is either a metric name string or a single-key dict `{name: {param: value}}`. See [Metrics Reference](metrics.md). |
| `regions` | list[str] | yes | `["global"]` | Regions over which scores are computed. Overrides any region set on individual streams. See [Regions Reference](regions.md). |
| `summary_dir` | str | yes | `<repo_root>/plots/` | Output directory for summary plots. |
| `score_plots` | list[str] | yes | `[]` | Score visualizations to produce. See values below. |
| `baseline` | str | yes | - | `run_id` used as reference for ratio and improvement calculations. |
| `plot_ensemble` | str\|bool | yes | `false` | Ensemble spread style: `false`, `"std"`, `"minmax"`, or `"members"`. |
| `print_summary` | bool | yes | `false` | Print score values to stdout. |
| `log_scale` | bool | yes | `false` | Logarithmic y-axis on summary line plots. |
| `add_grid` | bool | yes | `false` | Add background grid to summary line plots. |
| `agg_dims` | str\|list[str] | yes | `"ipoint"` | Advanced aggregation dims. Supported: `"ipoint"`, `"sample"`, `"forecast_step"`, `"ensemble"`. |

## `score_plots` values

| Value | Produces |
|-------|----------|
| `metric_plots` | Standard metric plot per region x stream x channel. |
| `ratio` | Ratio plots relative to `baseline`. |
| `heatmap` | Heat-map plots over lead time and channel. |
| `scorecard` | Score-card summaries. |
| `bar` | Bar plots of scores. |
| `score_map` | 2D spatial maps per forecast step (zarr only). |
| `score_animation` | Animations from score maps. |
| `init_hour` | Score vs initialization hour (zarr only). |

`psd` and `qq_analysis` require `metric_plots` to render their dedicated outputs.

Deprecated boolean flags are still read if `score_plots` is absent (with warnings):

- `summary_plots` -> `metric_plots`
- `ratio_plots` -> `ratio`
- `heat_maps` -> `heatmap`
- `score_cards` -> `scorecard`
- `bar_plots` -> `bar`
- `plot_score_maps` -> `score_map`
- `plot_score_animations` -> `score_animation`
- `plot_score_init_time_series` / `plot_score_init_timeseries` -> `init_hour`
