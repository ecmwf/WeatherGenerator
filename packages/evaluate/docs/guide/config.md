# Fast Evaluation Config Reference

This document is the reference for all configuration options accepted by
`uv run evaluate --config <your_config.yml>`.

A working template to copy and edit is `config/eval_config.yml`

---

## Table of Contents

1. [Overview: config layout](#1-overview-config-layout)
2. [Top-level keys](#2-top-level-keys)
3. [`global_plotting_options`](#3-global_plotting_options)
4. [`evaluation`](#4-evaluation)
5. [`default_streams`](#5-default_streams)
6. [`run_ids`](#6-run_ids)
   - [Common run keys](#61-common-run-keys)
   - [Run types](#62-run-types)
     - [Type: `zarr` (default)](#621-type-zarr-default)
     - [Type: `json`](#622-type-json)
     - [Type: `merge`](#623-type-merge)
     - [Type: `jsonmerge`](#624-type-jsonmerge)
     - [Type: `csv`](#625-type-csv)
7. [Stream config block](#7-stream-config-block)
   - [`evaluation` sub-block](#71-evaluation-sub-block)
   - [`plotting` sub-block](#72-plotting-sub-block)
   - [Regridding](#73-regridding)
   - [Climatology](#74-climatology)
8. [Metrics reference](#8-metrics-reference)
   - [Special output metrics](#special-output-metrics-psd-qq_analysis-rank_histogram)
9. [Regions reference](#9-regions-reference)
10. [Score caching (JSON files)](#10-score-caching-json-files)
11. [CSV format for pre-computed scores](#11-csv-format-for-pre-computed-scores)
12. [CLI overrides](#12-cli-overrides)

---

## 1. Overview: config layout

```
max_workers: ...                  # optional top-level cap on parallelism

global_plotting_options:          # optional — applied to all runs/streams
  ...

evaluation:                       # required — scoring and summary plot settings
  metrics: [...]
  regions: [...]
  ...

default_streams:                  # optional — stream config used when a run_id does not
  ERA5:                           #   specify its own streams
    channels: [...]
    evaluation: ...
    plotting: ...
  CERRA:
    ...

run_ids:                          # required — one entry per run to evaluate
  <run_id>:
    label: "..."
    results_base_dir: "..."
    # streams:  optional; if absent, default_streams is used
  ...
```

---

## 2. Top-level keys

| Key | Type | Optional | Default | Description |
|-----|------|----------|---------|-------------|
| `max_workers` | int | yes | — | Hard cap on parallel workers used for I/O, scoring, and plotting. Applied to all runs. Useful on shared nodes to avoid oversubscription. When absent, the number of workers is chosen automatically. |
| `private_paths` | dict | yes | — | HPC-specific private path overrides. Advanced option — only needed on certain clusters. See platform config docs. |

---

## 3. `global_plotting_options`

Applied to all runs. Stream-level blocks inside this section allow per-stream overrides
(e.g. colorscale limits). All keys are optional; see the annotated example at the end of this section.

### Global image / animation options

| Key | Type | Optional | Default | Description |
|-----|------|----------|---------|-------------|
| `regions` | list[str] | yes | `["global"]` | Regions for which 2D map plots are generated. See [section 9](#9-regions-reference) for supported values. |
| `image_format` | str | yes | `"png"` | File format for all saved images. Options: `"png"`, `"pdf"`, `"svg"`, `"eps"`, `"jpg"`. |
| `animation_format` | str | yes | `"gif"` | File format for animations. Options: `"gif"`, `"mp4"`. |
| `dpi_val` | int | yes | `300` | DPI for all saved images. |
| `fps` | int | yes | `2` | Frames per second for animations. |
| `n_bins` | int | yes | `50` | Number of bins used in histogram plots. |
| `log_x` | bool | yes | `false` | Log scale on the x-axis of histogram plots. |
| `log_y` | bool | yes | `false` | Log scale on the y-axis of histogram plots. |
| `fig_size` | [float, float] | yes | `null` | Figure size `[width, height]` in inches. When unset, matplotlib's default size is used for map/histogram plots; summary line plots use `[8, 10]`. |

### Per-stream appearance options (e.g. `ERA5:`)

Any stream name can appear as a key inside `global_plotting_options` to set stream-specific rendering defaults. Within a stream block, options can additionally be set per channel. The channel keys may be either an exact channel name (e.g. `2t`) or a glob pattern (e.g. `"q_*"`). 

| Key | Type | Optional | Default | Description |
|-----|------|----------|---------|-------------|
| `use_datashader` | bool | yes | `false` | Use [datashader](https://datashader.org/) for faster rasterised rendering of very dense scatter plots (recommended for grids with > 100k points, e.g. N320 or CERRA, where individual scatter points would overlap). Falls back to matplotlib scatter if not installed. |
| `marker_size` | float | yes | stream-dependent | Base scatter-plot marker size (matplotlib `s` units, i.e. pt²). Stream defaults: ERA5 → 2.5, IMERG → 0.25, CERRA → 0.1, others → 0.5. |
| `scale_marker_size` | bool | yes | `false` | Scale marker size by `1/cos²(lat)` to compensate for point clustering at high latitudes. |
| `marker` | str | yes | `"o"` | Matplotlib marker style. Common values: `"o"` (circle), `"s"` (square), `"."` (small dot), `"^"` (triangle up), `","` (pixel). See [matplotlib marker reference](https://matplotlib.org/stable/api/markers_api.html). |
| `alpha` | float | yes | — | Marker alpha (transparency), `0.0`–`1.0`. |
| `colormap` | str | yes | `"coolwarm"` | Matplotlib colormap name for 2D maps. Examples: `"viridis"`, `"RdBu_r"`, `"plasma"`. See [matplotlib colormaps](https://matplotlib.org/stable/gallery/color/colormap_reference.html). |
| `colors` | list[str] | yes | — | Explicit list of colors used to build a discrete `ListedColormap` (e.g. hex codes like "#BEDAE5"; `"none"` is allowed for transparency). Takes precedence over `colormap` and is typically combined with `levels`. Ignored for bias maps, which always use `coolwarm` for visual consistency. |
| `levels` | list[float] | yes | — | Explicit colorscale boundary values (e.g. `[-10, -5, 0, 5, 10]`). When set, a `BoundaryNorm` is applied and `vmin`/`vmax` are ignored. |
| `colorbar_scale` | str | yes | `"linear"` | Colorbar scaling: `"linear"`, `"log"`, or `"symlog"`. `"log"` is intended for single-sign skewed distributions (e.g. `tp`) and requires `vmin > 0`, otherwise it warns and falls back to `"linear"`. `"symlog"` handles skewed double-sign variables (e.g. `u`, `v`) and is default for non-linear bias plots since they are signed. Unknown values warn and fall back to `"linear"`. Explicit levels (non-bias plots) override this with a `BoundaryNorm`. |
| `add_healpix_grid` | bool | yes | `false` | Overlay a HEALPix grid on map plots. |
| `healpix_nside` | int | yes | `4` | HEALPix `nside` controlling grid resolution. Higher values produce a finer grid. |
| `healpix_color` | str | yes | `"black"` | Colour of the HEALPix grid lines. |
| `healpix_linewidth` | float | yes | `0.2` | Width of the HEALPix grid lines in points. |
| `healpix_linestyle` | str | yes | `"-"` | Line style of the HEALPix grid lines (e.g. `"-"`, `"--"`). |
| `healpix_step` | int | yes | `64` | Number of interpolation points per pixel boundary edge. Higher values produce smoother curves. |

Additional keys under a stream (or per-channel block) are forwarded to [matplotlib.axes.Axes.scatter](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html).
This can be useful for style tuning beyond the built-in options above.

Example:

```yaml
global_plotting_options:
  ERA5:
    # built-in keys parsed explicitly
    marker: "o"
    marker_size: 2.0
    alpha: 0.85

    # passed through to scatter (parsed["extra"])
    edgecolors: "none"
    zorder: 3
    rasterized: true

    # per-channel override + extra scatter kwargs
    2t:
      vmin: 250
      vmax: 305
      alpha: 0.9
      edgecolors: "black"
      linewidths: 0.05

    # per-channel override with glob pattern
    "q_*":
      levels: [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.015, 0.03]
```

Pass-through examples here are: `edgecolors`, `linewidths`, `zorder`, `alpha`.
If a key conflicts with an internally managed argument (e.g. `c`, `norm`, `cmap`, `s`, `marker`,
`transform`), the internal value will be used.

### Per-channel colorscale limits (e.g. `2t:`)

Under any per-stream block you can add entries keyed by channel name to set fixed colorscale
limits for 2D maps.

| Key | Type | Optional | Default | Description |
|-----|------|----------|---------|-------------|
| `vmin` | float | yes | — | Minimum value of the colorscale. When unset, matplotlib auto-scales. |
| `vmax` | float | yes | — | Maximum value of the colorscale. When unset, matplotlib auto-scales. |

### Annotated example

```yaml
global_plotting_options:
  regions: ["global", "europe"]
  image_format: "png"
  animation_format: "gif"
  log_colorbar: false
  dpi_val: 300
  fps: 2
  n_bins: 50
  log_x: false
  log_y: false
  ERA5:
    use_datashader: false
    marker_size: 2.0
    scale_marker_size: true
    marker: "o"
    alpha: 0.5
    colormap: "coolwarm"
    add_healpix_grid: false
    healpix_nside: 4
    healpix_color: "black"
    healpix_linewidth: 0.2
    healpix_linestyle: "-"
    healpix_step: 64
    2t:
      vmin: 250
      vmax: 300
      colormap: "RdBu_r"
    10u:
      vmin: -40
      vmax: 40
```

---

## 4. `evaluation`

Controls what to compute and how to visualise summary scores.

```yaml
evaluation:
  metrics: ["rmse", "mae"]
  regions: ["global", "nhem"]
  summary_plots: true
  ratio_plots: false
  heat_maps: false
  score_cards: false
  bar_plots: false
  summary_dir: "./plots/"
  plot_ensemble: "members"
  plot_score_maps: false
  plot_score_animations: false
  plot_score_init_time_series: false
  print_summary: false
  log_scale: false
  add_grid: false
  baseline: "my_run_id"
  # agg_dims: ["ipoint"]
```

| Key | Type | Optional | Default | Description |
|-----|------|----------|---------|-------------|
| `metrics` | list | no | — | Metrics to compute. Each item is either a metric name string or a single-key dict `{name: {param: value}}` for parametrised metrics. See [section 8](#8-metrics-reference). |
| `regions` | list[str] | yes | `["global"]` | Regions over which scores are computed. Overrides any region set on individual streams. See [section 9](#9-regions-reference). |
| `summary_dir` | str | yes | `<repo_root>/plots/` | Output directory for all summary (line/ratio/heatmap/etc.) plots. |
| `summary_plots` | bool | yes | `false` | Generate line plots of score vs forecast step, one per metric × region × stream × channel. |
| `ratio_plots` | bool | yes | `false` | Generate ratio plots (score relative to baseline). Requires `baseline` to be set. |
| `heat_maps` | bool | yes | `false` | Generate heat-map plots (score as a function of lead-time and channel). |
| `score_cards` | bool | yes | `false` | Generate score-card summary plots. |
| `bar_plots` | bool | yes | `false` | Generate bar plots of scores. |
| `baseline` | str | yes | — | `run_id` to use as the reference for ratio and improvement calculations. |
| `plot_ensemble` | str\|bool | yes | `false` | How to render ensemble spread on summary line plots. Options: `false` (no spread), `"std"` (mean ± std), `"minmax"` (shaded min–max), `"members"` (individual member lines). |
| `plot_score_maps` | bool | yes | `false` | Plot 2D spatial maps of scores per forecast step. **Slows down evaluation significantly.** |
| `plot_score_animations` | bool | yes | `false` | Animate score maps across forecast steps. Implies `plot_score_maps` must have data. |
| `plot_score_init_time_series` | bool | yes | `false` | Plot score timeseries grouped by initialisation hour of the day. |
| `print_summary` | bool | yes | `false` | Print score values to stdout. Can be very verbose for large runs. |
| `log_scale` | bool | yes | `false` | Use logarithmic y-axis on summary line plots. |
| `add_grid` | bool | yes | `false` | Add a background grid to summary line plots. |
| `agg_dims` | str\|list[str] | yes | `"ipoint"` | **Advanced.** Dimension(s) to aggregate (average) scores over. Supported values: `"ipoint"`, `"sample"`, `"forecast_step"`, `"ensemble"`. Default averages over spatial points only. Use with caution — averaging over sample or forecast_step hides temporal structure. |

---

## 5. `default_streams`

Defines the stream configuration used by any `run_id` that does not specify its own `streams`
block. The structure is identical to the per-run `streams` block described in [section 7](#7-stream-config-block).

```yaml
default_streams:
  ERA5:
    regions: ["global"]
    channels: ["2t", "10u", "z_500", "t_850"]
    regrid: true
    evaluation:
      forecast_step: "all"
      sample: "all"
      ensemble: "all"
    plotting:
      sample: [0, 1]
      forecast_step: [1, 2, 4, 8]
      ensemble: [0]
      plot_maps: true
      plot_bias: false
      plot_target: false
      plot_histograms: true
      plot_animations: false
  CERRA:
    regions: ["europe"]
    channels: ["z_500", "t_850", "u_850"]
    evaluation:
      forecast_step: "all"
      sample: "all"
    plotting:
      sample: [0]
      forecast_step: "all"
      plot_maps: true
      plot_histograms: true
      plot_animations: false
```

If a `run_id` does not define its own `streams` block, `default_streams` is used as-is. When a run
does include a `streams` block, `default_streams` is **completely ignored** for that run — all
required streams must be specified explicitly in the run's `streams` block.

See [section 7](#7-stream-config-block) for a full description of all keys.

---

## 6. `run_ids`

Each key under `run_ids` is a run identifier. The value is a configuration dict whose
required and optional keys depend on the `type` field.

### 6.1 Common run keys

These apply to all run types.

| Key | Type | Optional | Default | Description |
|-----|------|----------|---------|-------------|
| `label` | str | yes | run_id | Human-readable label used in plot legends. |
| `color` | str | yes | auto | Matplotlib colour string for this run in line/bar plots (e.g. `"magenta"`, `"#2ca02c"`). When absent, colours are assigned automatically. |
| `type` | str | yes | `"zarr"` | Reader type. Options: `"zarr"`, `"json"`, `"merge"`, `"jsonmerge"`, `"csv"`. |
| `streams` | dict | yes | — | Stream-specific config for this run. If absent, `default_streams` is used. When present, **`default_streams` is completely ignored for this run** — specify all required streams explicitly. |

### 6.2 Run types

The `type` field selects which reader is used. All five types share the common keys from [section 6.1](#61-common-run-keys).

#### 6.2.1 Type: `zarr` (default)

Standard run reading directly from WeatherGenerator Zarr output.

```yaml
run_ids:
  ar40mckx:
    label: "My run"
    results_base_dir: "./results/"
    mini_epoch: 0
    rank: "all"
    # streams:  optional; uses default_streams if omitted
```

| Key | Type | Optional | Default | Description |
|-----|------|----------|---------|-------------|
| `results_base_dir` | str | yes | private config | Base directory containing Zarr output folders. Required if `private_paths` is not set. |
| `runplot_base_dir` | str | yes | `results_base_dir` | Base directory for 2D map and histogram plots. |
| `metrics_base_dir` | str | yes | `results_base_dir` | Base directory for cached score JSON files. |
| `metrics_dir` | str | yes | `metrics_base_dir/evaluation` | Explicit path for score JSON files. Overrides `metrics_base_dir` if set. |
| `model_base_dir` | str | yes | — | Directory containing the model config files (used when `private_paths` is not set). |
| `mini_epoch` | int | yes | `0` | Epoch number used to identify the Zarr store. In inference this is always `0`. |
| `rank` | int\|str\|list | yes | `"all"` | Rank(s) of the Zarr store to read. Use `"all"` for multi-rank inference, an integer for a single rank, or a list of integers. |

#### 6.2.2 Type: `json`

Reads pre-computed scores from JSON files (no Zarr data required). Useful when the original
Zarr output has been deleted or is unavailable.

```yaml
run_ids:
  so67dku1:
    type: "json"                # <-------
    label: "Archived run"
    results_base_dir: "./results/"
    streams:
      ERA5:
        channels: ["z_500", "t_850"]
        evaluation:
          forecast_step: [2, 4, 6]
          sample: [0, 1, 2]
          ensemble: "all"
```

Uses the same path keys as the `zarr` type (`results_base_dir`, `metrics_dir`, etc.).
Plotting (maps, histograms, animations) is **not** available with this type.

#### 6.2.3 Type: `merge`

Stacks multiple Zarr runs over the ensemble dimension. Useful for creating a pseudo-ensemble
from several independent runs.

```yaml
run_ids:
  merge_test:
    type: "merge"               # <-------
    merge_run_ids:
      - so67dku4
      - c9cg8ql3
    merge_metrics_dir: "./merge_test/metrics/"
    label: "Merged ensemble"
    results_base_dir: "./results/"
    streams:
      ERA5:
        channels: ["z_500", "t_850"]
        evaluation:
          forecast_step: [2, 4, 6]
          sample: [0, 1, 2, 3]
          ensemble: "all"
```

| Key | Type | Optional | Default | Description |
|-----|------|----------|---------|-------------|
| `merge_run_ids` | list[str] | no | — | List of existing run_ids to merge. Each must be readable with the `zarr` reader. |
| `merge_metrics_dir` | str | no | — | Directory where merged score JSON files will be written and cached. |

#### 6.2.4 Type: `jsonmerge`

Same as `merge` but reads from pre-computed JSON score files instead of Zarr data.

```yaml
run_ids:
  merge_archived:
    type: "jsonmerge"           # <-------
    merge_run_ids:
      - so67dku4
      - c9cg8ql3
    merge_metrics_dir: "./merge_archived/metrics/"
    label: "Merged archived"
    results_base_dir: "./results/"
    streams:
      ERA5:
        channels: ["z_500", "t_850"]
        evaluation:
          forecast_step: [2, 4, 6]
          sample: [0, 1, 2, 3]
          ensemble: "all"
```

Same required keys as `merge`.

#### 6.2.5 Type: `csv`

Reads pre-computed scores from CSV files generated by external tools (e.g. ECMWF Quaver).
Only score line plots are produced; no maps, histograms, or animations.

```yaml
run_ids:
  pangu:
    type: "csv"                 # <-------
    label: "Pangu-Weather"
    metrics_dir: "<path to folder containing run_id sub-folder>"
    streams:
      ERA5:
        channels: ["2t", "q_850", "t_850", "z_500"]
        evaluation:
          forecast_step: "all"
          sample: "all"
```

| Key | Type | Optional | Default | Description |
|-----|------|----------|---------|-------------|
| `metrics_dir` | str | no | — | Path to the folder containing a sub-folder named `<run_id>/` with CSV files. |

See [section 11](#11-csv-format-for-pre-computed-scores) for the expected CSV column layout.

---

## 7. Stream config block

The same structure is used under `default_streams`, under `run_ids.<id>.streams`, and
inside `merge` type runs.

```yaml
ERA5:                                 # stream name
  regions: ["global", "nhem"]        # regions for maps (overrides global_plotting_options)
  channels: ["2t", "10u", "z_500"]
  offset: "1h"                       # optional
  regrid: true                       # optional
  climatology_path: "/path/..."      # optional
  evaluation:
    forecast_step: "all"
    sample: "all"
    ensemble: "all"
  plotting:
    forecast_step: [1, 2, 4, 8]
    sample: [0, 1]
    ensemble: [0]
    plot_maps: true
    plot_bias: false
    plot_target: false
    plot_histograms: true
    plot_animations: false
```

| Key | Type | Optional | Default | Description |
|-----|------|----------|---------|-------------|
| `regions` | list[str] | yes | `["global"]` | Regions for 2D maps for this stream. Overrides `global_plotting_options.regions`. |
| `channels` | list[str] | yes | all available | List of channel names to process (e.g. `["2t", "10u", "z_500"]`). |
| `offset` | str | yes | — | Timedelta offset used to infer initialisation time when `source_interval` is absent. Format examples: `"1h"`, `"30m"`, `"2h30m"`. |
| `regrid` | bool\|dict | yes | `false` | Re-project data to a regular lat/lon grid before scoring and plotting. `true` is shorthand for `{target_grid: [1.5, 1.5]}`. See [section 7.3](#73-regridding) for full dict options and caveats. |
| `climatology_path` | str | yes | auto | Explicit path to a climatology Zarr file required for `acc`, `rps`, `rpss`, `fact`, and `tact`. When absent, the code attempts auto-detection. See [section 7.4](#74-climatology). |

### 7.1 `evaluation` sub-block

Controls which data are loaded and scored.

| Key | Type | Optional | Default | Description |
|-----|------|----------|---------|-------------|
| `forecast_step` | str\|list[int] | yes | `"all"` | Forecast steps to score. `"all"` uses every available step. A list `[1, 2, 4]` selects specific steps. A range string `"1-50"` is equivalent to `[1, 2, ..., 50]`. |
| `sample` | str\|list[int] | yes | `"all"` | Samples (initialisation times) to score. `"all"` or a list of integers. |
| `ensemble` | str\|list[int] | yes | `"all"` | Ensemble members to include in scoring. `"all"` uses every member; `"mean"` uses the ensemble mean; a list `[0, 1, 2]` selects specific members. |

### 7.2 `plotting` sub-block

Controls which subset of the evaluated data is visualised with maps, histograms, and animations.
The `evaluation` and `plotting` sub-blocks are intentionally separate: you can score over a broad
set of forecast steps and samples while only generating plots for a representative subset, saving
both time and disk space.

| Key | Type | Optional | Default | Description |
|-----|------|----------|---------|-------------|
| `forecast_step` | str\|list[int] | yes | `"all"` | Forecast steps for which plots are created. Same syntax as `evaluation.forecast_step`. |
| `sample` | list[int] | yes | `"all"` | Samples for which plots are created. |
| `ensemble` | str\|list[int] | yes | `"all"` | Ensemble members for which maps/histograms are created. Same syntax as `evaluation.ensemble`. |
| `plot_maps` | bool | yes | `false` | Plot a 2D scatter map for each channel, valid time, and selected sample/ensemble member. |
| `plot_bias` | bool | yes | `true` | Plot the bias (prediction − target) as a 2D map alongside the prediction map. |
| `plot_target` | bool | yes | `true` | Also plot the target (ground truth) data using the same plotting options. |
| `plot_histograms` | bool\|str | yes | `false` | Plot histograms of target vs prediction. `true` or `"per-sample"` creates one histogram per sample; `"across-samples"` aggregates all samples into a single histogram. |
| `plot_animations` | bool | yes | `false` | Build an animation (GIF/MP4) cycling through forecast steps for each channel and sample. |

### 7.3 Regridding

When `regrid` is set, data is re-projected from the native model grid (e.g. an octahedral reduced
Gaussian grid) to a regular lat/lon grid before scoring and plotting, using
[earthkit.regrid](https://earthkit-regrid.readthedocs.io/).

```yaml
ERA5:
  regrid: true                            # shorthand: target_grid defaults to [1.5, 1.5]
  # or, for explicit control:
  regrid:
    target_grid: [1.5, 1.5]              # [lat_deg, lon_deg] for a regular lat/lon grid
    # target_grid: "O96"                 # or a named Gaussian grid
    # original_grid: "O96"              # optional: source grid if auto-detection fails
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `target_grid` | [float, float] \| str | `[1.5, 1.5]` | Target grid. A two-element list `[lat_deg, lon_deg]` for a regular lat/lon grid, or a named string such as `"O96"`. |
| `original_grid` | str | auto | Source grid name (e.g. `"O96"`). The code infers the grid from the number of spatial points; specify explicitly if auto-detection fails or produces a warning. |

> **When to regrid:** Useful when comparing runs at different native resolutions, or when using
> metrics that require a regular grid (e.g. `grad_amplitude`, PSD with `psd_method: "fft"`).
> Regridding adds I/O and interpolation overhead — for standard deterministic scoring it is
> often unnecessary.

> **Cache invalidation note:** Changing `target_grid` or `original_grid` does **not** automatically
> invalidate cached JSON score files. Delete the relevant JSON files before re-running if grid
> settings change (see [section 10](#10-score-caching-json-files)).

### 7.4 Climatology

A pre-computed climatology is required for the anomaly-based metrics `acc`, `rps`, `rpss`,
`fact`, and `tact`. The climatology must be a Zarr file with matching spatial grid and channel
names.

**Auto-detection:** If `climatology_path` is not set, the code locates the climatology using the
`data_path_aux` path from the model/inference config:

```
<data_path_aux>/climatology/<source_filename>_climatology.zarr
```

Auto-detection only works when the stream has a single source filename. If it fails, a warning
is logged and climatology-dependent metrics are skipped.

**Limitations:**
- For **gridded data** (regular lat/lon, after optional regridding): the climatology is
  interpolated to the model grid if resolutions differ.
- For **non-gridded / HEALPix data**: the climatology must already be on the same point layout
  as the model output; no spatial interpolation is applied.
- If the expected Zarr file does not exist, affected metrics return `NaN` without raising an error.

---

## 8. Metrics reference

Metrics are listed under `evaluation.metrics`. Each entry is either a plain string or a
single-key dict with a parameter sub-dict:

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

### Deterministic metrics (require prediction and target)

| Name | Description |
|------|-------------|
| `mae` | Mean Absolute Error |
| `mse` | Mean Squared Error |
| `rmse` | Root Mean Squared Error |
| `vrmse` | Variance-normalised RMSE |
| `l1` | L1 error norm |
| `l2` | L2 error norm |
| `bias` | Mean bias (prediction − target) |
| `psnr` | Peak Signal-to-Noise Ratio |
| `nse` | Nash–Sutcliffe Efficiency |
| `ets` | Equitable Threat Score. Default threshold per-variable (see `score.py`). Override with `thresh`. |
| `pss` | Peirce Skill Score. Override threshold with `thresh`. |
| `fbi` | Frequency Bias Index. Override threshold with `thresh`. |
| `grad_amplitude` | Ratio of spatial variability (gradient amplitude) between prediction and target. Requires a regular lat/lon grid. |
| `qq_analysis` | Quantile–quantile analysis. Produces Q-Q plots rather than line plots — see [special output metrics](#special-output-metrics-psd-qq_analysis-rank_histogram). |
| `psd` | Power Spectral Density. Produces PSD plots rather than line plots — see [special output metrics](#special-output-metrics-psd-qq_analysis-rank_histogram). |

### Metrics requiring alignment between consecutive forecast steps

> Cannot be used when coordinates change between steps (shuffled data is fine).

| Name | Description |
|------|-------------|
| `froct` | Forecast Rate of Change over Time |
| `troct` | Target Rate of Change over Time |

### Metrics requiring a pre-computed climatology

> Need either `climatology_path` in the stream config or a `data_path_aux` key in the model config.

| Name | Description |
|------|-------------|
| `acc` | Anomaly Correlation Coefficient |
| `rps` | Ranked Probability Score |
| `rpss` | Ranked Probability Skill Score |
| `fact` | Forecast Activity (standard deviation of forecast anomaly) |
| `tact` | Target Activity (standard deviation of target anomaly) |
| `seeps` | Stable Equitable Error in Probability Space ([Rodwell et al., 2011](https://journals.ametsoc.org/view/journals/mwre/140/8/mwr-d-11-00301.1.pdf)). Reported as the positively-oriented skill `1 − SEEPS_error` (**higher is better**; 1 = perfect). |

### Probabilistic metrics (require ensemble dimension)

| Name | Description |
|------|-------------|
| `ssr` | Spread–Skill Ratio |
| `crps` | Continuous Ranked Probability Score (via the [scores](https://scores.readthedocs.io/) package). Supports standard, fair, and threshold-weighted variants — see parameters below. |
| `rank_histogram` | Rank Histogram (Talagrand diagram). Produces a bar chart, not a score line plot — see [special output metrics](#special-output-metrics-psd-qq_analysis-rank_histogram). |
| `spread` | Ensemble Spread |

### Special output metrics: `psd`, `qq_analysis`, `rank_histogram` and `seeps`

The three metrics below do **not** produce standard score-vs-lead-time line plots. They are
handled by dedicated plotting functions and generate different output file types.

#### `psd` — Power Spectral Density

Computes the spatial power spectrum and plots variance as a function of total spherical harmonic
wavenumber (or zonal wavenumber for the FFT path). One curve per run/stream/channel is produced.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `psd_method` | `"sht"` | `"sht"` — Spherical Harmonic Transform; works on any separable grid (regular, octahedral, reduced Gaussian). `"fft"` — 1-D zonal FFT per latitude ring; **requires a regular lat/lon grid**. |
| `psd_regrid_resolution` | `1.0` | (FFT only) Resolution in degrees to regrid to before the FFT. |

```yaml
evaluation:
  metrics:
    - psd:
        psd_method: "sht"
    # or for the FFT path (requires regular grid or regrid: true):
    - psd:
        psd_method: "fft"
        psd_regrid_resolution: 1.0
```

#### `qq_analysis` — Quantile–Quantile Analysis

Computes and plots quantile–quantile curves comparing the distribution of predictions to targets.
Useful for detecting distributional biases (e.g. over-smoothing or wet-bias in precipitation).
One Q-Q curve per channel per run is produced. No additional parameters.

#### `rank_histogram` — Rank Histogram (Talagrand Diagram)

For each verification point the rank of the observation within the sorted ensemble is recorded.
A flat histogram indicates a well-calibrated ensemble; a U-shape indicates under-dispersion;
a dome shape indicates over-dispersion. Requires ensemble data (multi-member run or `merge`
type). No additional parameters.

### Metric parameters

Any metric that accepts a `thresh` parameter can be configured like:

```yaml
evaluation:
  metrics:
    - fbi:
        thresh: 280       # custom threshold (e.g. for 2t in Kelvin)
    - ets:
        thresh: 0.001     # custom threshold (e.g. for precipitation)
```

#### `crps` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | `"ecdf"` | CRPS variant. `"ecdf"` — standard CRPS from the empirical CDF. `"fair"` — debiased fair CRPS. `"tw_tail"` — threshold-weighted tail CRPS. `"tw_interval"` — threshold-weighted interval CRPS. |
| `fair` | `false` | If `true`, overrides `method` with `"fair"` (debiased CRPS). Shorthand for `method: "fair"`. |
| `threshold` | — | *(tw_tail only)* Threshold value. |
| `tail` | `"upper"` | *(tw_tail only)* Which tail to weight: `"upper"` or `"lower"`. |
| `lower_threshold` | — | *(tw_interval only)* Lower bound of the interval. |
| `upper_threshold` | — | *(tw_interval only)* Upper bound of the interval. |

```yaml
evaluation:
  metrics:
    - crps                           # standard CRPS (ecdf method)
    - crps:
        fair: true                   # fair/debiased CRPS
    - crps:
        method: "tw_tail"
        threshold: 0.1
        tail: "upper"                # weight upper tail (e.g. heavy precip)
    - crps:
        method: "tw_interval"
        lower_threshold: 0.0
        upper_threshold: 10.0
```

#### `seeps` score
The underlying [`scores`](https://scores.readthedocs.io/) implementation computes the Rodwell et al.
SEEPS **error** (negatively oriented, 0 = perfect). WeatherGenerator reports the
**positively-oriented** form `1 − SEEPS_error` (**higher is better**), matching the convention used
for reporting SEEPS at ECMWF (e.g. the [AIFS "it's raining data" blog](https://www.ecmwf.int/en/about/media-centre/aifs-blog/2024/its-raining-data)).
A perfect forecast scores 1, a climatology/no-skill forecast scores ~0, and values can be negative
where the forecast is worse than the penalty-matrix reference.

The `seeps` metric accepts two parameters:
```yaml
evaluation:
  metrics:
    - seeps:
        minimum_dry_prob: 0.1
        maximum_dry_prob: 0.85
```
Points where the climatological probability of a dry timestep (less than 0.2mm of rain) is below this minimum or above this maximum are excluded from the score computation. 
The default values given above are used in the literature and should typically not be changed. The other two parameters inherent in the method (the threshold for dryness, 
here 0.2mm and the conditional probability of heavy rain given a wet timestep, here 2/3) are baked into the climatological weights and cannot be changed by the user.  

---

## 9. Regions reference

Predefined bounding boxes `(lat_min, lat_max, lon_min, lon_max)`:

| Name | Lat range | Lon range | Notes |
|------|-----------|-----------|-------|
| `global` | −90 to 90 | −180 to 180 | Robinson projection for maps |
| `nhem` | 0 to 90 | −180 to 180 | Northern Hemisphere |
| `shem` | −90 to 0 | −180 to 180 | Southern Hemisphere |
| `tropics` | −30 to 30 | −180 to 180 | Tropical belt |
| `europe` | 35 to 70 | −10 to 40 | |
| `belgium` | 49 to 52 | 2 to 7 | |
| `arctic` | 50 to 90 | −180 to 180 | Stereographic projection for maps |
| `uwc-west` | 39 to 63 | −26 to 41 | UWC-West domain |
| `arome` | 37 to 56 | −12 to 16 | AROME domain |
| `icon` | 42 to 51 | −1 to 18 | ICON domain |

Regions are specified as lists of strings. They can appear in three places:

```yaml
# 1. global_plotting_options — affects map region selection for all streams
global_plotting_options:
  regions: ["global", "europe"]

# 2. evaluation — affects score calculation regions for all streams
evaluation:
  regions: ["global", "nhem", "tropics"]

# 3. per stream (highest precedence) — overrides global for a specific stream
default_streams:
  CERRA:
    regions: ["europe"]
```

**Precedence (highest to lowest):** per-stream `regions` → `evaluation.regions` → `global_plotting_options.regions`.
`evaluation.regions` controls which regions are **scored**; `global_plotting_options.regions`
controls which map extents are **plotted**. A per-stream `regions` key overrides both for that stream.

---

## 10. Score caching (JSON files)

To avoid recomputing scores on every run, results are saved to JSON files.
The path follows this pattern:

```
<metrics_dir>/<run_id>_<stream>_<region>_<metric>_chkpt<epoch:05d>.json
```

Where `metrics_dir` is resolved as follows (first match wins):

1. `run_ids.<id>.metrics_dir` (explicit path)
2. `run_ids.<id>.metrics_base_dir / "evaluation"`
3. `run_ids.<id>.results_base_dir / "evaluation"`
4. Platform-specific shared path from the private config

At runtime, the code checks whether a JSON file already exists for the requested
combination. If it does, the stored scores are loaded; otherwise they are computed and
saved for future use.

### Cache invalidation

The evaluation code automatically detects configuration changes and forces recomputation
when necessary. Cached scores are **automatically recomputed** when any of the following
change:
- `regrid`, `ensemble`, `rank`, `agg_dims`,

They are **recomputed only for the missing values** when the following parameters change:
- `derived_channels`, `forecast_steps`, `channels`, `sample`

**Manual cache invalidation required:**
The cache does **not** track changes to other parameters, so you must manually delete
the relevant JSON files, e.g. if any of these change:
- Climatology files
- Underlying Zarr data content (without forecast step changes)

---

## 11. CSV format for pre-computed scores

When using `type: "csv"`, CSV files must be placed under
`<metrics_dir>/<run_id>/` and follow this column layout:

```
,parameter,level,number,score,step,date,domain_name,value
0,t,925,0,rmse,0 days 12:00:00,2022-10-01 00:00:00,n.hem,0.031371
1,t,925,0,rmse,0 days 12:00:00,2022-10-01 12:00:00,n.hem,-0.01038
```

| Column | Description |
|--------|-------------|
| `parameter` | Variable short name (e.g. `t`, `z`, `u`) |
| `level` | Pressure level in hPa (e.g. `925`, `500`) |
| `number` | Ensemble member number (use `0` for deterministic) |
| `score` | Metric name in Quaver convention |
| `step` | Lead time as a timedelta string |
| `date` | Initialisation date-time |
| `domain_name` | Region name in Quaver convention (e.g. `n.hem`, `tropics`) |
| `value` | Score value |

Channel names are constructed as `<parameter>_<level>` (e.g. `t_925`).

---

## 12. CLI overrides

Individual config values can be overridden from the command line without editing the YAML:

```bash
uv run evaluate --config myconfig.yml \
  --options evaluation.summary_plots=true evaluation.regions=[global,nhem]
```

The `--options` flag uses OmegaConf dot-notation and does **not** support overriding
`run_ids` keys (use `--run-ids` for that):

```bash
# Restrict evaluation to a subset of run_ids
uv run evaluate --config myconfig.yml --run-ids ar40mckx c8g5katp
```

Upload scores to MLFlow after evaluation:

```bash
uv run evaluate --config myconfig.yml --push-metrics
```
