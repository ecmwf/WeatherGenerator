# global_plotting_options

Applied to all runs. Stream-level blocks inside this section allow per-stream overrides
(e.g. colorscale limits). All keys are optional.

## Global image / animation options

| Key | Type | Optional | Default | Description |
|-----|------|----------|---------|-------------|
| `regions` | list[str] | yes | `["global"]` | Regions for which 2D map plots are generated. See [Regions Reference](regions.md). |
| `image_format` | str | yes | `"png"` | File format for all saved images. Options: `"png"`, `"pdf"`, `"svg"`, `"eps"`, `"jpg"`. |
| `animation_format` | str | yes | `"mp4"` | File format for animations. Options: `"mp4"` (full truecolor), `"gif"` (256-color palette, some quality loss). |
| `log_colorbar` | bool | yes | `false` | Use a logarithmic colorscale on 2D map plots. |
| `dpi_val` | int | yes | `300` | DPI for all saved images. |
| `fps` | int | yes | `2` | Frames per second for animations. |
| `n_bins` | int | yes | `50` | Number of bins used in histogram plots. |
| `log_x` | bool | yes | `false` | Log scale on the x-axis of histogram plots. |
| `log_y` | bool | yes | `false` | Log scale on the y-axis of histogram plots. |
| `fig_size` | [float, float] | yes | `null` | Figure size `[width, height]` in inches. When unset, matplotlib's default size is used for map/histogram plots; summary line plots use `[8, 10]`. |

## Per-stream appearance options (e.g. ERA5)

Any stream name can appear as a key inside `global_plotting_options` to set stream-specific rendering defaults. Within a stream block, options can additionally be set per channel. Channel keys may be either an exact channel name (e.g. `2t`) or a glob pattern (e.g. `"q_*"`).

| Key | Type | Optional | Default | Description |
|-----|------|----------|---------|-------------|
| `use_datashader` | bool | yes | `false` | Use [datashader](https://datashader.org/) for faster rasterized rendering of dense scatter plots. Falls back to matplotlib scatter if not installed. |
| `marker_size` | float | yes | stream-dependent | Base scatter-plot marker size (matplotlib `s` units, pt^2). Stream defaults: ERA5 -> 2.5, IMERG -> 0.25, CERRA -> 0.1, others -> 0.5. |
| `scale_marker_size` | bool | yes | `false` | Scale marker size by `1/cos^2(lat)` to compensate for clustering at high latitudes. |
| `marker` | str | yes | `"o"` | Matplotlib marker style. Common values: `"o"`, `"s"`, `"."`, `"^"`, `","`. |
| `alpha` | float | yes | - | Marker alpha (transparency), `0.0`-`1.0`. |
| `colormap` | str | yes | `"coolwarm"` | Matplotlib colormap name for 2D maps. |
| `colors` | list[str] | yes | - | Explicit list of colors used to build a discrete `ListedColormap`. |
| `levels` | list[float] | yes | - | Explicit colorscale boundary values. When set, `BoundaryNorm` is applied and `vmin`/`vmax` are ignored. |
| `colorbar_scale` | str | yes | `"linear"` | Colorbar scaling: `"linear"`, `"log"`, or `"symlog"`. |
| `add_healpix_grid` | bool | yes | `false` | Overlay a HEALPix grid on map plots. |
| `healpix_nside` | int | yes | `4` | HEALPix `nside` controlling grid resolution. |
| `healpix_color` | str | yes | `"black"` | Colour of the HEALPix grid lines. |
| `healpix_linewidth` | float | yes | `0.2` | Width of the HEALPix grid lines in points. |
| `healpix_linestyle` | str | yes | `"-"` | Line style of the HEALPix grid lines. |
| `healpix_step` | int | yes | `64` | Number of interpolation points per pixel boundary edge. |

Additional keys under a stream (or per-channel block) are forwarded to [matplotlib.axes.Axes.scatter](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html).

## Per-channel colorscale limits (e.g. 2t)

| Key | Type | Optional | Default | Description |
|-----|------|----------|---------|-------------|
| `vmin` | float | yes | - | Minimum value of the colorscale. |
| `vmax` | float | yes | - | Maximum value of the colorscale. |

## Example

```yaml
global_plotting_options:
  regions: ["global", "europe"]
  image_format: "png"
  animation_format: "mp4"
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
