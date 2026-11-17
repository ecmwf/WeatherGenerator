# Regions Reference

Predefined bounding boxes `(lat_min, lat_max, lon_min, lon_max)`:

| Name | Lat range | Lon range | Notes |
|------|-----------|-----------|-------|
| `global` | -90 to 90 | -180 to 180 | Robinson projection for maps |
| `nhem` | 0 to 90 | -180 to 180 | Northern Hemisphere |
| `shem` | -90 to 0 | -180 to 180 | Southern Hemisphere |
| `tropics` | -30 to 30 | -180 to 180 | Tropical belt |
| `europe` | 35 to 70 | -10 to 40 | |
| `belgium` | 49 to 52 | 2 to 7 | |
| `arctic` | 50 to 90 | -180 to 180 | Stereographic projection for maps |
| `uwc-west` | 39 to 63 | -26 to 41 | UWC-West domain |
| `arome` | 37 to 56 | -12 to 16 | AROME domain |
| `icon` | 42 to 51 | -1 to 18 | ICON domain |

Regions can appear in three places:

```yaml
# 1. global_plotting_options
global_plotting_options:
  regions: ["global", "europe"]

# 2. evaluation
evaluation:
  regions: ["global", "nhem", "tropics"]

# 3. per stream
default_streams:
  CERRA:
    regions: ["europe"]
```

Precedence (highest to lowest): per-stream `regions` -> `evaluation.regions` -> `global_plotting_options.regions`.
