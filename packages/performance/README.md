# WeatherGenerator Performance Analysis Tools

This package contains tools for extracting and analyzing scaling performance data from WeatherGenerator training runs.

## Scripts

### extract_scaling_data.py

Extracts strong scaling metrics from WeatherGenerator training runs.

```bash
extract_scaling_data --logs-dir /path/to/logs --work-dir /path/to/work
```

### generate_scaling_plots.py

Generates scaling plots and tables from parquet/NDJSON data.

```bash
# Standard mode (single type)
generate_scaling_plots standard --type strong --input data.parquet

# Combined mode (separate files)
generate_scaling_plots combined \
  --strong-input strong.parquet \
  --weak-input weak.parquet

# Combined mode (single file with both types)
generate_scaling_plots standard --type strong,weak --input data.parquet
```

## Installation

This package is part of the WeatherGenerator workspace. To install:

```bash
# In the root WeatherGenerator directory
uv sync --extra performance
```

The scripts will be available as console scripts:
- `extract_scaling_data`
- `generate_scaling_plots`
