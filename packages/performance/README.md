# WeatherGenerator Performance Analysis Tools

This package contains tools for extracting and analyzing scaling performance data from WeatherGenerator training runs.

## Installation

Install the optional performance tools:

```bash
uv sync --extra performance
```

## Scripts

### extract_scaling_data.py

Extracts scaling metrics from WeatherGenerator training runs and writes parquet output.

```bash
extract_scaling_data --run-ids RUN_ID1 RUN_ID2 --output scaling.parquet
```

### generate_scaling_plots.py

Generates scaling plots and tables from parquet/NDJSON data using named columns from the input files.

```bash
generate_scaling_plots standard --input scaling.parquet --type strong --y-scale log
```

## Suggested workflow

1. Extract the scaling data into a parquet file (on your HPC).
2. Copy the parquet file to your local machine.
3. Generate plots from the parquet file.

Example:

```bash
extract_scaling_data --run-ids RUN_ID1 RUN_ID2 --output scaling.parquet
scp user@remote:/path/to/scaling.parquet .
generate_scaling_plots standard --input scaling.parquet --type strong --y-scale log
```
