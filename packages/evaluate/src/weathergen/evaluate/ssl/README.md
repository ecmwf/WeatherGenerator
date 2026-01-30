# Multi-Stream Self Supervised Learning Inference & Evaluation

This script runs **inference and evaluation for a multi-stream weather generation model** and produces quantitative metrics and diagnostic plots for each stream.

The workflow is designed to:

1. Run inference from an existing trained model run
2. Evaluate forecasts across multiple data streams
3. Print training, validation, and evaluation summaries

---

## Supported Streams

The script currently evaluates the following streams:

```python
streams = ["ERA5", "SurfaceCombined", "NPPATMS"]
```

Each stream has its own channels, metrics, and plotting configuration.

---

## Overview of the Pipeline

### 1. Inference (`infer_multi_stream`)

* Runs inference using a trained model (`run_id`)
* Creates a new run ID with `_inf` suffix
* Generates forecasts over a fixed date range
* Outputs results in Zarr format

**Inference period**

* Start: `2021-10-10`
* End: `2022-10-11`
* Samples per forecast: `10`

---

### 2. Evaluation Configuration (`get_evaluation_config`)

Defines:

* Metrics: `rmse`, `froct`
* Regions: `global`
* Plotting options (maps, histograms, summary plots)
* Stream-specific channels and evaluation settings

Each stream is evaluated independently but summarized jointly.

---

### 3. Evaluation (`evaluate_multi_stream_results`)

* Loads inference outputs
* Computes metrics per stream, region, and forecast step
* Saves plots and summaries under:

```
./results/<run_id>_inf/plots/summary/
```

---

### 4. Reporting Utilities

#### Print Losses (`print_losses`)

Prints training or validation losses for each stream:

* Uses `LossPhysical.<stream>.mse.avg`
* Supports `train` and `val` stages

#### Print Evaluation Results (`print_evaluation_results`)

Prints mean evaluation scores averaged over:

* samples
* forecast steps
* ensemble members

Results are grouped by:

* stream
* metric
* region

---

## File Structure Expectations

The script assumes the following directory layout:

```
.
├── models/
│   └── <run_id>
│       ├── <run_id>_latest.chkpt  
│       ├── model_<run_id>.json
├── results/
│   └── <run_id>_inf/
│       ├── metrics.json
│       └── plots/

```

---

## Running the Script

### Command Line

```bash
uv run --offline ssl_analysis --run-id <TRAIN_RUN_ID>
```

Optional verbose mode:

```bash
uv run --offline ssl_analysis --run_id <TRAIN_RUN_ID> --run-id <TRAIN_RUN_ID> --verbose
```

---

## Execution Flow (Main)

When executed, the script performs the following steps:

1. **Inference**

   * Creates a new inference run ID
   * Generates forecasts from the trained run

2. **Evaluation**

   * Computes metrics and generates plots

3. **Reporting**

   * Prints training losses from the original run
   * Prints validation losses from the inference run
   * Prints evaluation metrics for all streams

---

## Metrics & Outputs

### Metrics (for now)

* RMSE
* FROCT

### Outputs

* Summary plots
* Per-stream maps and histograms
* Console summaries of losses and evaluation scores

---

## Notes & Assumptions

* The original training run must already exist under `./results/`
* The `_inf` suffix is currently used for inference run naming
* Forecast steps and samples default to `"all"` during evaluation
