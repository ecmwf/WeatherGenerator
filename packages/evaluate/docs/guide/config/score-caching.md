# Score Caching (JSON files)

Scores are cached to avoid recomputing on every run.

Path pattern:

```text
<metrics_dir>/<run_id>_<stream>_<region>_<metric>_chkpt<epoch:05d>.json
```

`metrics_dir` resolution order:

1. `run_ids.<id>.metrics_dir`
2. `run_ids.<id>.metrics_base_dir / "evaluation"`
3. `run_ids.<id>.results_base_dir / "evaluation"`
4. Platform-specific shared path from private config

If JSON exists for a requested combination, cached scores are loaded.

## Cache invalidation

Automatic recomputation when these change:

- `regrid`
- `ensemble`
- `rank`
- `agg_dims`

Partial recomputation (missing values only) when these change:

- `derived_channels`
- `forecast_steps`
- `channels`
- `sample`

Manual cache invalidation is still required for other changes, e.g.:

- Climatology files
- Underlying Zarr data changes without forecast-step changes
