# utils/ — CLI, logging, helpers

- `cli.py` — CLI entry points and run stages (`Stage` enum). Config handling itself lives in `packages/common`, not here.
- `distributed.py` — process-group helpers (`is_root`, ...).
- `train_logger.py`, `metrics.py` — metrics writing/reading (polars); `plot_training.py`, `compare_run_configs.py` — run inspection tools.
- `performance.py` — `ThroughputTracker`; `validation_io.py` — validation output writing.
