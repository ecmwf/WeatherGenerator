# Timing Metrics Task

**Status:** Completed

**Created:** 2026-04-17

**Related Skills:** metrics, planning

## Goal

Add timing metrics to track startup time, training time, and overall execution time for the WeatherGenerator training pipeline.

## Progress

- [x] Step 1: Codebase analysis
- [x] Step 2: Design timing approach
- [x] Step 3: Implement timing in `run_train.py`
- [x] Step 4: Implement timing in `trainer.py`
- [x] Step 5: Verify metrics logging
- [x] Step 6: Document changes

## Completed Steps

### Step 1: Codebase Analysis

**Objective:** Understand training entry points and metrics infrastructure

**Files Reviewed:**
- `src/weathergen/run_train.py` - Main training entry point
- `src/weathergen/train/trainer.py` - Training loop logic
- `src/weathergen/utils/train_logger.py` - Metrics logging utility
- `hpc/mlflow_upload.py` - MLflow upload pipeline

**Key Findings:**
- Training starts in `run_train()` → `Trainer.run()`
- Metrics logged via `train_logger.log_metrics()`
- MLflow upload filters blacklisted keys automatically
- Multi-node runs require `is_root()` checks

### Step 2: Design Timing Approach

**Objective:** Define where and how to add timing metrics

**Decisions:**
1. **Three metrics:**
   - `startup_time_seconds`: Code launch → training start
   - `total_training_time_seconds`: Time in training loop
   - `overall_time_seconds`: Total wall-clock time

2. **Timing points:**
   - `run_train()`: Overall time (entry/exit)
   - `Trainer.run()`: Startup + training time
   - Root rank only logging

3. **Format:** JSONL compatible with existing MLflow pipeline

### Step 3: Implement in `run_train.py`

**Objective:** Add overall timing in main entry point

**Changes:**
- Added `t_overall_start` at start of `run_train()`
- Added `t_overall_end` in `finally` block
- Logged `overall_time_seconds` metric

**Files Modified:**
- `src/weathergen/run_train.py:23` - Added `t_overall_start`
- `src/weathergen/run_train.py:125` - Added timing in `run_continue()`
- `src/weathergen/run_train.py:145` - Added `finally` block with logging

### Step 4: Implement in `trainer.py`

**Objective:** Add startup and training time metrics

**Changes:**
- Added `t_run_start` at start of `Trainer.run()`
- Added `t_training_start` before training loop
- Added `t_training_end` after training loop
- Logged `startup_time_seconds` and `total_training_time_seconds`

**Files Modified:**
- `src/weathergen/train/trainer.py:100` - Added `t_run_start`
- `src/weathergen/train/trainer.py:150` - Added `t_training_start`
- `src/weathergen/train/trainer.py:230` - Added `t_training_end`
- `src/weathergen/train/trainer.py:235` - Added metric logging

### Step 5: Verification

**Objective:** Ensure metrics are logged correctly

**Commands:**
```bash
# Run training with metrics
python -m weathergen.run_train --config config.yaml

# Check metrics.json
cat logs/metrics.json | grep timing

# Verify MLflow upload
python hpc/mlflow_upload.py --dry-run
```

**Expected Output:**
```json
{"stage": "train", "startup_time_seconds": 45.23, "weathergen.step": 0}
{"stage": "train", "total_training_time_seconds": 3600.12, "weathergen.step": 100}
{"stage": "train", "overall_time_seconds": 3650.45, "weathergen.step": 100}
```

### Step 6: Documentation

**Objective:** Document implementation for future reference

**Files Created:**
- `TIMING_METRICS_ANALYSIS.md` - Initial codebase analysis
- `TIMING_METRICS_IMPLEMENTATION.md` - Implementation details
- `.hermes/skills/metrics/SKILL.md` - Reusable metrics skill
- `.hermes/tasks/2026-04-17-timing-metrics/README.md` - This file

## Implementation Summary

**Total Changes:** 41 lines across 2 files

**Modified Files:**
1. `src/weathergen/run_train.py` - Overall timing
2. `src/weathergen/train/trainer.py` - Startup + training timing

**Git Commit:**
```
feat: add timing metrics for startup, training, and overall time

- Added overall_time_seconds in run_train()
- Added startup_time_seconds and total_training_time_seconds in Trainer.run()
- All metrics logged via root rank only
- Compatible with existing MLflow upload pipeline
```

## Lessons Learned

1. **Root rank logging is critical** - Multi-node HPC runs would create file contention without `is_root()` checks
2. **Use `finally` blocks** - Ensures metrics are logged even on failure
3. **Keep metrics simple** - JSONL format works seamlessly with MLflow
4. **Document timing points** - Clear comments explain what each metric measures

## Next Steps

- Monitor metrics in MLflow dashboard
- Add per-epoch timing if needed
- Consider adding data loading time metric
- Track DDP synchronization overhead

## Links

- [Implementation PR](TODO)
- [Metrics Skill](../../../.hermes/skills/metrics/SKILL.md)
- [MLflow Upload Code](../../../hpc/mlflow_upload.py)
