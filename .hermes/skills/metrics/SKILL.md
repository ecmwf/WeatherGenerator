# Metrics Skill

Use this skill when adding metrics, logging, or monitoring to the codebase.

## When to Use

- Adding new metrics to track performance
- Implementing logging for debugging
- Integrating with MLflow or other experiment trackers
- Adding timing or profiling instrumentation

## Types of Metrics

### 1. Timing Metrics
Track execution time for:
- Startup/init phases
- Training/inference loops
- Overall run duration
- Individual operations

### 2. Performance Metrics
Track:
- Loss values
- Accuracy/precision/recall
- Throughput (samples/sec)
- Resource usage (GPU memory, CPU)

### 3. System Metrics
Track:
- DDP synchronization times
- Data loading times
- Checkpoint save/load times

## Implementation Pattern

### Step 1: Define Metric

Decide:
- **Name**: Clear, descriptive (e.g., `startup_time_seconds`)
- **Unit**: seconds, milliseconds, samples/sec, etc.
- **When logged**: Initialization, per-epoch, completion
- **Who logs**: Root rank only (for distributed training)

### Step 2: Add Timing Code

```python
import time
from weathergen.utils.distributed import is_root

# Start timing
t_start = time.time()

# ... code to measure ...

# Log metric (root rank only)
if is_root():
    elapsed = time.time() - t_start
    train_logger.log_metrics("train", {"metric_name": elapsed})
    logger.info(f"Metric: {elapsed:.2f} seconds")
```

### Step 3: Choose Timing Points

| Metric Type | Placement | Example |
|-------------|-----------|---------|
| **Startup time** | After init, before main loop | `trainer.run()` after data loader setup |
| **Training time** | Before/after training loop | `for epoch in epochs:` |
| **Overall time** | Entry/exit of main function | `run_train()` finally block |
| **Per-epoch time** | Inside epoch loop | After `train(epoch)` completes |

### Step 4: Ensure Root-Only Logging

For distributed training (DDP/FSDP):
```python
if is_root():
    # Only rank 0 writes to files/MLflow
    train_logger.log_metrics("train", {"metric": value})
```

### Step 5: Add to MLflow

Metrics written to `metrics.json` are automatically uploaded:
- Check `mlflow_upload.py` for filtering rules
- Avoid blacklisted keys (`weathergen.*`, `grad_norm.*`)
- Use simple numeric values (float/int)

### Step 6: Document Metric

Add to metrics reference:
```markdown
| Metric | Description | When Logged | Unit |
|--------|-------------|-------------|------|
| `startup_time_seconds` | Time from code launch to training start | After init | seconds |
| `total_training_time_seconds` | Time in training loop | After training | seconds |
| `overall_time_seconds` | Total wall-clock time | At completion | seconds |
```

## Common Patterns

### Timing a Code Block

```python
t_start = time.time()
try:
    # Code to measure
    result = expensive_operation()
finally:
    elapsed = time.time() - t_start
    if is_root():
        logger.info(f"Operation took {elapsed:.2f}s")
```

### Per-Iteration Timing

```python
for i, batch in enumerate(dataloader):
    t_iter_start = time.time()
    
    # Process batch
    loss = train_step(batch)
    
    if i % log_interval == 0:
        iter_time = time.time() - t_iter_start
        if is_root():
            train_logger.log_metrics("train", {"iter_time_ms": iter_time * 1000})
```

### Exception-Safe Timing

```python
t_start = time.time()
try:
    trainer.run(cf, devices)
finally:
    total_time = time.time() - t_start
    if is_root():
        train_logger.log_metrics("train", {"overall_time_seconds": total_time})
```

## Pitfalls

| Issue | Solution |
|-------|----------|
| **Multiple ranks logging** | Always use `is_root()` check |
| **Timer includes overhead** | Place timers as close to target code as possible |
| **Missing on failure** | Use `finally` blocks for critical metrics |
| **Too many metrics** | Filter blacklisted keys in MLflow upload |
| **Wrong units** | Be consistent (seconds for timing, not ms) |

## MLflow Integration

Metrics written to `metrics.json` are automatically picked up by `mlflow_upload.py`:

1. **Format**: JSONL (one JSON object per line)
2. **Stage**: Include `"stage": "train"` or `"stage": "val"`
3. **Timestamp**: Optional `weathergen.timestamp` field
4. **Step**: Optional `weathergen.step` field
5. **Filtering**: Blacklisted keys are dropped automatically

Example metrics.json line:
```json
{"stage": "train", "startup_time_seconds": 45.23, "weathergen.step": 100}
```

## Verification

After adding metrics:
1. Run training job
2. Check `metrics.json` for entries
3. Verify MLflow shows metrics
4. Confirm only one entry per metric (root rank)
5. Validate values are reasonable

## Related Skills

- `planning` - For designing metric strategy
- `implementation` - For coding changes
- `hpc-deployment` - For HPC-specific metrics
