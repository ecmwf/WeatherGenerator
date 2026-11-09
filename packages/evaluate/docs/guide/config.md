# Fast Evaluation Config Reference

FInd all configuration options accepted by `uv run evaluate --config <your_config.yml>` below.
A working template to copy and edit can be found in `config/eval_config.yml`.

<pre>
# Useful Links

- <a href="https://www.microsoft.com">Microsoft</a>
- <a href="https://github.com">GitHub</a>
- <a href="https://docs.python.org/3/">Python Documentation</a>
</pre>

```yaml
max_workers: ...                  # optional top-level cap on parallelism

global_plotting_options:          # optional - applied to all runs/streams
  ...

evaluation:                       # required - scoring and summary plot settings
  metrics: [...]
  regions: [...]
  ...

default_streams:                  # optional - stream config used when a run_id does not
  ERA5:                           #   specify its own streams
    channels: [...]
    evaluation: ...
    plotting: ...
  CERRA:
    ...

run_ids:                          # required - one entry per run to evaluate
  <run_id>:
    label: "..."
    results_base_dir: "..."
    # streams: optional; if absent, default_streams is used
  ...
```
## 

1. [Top-level keys](config/layout-and-top-level.md)
2. [Global Plotting Options](config/global-plotting-options.md)
3. [Evaluation](config/evaluation.md)
4. [`default_streams`, `run_ids`, and stream blocks](config/runs-and-streams.md)
5. [Metrics reference](config/metrics.md)
6. [Regions reference](config/regions.md)
7. [Score caching (JSON files)](config/score-caching.md)
8. [CSV format for pre-computed scores](config/csv-format.md)
9. [CLI overrides](config/cli-overrides.md)
