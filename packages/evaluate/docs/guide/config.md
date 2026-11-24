# Fast Evaluation Config Reference

FInd all configuration options accepted by `uv run evaluate --config <your_config.yml>` below.
A working template to copy and edit can be found in `config/eval_config.yml`.

<pre style="
    max-width:100%;
    overflow-x:auto;
    white-space:pre-wrap;
    word-break:break-word;
    background-color: #F8F8F8;
    padding-left:10px;
    padding-right:10px">

max_workers: ...          # optional top-level cap on parallelism 
                          <a href="./config/top-level.md">Top-Level Keys</a>

global_plotting_options:  # optional - applied to all runs/streams
                          <a href="./config/global-plotting-options.md">Global Plotting Options</a>
  ...

evaluation:               # required - scoring and summary plot settings 
                          <a href="./config/evaluation.md">Evaluation</a>
  metrics: [...]          <a href="./config/metrics.md">Metrics Reference</a>
  regions: [...]          <a href="./config/regions.md">Regions Reference</a>
  ...

default_streams:          # optional - stream config used when
                          # run_id does not specify its own streams
                          <a href="./config/runs-and-streams.md">default_streams, run_ids, and stream blocks</a> 
  ERA5:                  
    channels: [...]
    evaluation: ...
    plotting: ...
  CERRA:
    ...

run_ids:                  # required - one entry per run to evaluate
  <run_id>:
    label: "..."
    results_base_dir: "..."
    # streams: optional; if absent, default_streams is used
  ...

</pre>

Further configuration options:

- [Score caching (JSON files)](config/score-caching.md)
- [CSV format for pre-computed scores](config/csv-format.md)
- [CLI overrides](config/cli-overrides.md)
