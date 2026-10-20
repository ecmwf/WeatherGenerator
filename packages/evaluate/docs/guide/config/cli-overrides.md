# CLI Overrides

Override config values from command line without editing YAML:

```bash
uv run evaluate --config myconfig.yml \
  --options evaluation.score_plots=[metric_plots] evaluation.regions=[global,nhem]
```

`--options` uses OmegaConf dot-notation and does not support overriding `run_ids` keys.
Use `--run-ids` for that:

```bash
uv run evaluate --config myconfig.yml --run-ids ar40mckx c8g5katp
```

Upload scores to MLFlow:

```bash
uv run evaluate --config myconfig.yml --push-metrics
```
