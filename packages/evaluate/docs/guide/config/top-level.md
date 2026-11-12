# Top-Level Keys

| Key | Type | Optional | Default | Description |
|-----|------|----------|---------|-------------|
| `max_workers` | int | yes | - | Hard cap on parallel workers used for I/O, scoring, and plotting. Applied to all runs. Useful on shared nodes to avoid oversubscription. When absent, the number of workers is chosen automatically. |
| `private_paths` | dict | yes | - | HPC-specific private path overrides. Advanced option - only needed on certain clusters. See platform config docs. |
