# CSV Format for Pre-computed Scores

For `type: "csv"`, files must be under:

```text
<metrics_dir>/<run_id>/
```

Expected column layout:

```text
,parameter,level,number,score,step,date,domain_name,value
0,t,925,0,rmse,0 days 12:00:00,2022-10-01 00:00:00,n.hem,0.031371
1,t,925,0,rmse,0 days 12:00:00,2022-10-01 12:00:00,n.hem,-0.01038
```

| Column | Description |
|--------|-------------|
| `parameter` | Variable short name (e.g. `t`, `z`, `u`) |
| `level` | Pressure level in hPa (e.g. `925`, `500`) |
| `number` | Ensemble member number (`0` for deterministic) |
| `score` | Metric name in Quaver convention |
| `step` | Lead time as timedelta string |
| `date` | Initialization datetime |
| `domain_name` | Region name in Quaver convention |
| `value` | Score value |

Channel names are constructed as `<parameter>_<level>` (e.g. `t_925`).
