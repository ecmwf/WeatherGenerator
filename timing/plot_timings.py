from pathlib import Path

import pandas as pd
import polars as pl
import polars.selectors as ps

from weathergen.utils.metrics import read_metrics_file, get_train_metrics_path

BASE_PATH = Path(__file__).parent.parent / "results"
RUN_ID = "vk1xpn36"

stage = "validate"

if __name__ == "__main__":
    metrics_file = get_train_metrics_path(BASE_PATH, RUN_ID)
    timer = f"perf.timing.root.{stage}"
    metric = "mean"

    df = (
        read_metrics_file(metrics_file)
        .select(ps.starts_with(timer))
        .select(ps.ends_with(metric))
        .select(~ps.starts_with(f"{timer}.{metric}"))  # only show subtimings
        .to_pandas()
    )

    ax = df.plot(
        kind="bar",
        stacked=False,
        title=f"timing results: {RUN_ID} ({stage})",
        ylabel="time ($ns$)",
        xlabel="loop index"
    )
    ax.get_figure().savefig(f"timings_{stage}_{RUN_ID}.png")
