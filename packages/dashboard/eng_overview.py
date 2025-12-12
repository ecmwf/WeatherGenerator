import logging

import plotly.express as px
import polars as pl
import polars.selectors as ps
import streamlit as st
from polars import col as C

from weathergen.dashboard.colors import clusters_color_map, unknown_color
from weathergen.dashboard.metrics import all_runs, latest_runs, setup_mflow

_logger = logging.getLogger("eng_overview")


logging.basicConfig(level=logging.INFO)
_logger.info("Setting up MLFlow")
setup_mflow()


st.markdown("# Engineering overview")


runs = latest_runs()
all_runs_pdf = all_runs()

st.markdown("""The number of runs by month and by HPC.""")
# TODO: this is here just the number of root run ids.
#  Does not count how many tries or how many validation experiments were run.
all_runs_stats = (
    all_runs_pdf.sort("start_time")
    # Remove metrics and tags
    .select(~ps.starts_with("metrics"))
    .select(~ps.starts_with("params"))
    # Just keep roots
    .filter(C("tags.mlflow.parentRunId").is_null())
    # Put a month column
    .with_columns(pl.date(C("start_time").dt.year(), C("start_time").dt.month(), 1).alias("month"))
)


runs_lifecycle_stats = (
    # Remove metrics and params
    all_runs_pdf.select(~ps.starts_with("metrics"))
    .select(~ps.starts_with("params"))
    .filter(C("tags.run_id").is_not_null())
    # For each of the run_ids, keep status, time, all stages, hpc
    .group_by("tags.run_id")
    .agg(
        C("status").unique(), C("start_time").min(), C("tags.stage").unique(), C("tags.hpc").first()
    )
    .with_columns(
        # Filter mlflow status:
        # FAILED => failed
        # FINISHED => finished
        # else => running
        pl.when(C("status").list.contains("FAILED"))
        .then(pl.lit("failed"))
        .otherwise(
            pl.when(C("status").list.contains("FINISHED"))
            .then(pl.lit("finished"))
            .otherwise(pl.lit("running"))
        )
        .alias("synth_status"),
        # Has train/val/eval stages
        C("tags.stage").list.contains("train").alias("has_train_stage"),
        C("tags.stage").list.contains("val").alias("has_val_stage"),
        C("tags.stage").list.contains("eval").alias("has_eval_stage"),
    )
    # Put a month column
    .with_columns(pl.date(C("start_time").dt.year(), C("start_time").dt.month(), 1).alias("month"))
    # cast to str the week column: plotly will misinterpret it otherwise
    .with_columns(C("start_time").dt.week().cast(pl.String).alias("week"))
)


st.plotly_chart(
    px.bar(
        (all_runs_stats.group_by("month", "tags.hpc").agg(pl.count("run_id"))).to_pandas(),
        x="month",
        y="run_id",
        color="tags.hpc",
        color_discrete_map=clusters_color_map,
        color_discrete_sequence=[unknown_color],
    )
)


st.markdown(
    """
            
**Training runs by organization**

Note: some runs may not have org tags.

"""
)


st.plotly_chart(
    px.bar(
        all_runs_pdf.filter(C("tags.stage").eq("train"))
        .with_columns(
            pl.date(C("start_time").dt.year(), C("start_time").dt.month(), 1).alias("month")
        )
        .select(["month", C("params.wgtags.org").fill_null("unknown"), "tags.run_id"])
        .filter(C("tags.run_id").is_not_null())
        .group_by("params.wgtags.org", "month")
        .agg(pl.count("tags.run_id"))
        .to_pandas(),
        y="tags.run_id",
        x="month",
        color="params.wgtags.org",
        # hover_data=["start_time", "tags.uploader"],
    )
)


st.markdown(
    """
            
**The number of samples processed by HPC.**

(only includes runs for which evaluation data has been uploaded)

The number of samples is a good indicator of the amount of processing being done by experiment.

"""
)

process_samples_stats = (
    all_runs_pdf.sort("start_time")
    # Just keep start_time, run_id, mlflow_run_id, and num_samples
    .select(["start_time", "metrics.num_samples", "tags.run_id", "tags.hpc"])
    .drop_nulls()
    .with_columns(pl.date(C("start_time").dt.year(), C("start_time").dt.month(), 1).alias("month"))
    .group_by(["month", "tags.hpc"])
    .agg(C("metrics.num_samples").sum())
)

st.plotly_chart(
    px.bar(
        process_samples_stats.to_pandas(),
        y="metrics.num_samples",
        x="month",
        color="tags.hpc",
        color_discrete_map=clusters_color_map,
        color_discrete_sequence=[unknown_color],
    )
)


st.markdown(
    """
            
**The number of GPUs by run.**

(only includes runs for which evaluation data has been uploaded)

"""
)

st.plotly_chart(
    px.scatter(
        all_runs_pdf
        # world_size used to be called num_ranks
        .with_columns(
            pl.max_horizontal(
                pl.col("params.num_ranks").fill_null(0), pl.col("params.world_size").fill_null(0)
            )
            .cast(int)
            .alias("world_size")
        )
        .filter(pl.col("world_size") > 0)
        .select(["world_size", "start_time", "tags.hpc"])
        .to_pandas(),
        y="world_size",
        x="start_time",
        color="tags.hpc",
        # hover_data=["start_time", "tags.uploader"],
        color_discrete_map=clusters_color_map,
        color_discrete_sequence=[unknown_color],
        log_y=True,
    )
)


st.markdown(
    """
            
**Runs by final status**

Developers using older versions will show running forever.

"""
)

_status_colors = {"finished": "green", "failed": "red", "running": "lightblue"}

st.plotly_chart(
    px.bar(
        (
            runs_lifecycle_stats.group_by("week", "synth_status", "tags.hpc").agg(
                pl.count("tags.run_id")
            )
        ).to_pandas(),
        x="week",
        y="tags.run_id",
        color="synth_status",
        color_discrete_map=_status_colors,
    )
)


st.markdown(
    """
            
**Fraction of completed runs uploading training data**


"""
)

_present_colors = {True: "green", False: "lightgray"}

st.plotly_chart(
    px.bar(
        (
            runs_lifecycle_stats.filter(pl.col("synth_status") != "running")
            .group_by("week", "synth_status", "tags.hpc", "has_train_stage")
            .agg(pl.count("tags.run_id"))
        ).to_pandas(),
        x="week",
        y="tags.run_id",
        color="has_train_stage",
        color_discrete_map=_present_colors,
    )
)


st.markdown(
    """
            
**Fraction uploading evaluation data**

Developers using older versions will show running forever.

"""
)

st.plotly_chart(
    px.bar(
        (
            runs_lifecycle_stats.group_by("week", "synth_status", "tags.hpc", "has_eval_stage").agg(
                pl.count("tags.run_id")
            )
        ).to_pandas(),
        x="week",
        y="tags.run_id",
        color="has_eval_stage",
        color_discrete_map=_present_colors,
    )
)

all_metrics = sorted(all_runs_pdf.select(ps.starts_with("metrics.")).columns)

st.markdown(
    f"""
            
**List of MLFlow metrics by number of runs**

There is a hard limit of 1000 metrics per run in MLFlow.


Total number of metrics tracked: {len(all_metrics)}.
"""
)

st.dataframe(
    # The final dataframe is of the form:
    # - metric: str (the name of the metric)
    # - example_run_id: str (a run id associated with this metric)
    # - count: int
    # Unpivot take a list of columns and converts them to values
    all_runs_pdf.unpivot(ps.starts_with("metrics."), index="tags.run_id", variable_name="metric")
    .drop_nulls()
    .group_by("metric")
    .agg(
        pl.col("tags.run_id").first().alias("example_run_id"),
        pl.col("tags.run_id").count().alias("count"),
    )
    .sort(by="count", descending=True)
    .to_pandas()
)
