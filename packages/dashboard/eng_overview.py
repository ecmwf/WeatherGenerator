import logging

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import polars.selectors as ps
import streamlit as st
from plotly.subplots import make_subplots
from polars import col as C

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
    all_runs_pdf
    # Remove metrics and tags
    .select(~ps.starts_with("metrics"))
    .select(~ps.starts_with("params"))
    # Just keep roots
    .filter(C("tags.mlflow.parentRunId").is_null())
    # Put a month column
    .with_columns(pl.date(C("start_time").dt.year(), C("start_time").dt.month(), 1).alias("month"))
)


st.plotly_chart(
    px.bar(
        (all_runs_stats.group_by("month", "tags.hpc").agg(pl.count("run_id"))).to_pandas(),
        x="month",
        y="run_id",
        color="tags.hpc",
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
        all_runs_pdf.filter(pl.col("params.num_ranks").is_not_null())
        .select(["params.num_ranks", "start_time", "tags.hpc"])
        .to_pandas(),
        y="params.num_ranks",
        x="start_time",
        color="tags.hpc",
        # hover_data=["start_time", "tags.uploader"],
        log_y=True,
    )
)
