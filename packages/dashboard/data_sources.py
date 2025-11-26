import logging

import plotly.express as px
import polars as pl
import polars.selectors as ps
import streamlit as st
from polars import col as C

from weathergen.dashboard.metrics import all_runs, latest_runs, setup_mflow

_logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
_logger.info("Setting up MLFlow")
setup_mflow()


st.markdown("# Data sources usage")


runs = latest_runs()
all_runs_pdf = all_runs()


st.markdown(
    """
            
**The number of data sources by run.**

(only includes runs for which evaluation data has been uploaded)

"""
)

# The name of all the streams metrics
all_stream_columns = all_runs_pdf.select(ps.starts_with("metrics.stream")).columns

# Has [stream_name, tags.hpc, month, tags.run_id, start_time, tags.uploader]
streams_run_ids = (
    all_runs_pdf.select(
        ps.starts_with("metrics.stream") | ps.starts_with("start_time") | ps.starts_with("tags")
    )
    .unpivot(
        all_stream_columns,
        index=["start_time", "tags.run_id", "tags.hpc", "tags.uploader"],
        variable_name="metric",
        value_name="value",
    )
    .filter(pl.col("value").is_not_null())
    .with_columns(pl.col("metric").str.split(".").list.get(2).alias("stream_name"))
    .with_columns(pl.date(C("start_time").dt.year(), C("start_time").dt.month(), 1).alias("month"))
    .group_by("stream_name", "tags.hpc", "month", "tags.run_id", "start_time", "tags.uploader")
    .agg(pl.count("value"))
    .drop("value")
)

# The most used streams
st.table(
    streams_run_ids.select(["stream_name", "tags.run_id"])
    .group_by("stream_name")
    .agg(pl.count("tags.run_id").alias("num_runs"))
    .sort("num_runs", descending=True)
    .to_pandas()
)

st.plotly_chart(
    px.scatter(
        streams_run_ids.group_by("tags.run_id", "tags.hpc", "month", "start_time", "tags.uploader")
        .agg(pl.n_unique("stream_name").alias("num_streams"))
        .to_pandas(),
        y="num_streams",
        x="start_time",
        hover_data=["start_time", "tags.run_id", "tags.hpc", "tags.uploader"],
    )
)
