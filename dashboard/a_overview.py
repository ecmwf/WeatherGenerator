import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import polars.selectors as ps
import streamlit as st
from plotly.subplots import make_subplots
from polars import col as C
import logging

from weathergen.dashboard.metrics import all_runs, latest_runs, setup_mflow

_logger = logging.getLogger(__name__)


logging.basicConfig(level=logging.INFO)
_logger.info("Setting up MLFlow")
setup_mflow()


runs = latest_runs()

st.markdown("# Overview")


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

# st.dataframe(all_runs_stats.group_by("month", "tags.hpc").agg(pl.count("run_id")))


st.plotly_chart(
    px.bar(
        (all_runs_stats.group_by("month", "tags.hpc").agg(pl.count("run_id"))).to_pandas(),
        x="month",
        y="run_id",
        color="tags.hpc",
    )
)


st.markdown("# Training metrics")


accepted_metrics = [
    f"metrics.stream.{stream}.loss_mse.loss_avg"
    for stream in ["ERA5", "SurfaceCombined", "NPPATMS"]
] + ["metrics.num_samples"]


def make_plot(df):
    def filter_met(c: str) -> bool:
        return c in accepted_metrics

    plot_metrics = sorted([c for c in df.columns if filter_met(c)])
    hovertemplate = "".join(
        [
            f"{col}: %{{customdata[{idx}]}}<br>"
            if ("metrics" not in col and "params" not in col and "tags.mlflow" not in col)
            else ""
            for idx, col in enumerate(df.columns)
        ]
    )
    hovertemplate = "val: %{y}<br>" + hovertemplate
    num_plots = len(plot_metrics)
    fig = make_subplots(rows=num_plots, cols=1, subplot_titles=plot_metrics)
    for i, metric in enumerate(plot_metrics):
        s = go.Scatter(
            x=df["end_time"],
            y=df[metric],
            mode="markers",
            customdata=df,
            hovertemplate=hovertemplate,
        )
        fig.add_trace(s, row=i + 1, col=1)

    fig.update_yaxes(type="log")
    fig.update_layout(height=800, width=1024, showlegend=False)
    return fig


st.markdown("## Train")

st.plotly_chart(make_plot(runs.filter(pl.col("tags.stage") == "train")))

st.markdown("# Validation")

st.plotly_chart(make_plot(runs.filter(pl.col("tags.stage") == "val")))


train_runs = runs.filter(pl.col("tags.stage") == "train")
min_end_date = train_runs["start_time"].cast(pl.Float64).min()
max_end_date = train_runs["start_time"].cast(pl.Float64).max()
train_runs = train_runs.with_columns(
    (
        (pl.col("start_time").cast(pl.Float64) - pl.lit(min_end_date))
        / (pl.lit(max_end_date) - pl.lit(min_end_date))
    ).alias("idx")
)
train_runs["idx"]

st.plotly_chart(
    px.scatter(
        train_runs.to_pandas(),
        x="metrics.num_samples",
        y="metrics.loss_avg_0_mean",
        color="idx",
        hover_data=["start_time", "tags.hpc", "tags.uploader"],
        log_y=True,
        log_x=True,
    )
)
