import logging

import plotly.express as px
import polars as pl
import streamlit as st
from mlflow.client import MlflowClient

from weathergen.dashboard.metrics import ST_TTL_SEC, latest_runs, setup_mflow, stage_is_eval

_logger = logging.getLogger(__name__)

_logger.info("Setting up MLFlow")
client: MlflowClient = setup_mflow()

important_scores = [
    ("metrics.score.global.rmse.ERA5.2t", "deg K"),
]

st.markdown("""
            
# Main model training: evaluation scores

The evaluation scores logged during the main model training runs.

""")


@st.cache_data(ttl=ST_TTL_SEC, max_entries=2)
def get_runs_with_scores() -> pl.DataFrame:
    """
    The runs that have evaluation scores logged.
    - Only keep the eval stage runs
    - Only keep the metrics.score.* metrics
    """
    # Fully encapsulated logic to allow caching
    runs = latest_runs()
    eval_runs = runs.filter(stage_is_eval)
    # Keep all non-metrics columns, plus metrics.score.* columns
    # Do not keep gradient metrics or other metrics.
    target_cols = [
        col
        for col in eval_runs.columns
        if (col.startswith("metrics.score.") or not col.startswith("metrics"))
    ]
    eval_runs = eval_runs.select(target_cols)
    return eval_runs


eval_runs = get_runs_with_scores()

# The info columns to show on hover
info_cols = [
    "tags.hpc",
    "tags.uploader",
    "tags.run_id",
]


@st.cache_data(ttl=ST_TTL_SEC, max_entries=20)
def get_score_step_48h(score_col: str) -> pl.DataFrame:
    """
    Given a score name, return this score at the step corresponding to 48h.
    """
    score = score_col.replace("metrics.", "")
    # Caching since it makes multiple MLFlow calls
    eval_runs = get_runs_with_scores()
    step_48h = 8  # Each step = 6 hours => look for step = 8*6 = 48 hours
    score_data = (
        eval_runs.select(
            [
                pl.col("run_id"),  # The MLFlow run ID
                pl.col("start_time"),
                pl.col(score_col),
            ]
            + [pl.col(c) for c in info_cols]
        )
        .sort("start_time")
        .filter(pl.col(score_col).is_not_null())
    )
    _logger.info(f"Getting score data for {score_col} at 48h (step={step_48h}): len={len(score_data)}")

    # Iterate over the runs to get the metric at step 48h
    scores_dt: list[float | None] = []
    for row in score_data.iter_rows(named=True):
        mlflow_run_id = row["run_id"]
        _logger.info(f"Fetching metric history for run_id={mlflow_run_id}, score={score}")
        data = client.get_metric_history(
            run_id=mlflow_run_id,
            key=score,
        )
        # Find the value at step 48h
        value_48h: float | None = None
        for m in data:
            if m.step == step_48h:
                value_48h = m.value
                break
        scores_dt.append(value_48h)
    score_data = score_data.with_columns(pl.Series(name="score_48h", values=scores_dt)).filter(
        pl.col("score_48h").is_not_null()
    )
    return score_data


# The specific score of interest:
for score_col, unit in important_scores:
    score_data_48h = get_score_step_48h(score_col)
    score = score_col.replace("metrics.", "")

    st.markdown(f"""
    ## {score} at 48h
    The evaluation score at 48 hours into the forecast. Unit: {unit}
    """)
    tab1, tab2 = st.tabs(["chart", "data"])
    tab1.plotly_chart(
        px.scatter(
            score_data_48h.to_pandas(),
            x="start_time",
            y="score_48h",
            hover_data=info_cols,
        )
    )
    tab2.dataframe(score_data_48h.to_pandas())

st.markdown("""
# All latest evaluation scores
            
These scores are harder to compare: different experiments may have different forecast lengths.
            
""")

accepted_scores = sorted([col for col in eval_runs.columns if col.startswith("metrics.score.")])

for score_col in accepted_scores:
    score_data = (
        eval_runs.select(
            [
                pl.col("start_time"),
                pl.col(score_col),
            ]
            + [pl.col(c) for c in info_cols]
        )
        .sort("start_time")
        .filter(pl.col(score_col).is_not_null())
    )
    if score_data.is_empty():
        continue
    st.markdown(f"## {score_col}")
    st.plotly_chart(
        px.scatter(
            score_data.to_pandas(),
            x="start_time",
            y=score_col,
            hover_data=info_cols,
        )
    )
