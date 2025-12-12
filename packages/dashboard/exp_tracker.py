import logging
import os
from urllib.parse import quote

import polars as pl
import streamlit as st

from weathergen.dashboard.metrics import all_runs, get_experiment_id, setup_mflow

_logger = logging.getLogger("eng_overview")


logging.basicConfig(level=logging.INFO)
_logger.info("Setting up MLFlow")
setup_mflow()


st.markdown(
    """
# Experiment tracker

This pages gives insights into a single run, in particular
lineage information.            
            """
)

run_id = st.text_input(label="A run id, for example bl4butd8")

all_runs_pdf = all_runs()

# The current algorithm is brute-forcing the information search.


def find_all_relevant_runs(_run_id: str) -> list[str]:
    """
    Given a run_id, finds all the parent from_run_id
    """
    if _run_id == "":
        return []
    s: list[str] = []
    fringe = [_run_id]
    while len(fringe) > 0:
        current_run_id = fringe.pop()
        s.append(current_run_id)
        parent_runs = (
            all_runs_pdf.filter(pl.col("tags.run_id") == current_run_id)["tags.from_run_id"]
            .drop_nulls()
            .to_list()
        )
        for rid in parent_runs:
            if rid not in s:
                fringe.append(rid)
    return s


relevant_runs = find_all_relevant_runs(run_id)


def _experiment_url() -> str:
    # Set up by setup_mlflow
    host = os.environ["DATABRICKS_HOST"]
    exp_id = get_experiment_id()
    # The host URL may or may not have a final '/' and this confuses databricks.
    return f"{host}/ml/experiments/{exp_id}".replace("com//ml", "com/ml")


def _mlflow_search_link(run_ids: list[str]) -> str:
    txt = quote("tags.run_id IN (" + ", ".join([f"'{rid}'" for rid in run_ids]) + ")")
    return f"{_experiment_url()}/runs?searchFilter={txt}"


st.markdown(f"""
This experiment consists of {len(relevant_runs)} runs. 
[MLFlow link to all the runs]({_mlflow_search_link(relevant_runs)})
            """)


def _make_mlflow_link(col):
    return pl.lit(_experiment_url() + "/runs/") + col


for sub_run_id in relevant_runs:
    sub_runs = all_runs_pdf.filter(pl.col("tags.run_id") == sub_run_id).select(
        [
            _make_mlflow_link(pl.col("run_id")).alias("mlflow"),
            "tags.run_id",
            "tags.stage",
            "tags.uploader",
            "start_time",
            "tags.hpc",
            "params.wgtags.issue",
            "params.wgtags.exp",
        ]
    )
    st.markdown(
        f"""
## {sub_run_id}
                """
    )
    st.dataframe(
        sub_runs.to_pandas(),
        column_config={
            "mlflow": st.column_config.LinkColumn(
                display_text="mlflow",
            )
        },
        hide_index=True,
    )
