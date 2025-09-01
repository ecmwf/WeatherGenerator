"""
Downloads metrics from MLFlow.
"""

import streamlit as st
import plotly.express as px
import pandas as pd
import polars as pl
import streamlit as st
import plotly.express as px
import pandas as pd
import polars as pl
from functools import lru_cache
import os
import simple_cache
import polars.selectors as ps

import polars as pl
import mlflow
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mlflow
import mlflow.client
from mlflow.client import MlflowClient
from polars import col as C
from functools import lru_cache



phase = 'train'
exp_lifecycle = 'test'
project = "WeatherGenerator"
experiment_id = "384213844828345"
all_stages = ["train", "val"]

# Cache TTL
_ttl_sec = 120




class MlFlowUpload:
    tracking_uri = "databricks"
    registry_uri = "databricks-uc"
    experiment_name = "/Shared/weathergen-dev/core-model/defaultExperiment"


@lru_cache(maxsize=1)
def setup_mflow():
    # os.environ["DATABRICKS_HOST"] = None
    #os.environ["DATABRICKS_TOKEN"] = None
    mlflow.set_tracking_uri(MlFlowUpload.tracking_uri)
    mlflow.set_registry_uri(MlFlowUpload.registry_uri)
    mlflow_client = mlflow.client.MlflowClient(
        tracking_uri=MlFlowUpload.tracking_uri, registry_uri=MlFlowUpload.registry_uri
    )
    return mlflow_client



@simple_cache.cache_it(filename=".latest_runs_all_stages.cache", ttl=_ttl_sec)
def latest_runs():
    runs_pdf = pl.DataFrame(mlflow.search_runs(experiment_ids=[experiment_id],
                            filter_string=f"status='FINISHED' AND tags.completion_status = 'success'"))
    # print(runs_pdf  )
    runs_pdf = runs_pdf.filter(pl.col("tags.stage").is_in(all_stages))
    # print(runs_pdf  )
    latest_run_by_exp = (runs_pdf
                         .sort(by="end_time", descending=True)
                         .group_by(["tags.run_id", "tags.stage"])
                         .agg(pl.col("*").last())
                         .sort(by="tags.run_id"))
    print(len(runs_pdf))
    return latest_run_by_exp

@simple_cache.cache_it(filename=".all_runs.cache", ttl=_ttl_sec)
def all_runs():
    runs_pdf = pl.DataFrame(mlflow.search_runs(experiment_ids=[experiment_id],
                            filter_string=f"status='FINISHED' AND tags.completion_status = 'success'"))
    return runs_pdf

