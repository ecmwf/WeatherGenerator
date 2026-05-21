#!/usr/bin/env sh
# Sensible defaults for our use of Prefect:
# - start with uvx (no need to install prefect)
# - persist result in json format (easier to inspect results in the UI and to debug)
PREFECT_RESULTS_PERSIST_BY_DEFAULT=true PREFECT_RESULTS_DEFAULT_SERIALIZER=json uvx --with "prefect==3.7.0" prefect server start