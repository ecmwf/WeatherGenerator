# dashboard — internal streamlit dashboard

- NOT a uv workspace member (adding it breaks the streamlit deployment); has its own `uv.lock`/`.venv`. Run from this dir: `uv run --env-file=.env streamlit run dashboard.py`.
- `dashboard.py` — entry; pages: `atmo_training`, `atmo_eval`, `data_overview`, `data_sources`, `eng_overview`, `exp_tracker`.
- Deployment docs: WeatherGenerator-private wiki.
