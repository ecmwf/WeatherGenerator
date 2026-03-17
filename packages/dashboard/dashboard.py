import logging
import os

import streamlit as st
import streamlit_authenticator as stauth


@st.cache_resource
def get_logger():
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            level=logging.INFO,
        )
    print("_logger being returned", logger)  # noqa: T201
    return logger

_logger = get_logger()
_logger.info(f"Current page query param: {st.query_params.get('page')}")

user = os.getenv("USER_NAME")
password = os.getenv("USER_PASSWORD")
# Large ttl to avoid frequent logins.
auth_time_sec = int(os.getenv("AUTH_TIME_SEC", "18000"))

authenticator = stauth.Authenticate(
    {
        "usernames": {
            user: {
                "email": "noreply@weathergenerator.eu",
                "failed_login_attempts": 0,
                "logged_in": False,
                "first_name": "Test",
                "last_name": "Test",
                "password": password,
            }
        }
    },
    "authenticator_cookie",
    "authenticator_cookie_key",
    auth_time_sec,
)


def _make_page(script: str, title: str, url_path: str) -> st.Page:
    is_default = st.query_params.get("page") == url_path
    return st.Page(script, title=title, url_path=url_path, default=is_default)


pages = {
    "Engineering": [
        _make_page("eng_overview.py", title="overview", url_path="overview"),
        _make_page("exp_tracker.py", title="run details", url_path="run-details"),
    ],
    "Model:atmo": [
         _make_page("atmo_training.py", "training", "training"),
         _make_page("atmo_eval.py", "evaluation", "evaluation"),
    ],
    "Data": [
        _make_page("data_overview.py", "overview", "data-overview"),
        _make_page("data_sources.py", "sources", "data-sources"),
    ],
}
pg = st.navigation(pages)
# Only update query param when the user actually navigates, to avoid rerun loops.
if st.query_params.get("page") != pg.url_path and pg.url_path:
    _logger.info(f"Updating query param to {pg.url_path}")
    st.query_params["page"] = pg.url_path

try:
    authenticator.login()
except Exception as e:
    st.error(e)

if st.session_state.get("authentication_status") is False:
    st.error("Username/password is incorrect")
    st.stop()
elif not st.session_state.get("authentication_status"):
    st.warning("Please enter your username and password")
    st.stop()

pg.run()
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/e/e1/ECMWF_logo.svg")
st.sidebar.markdown("[weathergenerator.eu](https://weathergenerator.eu)")
# authenticator.logout()
