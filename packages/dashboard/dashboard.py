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


user = os.getenv("USER_NAME")
password = os.getenv("USER_PASSWORD")
auth_time_sec = int(os.getenv("AUTH_TIME_SEC", "1800"))

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


try:
    authenticator.login()
except Exception as e:
    st.error(e)


if st.session_state.get("authentication_status"):
    pg = st.navigation(
        {
            "Engineering": [st.Page("eng_overview.py", title="overview")],
            "Model:atmo": [
                st.Page("atmo_training.py", title="training"),
                st.Page("atmo_eval.py", title="evaluation"),
            ],
            "Data": [
                st.Page("data_overview.py", title="overview"),
                st.Page("data_sources.py", title="sources"),
            ],
        }
    )
    pg.run()
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/e/e1/ECMWF_logo.svg")
    st.sidebar.markdown("[weathergenerator.eu](https://weathergenerator.eu)")
    authenticator.logout()
elif st.session_state.get("authentication_status") is False:
    st.error("Username/password is incorrect")
elif st.session_state.get("authentication_status") is None:
    st.warning("Please enter your username and password")
