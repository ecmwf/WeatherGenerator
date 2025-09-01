import streamlit as st
import plotly.express as px
import pandas as pd
import polars as pl
import os

import streamlit_authenticator as stauth

user = os.getenv("USER_NAME")
password = os.getenv("USER_PASSWORD")

authenticator = stauth.Authenticate(
    {
        "usernames": {
            user : {
                "email" : "noreply@weathergenerator.eu",
                "failed_login_attempts": 0,
                "logged_in": False,
                "first_name": "Test",
                "last_name": "Test",
                "password": password
            }
        }
    },
    "authenticator_cookie",
    "authenticator_cookie_key",
    30
)


try:
    authenticator.login()
except Exception as e:
    st.error(e)


if st.session_state.get('authentication_status'):
    pg = st.navigation([st.Page("a_overview.py"),
                    st.Page("b_data.py")])
    pg.run()
    authenticator.logout()
elif st.session_state.get('authentication_status') is False:
    st.error('Username/password is incorrect')
elif st.session_state.get('authentication_status') is None:
    st.warning('Please enter your username and password')


