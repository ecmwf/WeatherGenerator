import streamlit as st

from weathergen.dashboard.metrics import test

st.text("Data" + test())
