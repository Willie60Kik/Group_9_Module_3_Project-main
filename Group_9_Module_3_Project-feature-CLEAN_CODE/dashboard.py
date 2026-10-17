from __future__ import annotations

import streamlit as st

from src.dashboard_common import clear_cached_data, show_sidebar_data_status


st.set_page_config(
    page_title="S&P 500 model research",
    page_icon=":material/query_stats:",
    layout="wide",
)

show_sidebar_data_status()
with st.sidebar:
    if st.button("Reload local data", icon=":material/refresh:"):
        clear_cached_data()

page = st.navigation(
    [
        st.Page(
            "app_pages/technical.py",
            title="Technical indicators",
            icon=":material/monitoring:",
            default=True,
        ),
        st.Page(
            "app_pages/ml_actual.py",
            title="ML vs actual",
            icon=":material/model_training:",
        ),
        st.Page(
            "app_pages/data.py",
            title="Data",
            icon=":material/table_view:",
        ),
    ],
    position="top",
)

st.title("S&P 500 trading research")
st.caption(
    "Independent technical analysis and macro-only ML evaluation using local data."
)
page.run()
