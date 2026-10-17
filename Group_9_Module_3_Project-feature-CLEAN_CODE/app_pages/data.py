from __future__ import annotations

import streamlit as st

from src.dashboard_common import DEFAULT_DATA, get_market_data


st.subheader("SPY data")
market = get_market_data()
st.dataframe(
    market.sort_values("Date", ascending=False),
    hide_index=True,
    alt="Local SPY price, returns, targets, and technical indicators",
)
st.download_button(
    "Download SPY data",
    data=market.to_csv(index=False).encode("utf-8"),
    file_name="spy_dashboard_data.csv",
    mime="text/csv",
)
st.caption(
    f"Source: `{DEFAULT_DATA.name}`. The target columns contain future returns "
    "and are retrospective labels, not inputs available on the listed date."
)
