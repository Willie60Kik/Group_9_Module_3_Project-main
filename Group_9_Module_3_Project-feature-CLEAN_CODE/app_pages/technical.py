from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard_common import get_market_data, show_market_context


st.subheader("Technical indicators")
st.caption(
    "This page is a standalone technical-analysis view. These indicators are "
    "not used as features by the macro-only ML models."
)
market = get_market_data()
latest = market.iloc[-1]

with st.container(horizontal=True):
    st.metric("Latest SPY close", f"${latest['Close']:,.2f}", border=True)
    st.metric("Current rule signal", latest["Rule_Signal"], border=True)
    st.metric("Current RSI (14)", f"{latest['RSI_14']:.1f}", border=True)

available_indicators = [
    "SMA 20",
    "SMA 50",
    "RSI (14)",
    "MACD",
    "MACD signal",
    "20-session volatility",
]
indicator_columns = {
    "SMA 20": "SMA_20",
    "SMA 50": "SMA_50",
    "RSI (14)": "RSI_14",
    "MACD": "MACD",
    "MACD signal": "MACD_Signal",
    "20-session volatility": "Volatility_20",
}

selected_indicators = st.multiselect(
    "Choose indicators to display",
    options=available_indicators,
    default=["SMA 20", "SMA 50", "RSI (14)"],
)
date_range = st.date_input(
    "Chart date range",
    value=(market["Date"].min().date(), market["Date"].max().date()),
    min_value=market["Date"].min().date(),
    max_value=market["Date"].max().date(),
    key="technical_date_range",
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

chart_data = market.loc[
    market["Date"].dt.date.between(start_date, end_date)
].copy()
if chart_data.empty:
    st.warning("No SPY observations fall within the selected date range.")
    st.stop()

chart_choice = st.radio(
    "Chart",
    ["SPY price", "Selected indicators"],
    horizontal=True,
    key="technical_chart_choice",
)
if chart_choice == "SPY price":
    price_columns = ["Close"] + [
        indicator_columns[name]
        for name in selected_indicators
        if name in {"SMA 20", "SMA 50"}
    ]
    st.line_chart(
        chart_data.set_index("Date")[price_columns],
        alt="SPY closing price with selected moving-average overlays",
    )
else:
    if not selected_indicators:
        st.info("Choose at least one indicator to draw the selected-indicators chart.")
    else:
        selected_columns = [indicator_columns[name] for name in selected_indicators]
        st.line_chart(
            chart_data.set_index("Date")[selected_columns],
            alt="Selected SPY technical indicators over time",
        )

st.subheader("Technical rule")
st.caption(
    "BUY when price is above SMA 50, RSI is above 50, and MACD is above its "
    "signal. SELL when any condition reverses; HOLD otherwise."
)
if latest["Rule_Signal"] == "BUY":
    st.success("Latest rule state: BUY")
elif latest["Rule_Signal"] == "SELL":
    st.warning("Latest rule state: SELL")
else:
    st.info("Latest rule state: HOLD")

show_market_context(
    "SPY price versus SMA 50, RSI (14), and MACD versus its signal line",
    signal_as_of=pd.Timestamp(latest["Date"]),
)

st.subheader("Recent indicator values")
table_columns = ["Date", "Close", "Rule_Signal"] + [
    indicator_columns[name] for name in selected_indicators
]
recent = chart_data[table_columns].tail(50).sort_values("Date", ascending=False)
st.dataframe(
    recent,
    hide_index=True,
    column_config={
        "Date": st.column_config.DateColumn("Date"),
        "Close": st.column_config.NumberColumn("Close", format="$%.2f"),
        "SMA_20": st.column_config.NumberColumn("SMA 20", format="$%.2f"),
        "SMA_50": st.column_config.NumberColumn("SMA 50", format="$%.2f"),
        "RSI_14": st.column_config.NumberColumn("RSI (14)", format="%.1f"),
        "Volatility_20": st.column_config.NumberColumn(
            "20-session volatility", format="%.2f%%"
        ),
        "MACD": st.column_config.NumberColumn("MACD", format="%.3f"),
        "MACD_Signal": st.column_config.NumberColumn("MACD signal", format="%.3f"),
    },
    alt="Recent SPY technical-indicator values and rule signals",
)
