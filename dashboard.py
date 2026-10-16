from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "spy_labeled_features.csv"
REQUIRED_COLUMNS = {
    "Date",
    "Close",
    "Volume",
    "Daily_Return",
    "SMA_20",
    "SMA_50",
    "Volatility_20",
    "RSI_14",
    "Future_Return_5d",
    "Target",
}
TARGET_NAMES = {-1: "SELL", 0: "HOLD", 1: "BUY"}


st.set_page_config(
    page_title="SPY Trading Research Dashboard",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_market_data(path: str) -> pd.DataFrame:
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Market data file not found: {data_path}")

    data = pd.read_csv(data_path, parse_dates=["Date"])
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(
            "The selected CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    data = data.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    numeric_columns = REQUIRED_COLUMNS.difference({"Date"})
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["Date", "Close", "SMA_50", "RSI_14", "Target"])

    if data.empty:
        raise ValueError("The market data file contains no usable SPY rows.")

    data["Target"] = data["Target"].astype(int)
    data["MACD"] = (
        data["Close"].ewm(span=12, adjust=False).mean()
        - data["Close"].ewm(span=26, adjust=False).mean()
    )
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

    buy = (
        (data["Close"] > data["SMA_50"])
        & (data["RSI_14"] > 50)
        & (data["MACD"] > data["MACD_Signal"])
    )
    sell = (
        (data["Close"] < data["SMA_50"])
        | (data["RSI_14"] < 50)
        | (data["MACD"] < data["MACD_Signal"])
    )
    data["Rule_Signal"] = "HOLD"
    data.loc[buy, "Rule_Signal"] = "BUY"
    data.loc[~buy & sell, "Rule_Signal"] = "SELL"
    data["Target_Name"] = data["Target"].map(TARGET_NAMES).fillna("Unknown")
    return data


st.title("SPY Trading Research Dashboard")
st.caption(
    "Local, historical analysis of SPY price data, technical signals, "
    "and five-trading-day target labels."
)

with st.sidebar:
    st.header("Data")
    st.caption("The dashboard reads the project CSV only; it does not call a market API.")
    if st.button("Reload project data"):
        load_market_data.clear()
    st.code(str(DEFAULT_DATA), language=None)

try:
    market = load_market_data(str(DEFAULT_DATA))
except (OSError, ValueError, pd.errors.ParserError) as exc:
    st.error(f"Unable to load the SPY dataset: {exc}")
    st.info("Run the notebook data preparation cells to create the labeled feature CSV.")
    st.stop()

min_date = market["Date"].min().date()
max_date = market["Date"].max().date()
with st.sidebar:
    selected_dates = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date = selected_dates[0] or min_date
    end_date = selected_dates[1] or selected_dates[0] or max_date
else:
    start_date = end_date = selected_dates

view = market.loc[
    market["Date"].dt.date.between(start_date, end_date)
].copy()
if view.empty:
    st.warning("No observations fall within the selected date range.")
    st.stop()

latest = view.iloc[-1]
window_return = (latest["Close"] / view.iloc[0]["Close"] - 1) * 100
daily_return = latest["Daily_Return"] * 100

metrics = st.columns(5)
metrics[0].metric("Latest close", f"${latest['Close']:,.2f}", help=str(latest["Date"].date()))
metrics[1].metric("Daily return", f"{daily_return:+.2f}%")
metrics[2].metric("Selected-period return", f"{window_return:+.2f}%")
metrics[3].metric("RSI (14)", f"{latest['RSI_14']:.1f}")
metrics[4].metric("Technical rule signal", latest["Rule_Signal"])

overview_tab, signals_tab, labels_tab, backtest_tab, data_tab = st.tabs(
    ["Overview", "Technical signals", "Target labels", "Backtest", "Data"]
)

with overview_tab:
    left, right = st.columns([2, 1])
    with left:
        st.subheader("SPY price and moving averages")
        price_chart = view.set_index("Date")[["Close", "SMA_20", "SMA_50"]]
        st.line_chart(price_chart, height=380)
    with right:
        st.subheader("Selected range")
        st.write(f"**From:** {view['Date'].min().date()}")
        st.write(f"**To:** {view['Date'].max().date()}")
        st.write(f"**Trading sessions:** {len(view):,}")
        st.write(f"**Volume on latest date:** {latest['Volume']:,.0f}")
        st.write(f"**Close vs. SMA 50:** {latest['Close'] / latest['SMA_50'] - 1:+.2%}")

    st.subheader("Daily trading volume")
    st.bar_chart(view.set_index("Date")[["Volume"]], height=220)

with signals_tab:
    st.subheader("Rule-based technical indicators")
    st.caption(
        "This indicator signal is not a Random Forest prediction and is not a live "
        "recommendation. BUY requires all three bullish conditions; SELL occurs when "
        "any condition turns bearish."
    )
    signal = latest["Rule_Signal"]
    if signal == "BUY":
        st.success("Latest rule signal: BUY — price above SMA 50, RSI above 50, and MACD above its signal.")
    elif signal == "SELL":
        st.warning("Latest rule signal: SELL — at least one of the bullish trend conditions is not met.")
    else:
        st.info("Latest rule signal: HOLD — no buy or sell condition was triggered.")

    indicator_columns = st.columns(3)
    indicator_columns[0].metric("SMA 20", f"${latest['SMA_20']:,.2f}")
    indicator_columns[1].metric("SMA 50", f"${latest['SMA_50']:,.2f}")
    indicator_columns[2].metric("20-session volatility", f"{latest['Volatility_20']:.2%}")

    st.subheader("RSI and MACD history")
    st.line_chart(view.set_index("Date")[["RSI_14"]], height=250)
    st.line_chart(view.set_index("Date")[["MACD", "MACD_Signal"]], height=250)

    st.subheader("Recent rule signals")
    recent = view.tail(30).copy()
    st.dataframe(
        recent[
            ["Date", "Close", "SMA_50", "RSI_14", "MACD", "MACD_Signal", "Rule_Signal"]
        ].sort_values("Date", ascending=False),
        hide_index=True,
        column_config={
            "Date": st.column_config.DateColumn("Date"),
            "Close": st.column_config.NumberColumn("Close", format="$%.2f"),
            "SMA_50": st.column_config.NumberColumn("SMA 50", format="$%.2f"),
            "RSI_14": st.column_config.NumberColumn("RSI 14", format="%.1f"),
            "MACD": st.column_config.NumberColumn("MACD", format="%.3f"),
            "MACD_Signal": st.column_config.NumberColumn("MACD signal", format="%.3f"),
        },
        width="stretch",
    )

with labels_tab:
    st.subheader("Historical five-day target labels")
    st.info(
        "These are retrospective labels based on realized returns over the following "
        "five trading days. They are not model predictions and are only known after "
        "that five-day period has elapsed."
    )
    target_columns = st.columns(3)
    proportions = view["Target_Name"].value_counts(normalize=True)
    for column, name in zip(target_columns, ("BUY", "HOLD", "SELL")):
        column.metric(f"{name} labels", f"{proportions.get(name, 0):.1%}")

    st.bar_chart(
        view["Target_Name"].value_counts().reindex(["BUY", "HOLD", "SELL"], fill_value=0),
        height=260,
    )
    st.subheader("Recent realized targets")
    recent_targets = view[["Date", "Future_Return_5d", "Target_Name"]].tail(50).copy()
    recent_targets["Realized next-five-day return (%)"] = (
        recent_targets.pop("Future_Return_5d") * 100
    )
    st.dataframe(
        recent_targets.sort_values("Date", ascending=False),
        hide_index=True,
        column_config={
            "Date": st.column_config.DateColumn("Date"),
            "Realized next-five-day return (%)": st.column_config.NumberColumn(
                "Realized next-five-day return", format="%+.2f%%"
            ),
            "Target_Name": st.column_config.TextColumn("Retrospective target"),
        },
        width="stretch",
    )

with backtest_tab:
    st.subheader("Notebook evaluation snapshot")
    st.caption(
        "These figures are the last documented notebook run. The dashboard does "
        "not retrain the Random Forest or recalculate this backtest."
    )
    backtest = pd.DataFrame(
        {
            "Strategy": [
                "Random Forest (long-or-cash)",
                "RSI/MACD/SMA rules (long-or-cash)",
                "Buy and hold",
            ],
            "Compounded return (%)": [163.4, 82.9, 507.0],
            "Maximum drawdown (%)": [-16.9, -13.9, -29.1],
        }
    )
    st.metric("Mean balanced accuracy", "38.6%", help="Constant-class baseline: 33.3%.")
    st.bar_chart(
        backtest.set_index("Strategy")[["Compounded return (%)"]],
        height=280,
    )
    st.dataframe(backtest, hide_index=True, width="stretch")
    st.warning(
        "All performance figures are historical and illustrative, not live results "
        "or a guarantee of future returns."
    )

with data_tab:
    st.subheader("Filtered source data")
    st.caption(f"Source: `{DEFAULT_DATA.name}` · {len(view):,} rows in selected range.")
    st.dataframe(
        view.sort_values("Date", ascending=False),
        hide_index=True,
        width="stretch",
    )
    csv_bytes = view.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data",
        data=csv_bytes,
        file_name="spy_dashboard_data.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Research and education only—not investment advice. The dashboard uses the "
    "latest date in the local dataset, which may not be current market data."
)
