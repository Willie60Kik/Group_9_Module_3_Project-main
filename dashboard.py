from __future__ import annotations

from pathlib import Path

import numpy as np
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
    "Price_SMA20_Ratio",
    "Price_SMA50_Ratio",
    "Future_Return_5d",
    "Target",
}
TARGET_NAMES = {-1: "SELL", 0: "HOLD", 1: "BUY"}
MODEL_FEATURES = [
    "Daily_Return",
    "Price_SMA20_Ratio",
    "Price_SMA50_Ratio",
    "Volatility_20",
    "RSI_14",
]
FOLD_MODEL_PARAMETERS = [
    {
        "max_depth": 4,
        "max_features": 0.8,
        "min_samples_leaf": 20,
        "n_estimators": 400,
    },
    {
        "max_depth": 4,
        "max_features": "sqrt",
        "min_samples_leaf": 5,
        "n_estimators": 400,
    },
    {
        "max_depth": 4,
        "max_features": 0.8,
        "min_samples_leaf": 5,
        "n_estimators": 400,
    },
    {
        "max_depth": 4,
        "max_features": 0.8,
        "min_samples_leaf": 5,
        "n_estimators": 400,
    },
    {
        "max_depth": 8,
        "max_features": "sqrt",
        "min_samples_leaf": 20,
        "n_estimators": 400,
    },
]
STRATEGY_NAMES = [
    "Random Forest (out-of-fold)",
    "RSI/MACD/SMA rules",
    "Buy and hold",
]


st.set_page_config(
    page_title="SPY Trading Research Dashboard",
    page_icon=":material/query_stats:",
    layout="wide",
)


def summarize_price_performance(close: pd.Series) -> dict[str, float]:
    """Summarize close-price performance for the selected date range."""
    close = close.dropna()
    if len(close) < 2 or close.iloc[0] <= 0:
        return {
            "price_return": float("nan"),
            "annualized_return": float("nan"),
            "annualized_volatility": float("nan"),
            "sharpe_ratio": float("nan"),
            "max_drawdown": float("nan"),
        }

    daily_returns = close.pct_change().dropna()
    daily_volatility = daily_returns.std(ddof=1)
    sharpe_ratio = (
        daily_returns.mean() / daily_volatility * (252**0.5)
        if pd.notna(daily_volatility) and daily_volatility > 0
        else float("nan")
    )
    price_return = close.iloc[-1] / close.iloc[0] - 1

    return {
        "price_return": float(price_return),
        "annualized_return": float(
            (1 + price_return) ** (252 / len(daily_returns)) - 1
        ),
        "annualized_volatility": float(daily_volatility * (252**0.5)),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float((close / close.cummax() - 1).min()),
    }


@st.cache_data(show_spinner="Generating chronological out-of-fold predictions...")
def generate_oof_predictions(
    data: pd.DataFrame,
    buy_threshold: float,
    sell_threshold: float,
) -> pd.DataFrame:
    """Fit notebook CV folds using labels built from the selected return thresholds."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import TimeSeriesSplit
    except ImportError as exc:
        raise RuntimeError(
            "Scikit-learn is required for the interactive model comparison. "
            "Install the project's notebook dependencies and reload the app."
        ) from exc

    features = data[MODEL_FEATURES]
    forward_returns = data["Future_Return_5d"]
    labels = pd.Series(
        np.select(
            [forward_returns > buy_threshold, forward_returns < -sell_threshold],
            [1, -1],
            default=0,
        ),
        index=data.index,
    )
    predictions = np.full(len(data), np.nan)
    splitter = TimeSeriesSplit(n_splits=5, gap=5)

    for fold, (train_positions, test_positions) in enumerate(splitter.split(features)):
        model = RandomForestClassifier(
            **FOLD_MODEL_PARAMETERS[fold],
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(features.iloc[train_positions], labels.iloc[train_positions])
        predictions[test_positions] = model.predict(features.iloc[test_positions])

    result = data[["Date", "Future_Return_5d", "Rule_Signal"]].copy()
    result["Target"] = labels
    result["Model_Prediction"] = predictions
    return result


def simulate_long_cash(
    actions: np.ndarray,
    forward_returns: np.ndarray,
    cost_per_side: float,
) -> np.ndarray:
    """Apply long-or-cash BUY/SELL/HOLD actions and charge turnover costs."""
    portfolio_returns = []
    invested = 0
    for action, market_return in zip(actions, forward_returns, strict=True):
        if action == 1:
            target_position = 1
        elif action == -1:
            target_position = 0
        else:
            target_position = invested

        turnover = abs(target_position - invested)
        net_return = (1 + target_position * market_return) * (
            1 - cost_per_side
        ) ** turnover - 1
        portfolio_returns.append(net_return)
        invested = target_position

    if portfolio_returns and invested:
        portfolio_returns[-1] = (1 + portfolio_returns[-1]) * (1 - cost_per_side) - 1
    return np.asarray(portfolio_returns)


def summarize_strategy_returns(
    name: str, returns: np.ndarray
) -> tuple[dict[str, float | str | int], np.ndarray]:
    """Summarize five-session portfolio returns and return its equity path."""
    equity = np.concatenate(([1.0], np.cumprod(1 + returns)))
    drawdown = equity / np.maximum.accumulate(equity) - 1
    volatility = returns.std(ddof=1) if len(returns) > 1 else float("nan")
    sharpe_ratio = (
        returns.mean() / volatility * (252 / 5) ** 0.5
        if pd.notna(volatility) and volatility > 0
        else float("nan")
    )
    annualized_return = (
        equity[-1] ** (252 / (5 * len(returns))) - 1
        if len(returns) > 1 and equity[-1] > 0
        else float("nan")
    )
    return (
        {
            "Strategy": name,
            "Compounded return": equity[-1] - 1,
            "Annualized return": annualized_return,
            "Maximum drawdown": drawdown.min(),
            "Annualized volatility": volatility * (252 / 5) ** 0.5,
            "Sharpe ratio": sharpe_ratio,
            "Five-session periods": len(returns),
        },
        equity[1:],
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
    data = data.dropna(
        subset=[
            "Date",
            "Close",
            "Volume",
            "Daily_Return",
            "SMA_20",
            "SMA_50",
            "Volatility_20",
            "RSI_14",
            "Price_SMA20_Ratio",
            "Price_SMA50_Ratio",
            "Future_Return_5d",
            "Target",
        ]
    )

    if data.empty:
        raise ValueError("The market data file contains no usable SPY rows.")

    if not data["Target"].isin(TARGET_NAMES).all():
        raise ValueError(
            "Target labels must contain only -1 (SELL), 0 (HOLD), or 1 (BUY)."
        )

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
    "Historical SPY price, risk, technical-rule signals, and five-day research "
    "results. Model predictions are chronological out-of-fold research estimates, "
    "not live signals or trade execution."
)

with st.sidebar:
    st.header("Data")
    st.caption(
        "The dashboard reads the project CSV only; it does not call a market API."
    )
    if st.button("Reload project data"):
        load_market_data.clear()
    st.code(str(DEFAULT_DATA), language=None)

try:
    market = load_market_data(str(DEFAULT_DATA))
except (OSError, ValueError, pd.errors.ParserError) as exc:
    st.error(f"Unable to load the SPY dataset: {exc}")
    st.info(
        "Run the notebook data preparation cells to create the labeled feature CSV."
    )
    st.stop()

min_date = market["Date"].min().date()
max_date = market["Date"].max().date()
with st.sidebar:
    st.caption(f"Dataset coverage: {min_date} to {max_date}")
    selected_dates = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    transaction_cost_bps = st.slider(
        "Transaction cost per side (basis points)",
        min_value=0,
        max_value=50,
        value=5,
        help="Applied on each entry or exit for the model and technical strategies.",
    )
    buy_threshold_pct = st.slider(
        "BUY threshold: forward return above (%)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="A five-session return above this threshold is labeled BUY.",
    )
    sell_threshold_pct = st.slider(
        "SELL threshold: forward loss below (%)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="A five-session return below the negative of this threshold is labeled SELL.",
    )
    st.caption(
        "Returns above BUY are BUY; returns below SELL are SELL; the inclusive "
        "breakeven band between them is HOLD."
    )

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date = selected_dates[0] or min_date
    end_date = selected_dates[1] or selected_dates[0] or max_date
else:
    start_date = end_date = selected_dates

buy_threshold = buy_threshold_pct / 100
sell_threshold = sell_threshold_pct / 100
sell_boundary_pct = -sell_threshold_pct if sell_threshold_pct else 0.0
hold_band_text = (
    f"HOLD / breakeven band: {sell_boundary_pct:+.1f}% to {buy_threshold_pct:+.1f}%"
)

view = market.loc[market["Date"].dt.date.between(start_date, end_date)].copy()
if view.empty:
    st.warning("No observations fall within the selected date range.")
    st.stop()
latest = view.iloc[-1]
performance = summarize_price_performance(view["Close"])

with st.container(horizontal=True):
    st.metric(
        "Latest close",
        f"${latest['Close']:,.2f}",
        help=str(latest["Date"].date()),
        border=True,
    )
    st.metric("Latest daily return", f"{latest['Daily_Return']:+.2%}", border=True)
    st.metric("20-session volatility", f"{latest['Volatility_20']:.2%}", border=True)
    st.metric("RSI (14)", f"{latest['RSI_14']:.1f}", border=True)
    st.metric("Technical rule signal", latest["Rule_Signal"], border=True)

overview_tab, signals_tab, labels_tab, backtest_tab, data_tab = st.tabs(
    ["Overview", "Technical signals", "Target labels", "Backtest", "Data"]
)

with overview_tab:
    st.subheader("Signal and holdings calculator")
    st.caption(
        "Choose a valuation date, then use a manual five-session return scenario or "
        "the technical-rule signal on that date. Enter your SPY holdings to estimate "
        "profit or loss at the selected date's close."
    )
    valuation_dates = view["Date"].dt.date.tolist()
    valuation_date = st.selectbox(
        "Valuation date",
        options=valuation_dates,
        index=len(valuation_dates) - 1,
        format_func=lambda selected_date: selected_date.strftime("%Y-%m-%d"),
    )
    valuation_row = view.loc[view["Date"].dt.date.eq(valuation_date)].iloc[-1]
    valuation_price = float(valuation_row["Close"])
    signal_source = st.radio(
        "Signal source",
        [
            "Selected-date return vs. purchase price",
            "Manual expected return (%)",
            "Technical-rule signal on valuation date",
        ],
        horizontal=True,
    )
    signal_inputs = st.columns(3)
    shares_held = signal_inputs[0].number_input(
        "SPY shares held",
        min_value=0.0,
        value=0.0,
        step=1.0,
        help="Enter the number of SPY ETF shares currently held, including fractions.",
    )
    purchase_price = signal_inputs[1].number_input(
        "Average purchase price per share ($)",
        min_value=0.01,
        value=valuation_price,
        step=0.01,
        format="%.2f",
        help="Your average cost per share, used to estimate profit or loss.",
    )
    expected_return_pct = None
    if signal_source == "Manual expected return (%)":
        expected_return_pct = signal_inputs[2].number_input(
            "Expected five-session return (%)",
            min_value=-100.0,
            max_value=100.0,
            value=0.0,
            step=0.1,
            help="Your what-if input; this value is not a model forecast.",
        )

    holding_value = shares_held * valuation_price
    purchase_value = shares_held * purchase_price
    unrealized_profit_loss = holding_value - purchase_value
    unrealized_return = valuation_price / purchase_price - 1

    if signal_source == "Selected-date return vs. purchase price":
        holding_return_pct = unrealized_return * 100
        if holding_return_pct > buy_threshold_pct:
            position_signal = "BUY"
            signal_reason = (
                f"The selected close is {holding_return_pct:+.2f}% versus your "
                f"average purchase price, above the BUY threshold of "
                f"{buy_threshold_pct:+.1f}%."
            )
        elif holding_return_pct < -sell_threshold_pct:
            position_signal = "SELL"
            signal_reason = (
                f"The selected close is {holding_return_pct:+.2f}% versus your "
                f"average purchase price, below the SELL threshold of "
                f"{-sell_threshold_pct:+.1f}%."
            )
        else:
            position_signal = "HOLD"
            signal_reason = (
                f"The selected close is {holding_return_pct:+.2f}% versus your "
                f"average purchase price, within the breakeven band."
            )
    elif signal_source == "Manual expected return (%)":
        if expected_return_pct > buy_threshold_pct:
            position_signal = "BUY"
            signal_reason = (
                f"Entered return {expected_return_pct:+.1f}% is above the "
                f"BUY threshold of {buy_threshold_pct:+.1f}%."
            )
        elif expected_return_pct < -sell_threshold_pct:
            position_signal = "SELL"
            signal_reason = (
                f"Entered return {expected_return_pct:+.1f}% is below the "
                f"SELL threshold of {-sell_threshold_pct:+.1f}%."
            )
        else:
            position_signal = "HOLD"
            signal_reason = (
                f"Entered return {expected_return_pct:+.1f}% is inside the "
                f"{hold_band_text.removeprefix('HOLD / breakeven band: ')} band."
            )
    else:
        position_signal = str(valuation_row["Rule_Signal"])
        signal_reason = f"Technical-rule status for {valuation_date}."

    if position_signal == "BUY":
        st.success(f"Status: **BUY** — {signal_reason}")
    elif position_signal == "SELL":
        st.warning(f"Status: **SELL** — {signal_reason}")
    else:
        st.info(f"Status: **HOLD** — {signal_reason}")

    st.caption(
        f"Valuation price: ${valuation_price:,.2f} closing price on {valuation_date}. "
        "The dashboard's headline KPIs above continue to show the latest date in "
        "the selected range."
    )
    with st.container(horizontal=True):
        st.metric("Shares held", f"{shares_held:,.4f}".rstrip("0").rstrip("."))
        st.metric("Average purchase price", f"${purchase_price:,.2f}")
        st.metric("Estimated holding value", f"${holding_value:,.2f}")
        profit_loss_display = (
            f"+${unrealized_profit_loss:,.2f}"
            if unrealized_profit_loss >= 0
            else f"-${abs(unrealized_profit_loss):,.2f}"
        )
        st.metric("Estimated unrealized P/L", profit_loss_display)
        st.metric("Unrealized return", f"{unrealized_return:+.2%}")
    st.caption(
        "Estimated profit or loss is based on closing prices and excludes dividends, "
        "commissions, taxes, and any currency conversion."
    )

    left, right = st.columns([2, 1])
    with left:
        st.subheader("SPY price and moving averages")
        price_chart = view.set_index("Date")[["Close", "SMA_20", "SMA_50"]]
        st.line_chart(
            price_chart,
            height=380,
            alt="SPY closing price with 20-session and 50-session moving averages",
        )
    with right:
        st.subheader("Selected range")
        st.write(f"**From:** {view['Date'].min().date()}")
        st.write(f"**To:** {view['Date'].max().date()}")
        st.write(f"**Trading sessions:** {len(view):,}")
        st.write(f"**Volume on latest date:** {latest['Volume']:,.0f}")
        st.write(f"**Close vs. SMA 50:** {latest['Close'] / latest['SMA_50'] - 1:+.2%}")

    st.subheader("Price risk and return")
    with st.container(horizontal=True):
        st.metric(
            "Selected-period price return",
            (
                f"{performance['price_return']:+.2%}"
                if pd.notna(performance["price_return"])
                else "N/A"
            ),
            border=True,
        )
        st.metric(
            "Annualized price return",
            (
                f"{performance['annualized_return']:+.2%}"
                if pd.notna(performance["annualized_return"])
                else "N/A"
            ),
            border=True,
        )
        st.metric(
            "Annualized volatility",
            (
                f"{performance['annualized_volatility']:.2%}"
                if pd.notna(performance["annualized_volatility"])
                else "N/A"
            ),
            border=True,
        )
        st.metric(
            "Maximum drawdown",
            (
                f"{performance['max_drawdown']:.2%}"
                if pd.notna(performance["max_drawdown"])
                else "N/A"
            ),
            border=True,
        )
        st.metric(
            "Sharpe ratio",
            (
                f"{performance['sharpe_ratio']:.2f}"
                if pd.notna(performance["sharpe_ratio"])
                else "N/A"
            ),
            help="Annualized from daily close-price returns; assumes a zero risk-free rate.",
            border=True,
        )

    drawdown = view.assign(Drawdown=view["Close"] / view["Close"].cummax() - 1)
    st.line_chart(
        drawdown.set_index("Date")[["Drawdown"]],
        height=220,
        alt="SPY close-price drawdown from the previous peak during the selected range",
    )
    st.caption(
        "Risk statistics use unadjusted closing prices, exclude dividends, and assume "
        "252 trading sessions per year. They describe the selected date range, not a strategy."
    )

    st.subheader("Daily trading volume")
    st.bar_chart(
        view.set_index("Date")[["Volume"]],
        height=220,
        alt="Daily SPY trading volume in the selected date range",
    )

with signals_tab:
    st.subheader("Rule-based technical indicators")
    st.caption(
        "This indicator signal is not a Random Forest prediction and is not a live "
        "recommendation. BUY requires all three bullish conditions; SELL occurs when "
        "any condition turns bearish."
    )
    signal = latest["Rule_Signal"]
    if signal == "BUY":
        st.success(
            "Latest rule signal: BUY — price above SMA 50, RSI above 50, and MACD above its signal."
        )
    elif signal == "SELL":
        st.warning(
            "Latest rule signal: SELL — at least one of the bullish trend conditions is not met."
        )
    else:
        st.info("Latest rule signal: HOLD — no buy or sell condition was triggered.")

    previous_signal = view["Rule_Signal"].shift()
    signal_changes = view["Rule_Signal"].ne(previous_signal) & previous_signal.notna()
    signal_counts = view["Rule_Signal"].value_counts(normalize=True)
    with st.container(horizontal=True):
        for name in ("BUY", "HOLD", "SELL"):
            st.metric(
                f"{name} sessions", f"{signal_counts.get(name, 0):.1%}", border=True
            )
        st.metric("Signal changes", f"{int(signal_changes.sum()):,}", border=True)

    indicator_columns = st.columns(3)
    indicator_columns[0].metric("SMA 20", f"${latest['SMA_20']:,.2f}")
    indicator_columns[1].metric("SMA 50", f"${latest['SMA_50']:,.2f}")
    indicator_columns[2].metric(
        "20-session volatility", f"{latest['Volatility_20']:.2%}"
    )

    st.subheader("RSI and MACD history")
    st.line_chart(
        view.set_index("Date")[["RSI_14"]],
        height=250,
        alt="SPY 14-session relative strength index over time",
    )
    st.line_chart(
        view.set_index("Date")[["MACD", "MACD_Signal"]],
        height=250,
        alt="SPY MACD line compared with its signal line over time",
    )

    st.subheader("Recent rule-signal changes")
    recent = view.loc[signal_changes].tail(20).copy()
    st.dataframe(
        recent.sort_values("Date", ascending=False)[
            ["Date", "Close", "SMA_50", "RSI_14", "MACD", "MACD_Signal", "Rule_Signal"]
        ],
        hide_index=True,
        alt="Most recent dates when the technical-rule signal changed",
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
        "that five-day period has elapsed. Change the BUY and SELL thresholds in the "
        "sidebar to redefine the target classes."
    )
    st.caption(hold_band_text)
    view["Selected_Target"] = np.select(
        [
            view["Future_Return_5d"] > buy_threshold,
            view["Future_Return_5d"] < -sell_threshold,
        ],
        [1, -1],
        default=0,
    )
    view["Selected_Target_Name"] = view["Selected_Target"].map(TARGET_NAMES)
    target_columns = st.columns(3)
    proportions = view["Selected_Target_Name"].value_counts(normalize=True)
    for column, name in zip(target_columns, ("BUY", "HOLD", "SELL")):
        column.metric(f"{name} labels", f"{proportions.get(name, 0):.1%}")

    st.bar_chart(
        view["Selected_Target_Name"]
        .value_counts()
        .reindex(["BUY", "HOLD", "SELL"], fill_value=0),
        height=260,
        alt="Count of historical five-session target labels",
    )
    st.subheader("Recent realized targets")
    recent_targets = (
        view[["Date", "Future_Return_5d", "Selected_Target_Name"]].tail(50).copy()
    )
    recent_targets["Realized next-five-day return (%)"] = (
        recent_targets.pop("Future_Return_5d") * 100
    )
    st.dataframe(
        recent_targets.sort_values("Date", ascending=False),
        hide_index=True,
        alt="Recent realized five-session returns and their retrospective labels",
        column_config={
            "Date": st.column_config.DateColumn("Date"),
            "Realized next-five-day return (%)": st.column_config.NumberColumn(
                "Realized next-five-day return", format="%+.2f%%"
            ),
            "Selected_Target_Name": st.column_config.TextColumn("Retrospective target"),
        },
        width="stretch",
    )

with backtest_tab:
    st.subheader("Date-range strategy comparison")
    st.caption(
        "Compare chronological out-of-fold Random Forest predictions, RSI/MACD/SMA "
        "rules, and buy-and-hold over the selected signal dates. All strategies use "
        "the same non-overlapping five-session forward-return periods."
    )
    st.caption(
        f"{hold_band_text}. The selected thresholds also define model training labels."
    )
    strategies = st.multiselect(
        "Strategies to compare",
        options=STRATEGY_NAMES,
        default=STRATEGY_NAMES,
    )

    try:
        oof = generate_oof_predictions(market, buy_threshold, sell_threshold)
    except (ImportError, RuntimeError, ValueError) as exc:
        st.error(f"Unable to generate model predictions: {exc}")
        st.stop()

    prediction_rows = oof.loc[oof["Model_Prediction"].notna()]
    if not prediction_rows.empty:
        st.caption(
            "Out-of-fold model coverage: "
            f"{prediction_rows['Date'].min().date()} to "
            f"{prediction_rows['Date'].max().date()}. Earlier rows have no held-out "
            "predictions and are excluded from this comparison."
        )
    trade_rows = prediction_rows.iloc[::5].copy()
    trade_rows = trade_rows.loc[
        trade_rows["Date"].dt.date.between(start_date, end_date)
    ].reset_index(drop=True)
    if trade_rows.empty:
        st.warning(
            "There are no out-of-fold five-session periods in this date range. "
            "Choose dates with available model predictions."
        )
    elif not strategies:
        st.info("Select at least one strategy to display its results.")
    else:
        from sklearn.metrics import balanced_accuracy_score

        cost_per_side = transaction_cost_bps / 10_000
        actions_by_strategy = {
            STRATEGY_NAMES[0]: trade_rows["Model_Prediction"].astype(int).to_numpy(),
            STRATEGY_NAMES[1]: trade_rows["Rule_Signal"]
            .map({"BUY": 1, "HOLD": 0, "SELL": -1})
            .to_numpy(dtype=int),
            STRATEGY_NAMES[2]: np.ones(len(trade_rows), dtype=int),
        }
        forward_returns = trade_rows["Future_Return_5d"].to_numpy(dtype=float)
        summary_rows = []
        equity_data = {"Date": trade_rows["Date"]}
        export_data = pd.DataFrame(
            {
                "Signal date": trade_rows["Date"],
                "BUY threshold (%)": buy_threshold_pct,
                "SELL threshold (%)": sell_threshold_pct,
                "Model prediction": trade_rows["Model_Prediction"].astype(int),
                "Realized target": trade_rows["Target"],
                "Realized next-five-session return": forward_returns,
            }
        )

        for strategy in strategies:
            strategy_returns = simulate_long_cash(
                actions_by_strategy[strategy],
                forward_returns,
                cost_per_side,
            )
            summary, equity = summarize_strategy_returns(strategy, strategy_returns)
            summary_rows.append(summary)
            equity_data[strategy] = equity
            export_data[f"{strategy} return"] = strategy_returns
            export_data[f"{strategy} equity"] = equity

        summary_frame = pd.DataFrame(summary_rows)
        percent_columns = [
            "Compounded return",
            "Annualized return",
            "Maximum drawdown",
            "Annualized volatility",
        ]
        summary_frame[percent_columns] *= 100
        model_score = balanced_accuracy_score(
            trade_rows["Target"],
            trade_rows["Model_Prediction"].astype(int),
        )
        with st.container(horizontal=True):
            st.metric("Five-session periods", f"{len(trade_rows):,}", border=True)
            st.metric(
                "Model balanced accuracy",
                f"{model_score:.1%}",
                help=(
                    "Calculated only on out-of-fold model predictions in the selected "
                    "range, compared with the realized five-session labels."
                ),
                border=True,
            )
            st.metric(
                "Transaction cost per side",
                f"{transaction_cost_bps} bps",
                border=True,
            )

        st.subheader("Growth of $1")
        st.line_chart(
            pd.DataFrame(equity_data).set_index("Date"),
            height=360,
            alt="Selected-date-range growth of one dollar for each selected strategy",
        )
        st.subheader("Risk and return comparison")
        st.dataframe(
            summary_frame,
            hide_index=True,
            alt="Selected-date-range compounded return and risk statistics by strategy",
            column_config={
                "Compounded return": st.column_config.NumberColumn(
                    format="%.2f%%",
                ),
                "Annualized return": st.column_config.NumberColumn(
                    format="%.2f%%",
                ),
                "Maximum drawdown": st.column_config.NumberColumn(
                    format="%.2f%%",
                ),
                "Annualized volatility": st.column_config.NumberColumn(
                    format="%.2f%%",
                ),
                "Sharpe ratio": st.column_config.NumberColumn(
                    format="%.2f",
                ),
                "Five-session periods": st.column_config.NumberColumn(
                    format="%d",
                ),
            },
            width="stretch",
        )
        st.download_button(
            "Download selected-period comparison",
            data=export_data.to_csv(index=False).encode("utf-8"),
            file_name="spy_strategy_comparison.csv",
            mime="text/csv",
        )

    st.info(
        "The model uses the notebook's five chronological folds and fold-specific "
        "hyperparameters; each prediction is made on a held-out fold, with a five-row "
        "gap between training and validation. Strategies are sampled every five "
        "out-of-fold signal dates, start in cash at the selected range, and use the "
        "same realized forward returns. Each signal is evaluated over the next five "
        "sessions, so the final signal's return can extend beyond the selected end date. "
        "The fold hyperparameters were tuned at the default +/-1% thresholds; changing "
        "thresholds redefines labels and refits folds, but does not retune those parameters."
    )
    st.warning(
        "Historical research only—not live predictions or investment advice. Results "
        "depend on the dataset, model, date range, five-session sampling, and assumed "
        "transaction costs; they do not include dividends, slippage, taxes, or all "
        "execution risks."
    )

with data_tab:
    st.subheader("Filtered source data")
    st.caption(f"Source: `{DEFAULT_DATA.name}` · {len(view):,} rows in selected range.")
    st.dataframe(
        view.sort_values("Date", ascending=False),
        hide_index=True,
        alt="Filtered historical SPY prices, technical features, and retrospective targets",
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
    f"Research and education only—not investment advice. Latest available observation: "
    f"{max_date}. The dataset may not reflect current market conditions."
)
