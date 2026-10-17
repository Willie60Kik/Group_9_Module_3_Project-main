from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard_common import (
    AVAILABLE_MACRO_FEATURES,
    DEFAULT_DATA,
    DEFAULT_FRED_DIR,
    directory_signature,
    file_signature,
    get_market_data,
    load_model_analysis,
    show_market_context,
)
from src.time_series_analysis import run_extended_time_series_analysis


st.subheader("Macro-only ML comparison")
st.caption(
    "SVM, HistGradientBoosting, XGBoost, and LSTM use selected FRED economic "
    "features only. SPY technical indicators remain on the Technical indicators page."
)
market = get_market_data()
market_signature = file_signature(DEFAULT_DATA)

with st.form("model_run_form"):
    fred_directory_text = st.text_input(
        "Folder containing the downloaded FRED CSV files",
        value=str(DEFAULT_FRED_DIR),
        help="For example: C:\\Users\\you\\Downloads\\Fred add files",
    )
    selected_features = st.multiselect(
        "Economic features for ML",
        options=AVAILABLE_MACRO_FEATURES,
        default=AVAILABLE_MACRO_FEATURES,
        help="Only these macroeconomic series and their changes are model inputs.",
    )
    left_settings, right_settings = st.columns(2)
    with left_settings:
        signal_threshold_percent = st.slider(
            "BUY / SELL predicted-return threshold (%)",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
            help=(
                "Predicted return above +threshold is BUY; below -threshold is "
                "SELL; between the two is HOLD. This changes ML signals, not the "
                "technical-rule conditions."
            ),
        )
        test_fraction_percent = st.slider(
            "Chronological holdout size (%)",
            min_value=10,
            max_value=40,
            value=20,
            step=5,
        )
    with right_settings:
        transaction_cost_percent = st.slider(
            "Transaction cost per side (%)",
            min_value=0.0,
            max_value=0.25,
            value=0.05,
            step=0.01,
        )
        lstm_epochs = st.slider(
            "LSTM training epochs",
            min_value=5,
            max_value=40,
            value=18,
            step=1,
        )
    run_models = st.form_submit_button(
        "Train models and compare", icon=":material/model_training:"
    )

fred_directory = Path(fred_directory_text).expanduser()
fred_signature = directory_signature(fred_directory)
analysis_key = (
    str(DEFAULT_DATA.resolve()),
    market_signature,
    str(fred_directory.resolve()),
    fred_signature,
    tuple(selected_features),
    signal_threshold_percent,
    test_fraction_percent,
    transaction_cost_percent,
    lstm_epochs,
)
if run_models:
    try:
        with st.spinner("Training SVM, HistGradientBoosting, XGBoost, and LSTM..."):
            result = load_model_analysis(
                str(DEFAULT_DATA),
                str(fred_directory),
                market_signature,
                fred_signature,
                tuple(selected_features),
                signal_threshold_percent / 100,
                test_fraction_percent / 100,
                transaction_cost_percent / 100,
                lstm_epochs,
            )
        st.session_state["model_analysis"] = result
        st.session_state["model_analysis_key"] = analysis_key
    except (OSError, ValueError, ImportError, pd.errors.ParserError) as exc:
        st.error(f"Unable to train and compare the models: {exc}")
        if isinstance(exc, ImportError):
            st.info(
                "Install dependencies in the active environment with "
                "`python -m pip install -r requirements.txt`."
            )

result = st.session_state.get("model_analysis")
if result is None or st.session_state.get("model_analysis_key") != analysis_key:
    st.info("Set the comparison options, then select **Train models and compare**.")
    st.stop()

as_of = pd.Timestamp(result["forecast_as_of"])
st.write(
    f"**Projection basis:** SPY close ${result['forecast_close']:,.2f} on "
    f"{as_of.date()} · next five trading sessions"
)
if (pd.Timestamp.now().normalize() - as_of).days > 7:
    st.warning(
        "The local SPY history is not current. These projections are based on the "
        "historical date above, not live forecasts."
    )
st.caption(
    f"Chronological holdout: {pd.Timestamp(result['test_start']).date()} to "
    f"{pd.Timestamp(result['test_end']).date()} · non-overlapping five-session "
    f"strategy windows · {result['transaction_cost_per_side']:.2f}% cost per side · "
    f"±{result['signal_threshold']:.1%} BUY/SELL threshold"
)

if result["missing_sources"]:
    st.warning("Optional FRED files not found: " + ", ".join(result["missing_sources"]))

st.subheader("Next five-session projections")
st.dataframe(
    result["forecasts"],
    hide_index=True,
    column_config={
        "Predicted five-session return (%)": st.column_config.NumberColumn(
            "Predicted return", format="%+.2f%%"
        ),
        "Projected SPY close": st.column_config.NumberColumn(
            "Implied close", format="$%.2f"
        ),
    },
    alt="Macro-only model projections for the next five SPY trading sessions",
)
show_market_context(
    "macro-only model predictions and the selected predicted-return threshold",
    signal_as_of=as_of,
)

st.subheader("Out-of-sample metrics")
st.caption(
    "CAGR uses compounded five-session strategy returns annualized over 252 "
    "trading sessions. Win rate is calculated on invested windows; Sharpe assumes "
    "a zero risk-free rate. Compare ML results against both benchmark rows."
)
st.dataframe(
    result["metrics"],
    hide_index=True,
    column_config={
        "Annualized volatility (%)": st.column_config.NumberColumn(
            "Volatility", format="%.2f%%"
        ),
        "Sharpe ratio": st.column_config.NumberColumn("Sharpe", format="%.2f"),
        "Equity return (%)": st.column_config.NumberColumn(
            "Equity return", format="%+.2f%%"
        ),
        "CAGR (%)": st.column_config.NumberColumn("CAGR", format="%+.2f%%"),
        "Win rate (%)": st.column_config.NumberColumn("Win rate", format="%.1f%%"),
        "Maximum drawdown (%)": st.column_config.NumberColumn(
            "Max drawdown", format="%.2f%%"
        ),
        "MAE (%)": st.column_config.NumberColumn("Return MAE", format="%.3f%%"),
        "Balanced signal accuracy (%)": st.column_config.NumberColumn(
            "Balanced accuracy", format="%.1f%%"
        ),
    },
    alt="Out-of-sample metrics for model strategies and benchmarks",
)

with st.expander("Evaluation criteria"):
    st.markdown(
        """
        - **CAGR / equity return:** higher is better after costs, but compare with
          buy-and-hold and the technical rule.
        - **Sharpe ratio:** higher indicates stronger return relative to variability.
        - **Drawdown / volatility:** lower indicates less downside and variability.
        - **Win rate:** interpret with average profit/loss; it can be high while
          total returns are poor.
        - **Return MAE:** lower is better; compare with the training-mean forecast.
        - **Balanced accuracy:** should beat the train-majority signal baseline.

        Prefer consistent improvements across return, risk, and forecast error.
        Tune on earlier validation data, not repeatedly against the final holdout.
        """
    )

st.subheader("Interactive time-series chart explorer")
st.caption(
    "Choose a chart and date range. Hover for values, drag to zoom, and use the "
    "legend to show or hide series. Regression intervals are empirical 90% "
    "prediction bands calibrated from earlier walk-forward residuals."
)
chart_min = pd.Timestamp(result["frame"]["Date"].min()).date()
chart_max = pd.Timestamp(result["frame"]["Date"].max()).date()
chart_range = st.date_input(
    "Chart date range",
    value=(chart_min, chart_max),
    min_value=chart_min,
    max_value=chart_max,
    key="ml_time_series_date_range",
)
if isinstance(chart_range, tuple) and len(chart_range) == 2:
    chart_start, chart_end = chart_range
else:
    chart_start = chart_end = chart_range

chart_choice = st.selectbox(
    "Chart type",
    [
        "Actual vs predicted with intervals",
        "Prediction scatter",
        "Daily-return lag plot",
        "Seasonal monthly returns",
        "Monthly returns by year",
        "Macro correlation matrix",
        "Strategy radar",
    ],
    key="ml_time_series_chart_type",
)

extended_key = ("extended_time_series", analysis_key)
extended_result = st.session_state.get("extended_time_series")
if st.session_state.get("extended_time_series_key") != extended_key:
    analysis_frame = result["frame"].merge(
        market[["Date", "Daily_Return"]],
        on="Date",
        how="left",
        validate="one_to_one",
    )
    with st.spinner("Preparing interactive time-series charts..."):
        extended_result = run_extended_time_series_analysis(
            frame=analysis_frame,
            feature_columns=result["feature_columns"],
            strategy_metrics=result["metrics"],
            holdout_fraction=float(result["test_fraction"]),
            interval_level=0.90,
            render_plots=False,
        )
    st.session_state["extended_time_series"] = extended_result
    st.session_state["extended_time_series_key"] = extended_key

def filter_chart_dates(data: pd.DataFrame) -> pd.DataFrame:
    return data.loc[
        pd.to_datetime(data["Date"]).dt.date.between(chart_start, chart_end)
    ].copy()


if chart_choice == "Actual vs predicted with intervals":
    plot_data = filter_chart_dates(extended_result["holdout_chart"])
    if plot_data.empty:
        st.info("No holdout forecast dates fall in the selected chart range.")
    else:
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=plot_data["Date"],
                y=plot_data["Linear regression lower bound (%)"],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=plot_data["Date"],
                y=plot_data["Linear regression upper bound (%)"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(31, 119, 180, 0.12)",
                name="Linear regression 90% interval",
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=plot_data["Date"],
                y=plot_data["Exponential smoothing lower bound (%)"],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=plot_data["Date"],
                y=plot_data["Exponential smoothing upper bound (%)"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(255, 127, 14, 0.12)",
                name="Smoothing 90% interval",
                hoverinfo="skip",
            )
        )
        for column, label in [
            ("Actual return (%)", "Actual"),
            ("Linear regression (%)", "Multivariate linear regression"),
            ("Exponential smoothing (%)", "Exponential smoothing"),
        ]:
            figure.add_trace(
                go.Scatter(
                    x=plot_data["Date"],
                    y=plot_data[column],
                    mode="lines",
                    name=label,
                )
            )
        figure.update_layout(
            title="Five-session SPY return predictions with empirical 90% intervals",
            xaxis_title="Forecast date",
            yaxis_title="Five-session return (%)",
            hovermode="x unified",
        )
        st.plotly_chart(figure, width="stretch", key="time_series_interval_chart")
elif chart_choice == "Prediction scatter":
    plot_data = filter_chart_dates(extended_result["holdout_chart"])
    if plot_data.empty:
        st.info("No holdout forecast dates fall in the selected chart range.")
    else:
        scatter_data = plot_data.melt(
            id_vars=["Date", "Actual return (%)"],
            value_vars=["Linear regression (%)", "Exponential smoothing (%)"],
            var_name="Forecast method",
            value_name="Predicted return (%)",
        )
        figure = px.scatter(
            scatter_data,
            x="Actual return (%)",
            y="Predicted return (%)",
            color="Forecast method",
            hover_data=["Date"],
            title="Actual vs predicted five-session SPY returns",
        )
        minimum = min(
            scatter_data["Actual return (%)"].min(),
            scatter_data["Predicted return (%)"].min(),
        )
        maximum = max(
            scatter_data["Actual return (%)"].max(),
            scatter_data["Predicted return (%)"].max(),
        )
        figure.add_shape(
            type="line",
            x0=minimum,
            y0=minimum,
            x1=maximum,
            y1=maximum,
            line={"dash": "dash", "color": "gray"},
        )
        st.plotly_chart(figure, width="stretch", key="time_series_scatter_chart")
elif chart_choice == "Daily-return lag plot":
    lag_data = extended_result["lag_chart"].rename_axis("Date").reset_index()
    lag_data = filter_chart_dates(lag_data)
    figure = px.scatter(
        lag_data,
        x="Previous session return (%)",
        y="Current session return (%)",
        hover_data=["Date"],
        title="SPY daily-return lag plot (one trading session)",
        opacity=0.55,
    )
    st.plotly_chart(figure, width="stretch", key="time_series_lag_chart")
elif chart_choice == "Seasonal monthly returns":
    seasonal_data = filter_chart_dates(extended_result["seasonal_chart"])
    figure = px.line(
        seasonal_data,
        x="Month",
        y="Monthly return (%)",
        color="Year",
        markers=True,
        title="Seasonal monthly SPY returns by year",
        labels={"Month": "Calendar month"},
    )
    figure.update_xaxes(dtick=1)
    st.plotly_chart(figure, width="stretch", key="time_series_seasonal_chart")
elif chart_choice == "Monthly returns by year":
    seasonal_data = filter_chart_dates(extended_result["seasonal_chart"])
    figure = px.box(
        seasonal_data,
        x="Year",
        y="Monthly return (%)",
        points="outliers",
        title="Distribution of monthly SPY returns by year",
    )
    st.plotly_chart(figure, width="stretch", key="time_series_box_chart")
elif chart_choice == "Macro correlation matrix":
    correlation = extended_result["correlation_chart"]
    figure = px.imshow(
        correlation,
        x=correlation.columns,
        y=correlation.index,
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu",
        title="Training-period FRED feature and target correlations",
        aspect="auto",
    )
    st.plotly_chart(figure, width="stretch", key="time_series_correlation_chart")
else:
    radar = extended_result["radar_chart"]
    radar_value_columns = [
        column for column in radar.columns if column != "Strategy"
    ]
    radar_long = radar.melt(
        id_vars="Strategy",
        value_vars=radar_value_columns,
        var_name="Metric",
        value_name="Relative score",
    )
    figure = px.line_polar(
        radar_long,
        r="Relative score",
        theta="Metric",
        color="Strategy",
        line_close=True,
        range_r=[0, 1],
        title="Relative holdout strategy profile (normalized 0-1)",
    )
    st.plotly_chart(figure, width="stretch", key="time_series_radar_chart")

st.subheader("Actual versus selected ML model")
model_names = [
    name
    for name in result["forecasts"]["Model"].tolist()
    if f"{name} predicted return" in result["scored"].columns
]
selected_model = st.selectbox("ML model", options=model_names)
comparison = result["scored"][
    ["Date", "Future_Return_5d", f"{selected_model} predicted return"]
].rename(
    columns={
        "Future_Return_5d": "Actual five-session return (%)",
        f"{selected_model} predicted return": "Predicted five-session return (%)",
    }
)
comparison["Actual five-session return (%)"] *= 100
comparison["Predicted five-session return (%)"] *= 100
error = (
    comparison["Actual five-session return (%)"]
    - comparison["Predicted five-session return (%)"]
)
with st.container(horizontal=True):
    st.metric("Model", selected_model, border=True)
    st.metric("Holdout return MAE", f"{error.abs().mean():.3f}%", border=True)
    st.metric(
        "Mean actual return",
        f"{comparison['Actual five-session return (%)'].mean():+.2f}%",
        border=True,
    )
    st.metric(
        "Mean predicted return",
        f"{comparison['Predicted five-session return (%)'].mean():+.2f}%",
        border=True,
    )
st.caption(
    "These daily forecast windows overlap. Portfolio metrics above use "
    "non-overlapping five-session windows. Realized values are only known after "
    "each five-session period has elapsed."
)
st.line_chart(
    comparison.set_index("Date"),
    alt=f"Actual and {selected_model} predicted five-session SPY returns",
)
st.dataframe(
    comparison.sort_values("Date", ascending=False).head(100),
    hide_index=True,
    column_config={
        "Date": st.column_config.DateColumn("Forecast date"),
        "Actual five-session return (%)": st.column_config.NumberColumn(
            "Actual return", format="%+.2f%%"
        ),
        "Predicted five-session return (%)": st.column_config.NumberColumn(
            "Predicted return", format="%+.2f%%"
        ),
    },
    alt=f"Recent actual and {selected_model} predicted returns",
)
st.download_button(
    "Download actual versus predicted CSV",
    data=comparison.to_csv(index=False).encode("utf-8"),
    file_name=f"{selected_model.lower().replace(' ', '_')}_actual_vs_predicted.csv",
    mime="text/csv",
)

st.subheader("Strategy equity curves")
st.line_chart(
    result["equity_curve"].set_index("Date"),
    alt="Holdout equity curves for ML strategies and comparison benchmarks",
)

st.caption(
    "FRED observations use conservative availability lags: one day for daily "
    "series, 45 days for monthly series, and 120 days for quarterly GDP. Standard "
    "FRED downloads may contain later revisions and are not point-in-time vintages."
)
