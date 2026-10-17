from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.modeling import HORIZON, SEQUENCE_LENGTH


def _finite_sample_quantile(errors: np.ndarray, level: float) -> float:
    if errors.size == 0:
        raise ValueError("Cannot estimate a prediction interval without calibration errors.")
    quantile_level = min(math.ceil((errors.size + 1) * level) / errors.size, 1.0)
    return float(np.quantile(errors, quantile_level, method="higher"))


def _ses_forecasts(
    target: pd.Series,
    positions: np.ndarray,
    alpha: float,
) -> np.ndarray:
    forecasts = []
    for position in positions:
        last_available_label = int(position) - HORIZON
        if last_available_label < 0:
            raise ValueError("Insufficient mature labels for exponential smoothing.")
        history = target.iloc[: last_available_label + 1].dropna()
        if history.empty:
            raise ValueError("No mature target labels available for smoothing.")
        forecasts.append(float(history.ewm(alpha=alpha, adjust=False).mean().iloc[-1]))
    return np.asarray(forecasts, dtype=float)


def _model() -> Pipeline:
    return make_pipeline(StandardScaler(), LinearRegression())


def _validate_analysis_inputs(
    frame: pd.DataFrame,
    feature_columns: list[str],
    strategy_metrics: pd.DataFrame,
) -> None:
    required = {"Date", "Close", "Daily_Return", "Future_Return_5d"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Time-series analysis data is missing columns: "
            + ", ".join(sorted(missing))
        )
    missing_features = set(feature_columns).difference(frame.columns)
    if missing_features:
        raise ValueError(
            "Time-series analysis is missing model features: "
            + ", ".join(sorted(missing_features))
        )
    if "Strategy" not in strategy_metrics.columns:
        raise ValueError("Strategy metrics must include a Strategy column.")
    if len(frame) < 250:
        raise ValueError("At least 250 aligned observations are required.")


def _make_plots(
    frame: pd.DataFrame,
    train_positions: np.ndarray,
    holdout_actual: np.ndarray,
    regression_predictions: np.ndarray,
    smoothing_predictions: np.ndarray,
    feature_columns: list[str],
    strategy_metrics: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    figure, axis = plt.subplots(figsize=(8, 7))
    axis.scatter(
        holdout_actual * 100,
        regression_predictions * 100,
        alpha=0.45,
        label="Multivariate linear regression",
    )
    axis.scatter(
        holdout_actual * 100,
        smoothing_predictions * 100,
        alpha=0.35,
        label="Exponential smoothing",
    )
    minimum = min(holdout_actual.min(), regression_predictions.min(), smoothing_predictions.min()) * 100
    maximum = max(holdout_actual.max(), regression_predictions.max(), smoothing_predictions.max()) * 100
    axis.plot([minimum, maximum], [minimum, maximum], "k--", label="Perfect prediction")
    axis.set(
        title="Actual vs predicted five-session SPY returns (holdout)",
        xlabel="Actual return (%)",
        ylabel="Predicted return (%)",
    )
    axis.legend()
    figure.tight_layout()
    plt.show()

    daily_returns = pd.to_numeric(frame["Daily_Return"], errors="coerce").dropna()
    lagged = pd.DataFrame(
        {"Previous session return": daily_returns.shift(1), "Current return": daily_returns}
    ).dropna()
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(
        lagged["Previous session return"] * 100,
        lagged["Current return"] * 100,
        alpha=0.3,
        s=14,
    )
    axis.axhline(0, color="black", linewidth=0.7)
    axis.axvline(0, color="black", linewidth=0.7)
    axis.set(
        title="Lag plot of daily SPY returns (lag 1 trading session)",
        xlabel="Previous session return (%)",
        ylabel="Current session return (%)",
    )
    figure.tight_layout()
    plt.show()

    close = pd.Series(
        pd.to_numeric(frame["Close"], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(pd.to_datetime(frame["Date"])),
    ).dropna()
    monthly_close = close.resample("ME").last()
    monthly_returns = monthly_close.pct_change().dropna().rename("Monthly return")
    monthly_frame = monthly_returns.to_frame()
    monthly_frame["Year"] = monthly_frame.index.year
    monthly_frame["Month"] = monthly_frame.index.month

    figure, axis = plt.subplots(figsize=(10, 5))
    for year, group in monthly_frame.groupby("Year"):
        axis.plot(
            group["Month"],
            group["Monthly return"] * 100,
            alpha=0.22,
            linewidth=0.8,
            color="tab:blue",
        )
    median_by_month = monthly_frame.groupby("Month")["Monthly return"].median() * 100
    axis.plot(
        median_by_month.index,
        median_by_month.values,
        marker="o",
        linewidth=2.5,
        color="darkorange",
        label="Median across years",
    )
    axis.set(
        title="Seasonal plot: monthly SPY returns by calendar month",
        xlabel="Calendar month",
        ylabel="Monthly return (%)",
        xticks=range(1, 13),
    )
    axis.legend()
    figure.tight_layout()
    plt.show()

    year_groups = [
        monthly_frame.loc[monthly_frame["Year"] == year, "Monthly return"].to_numpy() * 100
        for year in sorted(monthly_frame["Year"].unique())
    ]
    years = sorted(monthly_frame["Year"].unique())
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.boxplot(year_groups, tick_labels=years, showfliers=False)
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set(
        title="Box plot: monthly SPY return distribution by year",
        xlabel="Year",
        ylabel="Monthly return (%)",
    )
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    plt.show()

    correlation_data = frame.iloc[train_positions][
        feature_columns + ["Future_Return_5d"]
    ].astype(float)
    correlation = correlation_data.corr()
    figure, axis = plt.subplots(figsize=(max(9, len(correlation.columns) * 0.48), 8))
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(range(len(correlation.columns)))
    axis.set_yticks(range(len(correlation.columns)))
    axis.set_xticklabels(correlation.columns, rotation=90, fontsize=7)
    axis.set_yticklabels(correlation.columns, fontsize=7)
    axis.set_title("Training-period correlation matrix (FRED features and target)")
    figure.colorbar(image, ax=axis, label="Pearson correlation")
    figure.tight_layout()
    plt.show()

    radar_metrics = [
        ("Annualized volatility (%)", "Lower volatility"),
        ("Sharpe ratio", "Higher Sharpe"),
        ("Equity return (%)", "Higher return"),
        ("Win rate (%)", "Higher win rate"),
        ("Maximum drawdown (%)", "Smaller drawdown"),
    ]
    strategy_names = [
        "Technical indicator",
        "SVM (RBF)",
        "HistGradientBoosting",
        "XGBoost",
        "LSTM",
        "Buy and hold",
    ]
    available = strategy_metrics.loc[
        strategy_metrics["Strategy"].isin(strategy_names)
    ].set_index("Strategy")
    labels = [label for _, label in radar_metrics]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    figure, axis = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for strategy in strategy_names:
        if strategy not in available.index:
            continue
        scores = []
        for metric, _ in radar_metrics:
            if metric not in available.columns:
                raise ValueError(f"Strategy metrics are missing {metric!r}.")
            values = pd.to_numeric(available[metric], errors="coerce").dropna()
            if metric == "Maximum drawdown (%)":
                values = values.abs()
                score_values = values.max() - values
            elif metric == "Annualized volatility (%)":
                score_values = values.max() - values
            else:
                score_values = values - values.min()
            span = float(score_values.max() - score_values.min())
            if strategy not in score_values.index:
                scores.append(0.0)
            elif span == 0:
                scores.append(1.0)
            else:
                scores.append(float(score_values.loc[strategy] / span))
        scores += scores[:1]
        axis.plot(angles, scores, linewidth=1.4, label=strategy)
        axis.fill(angles, scores, alpha=0.04)
    axis.set_xticks(angles[:-1])
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 1)
    axis.set_title("Holdout strategy profile (relative normalized scores)")
    axis.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05), fontsize=8)
    figure.tight_layout()
    plt.show()


def run_extended_time_series_analysis(
    frame: pd.DataFrame,
    feature_columns: list[str],
    strategy_metrics: pd.DataFrame,
    *,
    holdout_fraction: float = 0.2,
    interval_level: float = 0.9,
    smoothing_alpha: float = 0.2,
    render_plots: bool = True,
) -> dict[str, object]:
    """Compare macro multivariate linear regression and SES with holdout plots."""
    _validate_analysis_inputs(frame, feature_columns, strategy_metrics)
    if not 0.1 <= holdout_fraction <= 0.5:
        raise ValueError("Holdout fraction must be between 10% and 50%.")
    if not 0.5 < interval_level < 1:
        raise ValueError("Prediction interval level must be between 50% and 100%.")
    if not 0 < smoothing_alpha <= 1:
        raise ValueError("Exponential-smoothing alpha must be in (0, 1].")

    ordered = frame.sort_values("Date").reset_index(drop=True)
    features = ordered[feature_columns].astype(float)
    target = ordered["Future_Return_5d"].astype(float)
    split = int(len(ordered) * (1 - holdout_fraction))
    train_positions = np.arange(SEQUENCE_LENGTH - 1, split - HORIZON)
    test_positions = np.arange(
        max(split, SEQUENCE_LENGTH - 1),
        len(ordered) - HORIZON,
    )
    if train_positions.size < 100 or test_positions.size < HORIZON:
        raise ValueError("Insufficient observations for the chronological holdout.")

    splitters = TimeSeriesSplit(n_splits=5, gap=HORIZON)
    regression_calibration_errors: list[float] = []
    smoothing_calibration_errors: list[float] = []
    train_features = features.iloc[train_positions]
    train_target = target.iloc[train_positions]
    for fold_train, fold_calibration in splitters.split(train_features):
        fitted = _model()
        fitted.fit(
            train_features.iloc[fold_train],
            train_target.iloc[fold_train],
        )
        calibrated_positions = train_positions[fold_calibration]
        calibrated_actual = target.iloc[calibrated_positions].to_numpy(dtype=float)
        calibrated_regression = fitted.predict(
            features.iloc[calibrated_positions]
        ).astype(float)
        regression_calibration_errors.extend(
            np.abs(calibrated_actual - calibrated_regression).tolist()
        )
        calibrated_smoothing = _ses_forecasts(
            target, calibrated_positions, smoothing_alpha
        )
        smoothing_calibration_errors.extend(
            np.abs(calibrated_actual - calibrated_smoothing).tolist()
        )

    regression_interval = _finite_sample_quantile(
        np.asarray(regression_calibration_errors),
        interval_level,
    )
    smoothing_interval = _finite_sample_quantile(
        np.asarray(smoothing_calibration_errors),
        interval_level,
    )

    regression = _model()
    regression.fit(train_features, train_target)
    actual = target.iloc[test_positions].to_numpy(dtype=float)
    regression_predictions = regression.predict(
        features.iloc[test_positions]
    ).astype(float)
    smoothing_predictions = _ses_forecasts(
        target, test_positions, smoothing_alpha
    )
    model_specs = [
        ("Multivariate linear regression", regression_predictions, regression_interval),
        ("Exponential smoothing", smoothing_predictions, smoothing_interval),
    ]

    prediction_table = pd.DataFrame(
        {
            "Date": ordered.iloc[test_positions]["Date"].to_numpy(),
            "Actual five-session return (%)": actual * 100,
        }
    )
    summary_rows = []
    for name, predictions, interval in model_specs:
        lower = predictions - interval
        upper = predictions + interval
        prediction_table[f"{name} predicted (%)"] = predictions * 100
        prediction_table[f"{name} lower {interval_level:.0%} bound (%)"] = lower * 100
        prediction_table[f"{name} upper {interval_level:.0%} bound (%)"] = upper * 100
        summary_rows.append(
            {
                "Model": name,
                "Holdout MAE (%)": mean_absolute_error(actual, predictions) * 100,
                "Holdout RMSE (%)": math.sqrt(
                    mean_squared_error(actual, predictions)
                )
                * 100,
                "Holdout R-squared": r2_score(actual, predictions),
                "Interval level": f"{interval_level:.0%}",
                "Lower error bound (%)": -interval * 100,
                "Upper error bound (%)": interval * 100,
                "Observed interval coverage (%)": np.mean(
                    (actual >= lower) & (actual <= upper)
                )
                * 100,
            }
        )

    latest_position = len(ordered) - 1
    mature_label_positions = np.arange(
        SEQUENCE_LENGTH - 1,
        len(ordered) - HORIZON,
    )
    latest_model = _model()
    latest_model.fit(
        features.iloc[mature_label_positions],
        target.iloc[mature_label_positions],
    )
    latest_regression = float(latest_model.predict(features.iloc[[latest_position]])[0])
    latest_smoothing = float(
        _ses_forecasts(
            target,
            np.asarray([latest_position]),
            smoothing_alpha,
        )[0]
    )
    next_forecast = pd.DataFrame(
        [
            {
                "Forecast as of": ordered.iloc[latest_position]["Date"],
                "Method": "Multivariate linear regression",
                "Predicted five-session return (%)": latest_regression * 100,
                "Lower empirical bound (%)": (latest_regression - regression_interval) * 100,
                "Upper empirical bound (%)": (latest_regression + regression_interval) * 100,
            },
            {
                "Forecast as of": ordered.iloc[latest_position]["Date"],
                "Method": "Exponential smoothing",
                "Predicted five-session return (%)": latest_smoothing * 100,
                "Lower empirical bound (%)": (latest_smoothing - smoothing_interval) * 100,
                "Upper empirical bound (%)": (latest_smoothing + smoothing_interval) * 100,
            },
        ]
    )

    holdout_chart = pd.DataFrame(
        {
            "Date": ordered.iloc[test_positions]["Date"].to_numpy(),
            "Actual return (%)": actual * 100,
            "Linear regression (%)": regression_predictions * 100,
            "Linear regression lower bound (%)": (
                regression_predictions - regression_interval
            )
            * 100,
            "Linear regression upper bound (%)": (
                regression_predictions + regression_interval
            )
            * 100,
            "Exponential smoothing (%)": smoothing_predictions * 100,
            "Exponential smoothing lower bound (%)": (
                smoothing_predictions - smoothing_interval
            )
            * 100,
            "Exponential smoothing upper bound (%)": (
                smoothing_predictions + smoothing_interval
            )
            * 100,
        }
    )
    daily_returns = pd.to_numeric(ordered["Daily_Return"], errors="coerce")
    lag_chart = pd.DataFrame(
        {
            "Previous session return (%)": daily_returns.shift(1) * 100,
            "Current session return (%)": daily_returns * 100,
        },
        index=pd.to_datetime(ordered["Date"]),
    ).dropna()
    monthly_close = pd.Series(
        pd.to_numeric(ordered["Close"], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(pd.to_datetime(ordered["Date"])),
    ).resample("ME").last()
    monthly_returns = monthly_close.pct_change().dropna().rename("Monthly return (%)")
    seasonal_chart = monthly_returns.mul(100).rename("Monthly return (%)").to_frame()
    seasonal_chart["Year"] = seasonal_chart.index.year.astype(str)
    seasonal_chart["Month"] = seasonal_chart.index.month
    seasonal_chart["Date"] = seasonal_chart.index
    correlation_chart = ordered.iloc[train_positions][
        feature_columns + ["Future_Return_5d"]
    ].astype(float).corr()
    radar_metrics = [
        ("Annualized volatility (%)", "Lower volatility"),
        ("Sharpe ratio", "Higher Sharpe"),
        ("Equity return (%)", "Higher return"),
        ("Win rate (%)", "Higher win rate"),
        ("Maximum drawdown (%)", "Smaller drawdown"),
    ]
    strategy_names = [
        "Technical indicator",
        "SVM (RBF)",
        "HistGradientBoosting",
        "XGBoost",
        "LSTM",
        "Buy and hold",
    ]
    strategy_frame = strategy_metrics.loc[
        strategy_metrics["Strategy"].isin(strategy_names)
    ].set_index("Strategy")
    radar_rows = []
    for strategy in strategy_names:
        if strategy not in strategy_frame.index:
            continue
        row: dict[str, object] = {"Strategy": strategy}
        for metric, label in radar_metrics:
            values = pd.to_numeric(strategy_frame[metric], errors="coerce").dropna()
            if metric in {"Maximum drawdown (%)", "Annualized volatility (%)"}:
                values = values.abs() if metric == "Maximum drawdown (%)" else values
                scores = values.max() - values
            else:
                scores = values - values.min()
            span = float(scores.max() - scores.min())
            row[label] = (
                1.0
                if span == 0 and strategy in scores.index
                else float(scores.loc[strategy] / span)
                if span > 0 and strategy in scores.index
                else 0.0
            )
        radar_rows.append(row)
    radar_chart = pd.DataFrame(radar_rows)

    if render_plots:
        _make_plots(
            ordered,
            train_positions,
            actual,
            regression_predictions,
            smoothing_predictions,
            feature_columns,
            strategy_metrics,
        )
    return {
        "holdout_summary": pd.DataFrame(summary_rows),
        "holdout_predictions": prediction_table,
        "next_forecast": next_forecast,
        "holdout_chart": holdout_chart,
        "lag_chart": lag_chart,
        "seasonal_chart": seasonal_chart,
        "correlation_chart": correlation_chart,
        "radar_chart": radar_chart,
        "calibration_observations": len(regression_calibration_errors),
        "interval_level": interval_level,
    }
