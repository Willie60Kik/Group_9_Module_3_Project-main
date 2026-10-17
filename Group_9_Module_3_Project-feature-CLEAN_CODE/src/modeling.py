from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


HORIZON = 5
SEQUENCE_LENGTH = 20
TEST_FRACTION = 0.2
TRANSACTION_COST = 0.0005
SIGNAL_THRESHOLD = 0.01

FRED_SOURCES: dict[str, tuple[str, int]] = {
    "fred effective fund interest dff.csv": ("DFF", 1),
    "fred inflation consumer index cpiaucsl.csv": ("CPIAUCSL", 45),
    "fred market yield us treasury 10 yrs gs10.csv": ("DGS10", 1),
    "fred nominal us dollar index dtwexbgs.csv": ("DTWEXBGS", 1),
    "fred oil price index dcoilwtico.csv": ("DCOILWTICO", 1),
    "fred production index output indpro.csv": ("INDPRO", 45),
    "fred real dollar index rtwexbgs.csv": ("RTWEXBGS", 45),
    "fred real gross domestic product gdpc1.csv": ("GDPC1", 120),
    "fred unemployment rate unrate.csv": ("UNRATE", 45),
    "fred us treasury interest 2 yrs dgs2.csv": ("DGS2", 1),
    "fred consumer sentiment index umcsent.csv": ("UMCSENT", 45),
}
EPU_FILENAME = "us economic uncertainty categorical_epu_data.csv"
PERCENT_CHANGE_SERIES = {
    "CPIAUCSL",
    "DTWEXBGS",
    "DCOILWTICO",
    "INDPRO",
    "RTWEXBGS",
    "GDPC1",
    "EPU",
}


def add_technical_rule(market: pd.DataFrame) -> pd.DataFrame:
    """Create the separate long-or-cash SMA/RSI/MACD technical rule."""
    result = market.copy().sort_values("Date").reset_index(drop=True)
    result["MACD"] = (
        result["Close"].ewm(span=12, adjust=False).mean()
        - result["Close"].ewm(span=26, adjust=False).mean()
    )
    result["MACD_Signal"] = result["MACD"].ewm(span=9, adjust=False).mean()

    buy = (
        (result["Close"] > result["SMA_50"])
        & (result["RSI_14"] > 50)
        & (result["MACD"] > result["MACD_Signal"])
    )
    sell = (
        (result["Close"] < result["SMA_50"])
        | (result["RSI_14"] < 50)
        | (result["MACD"] < result["MACD_Signal"])
    )
    result["Rule_Signal"] = np.select(
        [buy, ~buy & sell],
        ["BUY", "SELL"],
        default="HOLD",
    )
    return result


def _parse_fred_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(
        values.astype("string").str.strip(),
        format="mixed",
        dayfirst=True,
        errors="coerce",
    )


def _load_fred_series(
    path: Path,
    series_id: str,
    release_lag_days: int,
    market_dates: pd.DataFrame,
) -> pd.DataFrame:
    source = pd.read_csv(path)
    if "observation_date" not in source.columns or series_id not in source.columns:
        raise ValueError(
            f"{path.name} must contain 'observation_date' and '{series_id}' columns."
        )

    values = pd.DataFrame(
        {
            "Available_Date": _parse_fred_dates(source["observation_date"])
            + pd.Timedelta(days=release_lag_days),
            series_id: pd.to_numeric(source[series_id], errors="coerce"),
        }
    )
    values = (
        values.dropna()
        .sort_values("Available_Date")
        .drop_duplicates("Available_Date", keep="last")
    )
    return pd.merge_asof(
        market_dates.sort_values("Date"),
        values,
        left_on="Date",
        right_on="Available_Date",
        direction="backward",
    )[["Date", series_id]]


def _load_epu(path: Path, market_dates: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_csv(path)
    required = {"Year", "Month"}
    if not required.issubset(source.columns):
        raise ValueError(f"{path.name} must contain Year and Month columns.")
    epu_columns = [
        column
        for column in source.columns
        if column.lower().startswith("1. economic policy uncertainty")
    ]
    if not epu_columns:
        raise ValueError(
            f"{path.name} must contain an Economic Policy Uncertainty column."
        )

    observations = pd.to_datetime(
        {
            "year": pd.to_numeric(source["Year"], errors="coerce"),
            "month": pd.to_numeric(source["Month"], errors="coerce"),
            "day": 1,
        },
        errors="coerce",
    )
    values = pd.DataFrame(
        {
            "Available_Date": observations + pd.Timedelta(days=45),
            "EPU": pd.to_numeric(source[epu_columns[0]], errors="coerce"),
        }
    )
    values = (
        values.dropna()
        .sort_values("Available_Date")
        .drop_duplicates("Available_Date", keep="last")
    )
    return pd.merge_asof(
        market_dates.sort_values("Date"),
        values,
        left_on="Date",
        right_on="Available_Date",
        direction="backward",
    )[["Date", "EPU"]]


def load_macro_features(
    market_dates: pd.Series,
    fred_directory: str | Path,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Align released FRED observations to SPY dates without using future releases.

    Only selected U.S. economic series are used as model inputs. Technical
    indicators and SPY price/return columns are deliberately excluded.
    """
    folder = Path(fred_directory)
    if not folder.is_dir():
        raise NotADirectoryError(f"FRED data folder not found: {folder}")

    dates = pd.DataFrame(
        {"Date": pd.to_datetime(market_dates).dropna().sort_values().unique()}
    )
    if dates.empty:
        raise ValueError("No valid SPY dates were provided.")

    macro = dates.copy()
    used_sources: list[str] = []
    missing_sources: list[str] = []
    available_files = {path.name.lower(): path for path in folder.glob("*.csv")}

    for filename, (series_id, lag_days) in FRED_SOURCES.items():
        path = available_files.get(filename)
        if path is None:
            missing_sources.append(filename)
            continue
        aligned = _load_fred_series(path, series_id, lag_days, dates)
        macro = macro.merge(aligned, on="Date", how="left", validate="one_to_one")
        used_sources.append(series_id)

    epu_path = available_files.get(EPU_FILENAME)
    if epu_path is None:
        missing_sources.append(EPU_FILENAME)
    else:
        macro = macro.merge(
            _load_epu(epu_path, dates),
            on="Date",
            how="left",
            validate="one_to_one",
        )
        used_sources.append("EPU")

    if len(used_sources) < 3:
        raise ValueError(
            "At least three supported FRED series are required. "
            f"Found: {', '.join(used_sources) or 'none'}."
        )

    base_features = list(used_sources)
    for series_id in base_features:
        if series_id in PERCENT_CHANGE_SERIES:
            change = macro[series_id].pct_change(periods=21, fill_method=None)
            macro[f"{series_id}_change_21d"] = change.replace(
                [np.inf, -np.inf], np.nan
            )
        else:
            macro[f"{series_id}_change_21d"] = macro[series_id].diff(periods=21)

    feature_columns = base_features + [f"{name}_change_21d" for name in base_features]
    return macro, feature_columns, missing_sources


def _make_tabular_models() -> dict[str, object]:
    from xgboost import XGBRegressor

    return {
        "SVM (RBF)": make_pipeline(
            StandardScaler(),
            SVR(C=10.0, epsilon=0.002, kernel="rbf"),
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=120,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        ),
        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=180,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            n_jobs=2,
            random_state=42,
            verbosity=0,
        ),
    }


def _build_sequences(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            values[position - SEQUENCE_LENGTH + 1 : position + 1]
            for position in positions
        ]
    ).astype(np.float32)


def _lstm_predict(
    features: pd.DataFrame,
    target: pd.Series,
    fit_positions: np.ndarray,
    predict_positions: np.ndarray,
    scale_end: int,
    epochs: int = 18,
) -> np.ndarray:
    import torch
    from torch import nn

    if fit_positions.size == 0 or predict_positions.size == 0:
        raise ValueError("Not enough rows to train or evaluate the LSTM.")

    torch.manual_seed(42)
    torch.set_num_threads(1)
    scaler = StandardScaler()
    scaled = np.zeros(features.shape, dtype=np.float32)
    scaled[:scale_end] = scaler.fit_transform(features.iloc[:scale_end]).astype(
        np.float32
    )
    if scale_end < len(features):
        scaled[scale_end:] = scaler.transform(
            features.iloc[scale_end:]
        ).astype(np.float32)

    fit_positions = fit_positions[fit_positions >= SEQUENCE_LENGTH - 1]
    predict_positions = predict_positions[predict_positions >= SEQUENCE_LENGTH - 1]
    x_train = torch.from_numpy(_build_sequences(scaled, fit_positions))
    y_train = torch.from_numpy(
        target.iloc[fit_positions].to_numpy(dtype=np.float32)
    )
    x_predict = torch.from_numpy(_build_sequences(scaled, predict_positions))

    class ReturnLSTM(nn.Module):
        def __init__(self, input_size: int) -> None:
            super().__init__()
            self.recurrent = nn.LSTM(
                input_size=input_size,
                hidden_size=32,
                num_layers=1,
                batch_first=True,
            )
            self.output = nn.Linear(32, 1)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            sequence_output, _ = self.recurrent(inputs)
            return self.output(sequence_output[:, -1]).squeeze(-1)

    model = ReturnLSTM(features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    loss_function = nn.MSELoss()
    batch_size = 64

    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(x_train))
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            loss = loss_function(model(x_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        return model(x_predict).cpu().numpy().astype(float)


def _signal_from_return(predicted_return: float) -> str:
    return _signal_from_threshold(predicted_return, SIGNAL_THRESHOLD)


def _signal_from_threshold(predicted_return: float, threshold: float) -> str:
    if predicted_return > threshold:
        return "BUY"
    if predicted_return < -threshold:
        return "SELL"
    return "HOLD"


def _simulate_position(
    signals: pd.Series,
    realized_returns: pd.Series,
    transaction_cost: float,
    initially_invested: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    position = int(initially_invested)
    returns = np.zeros(len(signals), dtype=float)
    positions = np.zeros(len(signals), dtype=int)

    for index, signal in enumerate(signals):
        previous_position = position
        if signal == "BUY":
            position = 1
        elif signal == "SELL":
            position = 0
        positions[index] = position
        returns[index] = (
            position * float(realized_returns.iloc[index])
            - abs(position - previous_position) * transaction_cost
        )

    return returns, positions


def _performance_metrics(
    returns: np.ndarray,
    positions: np.ndarray,
    benchmark_signal: str | None = None,
) -> dict[str, float]:
    if len(returns) == 0:
        raise ValueError("Cannot calculate performance metrics on empty returns.")
    equity = np.cumprod(1 + returns)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]
    drawdown = equity / peaks - 1
    volatility = (
        float(np.std(returns, ddof=1) * np.sqrt(252 / HORIZON))
        if len(returns) > 1
        else 0.0
    )
    standard_deviation = (
        float(np.std(returns, ddof=1))
        if len(returns) > 1
        else 0.0
    )
    sharpe = (
        float(np.mean(returns) / standard_deviation * np.sqrt(252 / HORIZON))
        if standard_deviation > 0
        else 0.0
    )
    cagr = (
        float(equity[-1] ** (252 / (HORIZON * len(returns))) - 1)
        if equity[-1] > 0
        else -1.0
    )
    invested = positions == 1
    win_rate = float(np.mean(returns[invested] > 0)) if invested.any() else 0.0
    metrics = {
        "Annualized volatility (%)": volatility * 100,
        "Sharpe ratio": sharpe,
        "Equity return (%)": float((equity[-1] - 1) * 100),
        "CAGR (%)": cagr * 100,
        "Win rate (%)": win_rate * 100,
        "Maximum drawdown (%)": float(drawdown.min() * 100),
    }
    if benchmark_signal is not None:
        metrics["Balanced signal accuracy (%)"] = float(benchmark_signal)
    return metrics


def _compare_strategies(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    test_positions: np.ndarray,
    transaction_cost: float,
    signal_threshold: float,
    baseline_signal: str,
    baseline_return: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scored = data.iloc[test_positions].copy().reset_index(drop=True)
    for column in predictions.columns:
        if column != "Date":
            scored[column] = predictions[column].to_numpy()

    sampled = scored.iloc[::HORIZON].reset_index(drop=True)
    realized = sampled["Future_Return_5d"]
    model_names = [
        column.removesuffix(" signal")
        for column in predictions.columns
        if column.endswith(" signal")
    ]
    strategy_signals = {
        "Technical indicator": sampled["Rule_Signal"],
        **{name: sampled[f"{name} signal"] for name in model_names},
    }

    metrics_rows: list[dict[str, float | str]] = []
    equity_columns: dict[str, np.ndarray] = {"Date": sampled["Date"]}
    for name, signals in strategy_signals.items():
        returns, positions = _simulate_position(
            signals.reset_index(drop=True),
            realized.reset_index(drop=True),
            transaction_cost,
        )
        equity_columns[name] = np.cumprod(1 + returns)
        row: dict[str, float | str] = {"Strategy": name}
        row.update(_performance_metrics(returns, positions))
        metrics_rows.append(row)

    buy_hold_returns, buy_hold_positions = _simulate_position(
        pd.Series(["BUY"] * len(sampled)),
        realized.reset_index(drop=True),
        transaction_cost,
        initially_invested=False,
    )
    equity_columns["Buy and hold"] = np.cumprod(1 + buy_hold_returns)
    buy_hold_row: dict[str, float | str] = {"Strategy": "Buy and hold"}
    buy_hold_row.update(_performance_metrics(buy_hold_returns, buy_hold_positions))
    metrics_rows.append(buy_hold_row)

    mae = {
        column.removesuffix(" predicted return"): mean_absolute_error(
            scored["Future_Return_5d"], scored[column]
        )
        for column in predictions.columns
        if column.endswith(" predicted return")
    }
    actual_signal = scored["Future_Return_5d"].map(
        lambda value: _signal_from_threshold(float(value), signal_threshold)
    )
    balanced_accuracy = {
        column.removesuffix(" signal"): balanced_accuracy_score(
            actual_signal, scored[column]
        )
        for column in predictions.columns
        if column.endswith(" signal")
    }
    for row in metrics_rows:
        strategy = str(row["Strategy"])
        row["MAE (%)"] = mae.get(strategy, np.nan) * 100
        row["Balanced signal accuracy (%)"] = balanced_accuracy.get(
            strategy, np.nan
        ) * 100
    metrics_rows.append(
        {
            "Strategy": "Train-majority signal baseline",
            "Annualized volatility (%)": np.nan,
            "Sharpe ratio": np.nan,
            "Equity return (%)": np.nan,
            "CAGR (%)": np.nan,
            "Win rate (%)": np.nan,
            "Maximum drawdown (%)": np.nan,
            "MAE (%)": mean_absolute_error(
                scored["Future_Return_5d"],
                np.full(len(scored), baseline_return),
            )
            * 100,
            "Balanced signal accuracy (%)": balanced_accuracy_score(
                actual_signal, [baseline_signal] * len(actual_signal)
            )
            * 100,
        }
    )

    return (
        pd.DataFrame(metrics_rows),
        pd.DataFrame(equity_columns),
        scored,
    )


def run_model_comparison(
    market: pd.DataFrame,
    fred_directory: str | Path,
    transaction_cost: float = TRANSACTION_COST,
    signal_threshold: float = SIGNAL_THRESHOLD,
    test_fraction: float = TEST_FRACTION,
    selected_features: list[str] | None = None,
    lstm_epochs: int = 18,
) -> dict[str, object]:
    """Train separate macro-only models, compare strategies, and project five sessions."""
    required = {"Date", "Close", "Future_Return_5d", "SMA_50", "RSI_14"}
    missing = required.difference(market.columns)
    if missing:
        raise ValueError(
            "SPY dataset is missing required columns: " + ", ".join(sorted(missing))
        )
    if transaction_cost < 0:
        raise ValueError("Transaction cost cannot be negative.")
    if not 0 < signal_threshold < 1:
        raise ValueError("Signal threshold must be between 0 and 100 percent.")
    if not 0.1 <= test_fraction <= 0.5:
        raise ValueError("Test fraction must be between 10% and 50%.")
    if not 1 <= lstm_epochs <= 100:
        raise ValueError("LSTM epochs must be between 1 and 100.")

    technical = add_technical_rule(market)
    macro, feature_columns, missing_sources = load_macro_features(
        technical["Date"], fred_directory
    )
    if selected_features is not None:
        unknown_features = set(selected_features).difference(
            column
            for column in feature_columns
            if not column.endswith("_change_21d")
        )
        if unknown_features:
            raise ValueError(
                "Unsupported FRED model feature(s): "
                + ", ".join(sorted(unknown_features))
            )
        if not selected_features:
            raise ValueError("Select at least one FRED feature for ML training.")
        feature_columns = selected_features + [
            f"{name}_change_21d" for name in selected_features
        ]
    frame = technical[
        ["Date", "Close", "Future_Return_5d", "Rule_Signal"]
    ].merge(macro, on="Date", how="inner", validate="one_to_one")
    frame = frame.dropna(subset=feature_columns + ["Future_Return_5d"])
    frame = frame.sort_values("Date").reset_index(drop=True)
    if len(frame) < 500:
        raise ValueError(
            f"Only {len(frame)} usable observations remain after aligning FRED "
            "release dates. At least 500 are required for this comparison."
        )

    features = frame[feature_columns].astype(float)
    target = frame["Future_Return_5d"].astype(float)
    split = int(len(frame) * (1 - test_fraction))
    train_positions = np.arange(SEQUENCE_LENGTH - 1, split - HORIZON)
    test_positions = np.arange(
        max(split, SEQUENCE_LENGTH - 1),
        len(frame) - HORIZON,
    )
    if train_positions.size < 100 or test_positions.size < HORIZON:
        raise ValueError("Not enough usable rows for chronological training and testing.")
    target_signal = target.iloc[train_positions].map(
        lambda value: _signal_from_threshold(float(value), signal_threshold)
    )
    baseline_return = float(target.iloc[train_positions].mean())
    signal_classes = ("BUY", "HOLD", "SELL")
    baseline_signal = target_signal.value_counts().reindex(
        signal_classes, fill_value=0
    ).idxmax()

    prediction_frame = pd.DataFrame({"Date": frame.iloc[test_positions]["Date"].values})
    forecast_position = len(frame) - 1
    forecast_results: list[dict[str, float | str]] = []

    for name, estimator in _make_tabular_models().items():
        estimator.fit(features.iloc[train_positions], target.iloc[train_positions])
        prediction_frame[f"{name} predicted return"] = estimator.predict(
            features.iloc[test_positions]
        )

        forecast_estimator = _make_tabular_models()[name]
        known_label_positions = np.arange(SEQUENCE_LENGTH - 1, len(frame) - HORIZON)
        forecast_estimator.fit(
            features.iloc[known_label_positions],
            target.iloc[known_label_positions],
        )
        forecast_value = float(
            forecast_estimator.predict(features.iloc[[forecast_position]])[0]
        )
        forecast_results.append(
            {
                "Model": name,
                "Predicted five-session return (%)": forecast_value * 100,
                "Signal": _signal_from_threshold(forecast_value, signal_threshold),
                "Projected SPY close": float(frame.iloc[-1]["Close"] * (1 + forecast_value)),
            }
        )

    lstm_test = _lstm_predict(
        features,
        target,
        train_positions,
        test_positions,
        scale_end=int(train_positions[-1] + 1),
        epochs=lstm_epochs,
    )
    prediction_frame["LSTM predicted return"] = lstm_test
    known_label_positions = np.arange(SEQUENCE_LENGTH - 1, len(frame) - HORIZON)
    lstm_forecast = _lstm_predict(
        features,
        target,
        known_label_positions,
        np.array([forecast_position]),
        scale_end=len(frame) - HORIZON,
        epochs=lstm_epochs,
    )[0]
    forecast_results.append(
        {
            "Model": "LSTM",
            "Predicted five-session return (%)": float(lstm_forecast * 100),
            "Signal": _signal_from_threshold(float(lstm_forecast), signal_threshold),
            "Projected SPY close": float(frame.iloc[-1]["Close"] * (1 + lstm_forecast)),
        }
    )

    for name in ("SVM (RBF)", "HistGradientBoosting", "XGBoost", "LSTM"):
        prediction_frame[f"{name} signal"] = prediction_frame[
            f"{name} predicted return"
        ].map(lambda value: _signal_from_threshold(float(value), signal_threshold))

    metrics, equity_curve, scored = _compare_strategies(
        frame,
        prediction_frame,
        test_positions,
        transaction_cost,
        signal_threshold,
        baseline_signal,
        baseline_return,
    )
    return {
        "frame": frame,
        "feature_columns": feature_columns,
        "missing_sources": missing_sources,
        "metrics": metrics,
        "equity_curve": equity_curve,
        "scored": scored,
        "predictions": prediction_frame,
        "forecasts": pd.DataFrame(forecast_results),
        "test_start": frame.iloc[test_positions[0]]["Date"],
        "test_end": frame.iloc[test_positions[-1]]["Date"],
        "forecast_as_of": frame.iloc[forecast_position]["Date"],
        "forecast_close": float(frame.iloc[forecast_position]["Close"]),
        "transaction_cost_per_side": transaction_cost * 100,
        "signal_threshold": signal_threshold,
        "test_fraction": test_fraction,
        "baseline_signal": baseline_signal,
    }
