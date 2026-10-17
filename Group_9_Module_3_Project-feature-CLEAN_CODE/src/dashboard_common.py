from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.modeling import FRED_SOURCES, add_technical_rule, run_model_comparison
from src.market_context import show_market_context


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "spy_labeled_features.csv"
DEFAULT_FRED_DIR = Path.home() / "Downloads" / "Fred add files"
REQUIRED_COLUMNS = {
    "Date",
    "Close",
    "High",
    "Low",
    "Open",
    "Volume",
    "Daily_Return",
    "SMA_20",
    "SMA_50",
    "Volatility_20",
    "RSI_14",
    "Future_Return_5d",
}


@st.cache_data(show_spinner=False, max_entries=4)
def load_market_data(path: str, modified_ns: int) -> pd.DataFrame:
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"SPY data file not found: {data_path}")

    data = pd.read_csv(data_path, parse_dates=["Date"])
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(
            "The selected CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    data = data.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    for column in REQUIRED_COLUMNS.difference({"Date"}):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(
        subset=["Date", "Close", "SMA_20", "SMA_50", "RSI_14", "Future_Return_5d"]
    )
    if data.empty:
        raise ValueError("The SPY data file contains no usable rows.")
    return add_technical_rule(data)


@st.cache_data(show_spinner=False, max_entries=4)
def load_model_analysis(
    market_path: str,
    fred_directory: str,
    market_signature: tuple[int, int],
    fred_signature: tuple[tuple[str, int, int], ...],
    selected_features: tuple[str, ...],
    signal_threshold: float,
    test_fraction: float,
    transaction_cost: float,
    lstm_epochs: int,
) -> dict[str, object]:
    del market_signature, fred_signature
    market_path_obj = Path(market_path)
    market = load_market_data(
        str(market_path_obj),
        market_path_obj.stat().st_mtime_ns,
    )
    return run_model_comparison(
        market,
        fred_directory,
        transaction_cost=transaction_cost,
        signal_threshold=signal_threshold,
        test_fraction=test_fraction,
        selected_features=list(selected_features),
        lstm_epochs=lstm_epochs,
    )


def file_signature(path: Path) -> tuple[int, int]:
    if not path.is_file():
        return (0, 0)
    stats = path.stat()
    return stats.st_size, stats.st_mtime_ns


def directory_signature(path: Path) -> tuple[tuple[str, int, int], ...]:
    if not path.is_dir():
        return ()
    return tuple(
        sorted(
            (item.name, item.stat().st_size, item.stat().st_mtime_ns)
            for item in path.glob("*.csv")
            if item.is_file()
        )
    )


def get_market_data() -> pd.DataFrame:
    signature = file_signature(DEFAULT_DATA)
    try:
        market = load_market_data(str(DEFAULT_DATA), signature[1])
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Unable to load the SPY dataset: {exc}")
        st.info("Run the notebook data-preparation cells to create the labeled SPY CSV.")
        st.stop()

    latest = market.iloc[-1]
    st.caption(
        f"Local SPY data: {market['Date'].min().date()} to "
        f"{latest['Date'].date()} · {len(market):,} observations"
    )
    return market


def clear_cached_data() -> None:
    load_market_data.clear()
    load_model_analysis.clear()
    st.session_state.pop("model_analysis", None)
    st.session_state.pop("model_analysis_key", None)


def show_sidebar_data_status() -> None:
    with st.sidebar:
        st.header("Local data")
        st.code(str(DEFAULT_DATA), language=None)
        st.caption(f"FRED folder: {DEFAULT_FRED_DIR}")


AVAILABLE_MACRO_FEATURES = sorted(
    {series_id for series_id, _ in FRED_SOURCES.values()} | {"EPU"}
)
