from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.modeling import (
    _compare_strategies,
    _performance_metrics,
    _signal_from_return,
    _signal_from_threshold,
    load_macro_features,
)


class ModelingTests(unittest.TestCase):
    def test_fred_values_are_not_available_before_release_lag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "Fred Effective Fund Interest DFF.csv").write_text(
                "observation_date,DFF\n1/1/2023,3.0\n2/1/2023,3.1\n",
                encoding="utf-8",
            )
            (folder / "FRED INFLATION CONSUMER INDEX CPIAUCSL.csv").write_text(
                "observation_date,CPIAUCSL\n1/1/2023,100\n1/2/2023,101\n",
                encoding="utf-8",
            )
            (folder / "Fred Production Index Output INDPRO.csv").write_text(
                "observation_date,INDPRO\n1/1/2023,100\n1/2/2023,101\n",
                encoding="utf-8",
            )
            dates = pd.Series(pd.to_datetime(["2023-01-01", "2023-01-02", "2023-02-16"]))

            macro, features, _ = load_macro_features(dates, folder)

        self.assertTrue(pd.isna(macro.loc[0, "DFF"]))
        self.assertEqual(macro.loc[1, "DFF"], 3.0)
        self.assertEqual(macro.loc[2, "CPIAUCSL"], 100.0)
        self.assertTrue(all("RSI" not in column for column in features))
        self.assertTrue(all("SMA" not in column for column in features))
        self.assertIn("DFF_change_21d", features)

    def test_return_thresholds_are_exclusive_and_hold_is_inclusive(self) -> None:
        self.assertEqual(_signal_from_return(0.01001), "BUY")
        self.assertEqual(_signal_from_return(0.01), "HOLD")
        self.assertEqual(_signal_from_return(-0.01), "HOLD")
        self.assertEqual(_signal_from_return(-0.01001), "SELL")
        self.assertEqual(_signal_from_threshold(0.20, 0.20), "HOLD")
        self.assertEqual(_signal_from_threshold(0.20001, 0.20), "BUY")
        self.assertEqual(_signal_from_threshold(-0.20001, 0.20), "SELL")

    def test_metrics_use_invested_windows_and_compounded_equity(self) -> None:
        returns = np.array([0.02, -0.01, 0.03])
        positions = np.array([1, 1, 0])

        metrics = _performance_metrics(returns, positions)

        self.assertAlmostEqual(
            metrics["Equity return (%)"],
            ((1.02 * 0.99 * 1.03) - 1) * 100,
        )
        expected_cagr = (1.02 * 0.99 * 1.03) ** (252 / (5 * 3)) - 1
        self.assertAlmostEqual(metrics["CAGR (%)"], expected_cagr * 100)
        self.assertEqual(metrics["Win rate (%)"], 50.0)
        self.assertLess(metrics["Maximum drawdown (%)"], 0)
        self.assertTrue(np.isfinite(metrics["Sharpe ratio"]))

    def test_strategy_comparison_includes_technical_and_model_signals(self) -> None:
        data = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=10, freq="B"),
                "Future_Return_5d": [0.02, 0.01, -0.02, 0.01, 0.02] * 2,
                "Rule_Signal": ["BUY", "HOLD", "SELL", "BUY", "HOLD"] * 2,
            }
        )
        predictions = pd.DataFrame(
            {
                "Date": data["Date"],
                "SVM (RBF) predicted return": [0.02] * 10,
                "SVM (RBF) signal": ["BUY"] * 10,
            }
        )

        metrics, equity, _ = _compare_strategies(
            data,
            predictions,
            np.arange(10),
            transaction_cost=0.0005,
            signal_threshold=0.01,
            baseline_signal="HOLD",
            baseline_return=0.0,
        )

        self.assertEqual(
            set(metrics["Strategy"]),
            {
                "Technical indicator",
                "SVM (RBF)",
                "Buy and hold",
                "Train-majority signal baseline",
            },
        )
        self.assertIn("SVM (RBF)", equity.columns)
        self.assertEqual(len(equity), 2)
        baseline = metrics.loc[
            metrics["Strategy"] == "Train-majority signal baseline"
        ].iloc[0]
        self.assertTrue(np.isfinite(baseline["MAE (%)"]))


if __name__ == "__main__":
    unittest.main()
