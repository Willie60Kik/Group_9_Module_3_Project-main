from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.time_series_analysis import (
    _finite_sample_quantile,
    _ses_forecasts,
    run_extended_time_series_analysis,
)


class TimeSeriesAnalysisTests(unittest.TestCase):
    def test_ses_only_uses_five_session_mature_labels(self) -> None:
        values = pd.Series([0.01, 0.02, 0.03, 0.9, -0.8, 0.7, -0.6, 0.5])

        forecast = _ses_forecasts(values, np.asarray([7]), alpha=1.0)

        self.assertEqual(forecast[0], 0.03)

    def test_empirical_interval_quantile_uses_upper_order_statistic(self) -> None:
        result = _finite_sample_quantile(np.asarray([0.1, 0.2, 0.3, 0.4]), 0.8)

        self.assertEqual(result, 0.4)

    def test_empirical_interval_requires_calibration_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "calibration errors"):
            _finite_sample_quantile(np.asarray([]), 0.9)

    def test_extended_analysis_returns_interactive_chart_data(self) -> None:
        rng = np.random.default_rng(7)
        count = 320
        frame = pd.DataFrame(
            {
                "Date": pd.bdate_range("2018-01-01", periods=count),
                "Close": 100 * np.cumprod(1 + rng.normal(0.0002, 0.01, count)),
                "Daily_Return": rng.normal(0.0002, 0.01, count),
                "Future_Return_5d": rng.normal(0.001, 0.02, count),
                "MACRO_A": rng.normal(size=count),
                "MACRO_B": rng.normal(size=count),
            }
        )
        strategies = pd.DataFrame(
            [
                {
                    "Strategy": name,
                    "Annualized volatility (%)": 12.0 + index,
                    "Sharpe ratio": 0.5 + index * 0.1,
                    "Equity return (%)": 5.0 + index,
                    "Win rate (%)": 50.0 + index,
                    "Maximum drawdown (%)": -10.0 + index,
                }
                for index, name in enumerate(
                    [
                        "Technical indicator",
                        "SVM (RBF)",
                        "HistGradientBoosting",
                        "XGBoost",
                        "LSTM",
                        "Buy and hold",
                    ]
                )
            ]
        )

        result = run_extended_time_series_analysis(
            frame,
            ["MACRO_A", "MACRO_B"],
            strategies,
            render_plots=False,
        )

        self.assertEqual(len(result["holdout_chart"]), len(result["holdout_predictions"]))
        self.assertIn("Linear regression lower bound (%)", result["holdout_chart"])
        self.assertIn("Previous session return (%)", result["lag_chart"])
        self.assertIn("Month", result["seasonal_chart"])
        self.assertEqual(result["correlation_chart"].shape, (3, 3))
        self.assertIn("Strategy", result["radar_chart"])
        self.assertEqual(len(result["next_forecast"]), 2)


if __name__ == "__main__":
    unittest.main()
