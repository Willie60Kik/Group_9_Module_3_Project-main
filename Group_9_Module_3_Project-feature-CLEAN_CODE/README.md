# S&P 500 Machine-Learning Trading Signal Project

## Project plan

The project tests whether historical SPY market data can support useful trading
signals. The machine-learning model is the primary approach. A technical-indicator
strategy is a baseline for comparison, not a replacement for the ML model.

1. **Collect market data** — Obtain daily SPY price and volume data. Record the
   source and date range, and handle missing or invalid observations.
2. **Prepare features** — Calculate lagged returns, moving averages, price-to-
   average ratios, volatility, RSI, and other indicators using information
   available on each prediction date only.
3. **Define the prediction target and trading decisions** — Forecast SPY's
   cumulative return over the next five trading days, and assign the target
   class using the rules below. At each prediction date, use only information
   available on that date. Keep future returns strictly in the target; never
   include them in model features.
4. **Train and validate chronologically** — Use older observations for training
   and newer observations for validation and final testing. Tune the model
   without using the held-out test period.
5. **Evaluate and simulate trading** — Report classification metrics, including
   balanced accuracy and a simple baseline. Apply the BUY/SELL/HOLD position
   rules above to model predictions, then compare out-of-sample returns with
   buy-and-hold and a separately defined rule-based RSI/MACD/SMA strategy.
   Include transaction costs and state any other simplifying assumptions.
6. **Present results and learn** — Summarize model performance, backtest
   outcomes, limitations, and next steps. A dashboard can display signals,
   performance, and benchmark comparisons; alerts are optional and should only
   be added if they are in project scope.

## Trading decision

The intended strategy is long-or-cash, not short-selling. Classify the
predicted five-trading-day cumulative SPY return and act on the model's predicted
class as follows:

| Predicted return | Signal | Portfolio action |
| --- | --- | --- |
| Greater than +1% | **BUY** | Enter or maintain a long SPY position |
| From -1% through +1%, inclusive | **HOLD** | Do not initiate a trade; keep any existing position unchanged |
| Less than -1% | **SELL** | Exit the SPY position to cash |

For the notebook's labels, these signals correspond to classes `1`, `0`, and
`-1`, respectively. The backtest should apply these actions to model
predictions—not to the realized future return—and account for entry/exit costs.

The comparison rule strategy enters or maintains a long position when the SPY
close is above its 50-day moving average, RSI(14) is above 50, and the MACD line
is above its signal line. It exits to cash when any of those conditions reverses;
otherwise, it holds the current position. It does not short SPY.

## Model comparison

The dashboard trains four regressors to predict the next five-session SPY
return: an RBF Support Vector Regressor, HistGradientBoosting, XGBoost, and an
LSTM. The ML predictors are selected FRED macroeconomic series and their
21-session changes only. SPY price-derived technical indicators are excluded
from those model inputs.

The technical baseline is a separate long-or-cash SMA/RSI/MACD rule. ML
predictions are converted to BUY/HOLD/SELL using adjustable five-session return
thresholds from +1%/-1% up to +20%/-20%. Model strategies, the technical rule,
and buy-and-hold are compared on the same chronological holdout using
non-overlapping five-session periods and a 0.05% cost per side. The dashboard
reports annualized volatility,
Sharpe ratio (zero risk-free-rate assumption), compounded equity return, CAGR,
win rate on invested periods, maximum drawdown, return MAE, and balanced signal
accuracy. A train-majority signal and train-mean return forecast are included as
simple model baselines.

The ML view is interactive: choose FRED features, a BUY/SELL predicted-return
threshold from 1% to 20%, the chronological holdout size, transaction costs,
and LSTM epochs, then retrain the comparison. A prediction above the selected
positive threshold is BUY, below its negative is SELL, and between the two is
HOLD. The threshold controls ML signals; it does not alter the separate
technical rule. CAGR is annualized from compounded non-overlapping
five-session returns using 252 trading sessions per year. It is not a promise of
future performance. Select any trained model to inspect its held-out
five-session predictions beside the realized returns for the same dates, view
the comparison chart/table, or download that model's actual-versus-predicted
CSV. Those detailed predictions use daily (overlapping) forecast windows;
portfolio metrics continue to use non-overlapping windows.

Economic series are merged by their observation dates only after conservative
availability lags (one day for daily series, 45 days for monthly series, and
120 days for quarterly GDP). The test split is chronological and training
labels whose five-session return overlaps the test period are purged. Reported
projections use the most recent SPY observation in the local file; they are not
live forecasts. Standard FRED downloads can contain historical revisions, so
availability lags alone do not make the inputs point-in-time vintages. Results
are educational and do not include every execution cost, tax, or investment risk.

The Technical indicators and ML vs actual pages also show the S&P 500 Shiller
CAPE (cyclically adjusted P/E) compared with its historical average, plus recent
S&P 500/SPY headlines from Google News RSS. CAPE is relative historical
valuation, not a fair-value estimate. Headlines and valuation are context only:
BUY/SELL/HOLD signals are based on the selected ML prediction threshold or, on
the technical page, SMA/RSI/MACD conditions; neither currently uses news or CAPE
as a decision input. The context feeds require an internet connection and are
cached temporarily. Their as-of dates are shown separately from the local SPY
dataset date; when dates differ, do not treat the feeds as contemporaneous
evidence for the local model signal.

## Project files

- `notebooks/Trading_Signal_Model_v1.ipynb` — SPY data preparation and
  exploratory analysis, rolling validation, 80/20 macro-only ML test, and
  time-series plots/forecast comparisons.
- `dashboard.py` — Streamlit navigation entry point.
- `app_pages/technical.py` — separate interactive technical-indicator page.
- `app_pages/ml_actual.py` — macro-only ML page with selectable actual-versus-
  predicted comparison and interactive time-series chart explorer.
- `app_pages/data.py` — local SPY source-data page.
- `src/dashboard_common.py` — shared cached data and model helpers for pages.
- `src/market_context.py` — S&P 500 CAPE valuation and recent headline context.
- `src/time_series_analysis.py` — regression, exponential smoothing, prediction
  intervals, and time-series chart data shared by notebook and dashboard.
- `src/modeling.py` — FRED alignment, model fitting, five-session forecasts,
  and backtest metrics.
- `data/spy_features.csv` and `data/spy_labeled_features.csv` — prepared
  feature and target datasets.
- `requirements.txt` — dashboard and ML runtime dependencies.
- `Solution_Diagram.jpeg` — current workflow diagram.

## Interactive dashboard

Install dependencies in the active Python environment, then launch the app from
the project root:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run dashboard.py
```

Use the top navigation to switch between **Technical indicators**, **ML vs
actual**, and **Data**. On the technical page, choose which indicators to display
and set the chart date range. The ML page keeps its model selection and actual
versus predicted return comparison separate. It defaults to
`Downloads/Fred add files`; change the folder in the page if your FRED CSVs are
elsewhere. It reads named CSV series from that folder and the local SPY
labeled-feature CSV. It does not use DuckDB, call a market-data API, place
trades, or send alerts.
