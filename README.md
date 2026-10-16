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

## Current result

Using five-fold chronological evaluation, the tuned Random Forest achieved
38.6% mean balanced accuracy versus 33.3% for the constant-class baseline. In
the simplified out-of-sample simulation, the long-or-cash model strategy
compounded to 163.4%, the RSI/MACD/SMA rule strategy to 82.9%, and buy-and-hold
to 507.0%. The model outperformed the rule-based strategy but substantially
underperformed buy-and-hold, so these results do not support using it for
trading.

The backtest samples non-overlapping five-trading-day periods and assumes
transaction costs of 0.05% per side. These historical results are illustrative,
not a guarantee of future performance; they do not model all execution effects,
taxes, or investment risks.

## Project files

- `notebooks/Trading_Signal_Model.ipynb` — analysis, model training, and
  backtesting.
- `dashboard.py` — local interactive dashboard for SPY price history,
  technical-rule signals, target labels, and documented backtest results.
- `data/spy_features.csv` and `data/spy_labeled_features.csv` — prepared
  feature and target datasets.
- `Solution_Diagram.jpeg` — current workflow diagram.

## Interactive dashboard

Run the app from the project root with `streamlit run dashboard.py` after
Streamlit and pandas are available in your Python environment. The dashboard
reads the local labeled feature CSV only; it does not call a market-data API,
place trades, or send alerts. It shows a technical-rule signal separately from
the retrospective target labels, which are not model predictions. The
backtest figures are a documented notebook snapshot and are not recalculated
by the dashboard.

