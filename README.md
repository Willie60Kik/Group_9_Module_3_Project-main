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

These figures are the notebook's recorded run. The interactive dashboard
recalculates results from the current labeled CSV and selected date range, so its
out-of-fold performance may differ from this snapshot.

## Project files

- `notebooks/Trading_Signal_Model.ipynb` — analysis, model training, and
  backtesting.
- `dashboard.py` — local interactive dashboard for SPY price history,
  price-based risk and return, technical-rule signals, retrospective target
  labels, and documented backtest results.
- `data/spy_features.csv` and `data/spy_labeled_features.csv` — prepared
  feature and target datasets.
- `Solution_Diagram.jpeg` — current workflow diagram.

## Interactive dashboard

Run the app from the project root with `streamlit run dashboard.py` after
Streamlit, pandas, and scikit-learn are available in your Python environment. The dashboard
reads the local labeled feature CSV only; it does not call a market-data API,
place trades, or send alerts. It shows a technical-rule signal separately from
the retrospective target labels, which are not model predictions. The
overview calculates close-price returns, annualized volatility, Sharpe ratio,
and maximum drawdown for the selected date range. These price statistics
exclude dividends; the Sharpe ratio assumes a zero risk-free rate and 252
trading sessions per year.

The Overview tab includes a signal and holdings calculator. Select a valuation
date within the filtered range, then choose a manual expected five-session return
or that date's technical-rule signal, or compare the selected-date close with
the entered average purchase price. The dashboard displays BUY, SELL, or HOLD
using the current sidebar thresholds; the price-versus-purchase-price option
shows BUY above the threshold, SELL below the loss threshold, and HOLD within
the breakeven band. Enter SPY ETF shares held and the average purchase price per
share to estimate the position's market value and unrealized profit or loss at
the selected date's close. Manual return is a user-entered scenario, not a
model prediction. Profit/loss excludes dividends, commissions, taxes, and
currency conversion. The headline KPIs continue to show the latest date in the
selected range.

The Backtest tab compares out-of-fold Random Forest predictions, the RSI/MACD/SMA
rules, and buy-and-hold over the selected signal-date range. It fits the notebook's
five chronological folds with a five-row training/validation gap and the fold-specific
hyperparameters recorded by the notebook. Strategies are evaluated on aligned,
non-overlapping five-session forward-return periods with an adjustable per-side
transaction cost; the simulation starts in cash at the selected range. The dashboard
reports strategy returns, annualized volatility, Sharpe ratio, and maximum drawdown
for that selected window and allows the comparison data to be downloaded.

The sidebar also lets users set separate BUY and SELL five-session return thresholds
(0%–10%). Returns above the BUY threshold are labeled BUY; returns below the negative
SELL threshold are labeled SELL; returns between those boundaries, including both
boundaries, are labeled HOLD (the breakeven band). Changing thresholds recalculates
the historical target-label chart and retrains the chronological out-of-fold model
with the resulting labels. The technical-rule strategy remains a separate RSI/MACD/SMA
benchmark and is not changed by the return thresholds. The saved fold hyperparameters
were tuned using the default +/-1% thresholds, so custom-threshold comparisons should
be treated as exploratory unless the notebook is retuned for those thresholds.

These out-of-fold estimates are for historical research, not live predictions. The
last signal's five-session return may extend beyond the selected end date. Strategy
results use unadjusted close prices, exclude dividends, and do not model slippage,
taxes, or all execution risks. Rerun the notebook's model tuning if the dataset or
model methodology changes so the saved fold parameters remain appropriate.
