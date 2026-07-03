# Congrats Congrats Money Money — Precog Quant Task 2026

End-to-end algo-trading pipeline: data cleaning → features → walk-forward
modeling → backtesting → statistical arbitrage overlay.

Every stage is a **plain, directly-runnable Python script** (no notebook
required) — just `python pipeline/0N_*.py` in order.

## 1. Data

Official dataset: https://www.kaggle.com/datasets/iamspace/precog-quant-task-2026

`data/load_data.py` fetches it via **kagglehub** (the same pattern Kaggle's own
"Copy code" button gives you):

```bash
python data/load_data.py
```

This requires Kaggle API credentials on whatever machine runs it — see
https://www.kaggle.com/docs/api (either `~/.kaggle/kaggle.json`, or the
`KAGGLE_USERNAME` / `KAGGLE_KEY` env vars). **If no credentials/network route to
Kaggle are available, it automatically falls back** to a synthetic stand-in
dataset (`data/generate_synthetic_data.py`) with the exact same schema
(`date, ticker, open, high, low, close, volume`), so the pipeline still runs
end-to-end for demonstration. This fallback is what the results shipped in
`outputs/` were generated from, since the environment that built this project
has no network route to Kaggle's API. You do not need to run `load_data.py`
yourself — `pipeline/01_data_feature_engineering.py` calls it automatically the
first time `data/daily_prices.csv` doesn't exist yet, and caches the result.

The synthetic fallback isn't naive noise — it encodes a market + 6 sector
factors across 30 tickers over 6 years, per-name momentum/reversal dynamics
that flip regime halfway through the sample (so Part 2's walk-forward
retraining has a real regime shift to adapt to), 3 tickers pairs built to be
genuinely cointegrated (for Part 4), and injected data-quality issues (missing
values, fat-finger ticks, frozen prices, bad volume, duplicate rows) for Part
1 to catch.

**To force a refresh** (e.g. once you have Kaggle credentials configured), delete
`data/daily_prices.csv` and re-run `pipeline/01_data_feature_engineering.py`, or
run `python data/load_data.py` directly.

## 2. Structure

```
data/
  load_data.py                 real-data loader (kagglehub) + synthetic fallback
  generate_synthetic_data.py   fallback-only synthetic data generator
  daily_prices.csv             cached OHLCV panel (generated on first run)
scripts/                       shared, importable pipeline logic
  data_utils.py                 Part 1: loading + cleaning
  features.py                   Part 1: feature engineering
  modeling.py                   Part 2: walk-forward model + strategy logic
  backtest.py                   Part 3: simulation engine + metrics
  stat_arb.py                   Part 4: correlation screen + cointegration + lead-lag
pipeline/                      the 4 deliverable scripts -- run these, in order
  01_data_feature_engineering.py
  02_modeling_strategy.py
  03_backtesting.py
  04_stat_arb.py
outputs/
  figures/                      saved plots (PNG)
  results/                      intermediate parquet/CSV artifacts passed between stages
```

## 3. Run it

```bash
pip install -r requirements.txt
python pipeline/01_data_feature_engineering.py
python pipeline/02_modeling_strategy.py
python pipeline/03_backtesting.py
python pipeline/04_stat_arb.py
```

Each stage saves its outputs to `outputs/results/` (read by the next stage)
and `outputs/figures/` (final plots), and prints all tables/metrics/discussion
to stdout as it runs — no notebook viewer needed to see the results, though
you're welcome to open the `.py` files in Jupyter too (they run fine as
scripts either way).

## 4. Key design decisions, briefly

- **Cleaning (Part 1)**: per-ticker, causal only (no future data used to patch
  the past), robust rolling-MAD outlier detection on returns (std-based
  z-scores get blown up by the very fat-finger spikes they're supposed to
  catch), capped forward-fill for short gaps, drop for long ones.
- **Features (Part 1)**: a deliberately small set (momentum/reversal/vol/
  volume/range), all cross-sectionally rank-normalized per day.
- **Target (Part 2)**: 5-day forward log return (regression), so magnitude
  survives into the weighting step.
- **Model (Part 2)**: rank-averaged ensemble of Ridge + LightGBM, refit
  walk-forward every 21 trading days on a purged expanding window.
- **Strategy (Part 2)**: daily top-6/bottom-6 dollar-neutral long/short book.
- **Backtest (Part 3)**: $1,000,000 capital, 10bps/turnover cost, careful
  next-day (not same-day) return alignment to avoid look-ahead bias — with
  and without costs, plus a cost-sensitivity sweep and a lower-turnover
  (5-day rebalance) variant.
- **Stat arb (Part 4)**: correlation screen → Engle-Granger cointegration
  test → lead-lag cross-correlation → sketch of how a pairs sleeve would be
  netted against the main book rather than blended into one signal.

## 5. Results summary (on the current data — re-run to update after swapping in the real dataset)

- Mean daily IC of the ensemble signal is small, noisy, and drifts over time
  (~0.02, ~55% of days positive) — realistic for a cross-sectional equity
  signal, not a red flag.
- Gross of transaction costs the long/short book shows a Sharpe of ~1.9; net
  of a realistic 10bps/turnover cost that collapses to ~0.3, because daily
  full-book rebalancing runs ~83% average daily turnover against a modest
  edge. A 5-day-rebalance variant cuts turnover to ~10% and actually improves
  net Sharpe to ~0.6 — reacting to the signal a bit slower but paying far less
  in costs.
- The cointegration test correctly recovers all 3 pairs built to be
  cointegrated in the synthetic data, with sensible ~13-18 day half-lives, and
  correctly flags them as sitting within their own nearest-neighbor cluster.

Doubts document: https://docs.google.com/document/d/1ybowfIuIkde2ggIqVEkyZ84t4IpuDXBiX0Wk5ZOZs3E/edit?usp=sharing
