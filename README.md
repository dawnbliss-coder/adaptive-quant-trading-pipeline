### Precog Quant Task 2026 — End-to-End Algorithmic Trading Pipeline

An end-to-end quant research pipeline for a universe of anonymized equities: data
cleaning → feature engineering → walk-forward model training → backtesting with
realistic costs → a statistical arbitrage overlay.

Every stage is a **plain, directly-runnable Python script** — no notebook viewer
required, just `python pipeline/0N_*.py` in order.

---

## Contents

- [Data](#data)
- [Project structure](#project-structure)
- [Setup & run](#setup--run)
- [Methodology](#methodology)
- [Results](#results)
- [Known limitations](#known-limitations)

---

## Data

Official dataset: [`iamspace/precog-quant-task-2026`](https://www.kaggle.com/datasets/iamspace/precog-quant-task-2026) on Kaggle.

```bash
python data/load_data.py
```
---

## Project structure

```
data/
  load_data.py                 real-data loader (kagglehub) + synthetic fallback
  generate_synthetic_data.py   fallback-only synthetic data generator
  daily_prices.csv             cached OHLCV panel (generated on first run)
scripts/                       shared, importable pipeline logic
  data_utils.py                  Part 1 — loading + cleaning
  features.py                    Part 1 — feature engineering
  modeling.py                    Part 2 — walk-forward model + strategy logic
  backtest.py                    Part 3 — simulation engine + metrics
  stat_arb.py                    Part 4 — correlation screen + cointegration + lead-lag
pipeline/                      the 4 deliverable scripts — run these, in order
  01_data_feature_engineering.py
  02_modeling_strategy.py
  03_backtesting.py
  04_stat_arb.py
outputs/
  figures/                      saved plots (PNG)
  results/                      intermediate parquet/CSV artifacts passed between stages
```

---

## Setup & run

```bash
git clone <this-repo>
cd quant_task_v2
pip install -r requirements.txt

python pipeline/01_data_feature_engineering.py
python pipeline/02_modeling_strategy.py
python pipeline/03_backtesting.py
python pipeline/04_stat_arb.py
```
---

## Methodology

**Part 1 — Cleaning & features.** Cleaning is done per-ticker and strictly
causally (never using future data to patch the past): duplicate rows dropped,
OHLC sanity enforced, non-positive prices/negative volume nulled, a rolling-MAD
(not std) outlier filter catches fat-finger spikes without being blown up by
the very outliers it's looking for, stale/frozen price runs detected, gaps
forward-filled up to a 5-day cap (longer gaps dropped rather than fabricated).
Features are a deliberately small set — momentum (5/21/63d), reversal (1/5d),
realized vol + vol-of-vol, Garman-Klass range, dollar-volume z-score, Amihud
illiquidity — all cross-sectionally rank-normalized per day.

**Part 2 — Modeling & strategy.** Target is 5-day forward log return
(regression, to preserve magnitude for weighting). Two structurally different
models — Ridge (linear) and a gradient-boosted tree (LightGBM, or a
scikit-learn fallback) — are trained and their predictions rank-averaged into
one signal each day, addressing the brief's "single model instability" hint.
Training is **walk-forward**: refit every 21 trading days on a purged,
~500-day expanding window, always strictly causal, addressing the "markets
evolve" hint. Strategy logic: daily top-6 / bottom-6 dollar-neutral,
equal-weighted long/short book.

**Part 3 — Backtesting.** $1,000,000 capital, 10bps cost per unit of turnover,
no leverage (gross exposure capped at 2.0 = 100% long + 100% short). Careful
next-day (not same-day) return alignment to avoid look-ahead bias — a weight
decided using information through day *t* can only earn the return from *t*
to *t+1*. Reports Sharpe, drawdown, turnover, and return with and without
costs, plus a cost-sensitivity sweep and a lower-turnover (5-day rebalance)
variant.

**Part 4 — Statistical arbitrage overlay.** Correlation is used only as a
cheap first screen (it doesn't establish that price *levels* stay tethered);
the real test is **Engle-Granger cointegration** (OLS hedge ratio + ADF test
on the residual spread) on the correlation-screened shortlist, plus a
lead-lag cross-correlation check and a sketch of how a pairs sleeve would be
netted against the main book rather than blended into one signal.

---

## Results

**Signal quality:** mean daily IC ≈ 0.02, positive on ~55% of days — small but
real, and consistent with how weak/noisy genuine cross-sectional equity
signals typically look.

**Backtest, full book, daily rebalance:**

| | Gross (no costs) | Net of 10bps costs |
|---|---|---|
| Sharpe Ratio | 1.78 | 0.21 |
| Annualized Return | 54.9% | 2.0% |
| Max Drawdown | -25.6% | -48.3% |
| Avg Daily Turnover | 82.9% | 82.9% |

High turnover erodes most of the gross edge. The cost-sensitivity sweep puts
the break-even cost level around 12–15bps.

**Lower-turnover variant (5-day rebalance, net of 10bps costs):** Sharpe
improves to 0.66 as turnover drops to ~10%, at the cost of a somewhat deeper
drawdown (-53.3%) — trading reaction speed for cost savings.

**Stat arb:** correctly recovers the 3 pairs built to be cointegrated (out of
435 candidate pairs, p < 0.05, half-lives of 13–18 days), with zero false
positives at that threshold.

---
