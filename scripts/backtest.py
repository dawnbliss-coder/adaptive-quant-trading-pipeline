"""
backtest.py -- Part 3: simulation + performance metrics
-----------------------------------------------------------
The backtester takes daily target weights (from modeling.signal_to_weights) and:
  - Rebalances daily to the new target weights (positions are held ~1 day since
    the signal targets 5-day forward returns but is recomputed daily -- this is a
    deliberately simple, high-frequency-ish rebalance to make transaction-cost
    sensitivity visible; a lower-turnover variant is also run for comparison).
  - Charges `cost_bps` (default 10bps = 0.10%) on the DOLLAR VALUE TRADED each day
    (i.e. on the change in each position's weight, not on total exposure), which
    is the standard convention: costs scale with turnover, not portfolio size.
  - Applies capital scaling ($1,000,000 initial capital, gross exposure normalized
    to sum(|weight|) = 1 -> the long/short book is capped at 100% gross by default,
    i.e. no leverage).
"""
import numpy as np
import pandas as pd


def run_backtest(weights: pd.DataFrame, capital: float = 1_000_000.0,
                  cost_bps: float = 10.0) -> dict:
    """
    weights: long DataFrame [date, ticker, weight, fwd_ret_5] where fwd_ret_5 is the
    *5-day forward return realized starting the day after `date`*. We convert this
    into a daily-equivalent simple return for a clean daily PnL series: since weights
    are recomputed and the book is rebalanced every day, we approximate each day's
    return contribution as weight * (daily component of fwd_ret_5), which for our
    walk-forward setup is well approximated by using the 1-day-ahead realized return
    of the SAME ticker (recomputed from the panel) rather than the 5-day forward
    label. This function expects a `daily_ret` column already merged in -- see
    notebooks/03_backtesting for how it's built from the cleaned price panel.
    """
    cost_rate = cost_bps / 1e4
    piv_w = weights.pivot(index="date", columns="ticker", values="weight").fillna(0.0)
    piv_r = weights.pivot(index="date", columns="ticker", values="daily_ret").fillna(0.0)
    piv_w, piv_r = piv_w.align(piv_r, join="outer", axis=None, fill_value=0.0)

    dates = piv_w.index
    prev_w = pd.Series(0.0, index=piv_w.columns)
    equity = capital
    curve, turnovers, gross_rets, net_rets = [], [], [], []

    for d in dates:
        w = piv_w.loc[d]
        r = piv_r.loc[d]
        gross_ret = float((w * r).sum())
        traded = (w - prev_w).abs().sum()
        turnover = float(traded) / 2.0   # standard convention: buys+sells / 2
        cost = float(traded) * cost_rate
        net_ret = gross_ret - cost
        equity *= (1 + net_ret)
        curve.append(equity)
        turnovers.append(turnover)
        gross_rets.append(gross_ret)
        net_rets.append(net_ret)
        prev_w = w

    result = pd.DataFrame({
        "date": dates, "equity": curve, "turnover": turnovers,
        "gross_ret": gross_rets, "net_ret": net_rets,
    }).set_index("date")
    return result


def performance_metrics(result: pd.DataFrame, ret_col: str = "net_ret",
                         periods_per_year: int = 252) -> dict:
    r = result[ret_col].dropna()
    ann_ret = (1 + r).prod() ** (periods_per_year / len(r)) - 1
    ann_vol = r.std() * np.sqrt(periods_per_year)
    sharpe = (r.mean() / r.std()) * np.sqrt(periods_per_year) if r.std() > 0 else np.nan

    equity = (1 + r).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()
    avg_dd = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0.0

    total_return = equity.iloc[-1] - 1
    avg_turnover = result["turnover"].mean()

    return {
        "Annualized Return": ann_ret,
        "Annualized Vol": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Avg Drawdown": avg_dd,
        "Total Return": total_return,
        "Avg Daily Turnover": avg_turnover,
        "Final Equity ($)": result["equity"].iloc[-1] if "equity" in result else np.nan,
        "N days": len(r),
    }
