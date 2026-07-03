"""
03_backtesting.py -- Part 3: Backtesting & Performance Analysis
================================================================================
Run directly:  python pipeline/03_backtesting.py
(requires 01_data_feature_engineering.py and 02_modeling_strategy.py to have run first)

Simulation: $1,000,000 initial capital, 10bps transaction cost per unit of
turnover, no leverage. Careful next-day (not same-day) return alignment to
avoid look-ahead bias. Reports Sharpe, drawdown, turnover, return -- with and
without costs -- plus a cost-sensitivity sweep and a lower-turnover variant.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
FIG_DIR = os.path.join(ROOT, "outputs", "figures")
RESULTS_DIR = os.path.join(ROOT, "outputs", "results")

from data_utils import to_panel  # noqa: E402
from backtest import run_backtest, performance_metrics  # noqa: E402

plt.style.use("seaborn-v0_8-whitegrid")


def format_metrics_table(metrics_dict: dict) -> pd.DataFrame:
    raw = pd.DataFrame(metrics_dict)
    pct_rows = ["Annualized Return", "Annualized Vol", "Max Drawdown", "Avg Drawdown", "Total Return"]
    fmt = {}
    for row in raw.index:
        if row in pct_rows:
            fmt[row] = raw.loc[row].apply(lambda x: f"{x:.2%}")
        elif row == "Sharpe Ratio":
            fmt[row] = raw.loc[row].apply(lambda x: f"{x:.2f}")
        elif row == "Avg Daily Turnover":
            fmt[row] = raw.loc[row].apply(lambda x: f"{x:.1%}")
        elif row == "Final Equity ($)":
            fmt[row] = raw.loc[row].apply(lambda x: f"${x:,.0f}")
        else:
            fmt[row] = raw.loc[row]
    return pd.DataFrame(fmt).T


def main():
    weights_path = os.path.join(RESULTS_DIR, "weights.parquet")
    clean_path = os.path.join(RESULTS_DIR, "clean_prices.parquet")
    if not (os.path.exists(weights_path) and os.path.exists(clean_path)):
        raise FileNotFoundError(
            "Missing inputs -- run pipeline/01_data_feature_engineering.py and "
            "pipeline/02_modeling_strategy.py first.")

    weights = pd.read_parquet(weights_path)
    clean = pd.read_parquet(clean_path)
    panel = to_panel(clean, "close")

    # ---- 3.1 Align next-day returns (avoid look-ahead) -------------------------
    # A weight decided using info up to day t can only earn the return from
    # t -> t+1, NOT the return realized by day t itself (already baked into the
    # signal that produced the weight -- crediting it back is look-ahead bias).
    ret_panel = np.log(panel).diff().shift(-1)
    ret_long = ret_panel.reset_index().melt(id_vars="date", var_name="ticker", value_name="daily_ret")
    w = weights.merge(ret_long, on=["date", "ticker"], how="left")

    # ---- 3.2 Backtest, with and without costs -----------------------------------
    res_with_costs = run_backtest(w, capital=1_000_000, cost_bps=10.0)
    res_no_costs = run_backtest(w, capital=1_000_000, cost_bps=0.0)
    metrics_with = performance_metrics(res_with_costs)
    metrics_no = performance_metrics(res_no_costs, ret_col="gross_ret")

    summary = format_metrics_table({"With 10bps costs": metrics_with, "No transaction costs": metrics_no})
    print("Performance summary:")
    print(summary, "\n")

    # ---- 3.3 Cumulative PnL vs equal-weight benchmark ---------------------------
    ew_ret = np.log(panel).diff().mean(axis=1).reindex(res_with_costs.index)
    ew_equity = 1_000_000 * (1 + ew_ret.fillna(0)).cumprod()

    fig, ax = plt.subplots(figsize=(12, 5.5))
    res_with_costs["equity"].plot(ax=ax, label="Strategy (net of 10bps costs)", color="seagreen", lw=1.6)
    res_no_costs["equity"].plot(ax=ax, label="Strategy (gross, no costs)", color="steelblue", lw=1.2, ls="--")
    ew_equity.plot(ax=ax, label="Equal-weight benchmark (buy & hold)", color="gray", lw=1.4)
    ax.set_ylabel("Equity ($)")
    ax.set_title("Cumulative PnL — Strategy vs. Equal-Weight Benchmark")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "03_cumulative_pnl.png"), dpi=130)
    plt.close(fig)

    # ---- 3.4 Drawdown -----------------------------------------------------------
    eq = (1 + res_with_costs["net_ret"]).cumprod()
    dd = eq / eq.cummax() - 1
    fig, ax = plt.subplots(figsize=(12, 3.5))
    dd.plot(ax=ax, color="firebrick")
    ax.fill_between(dd.index, dd.values, 0, color="firebrick", alpha=0.2)
    ax.set_title(f"Strategy Drawdown (max = {dd.min():.1%})")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "03_drawdown.png"), dpi=130)
    plt.close(fig)
    print("Saved figures: outputs/figures/03_cumulative_pnl.png, 03_drawdown.png\n")

    # ---- 3.5 Cost sensitivity sweep + lower-turnover variant --------------------
    cost_sweep = []
    for bps in [0, 2, 5, 10, 20, 40]:
        r = run_backtest(w, cost_bps=bps)
        m = performance_metrics(r)
        cost_sweep.append({"cost_bps": bps, "Sharpe": m["Sharpe Ratio"],
                            "Ann. Return": m["Annualized Return"], "Total Return": m["Total Return"]})
    cost_sweep = pd.DataFrame(cost_sweep)
    print("Cost sensitivity sweep:")
    print(cost_sweep, "\n")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(cost_sweep["cost_bps"], cost_sweep["Sharpe"], marker="o", color="darkorange")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Cost (bps per unit turnover)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Strategy Sharpe vs. transaction cost assumption")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "03_cost_sensitivity.png"), dpi=130)
    plt.close(fig)
    print("Saved figure: outputs/figures/03_cost_sensitivity.png\n")

    # 5-day rebalance variant (holds each day's target weights for 5 days)
    piv_target = weights.pivot(index="date", columns="ticker", values="weight")
    piv_target_slow = piv_target.iloc[::5].reindex(piv_target.index).ffill()
    w_slow_long = piv_target_slow.reset_index().melt(id_vars="date", var_name="ticker", value_name="weight")
    w_slow_long = w_slow_long.merge(ret_long, on=["date", "ticker"], how="left")
    res_slow = run_backtest(w_slow_long, cost_bps=10.0)
    metrics_slow = performance_metrics(res_slow)

    compare = format_metrics_table({
        "Daily rebalance (10bps)": metrics_with, "5-day rebalance (10bps)": metrics_slow})
    print("Daily vs. 5-day rebalance (both net of 10bps costs):")
    print(compare, "\n")

    print("Discussion -- did the strategy survive transaction costs?")
    print(f"  Gross Sharpe: {metrics_no['Sharpe Ratio']:.2f}  ->  "
          f"Net-of-cost Sharpe: {metrics_with['Sharpe Ratio']:.2f}")
    print("  High daily turnover (see 'Avg Daily Turnover' above) is what erodes the edge; "
          "the cost sweep shows roughly where the break-even cost level sits, and the "
          "5-day-rebalance variant trades reaction speed for lower costs.")

    res_with_costs.to_csv(os.path.join(RESULTS_DIR, "backtest_with_costs.csv"))
    res_no_costs.to_csv(os.path.join(RESULTS_DIR, "backtest_no_costs.csv"))
    summary.to_csv(os.path.join(RESULTS_DIR, "performance_summary.csv"))
    print("\nSaved backtest results to outputs/results/")


if __name__ == "__main__":
    main()
