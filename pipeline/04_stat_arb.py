"""
04_stat_arb.py -- Part 4: Statistical Arbitrage Overlay
================================================================================
Run directly:  python pipeline/04_stat_arb.py
(requires 01_data_feature_engineering.py to have run first)

Pipeline: correlation screen -> Engle-Granger cointegration test -> lead-lag
cross-correlation -> sector-proxy context -> implementation sketch for folding
a pairs sleeve into the main long/short book from Parts 2-3.
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
from stat_arb import correlation_screen, cointegration_test, lead_lag  # noqa: E402

plt.style.use("seaborn-v0_8-whitegrid")


def plot_pair(panel, a, b, hedge_ratio, ax_price, ax_spread):
    pa, pb = np.log(panel[a]), np.log(panel[b])
    idx = pa.dropna().index.intersection(pb.dropna().index)
    pa, pb = pa.loc[idx], pb.loc[idx]
    spread = pa - hedge_ratio * pb
    z = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()

    ax_price.plot(idx, pa, label=f"log({a})", lw=1.1)
    ax_price.plot(idx, hedge_ratio * pb, label=f"{hedge_ratio:.2f}·log({b})", lw=1.1)
    ax_price.set_title(f"{a} vs {b} — hedge-ratio-scaled log price")
    ax_price.legend(fontsize=8)

    ax_spread.plot(idx, z, color="purple", lw=0.9)
    ax_spread.axhline(0, color="black", lw=0.7)
    ax_spread.axhline(2, color="red", ls="--", lw=0.7)
    ax_spread.axhline(-2, color="red", ls="--", lw=0.7)
    ax_spread.set_title(f"{a}/{b} spread z-score (60d rolling)")


def main():
    clean_path = os.path.join(RESULTS_DIR, "clean_prices.parquet")
    if not os.path.exists(clean_path):
        raise FileNotFoundError("Run pipeline/01_data_feature_engineering.py first.")
    clean = pd.read_parquet(clean_path)
    panel = to_panel(clean, "close")

    # ---- 4.1 Correlation screen -----------------------------------------------
    candidates = correlation_screen(panel, top_k=60)
    print("Top correlation-screened candidate pairs:")
    print(candidates.head(10), "\n")

    # ---- 4.2 Cointegration (Engle-Granger) --------------------------------------
    coint_results = cointegration_test(panel, candidates, p_thresh=0.05)
    sig_pairs = coint_results[coint_results["significant"]].sort_values("coint_pvalue")
    print(f"{len(sig_pairs)} / {len(coint_results)} screened pairs show significant "
          f"cointegration (Engle-Granger p < 0.05):")
    print(sig_pairs, "\n")
    print("Caveat: testing many pairs is a multiple-testing problem -- treat any pair "
          "found here as a hypothesis to monitor out-of-sample, not a guarantee.\n")

    # ---- 4.3 Visualize spreads for the top candidates ---------------------------
    top3 = sig_pairs.head(3) if len(sig_pairs) >= 3 else coint_results.head(3)
    fig, axes = plt.subplots(len(top3), 2, figsize=(13, 3.3 * len(top3)))
    if len(top3) == 1:
        axes = axes.reshape(1, -1)
    for row, (_, r) in zip(axes, top3.iterrows()):
        plot_pair(panel, r["a"], r["b"], r["hedge_ratio"], row[0], row[1])
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "04_pair_spreads.png"), dpi=130)
    plt.close(fig)

    # ---- 4.4 Lead-lag structure ---------------------------------------------------
    fig, axes = plt.subplots(1, len(top3), figsize=(5 * len(top3), 4))
    if len(top3) == 1:
        axes = [axes]
    for ax, (_, r) in zip(axes, top3.iterrows()):
        xc = lead_lag(panel, r["a"], r["b"])
        ax.bar(xc["lag"], xc["xcorr"], color="teal")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"{r['a']} vs {r['b']}")
        ax.set_xlabel("lag (days), positive = b lags a")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "04_lead_lag.png"), dpi=130)
    plt.close(fig)
    print("Saved figures: outputs/figures/04_pair_spreads.png, 04_lead_lag.png\n")

    # ---- 4.5 Sector-proxy context --------------------------------------------------
    ret = np.log(panel).diff().dropna()
    full_corr = ret.corr()
    print("Nearest-neighbor context (proxy for 'same sector' in an anonymized universe):")
    for _, r in top3.iterrows():
        a, b = r["a"], r["b"]
        nearest_a = full_corr[a].drop(a).nlargest(3)
        nearest_b = full_corr[b].drop(b).nlargest(3)
        same_cluster = b in nearest_a.index or a in nearest_b.index
        verdict = "appear to be in the same cluster" if same_cluster else "are not each other's nearest neighbor"
        print(f"  {a}-{b}: nearest neighbors of {a} = {list(nearest_a.index)}, "
              f"nearest neighbors of {b} = {list(nearest_b.index)}, {verdict}")

    # ---- 4.6 Implementation idea ----------------------------------------------------
    print("\nImplementation idea -- folding stat-arb into the main portfolio:")
    print("""
  1. Separate risk budgets: run the cross-sectional alpha book (Parts 2-3) and
     the stat-arb book as two sleeves with independent capital allocations
     (e.g. 80/20), rather than blending scores -- they operate on different
     mathematical objects (single-name expected return vs. pair-spread
     deviation).
  2. Position-level netting: net the two sleeves' target weights PER TICKER
     before sending anything to the backtester/broker, so they don't
     independently rack up transaction costs trading against each other.
  3. Sizing the spread trade: size each pair position proportionally to the
     spread z-score (enter |z|>2, scale linearly, exit at |z|<0.5), inverse-
     scaled by the spread's rolling volatility so pairs with different half-
     lives/vols contribute comparable risk.
  4. Regime gate: only trade a pair while its rolling cointegration p-value
     stays below threshold on an expanding basis -- turn the overlay off for
     a pair once its statistical relationship breaks down.
""")

    coint_results.to_csv(os.path.join(RESULTS_DIR, "stat_arb_candidates.csv"), index=False)
    print("Saved outputs/results/stat_arb_candidates.csv")


if __name__ == "__main__":
    main()
