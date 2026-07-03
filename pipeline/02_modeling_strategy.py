"""
02_modeling_strategy.py -- Part 2: Model Training & Strategy Formulation
================================================================================
Run directly:  python pipeline/02_modeling_strategy.py
(requires 01_data_feature_engineering.py to have been run first)

Target: 5-day forward cumulative log return (regression).
  - Hint A (single-model instability): rank-averaged ensemble of Ridge (linear)
    + LightGBM (nonlinear).
  - Hint B (markets evolve): walk-forward, purged, expanding-window retraining
    (refit every 21 trading days, always strictly causal).
Strategy logic: daily top-6 / bottom-6 dollar-neutral long/short book.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
FIG_DIR = os.path.join(ROOT, "outputs", "figures")
RESULTS_DIR = os.path.join(ROOT, "outputs", "results")

from modeling import walk_forward_predict, signal_to_weights  # noqa: E402

plt.style.use("seaborn-v0_8-whitegrid")

FEAT_COLS = ["mom_5", "mom_21", "mom_63", "rev_1", "rev_5",
             "vol_21", "vol_of_vol_21", "range_vol_21",
             "dollar_vol_z_21", "amihud_illiq_21"]


def main():
    feat_path = os.path.join(RESULTS_DIR, "features.parquet")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            "features.parquet not found -- run pipeline/01_data_feature_engineering.py first.")
    feat = pd.read_parquet(feat_path)
    print(f"Loaded feature panel: {feat.shape}\n")

    # ---- 2.1 Walk-forward training -------------------------------------------
    pred = walk_forward_predict(feat, FEAT_COLS, train_window=500, refit_every=21, purge=5)
    print(f"Predictions cover {pred['date'].min().date()} -> {pred['date'].max().date()} "
          f"({pred['date'].nunique()} trading days, {len(pred):,} stock-days)\n")

    # ---- 2.2 Signal quality: Information Coefficient --------------------------
    daily_ic = pred.groupby("date").apply(
        lambda g: g["signal"].corr(g["fwd_ret_5"], method="spearman"))
    print(f"Mean daily IC: {daily_ic.mean():.4f} | "
          f"IC IR (mean/std): {daily_ic.mean() / daily_ic.std():.3f} | "
          f"%% days IC>0: {(daily_ic > 0).mean():.1%}\n")

    fig, ax = plt.subplots(figsize=(11, 4))
    daily_ic.rolling(63).mean().plot(ax=ax, color="darkorange", lw=1.3)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Rolling 63-day mean Information Coefficient")
    ax.set_ylabel("Spearman IC")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "02_rolling_ic.png"), dpi=130)
    plt.close(fig)
    print("Saved figure: outputs/figures/02_rolling_ic.png")
    print("(A weak, noisy, drifting IC is normal for cross-sectional equity signals -- "
          "not a red flag by itself. Watch for it tracking drawdowns in Part 3.)\n")

    # ---- 2.3 Strategy logic: top/bottom-6 dollar-neutral book -----------------
    weights = signal_to_weights(pred, n_long=6, n_short=6)
    max_imbalance = weights.groupby("date")["weight"].sum().abs().max()
    gross = weights.groupby("date")["weight"].apply(lambda w: w.abs().sum())
    print(f"Book is dollar-neutral each day (max |sum of weights| across all days): "
          f"{max_imbalance:.2e} (should be ~0)")
    print("Gross exposure per day (should be ~2.0 = 100% long + 100% short):")
    print(gross.describe(), "\n")

    weights.to_parquet(os.path.join(RESULTS_DIR, "weights.parquet"))
    pred.to_parquet(os.path.join(RESULTS_DIR, "predictions.parquet"))
    print("Saved outputs/results/weights.parquet and predictions.parquet")


if __name__ == "__main__":
    main()
