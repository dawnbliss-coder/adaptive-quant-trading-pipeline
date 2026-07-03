"""
01_data_feature_engineering.py -- Part 1: Data Cleaning & Feature Engineering
================================================================================
Run directly:  python pipeline/01_data_feature_engineering.py

Loads daily_prices.csv (real Kaggle dataset via data/load_data.py if you've
set up Kaggle API credentials, otherwise the synthetic stand-in), assesses
data quality, cleans it, builds a small cross-sectionally rank-normalized
feature set, and saves the results for the next stage.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless-safe: save figures to disk, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
FIG_DIR = os.path.join(ROOT, "outputs", "figures")
RESULTS_DIR = os.path.join(ROOT, "outputs", "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

from data_utils import load_raw, clean_prices  # noqa: E402
from features import build_features  # noqa: E402

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.width", 120)


def main():
    # ---- 1.1 Load & assess raw data quality --------------------------------
    raw = load_raw()
    print(f"Rows: {len(raw):,} | Tickers: {raw.ticker.nunique()} | "
          f"Date range: {raw.date.min().date()} -> {raw.date.max().date()}\n")

    quality = pd.DataFrame({
        "n_missing": raw[["open", "high", "low", "close", "volume"]].isna().sum(),
        "pct_missing": (raw[["open", "high", "low", "close", "volume"]].isna().mean() * 100).round(3),
    })
    print("Missing values by column:")
    print(quality, "\n")
    print("Duplicate rows:", raw.duplicated().sum())
    print("Non-positive close prices:", (raw["close"] <= 0).sum())
    print("Negative/zero volume:", (raw["volume"] <= 0).sum(), "\n")

    example_ticker = raw["ticker"].iloc[0]
    sub = raw[raw["ticker"] == example_ticker].sort_values("date")
    log_ret = np.log(sub["close"]).diff()
    suspicious = sub.loc[log_ret.abs().nlargest(5).index, ["date", "ticker", "close"]]
    print(f"Top-5 largest |log return| jumps for {example_ticker} (candidate bad ticks):")
    print(suspicious, "\n")

    # ---- 1.2 Clean -----------------------------------------------------------
    clean = clean_prices(raw)
    print(f"Rows before cleaning: {len(raw):,} -> after: {len(clean):,}")
    print(f"Remaining NaNs in close: {clean['close'].isna().sum()}\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    raw_sub = raw[raw["ticker"] == example_ticker].sort_values("date")
    clean_sub = clean[clean["ticker"] == example_ticker].sort_values("date")
    axes[0].plot(raw_sub["date"], raw_sub["close"], lw=0.8, color="crimson")
    axes[0].set_title(f"{example_ticker} — RAW close (bad ticks/stale runs visible)")
    axes[1].plot(clean_sub["date"], clean_sub["close"], lw=0.8, color="seagreen")
    axes[1].set_title(f"{example_ticker} — CLEANED close")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "01_clean_vs_raw.png"), dpi=130)
    plt.close(fig)
    print(f"Saved figure: outputs/figures/01_clean_vs_raw.png")

    # ---- 1.3 Feature engineering ----------------------------------------------
    feat, feat_cols = build_features(clean)
    print(f"\nFeature panel shape: {feat.shape}")
    print(feat[["date", "ticker"] + feat_cols].dropna().head(), "\n")

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.flat, feat_cols[:6]):
        sns.histplot(feat[col].dropna(), bins=60, ax=ax, color="steelblue")
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "01_feature_distributions.png"), dpi=130)
    plt.close(fig)

    rank_cols = [c + "_rank" for c in feat_cols]
    corr = feat[rank_cols].corr()
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                xticklabels=feat_cols, yticklabels=feat_cols)
    plt.title("Cross-sectional-rank feature correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "01_feature_corr.png"), dpi=130)
    plt.close()
    print("Saved figures: outputs/figures/01_feature_distributions.png, 01_feature_corr.png")
    print("(mom_21/mom_63 and rev_1/rev_5 are near-mirror-images of each other -- "
          "expected, since momentum/reversal share overlapping return windows.)")

    # ---- save artifacts for the next stage -------------------------------------
    feat.to_parquet(os.path.join(RESULTS_DIR, "features.parquet"))
    clean.to_parquet(os.path.join(RESULTS_DIR, "clean_prices.parquet"))
    print("\nSaved outputs/results/features.parquet and clean_prices.parquet")


if __name__ == "__main__":
    main()
