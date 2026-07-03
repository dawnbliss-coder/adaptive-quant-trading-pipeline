"""
features.py -- Part 1: feature engineering
--------------------------------------------
Every feature is computed causally (only data up to and including day t) and
cross-sectionally rank-normalized within the universe on each day, which is the
standard quant-research trick for two reasons:
  1. It makes features comparable across stocks with wildly different price/vol
     scales without needing per-stock normalization constants that drift over time.
  2. Tree/linear models trained on cross-sectional ranks generalize much better
     out-of-sample than models trained on raw levels, because "cheap relative to
     peers today" is a more stable relationship than any absolute threshold.

Feature families (kept deliberately small -- the brief explicitly warns against
throwing the kitchen sink at it):
  - Momentum:     5/21/63-day cumulative returns (short/medium-term momentum)
  - Reversal:     1-day and 5-day return (short-term mean reversion signal)
  - Volatility:   21-day realized vol, and vol-of-vol (21d std of 5d rolling vol)
  - Volume:       21-day dollar-volume z-score, volume/price co-movement (Amihud
                   illiquidity proxy: |return| / dollar volume)
  - Range:        Garman-Klass-style intraday range as a cheap realized-vol proxy
  - Cross-section: rank of each feature within the universe, each day
"""
import numpy as np
import pandas as pd


def _amihud(g: pd.DataFrame) -> pd.Series:
    dollar_vol = (g["close"] * g["volume"]).replace(0, np.nan)
    return (g["log_ret"].abs() / dollar_vol).rolling(21, min_periods=10).mean()


def build_features(cleaned: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for tk, g in cleaned.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        f = pd.DataFrame({"date": g["date"], "ticker": tk})

        f["mom_5"] = g["log_ret"].rolling(5).sum()
        f["mom_21"] = g["log_ret"].rolling(21).sum()
        f["mom_63"] = g["log_ret"].rolling(63).sum()
        f["rev_1"] = -g["log_ret"]
        f["rev_5"] = -g["log_ret"].rolling(5).sum()

        f["vol_21"] = g["log_ret"].rolling(21, min_periods=10).std()
        vol_5 = g["log_ret"].rolling(5, min_periods=3).std()
        f["vol_of_vol_21"] = vol_5.rolling(21, min_periods=10).std()

        gk_range = np.log(g["high"] / g["low"].replace(0, np.nan)) ** 2
        f["range_vol_21"] = np.sqrt(gk_range.rolling(21, min_periods=10).mean() / (4 * np.log(2)))

        dollar_vol = g["close"] * g["volume"]
        f["dollar_vol_z_21"] = (
            (dollar_vol - dollar_vol.rolling(21, min_periods=10).mean())
            / dollar_vol.rolling(21, min_periods=10).std()
        )
        f["amihud_illiq_21"] = _amihud(g)

        # forward-looking targets (label), NOT a feature -- kept alongside for convenience
        f["fwd_ret_5"] = g["log_ret"].shift(-1).rolling(5).sum().shift(-4)
        # ^ sum of the next 5 days' returns, aligned to day t (uses only t+1..t+5)

        frames.append(f)

    feat = pd.concat(frames, ignore_index=True)

    # cross-sectional rank-normalization (per day, across the universe), features only
    feat_cols = ["mom_5", "mom_21", "mom_63", "rev_1", "rev_5",
                 "vol_21", "vol_of_vol_21", "range_vol_21",
                 "dollar_vol_z_21", "amihud_illiq_21"]
    for col in feat_cols:
        feat[col + "_rank"] = feat.groupby("date")[col].rank(pct=True)

    return feat, feat_cols
