"""
data_utils.py -- Part 1: loading + cleaning
--------------------------------------------
Design principle: clean PER TICKER, never across tickers (a stale-price run in one
name shouldn't influence another), and never leak future information into the past
(no forward-fill from the future, no using full-sample stats for anomaly thresholds
that a live trader wouldn't have had on that day).
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data"))
RAW_PATH = os.path.join(ROOT, "data", "daily_prices.csv")


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """
    Loads daily_prices.csv, fetching it first (real Kaggle dataset, falling
    back to the synthetic stand-in) via data/load_data.py if it isn't cached
    on disk yet.
    """
    if not os.path.exists(path):
        from load_data import load_daily_prices
        return load_daily_prices()
    return pd.read_csv(path, parse_dates=["date"])


def clean_prices(df: pd.DataFrame, mad_z_thresh: float = 8.0) -> pd.DataFrame:
    """
    Cleans raw OHLCV data. Steps, in order, applied independently per ticker:
      1. Drop exact duplicate rows.
      2. Enforce OHLC sanity: high = max(o,h,l,c), low = min(o,h,l,c).
      3. Non-positive prices/volume -> NaN (can't be real).
      4. Return-based outlier detection: flag |log return| that's an extreme
         rolling-MAD (median absolute deviation) outlier -> treat close as NaN.
         MAD is used instead of std because it's robust to the very outliers
         we're trying to detect (a 10x fat-finger tick would blow up a std-based
         z-score threshold and hide itself).
      5. Detect stale-price runs (>=3 consecutive identical closes) -> NaN
         (frozen/stale feed, not a real flat period at penny precision).
      6. Reconstruct NaNs with a *causal* fill: forward-fill close (last known
         traded price), then rebuild O/H/L as that same price and volume as 0
         for genuinely missing bars, capped at 5 consecutive days -- beyond
         that we drop the ticker-day rather than fabricate a trend.
      7. Recompute log returns from the cleaned close series.
    """
    df = df.drop_duplicates()
    out = []
    for tk, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)

        o, h, l, c = g["open"], g["high"], g["low"], g["close"]
        g["high"] = pd.concat([o, h, l, c], axis=1).max(axis=1)
        g["low"] = pd.concat([o, h, l, c], axis=1).min(axis=1)

        for col in ["open", "high", "low", "close"]:
            g.loc[g[col] <= 0, col] = np.nan
        g.loc[g["volume"] < 0, "volume"] = np.nan

        log_ret = np.log(g["close"]).diff()
        roll_med = log_ret.rolling(60, min_periods=20).median()
        roll_mad = (log_ret - roll_med).abs().rolling(60, min_periods=20).median()
        robust_z = 0.6745 * (log_ret - roll_med) / roll_mad.replace(0, np.nan)
        bad = robust_z.abs() > mad_z_thresh
        g.loc[bad.fillna(False), "close"] = np.nan

        same_as_prev = g["close"].diff().eq(0)
        run_id = (~same_as_prev).cumsum()
        run_len = g.groupby(run_id)["close"].transform("size")
        stale = same_as_prev & (run_len >= 3)
        g.loc[stale, "close"] = np.nan

        gap = g["close"].isna()
        gap_id = (~gap).cumsum()
        gap_len = g.groupby(gap_id)["close"].transform(lambda s: s.isna().cumsum())
        fillable = gap & (gap_len <= 5)
        g["close"] = g["close"].ffill()
        g.loc[gap & ~fillable, "close"] = np.nan
        g["open"] = g["open"].fillna(g["close"])
        g["high"] = g["high"].fillna(g["close"])
        g["low"] = g["low"].fillna(g["close"])
        g["volume"] = g["volume"].fillna(0)

        g = g.dropna(subset=["close"]).reset_index(drop=True)
        g["log_ret"] = np.log(g["close"]).diff()
        out.append(g)

    cleaned = pd.concat(out, ignore_index=True)
    cleaned = cleaned.sort_values(["ticker", "date"]).reset_index(drop=True)
    return cleaned


def to_panel(df: pd.DataFrame, value_col: str = "close") -> pd.DataFrame:
    """Long -> wide (date x ticker) panel."""
    return df.pivot(index="date", columns="ticker", values=value_col).sort_index()
