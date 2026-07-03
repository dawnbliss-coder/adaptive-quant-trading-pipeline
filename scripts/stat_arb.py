"""
stat_arb.py -- Part 4: relative-value / pairs discovery
-----------------------------------------------------------
Pipeline:
  1. Baseline: pairwise Pearson correlation of daily returns across the whole
     universe -- fast, but correlation of RETURNS says two things move together
     day-to-day; it says nothing about whether their PRICE LEVELS stay tethered
     over time (two independent random walks can have correlated short bursts by
     chance and still drift arbitrarily far apart).
  2. What we actually want for stat arb is COINTEGRATION: a linear combination of
     the two price series that is stationary (mean-reverting), which is the
     property that makes a pairs trade well-defined (the spread has to come back).
     We test this with the Engle-Granger two-step method (OLS hedge ratio, then
     ADF test on the residual spread) on the highest-correlation candidate pairs
     from step 1 (screening on correlation first keeps the number of ADF tests --
     and false-discovery risk from multiple testing -- manageable).
  3. Lead-lag: for pairs that pass, we check the cross-correlation of returns at
     lags -10..+10 days to see whether one name's move tends to precede the
     other's (informs which leg to react faster on when trading the spread).
  4. Sector / rank context: we report which sector each pair belongs to, since a
     pair cointegrating WITHIN sector is easy to rationalize (shared fundamentals);
     a pair cointegrating ACROSS sectors is more interesting/surprising and worth
     a sanity check against being a statistical artifact.
"""
import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller


def correlation_screen(price_panel: pd.DataFrame, top_k: int = 60) -> pd.DataFrame:
    ret = np.log(price_panel).diff().dropna()
    corr = ret.corr()
    pairs = []
    for a, b in itertools.combinations(corr.columns, 2):
        pairs.append((a, b, corr.loc[a, b]))
    df = pd.DataFrame(pairs, columns=["a", "b", "corr"])
    return df.reindex(df["corr"].abs().sort_values(ascending=False).index).head(top_k).reset_index(drop=True)


def cointegration_test(price_panel: pd.DataFrame, candidates: pd.DataFrame,
                        p_thresh: float = 0.05) -> pd.DataFrame:
    rows = []
    for _, row in candidates.iterrows():
        a, b = row["a"], row["b"]
        pa, pb = price_panel[a].dropna(), price_panel[b].dropna()
        idx = pa.index.intersection(pb.index)
        pa, pb = np.log(pa.loc[idx]), np.log(pb.loc[idx])
        if len(idx) < 250:
            continue
        score, pvalue, _ = coint(pa, pb)
        # hedge ratio via OLS, then ADF on the residual spread for a human-readable half-life
        beta = np.polyfit(pb, pa, 1)[0]
        spread = pa - beta * pb
        adf_stat, adf_p, *_ = adfuller(spread)
        # half-life of mean reversion from an AR(1) fit on the spread
        spread_lag = spread.shift(1).dropna()
        spread_now = spread.loc[spread_lag.index]
        phi = np.polyfit(spread_lag, spread_now, 1)[0]
        half_life = -np.log(2) / np.log(abs(phi)) if 0 < abs(phi) < 1 else np.nan

        rows.append({
            "a": a, "b": b, "corr": row["corr"], "coint_pvalue": pvalue,
            "adf_pvalue": adf_p, "hedge_ratio": beta, "half_life_days": half_life,
        })
    out = pd.DataFrame(rows).sort_values("coint_pvalue")
    out["significant"] = out["coint_pvalue"] < p_thresh
    return out


def lead_lag(price_panel: pd.DataFrame, a: str, b: str, max_lag: int = 10) -> pd.DataFrame:
    ret = np.log(price_panel[[a, b]]).diff().dropna()
    lags = range(-max_lag, max_lag + 1)
    xcorr = [ret[a].corr(ret[b].shift(lag)) for lag in lags]
    return pd.DataFrame({"lag": list(lags), "xcorr": xcorr})
