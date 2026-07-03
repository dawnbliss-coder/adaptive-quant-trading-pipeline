"""
modeling.py -- Part 2: prediction + strategy logic
-----------------------------------------------------
Target: 5-day forward cumulative log return (regression), per Part 2's guidance
to pick either classification or regression -- regression is chosen because it
preserves magnitude information that's directly usable for portfolio weighting,
not just direction.

Hint A (single model instability): we ensemble two structurally different model
families -- Ridge (linear, regularized, captures smooth cross-sectional structure)
and LightGBM (nonlinear, captures interactions/thresholds) -- and average their
cross-sectional RANKS (not raw predictions, since the two models' output scales
aren't comparable). Averaging ranks from de-correlated models is a simple, robust
form of ensembling that's hard for either model's idiosyncratic errors to dominate.

Hint B (markets evolve): instead of one static train/test split, we use walk-forward
/ purged expanding-window retraining. The model is refit every REFIT_EVERY trading
days using only data available up to that point, and only ever predicts forward in
time. This is what lets the model adapt if a relationship (e.g. which names are
momentum vs. mean-reverting) changes mid-sample -- which we deliberately built into
the synthetic data (see generate_data.py) to make this matter.

We also PURGE the last 5 days of each training window (since fwd_ret_5 on those
days peeks into the test period) to avoid leakage at the train/test boundary.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import lightgbm as lgb

RANDOM_STATE = 7


def walk_forward_predict(feat: pd.DataFrame, feat_cols: list[str],
                          train_window: int = 500, refit_every: int = 21,
                          purge: int = 5) -> pd.DataFrame:
    rank_cols = [c + "_rank" for c in feat_cols]
    feat = feat.dropna(subset=rank_cols).copy()
    dates = np.sort(feat["date"].unique())

    preds = []
    start_idx = train_window + purge
    for t0 in range(start_idx, len(dates), refit_every):
        train_end = dates[t0 - refit_every - purge] if t0 - refit_every - purge >= 0 else dates[0]
        train_start_i = max(0, t0 - train_window - refit_every)
        train_dates = dates[train_start_i: t0 - refit_every - purge]
        test_dates = dates[t0 - refit_every: min(t0, len(dates))]
        if len(train_dates) < 100 or len(test_dates) == 0:
            continue

        train = feat[feat["date"].isin(train_dates) & feat["fwd_ret_5"].notna()]
        test = feat[feat["date"].isin(test_dates)]
        if len(train) < 200 or len(test) == 0:
            continue

        Xtr, ytr = train[rank_cols], train["fwd_ret_5"].values
        Xte = test[rank_cols]

        ridge = Ridge(alpha=5.0, random_state=RANDOM_STATE).fit(Xtr, ytr)
        gbm = lgb.LGBMRegressor(
            n_estimators=150, max_depth=4, num_leaves=15,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=50, random_state=RANDOM_STATE, verbosity=-1,
        ).fit(Xtr, ytr)

        pred_ridge = ridge.predict(Xte)
        pred_gbm = gbm.predict(Xte)

        chunk = test[["date", "ticker", "fwd_ret_5"]].copy()
        chunk["pred_ridge"] = pred_ridge
        chunk["pred_gbm"] = pred_gbm
        # rank-average ensemble (Hint A)
        chunk["rank_ridge"] = chunk.groupby("date")["pred_ridge"].rank(pct=True)
        chunk["rank_gbm"] = chunk.groupby("date")["pred_gbm"].rank(pct=True)
        chunk["signal"] = 0.5 * chunk["rank_ridge"] + 0.5 * chunk["rank_gbm"]
        preds.append(chunk)

    return pd.concat(preds, ignore_index=True)


def signal_to_weights(pred: pd.DataFrame, n_long: int = 6, n_short: int = 6) -> pd.DataFrame:
    """
    Strategy logic: on each day, go long the top-`n_long` ranked names and short
    the bottom-`n_short`, dollar-neutral, equal-weighted within each leg. This is
    a standard long/short cross-sectional strategy -- market-neutral by construction,
    so PnL should reflect stock-picking skill rather than beta exposure.
    """
    out = []
    for d, g in pred.groupby("date"):
        g = g.sort_values("signal", ascending=False)
        longs = g.head(n_long).copy()
        shorts = g.tail(n_short).copy()
        longs["weight"] = 1.0 / len(longs)
        shorts["weight"] = -1.0 / len(shorts)
        out.append(pd.concat([longs, shorts]))
    w = pd.concat(out, ignore_index=True)
    return w[["date", "ticker", "signal", "weight", "fwd_ret_5"]]
