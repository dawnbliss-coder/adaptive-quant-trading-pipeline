"""
generate_synthetic_data.py
===========================
Fallback data source, used ONLY if the real Kaggle dataset can't be loaded
(see load_data.py). Produces a daily_prices.csv with the same schema as the
official dataset (date, ticker, open, high, low, close, volume), deliberately
built with realistic structure so the rest of the pipeline has something real
to find:
  - A market-wide common factor + 6 sector factors, 30 tickers, 6 years daily
  - Per-stock idiosyncratic AR(1) drift that FLIPS regime halfway through the
    sample for a subset of names (so Part 2's walk-forward retraining has a
    genuine regime shift to adapt to)
  - 3 explicitly cointegrated ticker pairs, built via an Engle-Granger-testable
    OU-spread construction (for Part 4)
  - Injected data-quality problems: missing values, fat-finger 10x/0.1x ticks,
    frozen/stale price runs, negative/zero volume, duplicate rows (for Part 1)

Not used at all if the real Kaggle dataset loads successfully.
"""
import numpy as np
import pandas as pd


def generate(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n_sectors, tickers_per_sector = 6, 5
    n = n_sectors * tickers_per_sector          # 30 tickers
    years = 6
    t_days = years * 252

    dates = pd.bdate_range("2020-01-02", periods=t_days)
    tickers = [f"A{str(i).zfill(3)}" for i in range(n)]
    sector_of = {tk: i // tickers_per_sector for i, tk in enumerate(tickers)}

    # ---- latent factors ----
    mkt = rng.normal(0.0003, 0.010, t_days)
    sector_fac = rng.normal(0.0, 0.006, (n_sectors, t_days)) + 0.35 * mkt

    beta = rng.uniform(0.6, 1.6, n)
    sector_beta = rng.uniform(0.4, 1.2, n)
    idio_phi = rng.uniform(-0.15, 0.25, n)
    flips = rng.choice(n, size=8, replace=False)
    vol = rng.uniform(0.012, 0.035, n)

    log_ret = np.zeros((n, t_days))
    idio_prev = np.zeros(n)
    for t in range(t_days):
        phi_t = idio_phi.copy()
        if t > t_days // 2:
            phi_t[flips] *= -1.0
        shock = rng.standard_t(df=5, size=n) * vol / np.sqrt(5 / 3)
        idio = phi_t * idio_prev + shock
        idio_prev = idio
        sec = np.array([sector_fac[sector_of[tk], t] for tk in tickers])
        log_ret[:, t] = beta * mkt[t] + sector_beta * sec + idio

    # ---- 3 truly cointegrated pairs (Engle-Granger testable) ----
    for idx_a, idx_b in [(0, 1), (10, 11), (20, 21)]:
        ra = rng.normal(0.0003, 0.014, t_days)
        log_price_a = np.log(60.0) + np.cumsum(ra)
        kappa, mu, sigma = 0.05, 0.0, 0.02
        s = np.zeros(t_days)
        for t in range(1, t_days):
            s[t] = s[t - 1] + kappa * (mu - s[t - 1]) + rng.normal(0, sigma)
        log_price_b = log_price_a - s + 0.05
        log_ret[idx_a, :] = np.diff(log_price_a, prepend=log_price_a[0])
        log_ret[idx_b, :] = np.diff(log_price_b, prepend=log_price_b[0])

    # ---- build OHLCV ----
    start_price = rng.uniform(15, 250, n)
    close = start_price[:, None] * np.exp(np.cumsum(log_ret, axis=1))

    frames = []
    for i, tk in enumerate(tickers):
        c = close[i]
        o = np.empty(t_days)
        o[0] = c[0] * (1 + rng.normal(0, 0.001))
        o[1:] = c[:-1] * (1 + rng.normal(0, 0.0015, t_days - 1))
        rng_pct = np.abs(rng.normal(0, 0.006, t_days)) + 0.003
        h = np.maximum(o, c) * (1 + rng_pct)
        l = np.minimum(o, c) * (1 - rng_pct)
        base_vol = rng.uniform(2e5, 3e6)
        volu = np.maximum(1000, base_vol * (1 + 4 * np.abs(log_ret[i])) * (1 + rng.normal(0, 0.2, t_days)))
        frames.append(pd.DataFrame({
            "date": dates, "ticker": tk, "open": o, "high": h, "low": l,
            "close": c, "volume": volu.astype(int),
        }))
    df = pd.concat(frames, ignore_index=True)

    # ---- inject realistic data-quality problems ----
    n_rows = len(df)
    for col in ["open", "high", "low", "close", "volume"]:
        idx = rng.choice(n_rows, size=int(0.0015 * n_rows), replace=False)
        df.loc[idx, col] = np.nan
    idx = rng.choice(n_rows, size=25, replace=False)
    df.loc[idx, "close"] = df.loc[idx, "close"] * rng.choice([10, 0.1], size=25)
    for tk in rng.choice(df["ticker"].unique(), size=5, replace=False):
        sub = df[df["ticker"] == tk].sort_values("date")
        start = rng.integers(50, len(sub) - 10)
        freeze_val = sub.iloc[start]["close"]
        stale_idx = sub.iloc[start:start + rng.integers(3, 8)].index
        df.loc[stale_idx, "close"] = freeze_val
    idx = rng.choice(n_rows, size=15, replace=False)
    df.loc[idx, "volume"] = rng.choice([-1, 0], size=15)
    dup_idx = rng.choice(n_rows, size=10, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    df = generate()
    print(f"Generated synthetic dataset: {len(df):,} rows, {df.ticker.nunique()} tickers, "
          f"{df.date.min().date()} -> {df.date.max().date()}")
