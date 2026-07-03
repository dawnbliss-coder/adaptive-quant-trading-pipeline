"""
load_data.py
=============
Single entry point for getting daily_prices.csv. Tries the REAL Kaggle dataset
first via kagglehub (exactly the loading pattern Kaggle's own "New Notebook ->
Copy code" button gives you), and only falls back to the synthetic generator
if that fails (e.g. no Kaggle API credentials configured, or no network route
to Kaggle -- which is the case in the sandbox this pipeline was built in).

Usage:
    python data/load_data.py            # fetch (or refresh) data/daily_prices.csv
    from load_data import load_daily_prices   # or import it from another script

To use the REAL dataset on your own machine:
  1. pip install kagglehub[pandas-datasets]
  2. Set up Kaggle API credentials: https://www.kaggle.com/docs/api
     (either ~/.kaggle/kaggle.json, or the KAGGLE_USERNAME / KAGGLE_KEY env vars)
  3. Run this script. It downloads iamspace/precog-quant-task-2026 and caches
     it to data/daily_prices.csv. Every other script in this project only reads
     that cached CSV, so nothing downstream needs to change.
"""
import glob
import os
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(ROOT, "daily_prices.csv")
DATASET = "iamspace/precog-quant-task-2026"


def _load_from_kaggle() -> pd.DataFrame:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter

    # The exact file name inside the dataset isn't fixed in the task brief
    # ("final file name may differ"), so we download the dataset and
    # auto-detect the CSV rather than hardcoding a path.
    path = kagglehub.dataset_download(DATASET)
    csvs = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    if not csvs:
        raise FileNotFoundError(f"No CSV found in downloaded dataset at {path}")
    preferred = [c for c in csvs if "price" in os.path.basename(c).lower()]
    target = preferred[0] if preferred else csvs[0]

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        DATASET,
        os.path.relpath(target, path),
    )
    return df


def load_daily_prices(force_refresh: bool = False) -> pd.DataFrame:
    if not force_refresh and os.path.exists(CACHE_PATH):
        return pd.read_csv(CACHE_PATH, parse_dates=["date"])

    try:
        df = _load_from_kaggle()
        print(f"Loaded REAL dataset from Kaggle ({DATASET}): {len(df):,} rows")
    except Exception as e:
        print(f"[warning] Could not load the real Kaggle dataset ({type(e).__name__}: {e})")
        print("[warning] Falling back to the synthetic stand-in dataset "
              "(see data/generate_synthetic_data.py).")
        from generate_synthetic_data import generate
        df = generate()

    df["date"] = pd.to_datetime(df["date"])
    df.to_csv(CACHE_PATH, index=False)
    return df


if __name__ == "__main__":
    df = load_daily_prices(force_refresh=True)
    print(df.head())
    print(f"\nCached to {CACHE_PATH}")
