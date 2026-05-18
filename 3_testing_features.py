# %%
import pandas as pd
import numpy as np

import datetime
import os, sys
import importlib
import utils
importlib.reload(utils)

from utils import plot_series, plot_series_with_names, plot_series_bar
from utils import plot_dataframe
from utils import get_universe_adjusted_series, scale_weights_to_one, scale_to_book_long_short
from utils import generate_portfolio, backtest_portfolio
from utils import match_implementations

import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")

# %%
from io import BytesIO
import requests
import pandas as pd
from tqdm import tqdm
import ast

def download_with_progress(url):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    buffer = BytesIO()
    with tqdm(total=total_size, unit='B', unit_scale=True, desc=url.split("/")[-1]) as pbar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
            if chunk:
                buffer.write(chunk)
                pbar.update(len(chunk))

    buffer.seek(0)
    return buffer

# %%
BASE_DIR = "../"
DATA_DIR = os.path.join(BASE_DIR, "stores")

features = pd.read_parquet(os.path.join(DATA_DIR, "features.parquet"))

universe = pd.read_parquet(os.path.join(DATA_DIR, "universe_5m.parquet"))

returns = pd.read_parquet(os.path.join(DATA_DIR, "returns.parquet"))

# %%
sector_mapping = pd.read_csv(os.path.join(BASE_DIR, "top_5000_us_by_marketcap.csv")).set_index("symbol").sector

# %%
# Testing the backtest_portfolio function for a specific signal

def generate_portfolio_vectorized(
    entire_features: pd.DataFrame,
    universe: pd.DataFrame,
    start_date: str,
    end_date: str,
    signal_column: str,
    neutralize_by: str = None,
    sector_mapping: pd.DataFrame = None
):
    # Validate date format
    try:
        start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        cutoff_date = datetime.datetime.strptime('2005-01-01', '%Y-%m-%d')
    except ValueError:
        raise ValueError("start_date and end_date must be strings in 'YYYY-MM-DD' format.")

    # Ensure start_date is before end_date
    if start_dt >= end_dt:
        raise ValueError("start_date must be earlier than end_date.")

    # Ensure start_date is not before '2005-01-01'
    if start_dt < cutoff_date:
        raise ValueError("start_date must be later than '2005-01-01'.")

    # Get trading days within the date range
    trading_days = universe.index[(universe.index >= start_dt) & (universe.index <= end_dt)]

    if len(trading_days) == 0:
        raise ValueError("No Trading Days in the specified dates")

    portfolio = 0

    universe_boolean = universe.loc[:end_date].astype(bool)

    features_ = entire_features.loc[:end_date]

    signal1 = features_[signal_column].shift(5)
    signal1 = signal1.where(universe_boolean, np.nan)
    signal1 = signal1.rank(axis=1, method="min", ascending=True)
    if neutralize_by == "market":
        signal1 = signal1.sub(signal1.mean(axis=1), axis=0)
        signal1 = signal1.sub(signal1.mean(axis=1), axis=0)
        signal1 = signal1.div(signal1.abs().sum(axis=1), axis=0)
        signal1 = signal1.fillna(0)
    if neutralize_by == "sector" and sector_mapping is not None:
        sector_mapping = sector_mapping.reindex(signal1.columns, fill_value="Others")
        sector_mapping = sector_mapping.loc[signal1.columns]
        signal1 = signal1.sub(signal1.groupby(sector_mapping, axis=1).transform("mean"), axis=0)
        signal1 = signal1.div(signal1.abs().sum(axis=1), axis=0)
        signal1 = signal1.fillna(0)

    portfolio = -1 * signal1

    portfolio = portfolio.div(portfolio.abs().sum(axis=1), axis=0)

    return portfolio.fillna(0).loc[start_date:end_date]

# %%
benchmark_portfolio_vectorized = generate_portfolio_vectorized(
    features,
    universe,
    "2010-01-01",
    "2026-01-01",
    "accumulation_distribution_index",
    "sector",
    sector_mapping
)

# %%
benchmark_portfolio_vectorized

# %%
sr_vectorized, pnl_vectorized = backtest_portfolio(benchmark_portfolio_vectorized.loc["2010":"2019"], returns.loc["2010":"2019"], universe.loc["2010":"2019"], True, True)

# %%
feature_results = pd.DataFrame(columns=["feature", "sharpe_ratio"])
feature_pnls = {}
for col in features.columns.get_level_values(0).unique():
    print(f"Testing feature: {col}")
    portfolio_vectorized = generate_portfolio_vectorized(
        features,
        universe,
        "2010-01-01",
        "2026-01-01",
        col,
        "sector",
        sector_mapping
    )
    sr_vectorized, pnl_vectorized = backtest_portfolio(portfolio_vectorized.loc["2010":"2019"], returns.loc["2010":"2019"], universe.loc["2010":"2019"], False, False)
    new_row = pd.DataFrame([{
        "feature": col,
        "sharpe_ratio": sr_vectorized
    }])
    feature_pnls[col] = pnl_vectorized

    feature_results = pd.concat([feature_results, new_row], ignore_index=True)

# %%
top_signals = feature_results.sort_values(by="sharpe_ratio", ascending=False)
top_signals

# %%
selected_signals = pd.DataFrame(feature_pnls).loc[:, ['ichimoku_conversion', 'accumulation_distribution_index', 'average_true_range']]

# %%
selected_signals.corr()

# %%
selected_signals.to_parquet("selected_signals.parquet")


