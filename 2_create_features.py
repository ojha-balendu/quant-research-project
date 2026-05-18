# %%
import pandas as pd

from technical_indicators import calculate_all_indicators_parallel
import warnings
warnings.filterwarnings("ignore")
import requests
from io import BytesIO
import os

# %%
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "stores_created")

os.makedirs(DATA_DIR, exist_ok=True)

# %%
pv = pd.read_parquet(os.path.join(BASE_DIR, "all_prices_5000_tickers.parquet"), engine="pyarrow")
# %%
# Calculate technical indicators using parellel processing
# Please read file 'technical_indicators.py' for details on the indicators being calculated
# Note this is a computationally intensive step and may take some time to complete

indicators = calculate_all_indicators_parallel(pv, n_jobs=-1)

# If indicators is a dict → merge it
if isinstance(indicators, dict):
    indicators = pd.concat(indicators, axis=1)

# Optional but recommended: name levels
if isinstance(indicators.columns, pd.MultiIndex):
    indicators.columns.names = ["feature", "ticker"]

# Downcast to save memory
indicators = indicators.astype("float32")

# Save
indicators.to_parquet(
    os.path.join(DATA_DIR, "features.parquet"),
    compression="zstd",
    engine="pyarrow"
)

