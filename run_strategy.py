import pandas as pd
import numpy as np
import os
import datetime
from utils import scale_weights_to_one, scale_to_book_long_short

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "stores_created")

print("Loading data...")
features = pd.read_parquet(os.path.join(DATA_DIR, "features.parquet"))
universe = pd.read_parquet(os.path.join(DATA_DIR, "universe_5m.parquet"))

print("Calculating signals...")
latest_date = features.index.max()
features_latest = features.loc[[latest_date]]
universe_latest = universe.loc[[latest_date]].astype(bool)

# Calculate custom strategy
# 0.1 * ichimoku + 0.7 * macd
ichimoku = features_latest["ichimoku_conversion"] if "ichimoku_conversion" in features_latest.columns.get_level_values(0) else features_latest.xs("ichimoku_conversion", axis=1, level=0)
macd = features_latest["macd"] if "macd" in features_latest.columns.get_level_values(0) else features_latest.xs("macd", axis=1, level=0)

signal = 0.1 * ichimoku + 0.7 * macd
signal = signal.where(universe_latest.values, np.nan)

# Cross-sectional rank
signal_ranked = signal.rank(axis=1, method="min", ascending=True)

# Neutralize by market (demean and scale)
weights = signal_ranked.iloc[0].sub(signal_ranked.iloc[0].mean())
weights = weights.div(weights.abs().sum()).fillna(0)

# Create targets.csv
# Assuming $1,000,000 book size
target_notional = weights * 1_000_000

targets = pd.DataFrame({
    "internal_code": target_notional.index,
    "target_notional": target_notional.values,
    "currency": "USD"
})
# Only keep non-zero targets or universe
targets = targets[targets["target_notional"] != 0]

output_dir = os.path.join(BASE_DIR, "targets")
os.makedirs(output_dir, exist_ok=True)
targets_path = os.path.join(output_dir, "targets.csv")
targets.to_csv(targets_path, index=False)

print(f"Targets saved to {targets_path}")
