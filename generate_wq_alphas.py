import os
import pandas as pd
import numpy as np
import warnings
import importlib.util

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "stores_created")

# Dynamically import the engine
wq_path = os.path.join(BASE_DIR, "5_test_worldquant_alphas.py")
spec = importlib.util.spec_from_file_location("wq_module", wq_path)
wq_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wq_module)
WorldQuantAlphas = wq_module.WorldQuantAlphas

print("Loading Pricing Data...")
pv = pd.read_parquet(os.path.join(BASE_DIR, "all_prices_5000_tickers.parquet"), engine="pyarrow")
returns = pd.read_parquet(os.path.join(DATA_DIR, "returns.parquet"))
sector_mapping = pd.read_csv(os.path.join(BASE_DIR, "top_5000_us_by_marketcap.csv")).set_index("symbol")["sector"]

print("Initializing WorldQuant Engine...")
df_volume = pv['Volume']
df_vwap = (pv['High'] + pv['Low'] + pv['Adj Close']) / 3

wq_engine = WorldQuantAlphas(pv, returns, df_volume, df_vwap, sector_mapping)

# The exact subset of unique alphas required by your sector mapping
required_alphas = [
    'alpha_001', 'alpha_006', 'alpha_016', 'alpha_023',
    'alpha_029', 'alpha_035', 'alpha_058', 'alpha_067',
    'alpha_075', 'alpha_100', 'alpha_101'
]

print(f"Generating subset of {len(required_alphas)} required WorldQuant Alphas...")
alpha_dict = {}

# Iterate only through the required alphas instead of all 101
for m in required_alphas:
    try:
        result = getattr(wq_engine, m)()
        
        # Safely convert raw numpy arrays back to DataFrames if needed
        if isinstance(result, np.ndarray):
            result = pd.DataFrame(result, index=pv['Adj Close'].index, columns=pv['Adj Close'].columns)
            
        alpha_dict[m] = result
    except Exception as e:
        print(f"Skipping {m} due to error: {e}")

wq_features = pd.concat(alpha_dict, axis=1)
wq_features.columns.names = ["feature", "ticker"]

print("Saving subset WQ Features to disk...")
wq_features = wq_features.astype("float32") # Downcast for memory
wq_features.to_parquet(os.path.join(DATA_DIR, "wq_features.parquet"), compression="zstd")
print("SUCCESS: Optimized wq_features.parquet saved!")
