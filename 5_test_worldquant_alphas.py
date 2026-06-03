import os
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import warnings
from joblib import Parallel, delayed
from utils import backtest_portfolio

warnings.filterwarnings("ignore")

# ==========================================
# 1. WorldQuant Alpha Engine
# ==========================================
class WorldQuantAlphas:
    def __init__(self, df_prices, df_returns, df_volume, df_vwap, df_adv20, sector_mapping):
        self.open = df_prices['Open']
        self.close = df_prices['Adj Close'] 
        self.high = df_prices['High']
        self.low = df_prices['Low']
        self.volume = df_volume
        self.returns = df_returns
        self.vwap = df_vwap
        self.adv20 = df_adv20
        # Align sector mapping to our columns
        self.sector_mapping = sector_mapping.reindex(self.close.columns).fillna('Unknown')
        
    # --- Operator Definitions ---
    def rank(self, x): return x.rank(axis=1, pct=True)
    def delay(self, x, d): return x.shift(d)
    def correlation(self, x, y, d): return x.rolling(window=d).corr(y)
    def covariance(self, x, y, d): return x.rolling(window=d).cov(y)
    def scale(self, x, a=1): return x.div(x.abs().sum(axis=1), axis=0) * a
    def delta(self, x, d): return x.diff(d)
    def signedpower(self, x, a): return np.sign(x) * (x.abs() ** a)
    def ts_min(self, x, d): return x.rolling(window=d).min()
    def ts_max(self, x, d): return x.rolling(window=d).max()
    def ts_argmax(self, x, d): return x.rolling(window=d).apply(np.argmax, raw=True) + 1
    def ts_argmin(self, x, d): return x.rolling(window=d).apply(np.argmin, raw=True) + 1
    
    def ts_rank(self, x, d):
        def rank_last(slice_):
            if np.isnan(slice_).any(): return np.nan
            return rankdata(slice_)[-1]
        return x.rolling(window=d).apply(rank_last, raw=True)

    def decay_linear(self, x, d):
        """Weighted moving average with linearly decaying weights d, d-1, ..., 1"""
        d = int(d)
        weights = np.arange(d, 0, -1)
        weights = weights / weights.sum()
        def apply_weights(slice_):
            if np.isnan(slice_).any(): return np.nan
            return np.dot(slice_, weights)
        return x.rolling(d).apply(apply_weights, raw=True)

    def indneutralize(self, x):
        """Cross-sectionally demeaned within each sector group"""
        return x.sub(x.groupby(self.sector_mapping, axis=1).transform('mean'), axis=0)

    # --- Alpha Formulations (1 to 35 + Key Complex Ones) ---
    def alpha_001(self):
        cond = self.returns < 0
        std_ret = self.returns.rolling(20).std()
        val = std_ret.where(cond, self.close)
        return self.rank(self.ts_argmax(self.signedpower(val, 2), 5)) - 0.5

    def alpha_002(self):
        term1 = self.rank(self.delta(np.log(self.volume + 1), 2))
        term2 = self.rank((self.close - self.open) / self.open)
        return -1 * self.correlation(term1, term2, 6)

    def alpha_003(self):
        return -1 * self.correlation(self.rank(self.open), self.rank(self.volume), 10)

    def alpha_004(self):
        return -1 * self.ts_rank(self.rank(self.low), 9)

    def alpha_005(self):
        term1 = self.rank(self.open - (self.vwap.rolling(10).sum() / 10))
        term2 = -1 * np.abs(self.rank(self.close - self.vwap))
        return term1 * term2

    def alpha_006(self):
        return -1 * self.correlation(self.open, self.volume, 10)

    def alpha_007(self):
        cond = self.adv20 < self.volume
        true_val = -1 * self.ts_rank(np.abs(self.delta(self.close, 7)), 60) * np.sign(self.delta(self.close, 7))
        return true_val.where(cond, -1)

    def alpha_008(self):
        prod = self.open.rolling(5).sum() * self.returns.rolling(5).sum()
        return -1 * self.rank(prod - self.delay(prod, 10))

    def alpha_009(self):
        delta_c = self.delta(self.close, 1)
        cond1 = self.ts_min(delta_c, 5) > 0
        cond2 = self.ts_max(delta_c, 5) < 0
        return delta_c.where(cond1, delta_c.where(cond2, -delta_c))

    def alpha_010(self):
        delta_c = self.delta(self.close, 1)
        cond1 = self.ts_min(delta_c, 4) > 0
        cond2 = self.ts_max(delta_c, 4) < 0
        return self.rank(delta_c.where(cond1, delta_c.where(cond2, -delta_c)))

    def alpha_011(self):
        vwap_close = self.vwap - self.close
        term1 = self.rank(self.ts_max(vwap_close, 3)) + self.rank(self.ts_min(vwap_close, 3))
        term2 = self.rank(self.delta(self.volume, 3))
        return term1 * term2

    def alpha_012(self):
        return np.sign(self.delta(self.volume, 1)) * (-1 * self.delta(self.close, 1))

    def alpha_013(self):
        return -1 * self.rank(self.covariance(self.rank(self.close), self.rank(self.volume), 5))

    def alpha_014(self):
        return -1 * self.rank(self.delta(self.returns, 3)) * self.correlation(self.open, self.volume, 10)

    def alpha_015(self):
        corr = self.correlation(self.rank(self.high), self.rank(self.volume), 3)
        return -1 * corr.rolling(3).sum()

    def alpha_016(self):
        return -1 * self.rank(self.covariance(self.rank(self.high), self.rank(self.volume), 5))

    def alpha_017(self):
        term1 = -1 * self.rank(self.ts_rank(self.close, 10))
        term2 = self.rank(self.delta(self.delta(self.close, 1), 1))
        term3 = self.rank(self.ts_rank((self.volume / self.adv20), 5))
        return term1 * term2 * term3

    def alpha_018(self):
        term1 = self.close.diff(1).abs().rolling(5).std() + (self.close - self.open)
        term2 = self.correlation(self.close, self.open, 10)
        return -1 * self.rank(term1 + term2)

    def alpha_019(self):
        term1 = -1 * np.sign((self.close - self.delay(self.close, 7)) + self.delta(self.close, 7))
        term2 = 1 + self.rank(1 + self.returns.rolling(250).sum())
        return term1 * term2

    def alpha_020(self):
        term1 = -1 * self.rank(self.open - self.delay(self.high, 1))
        term2 = self.rank(self.open - self.delay(self.close, 1))
        term3 = self.rank(self.open - self.delay(self.low, 1))
        return term1 * term2 * term3

    def alpha_023(self):
        cond = (self.high.rolling(20).sum() / 20) < self.high
        return (-1 * self.delta(self.high, 2)).where(cond, 0)

    def alpha_024(self):
        cond = (self.delta((self.close.rolling(100).sum() / 100), 100) / self.delay(self.close, 100)) <= 0.05
        true_val = -1 * (self.close - self.ts_min(self.close, 100))
        false_val = -1 * self.delta(self.close, 3)
        return true_val.where(cond, false_val)

    def alpha_025(self):
        return self.rank(((-1 * self.returns) * self.adv20 * self.vwap) * (self.high - self.close))

    def alpha_026(self):
        corr = self.correlation(self.ts_rank(self.volume, 5), self.ts_rank(self.high, 5), 5)
        return -1 * self.ts_max(corr, 3)

    def alpha_028(self):
        return self.scale(self.correlation(self.adv20, self.low, 5) + ((self.high + self.low) / 2) - self.close)

    def alpha_032(self):
        term1 = self.scale((self.close.rolling(7).sum() / 7) - self.close)
        term2 = 20 * self.scale(self.correlation(self.vwap, self.delay(self.close, 5), 230))
        return term1 + term2

    def alpha_033(self):
        return self.rank(-1 * ((1 - (self.open / self.close)) ** 1))

    def alpha_041(self):
        return ((self.high * self.low)**0.5) - self.vwap

    def alpha_042(self):
        return self.rank((self.vwap - self.close)) / self.rank((self.vwap + self.close))

    def alpha_053(self):
        inner = ((self.close - self.low) - (self.high - self.close)) / (self.close - self.low)
        return -1 * self.delta(inner, 9)

    def alpha_054(self):
        # Adding a tiny constant to avoid division by zero
        return (-1 * ((self.low - self.close) * (self.open ** 5))) / (((self.low - self.high) * (self.close ** 5)) + 1e-6)

    def alpha_060(self):
        inner = (((self.close - self.low) - (self.high - self.close)) / ((self.high - self.low) + 1e-6)) * self.volume
        return -(1 * ((2 * self.scale(self.rank(inner))) - self.scale(self.rank(self.ts_argmax(self.close, 10)))))

    def alpha_101(self):
        return (self.close - self.open) / ((self.high - self.low) + 0.001)

    def generate_all(self):
        print("Generating Alpha Features...")
        alpha_dict = {}
        methods = [m for m in dir(self) if m.startswith('alpha_')]
        for m in methods:
            alpha_dict[m] = getattr(self, m)()
        
        features_df = pd.concat(alpha_dict, axis=1)
        features_df.columns.names = ["feature", "ticker"]
        return features_df

# ==========================================
# 2. Global Sector-Neutral Portfolio Generator
# ==========================================
def generate_global_sector_neutral_portfolio(
    entire_features: pd.DataFrame, universe: pd.DataFrame, 
    start_date: str, end_date: str, signal_column: str, 
    sector_mapping: pd.Series
):
    universe_boolean = universe.loc[start_date:end_date].astype(bool)
    
    signal1 = entire_features.xs(signal_column, axis=1, level=0).loc[start_date:end_date].shift(5)
    signal1 = signal1.where(universe_boolean, np.nan)
    
    signal1 = signal1.rank(axis=1, method="min", ascending=True)
    
    aligned_sectors = sector_mapping.reindex(signal1.columns).fillna('Unknown')
    signal1 = signal1.sub(signal1.groupby(aligned_sectors, axis=1).transform('mean'), axis=0)
    
    portfolio = -1 * signal1.fillna(0)
    return portfolio.div(portfolio.abs().sum(axis=1), axis=0).fillna(0)

# ==========================================
# 3. Parallelized Main Execution
# ==========================================
def process_feature(feature, features, universe, start_date, end_date, sector_mapping, returns_subset, sectors):
    """Worker function to process a single feature across all sectors"""
    portfolio = generate_global_sector_neutral_portfolio(
        features, universe, start_date, end_date, feature, sector_mapping
    )
    daily_stock_pnl = portfolio.shift(1) * returns_subset
    
    feature_results = []
    for sector in sectors:
        sector_tickers = sector_mapping[sector_mapping == sector].index.intersection(daily_stock_pnl.columns)
        if len(sector_tickers) == 0: continue
            
        sector_daily_pnl = daily_stock_pnl[sector_tickers].sum(axis=1)
        mean_pnl = sector_daily_pnl.mean()
        std_pnl = sector_daily_pnl.std()
        
        sharpe = np.sqrt(252) * (mean_pnl / std_pnl) if std_pnl != 0 and not np.isnan(std_pnl) else 0.0
        feature_results.append({"sector": sector, "feature": feature, "sharpe_ratio": sharpe})
        
    return feature_results

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "stores_created")

    print("Loading Data...")
    pv = pd.read_parquet(os.path.join(BASE_DIR, "all_prices_5000_tickers.parquet"), engine="pyarrow")
    universe = pd.read_parquet(os.path.join(DATA_DIR, "universe_5m.parquet"))
    returns = pd.read_parquet(os.path.join(DATA_DIR, "returns.parquet"))
    sector_mapping = pd.read_csv(os.path.join(BASE_DIR, "top_5000_us_by_marketcap.csv")).set_index("symbol")["sector"]

    df_volume = pv['Volume']
    df_vwap = (pv['High'] + pv['Low'] + pv['Adj Close']) / 3
    df_adv20 = df_volume.rolling(20).mean()

    # Pass sector_mapping to the engine so indneutralize() works perfectly
    wq_engine = WorldQuantAlphas(pv, returns, df_volume, df_vwap, df_adv20, sector_mapping)
    features = wq_engine.generate_all()

    start_date = "2025-12-01"
    end_date = "2026-06-01"
    sectors = sector_mapping.dropna().unique()
    features_list = features.columns.get_level_values(0).unique()
    returns_subset = returns.loc[start_date:end_date]

    print(f"\nRunning Parallel Sector-Attribution Backtest ({start_date} to {end_date})...")
    
    # Run backtests in parallel across all CPU cores
    all_results = Parallel(n_jobs=-1)(
        delayed(process_feature)(
            feature, features, universe, start_date, end_date, sector_mapping, returns_subset, sectors
        ) for feature in features_list
    )
    
    # Flatten the list of lists
    results = [item for sublist in all_results for item in sublist]

    results_df = pd.DataFrame(results)
    best_alphas_idx = results_df.groupby("sector")["sharpe_ratio"].idxmax()
    best_alphas = results_df.loc[best_alphas_idx].sort_values(by="sharpe_ratio", ascending=False)

    print("\n=== Top Alpha Formulas by Sector ===")
    print(best_alphas.to_string(index=False))
