import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import CONFIG

def detect_outliers_rolling_robust(series, window=20, threshold=3.5):
    """Detect outliers using a rolling robust method."""
    rolling_median = series.rolling(window=window, center=False, min_periods=1).median()
    rolling_mad = series.rolling(window=window, center=False, min_periods=1).apply(lambda x: np.median(np.abs(x - np.median(x))))
    robust_z = 0.6745 * (series - rolling_median) / rolling_mad
    return series[np.abs(robust_z) > threshold]

def winsorize_rolling_robust(series, window=20, threshold=3.5):
    """Winsorize a series using a rolling robust method."""
    rolling_median = series.rolling(window=window, center=False, min_periods=1).median()
    rolling_mad = series.rolling(window=window, center=False, min_periods=1).apply(lambda x: np.median(np.abs(x - np.median(x))))
    lower_bound = rolling_median - threshold * (rolling_mad / 0.6745)
    upper_bound = rolling_median + threshold * (rolling_mad / 0.6745)
    winsorized = series.clip(lower=lower_bound, upper=upper_bound)
    return winsorized

data_path = CONFIG["outlier"]["csv_path"]
save_dir = CONFIG["outlier"]["output_dir"]
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
df = pd.read_csv(data_path, index_col=0, parse_dates=True)
winsorized_all = pd.DataFrame(index=df.index)
for stock in df.columns:
    print(f"\nAnalysing stock: {stock}")
    prices = df[stock].dropna()
    if prices.empty:
        print(f"Insufficient data for {stock}.")
        continue
    daily_log_returns = np.log(prices / prices.shift(1)).dropna()
    rolling_outliers_return_robust = detect_outliers_rolling_robust(daily_log_returns)
    winsorized_returns = winsorize_rolling_robust(daily_log_returns)
    ann_vol_original = daily_log_returns.std() * np.sqrt(252)
    ann_vol_winsorized = winsorized_returns.std() * np.sqrt(252)
    ann_vol = daily_log_returns.std() * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(daily_log_returns.index, daily_log_returns, label=f"Original Daily Log Returns (Annual Vol: {ann_vol:.2%})", color='blue')
    ax.plot(winsorized_returns.index, winsorized_returns, label=f"Winsorized Returns (Annual Vol: {ann_vol_winsorized:.2%})", color='green')
    ax.scatter(rolling_outliers_return_robust.index, rolling_outliers_return_robust, color='red', label="Detected Outliers (Rolling Robust)", marker='s')
    ax.set_title(f"{stock} - Daily Log Returns and Winsorized Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log Return")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    fig_save_path = os.path.join(save_dir, f"{stock}_winsorized_outliers.png")
    plt.savefig(fig_save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved for {stock}: {fig_save_path}")
    initial_price = prices.iloc[0]
    winsorized_prices = initial_price * np.exp(winsorized_returns.cumsum())
    winsorized_all[stock] = winsorized_prices
all_csv_save_path = os.path.join(save_dir, "all_stocks_winsorized.csv")
winsorized_all.to_csv(all_csv_save_path, index=True)
print(f"All stocks winsorized data saved: {all_csv_save_path}")
