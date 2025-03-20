import os
import pandas as pd
import numpy as np
import yfinance as yf
from config import CONFIG
from common import compute_sharpe_ratio, annualize_return, annualize_volatility

sp500config = CONFIG["sp500"]

def rank_companies_by_log_sharpe(tickers, start_date, end_date, risk_free_rate=sp500config["risk_free_rate"]):
    """Rank companies by their annualized Sharpe ratio computed from log returns."""
    csv_path = sp500config.get("csv_path", None)
    if csv_path and os.path.exists(csv_path):
        data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
        data = data.dropna(axis=1, how='any')
        if csv_path:
            data.to_csv(csv_path)
    daily_log_returns = np.log(data / data.shift(1)).dropna()
    ann_log_return = annualize_return(daily_log_returns)
    ann_vol = annualize_volatility(daily_log_returns)
    sharpe_ratios = pd.Series({ticker: compute_sharpe_ratio(ann_log_return[ticker], ann_vol[ticker], risk_free_rate) 
                               for ticker in ann_log_return.index})
    return sharpe_ratios

def get_sp500_tickers_by_sector(start_date, end_date, sharpe_threshold, invalid_tickers=None):
    """Retrieve S&P 500 companies by sector that meet the specified Sharpe threshold."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = pd.read_html(url)[0]
    sp500_table.drop_duplicates(subset=["Symbol"], inplace=True)
    start_year = int(start_date.split("-")[0])
    if "Founded" in sp500_table.columns:
        sp500_table["Founded"] = pd.to_numeric(sp500_table["Founded"].astype(str).str.extract(r'(\d{4})', expand=False), errors="coerce")
        sp500_table = sp500_table.dropna(subset=["Founded"])
        sp500_table = sp500_table[sp500_table["Founded"] <= start_year]
    if invalid_tickers is None:
        invalid_tickers = ['SW', 'CEG', 'GEHC', 'BF.B', 'BRK.B']
    sp500_table = sp500_table[~sp500_table["Symbol"].isin(invalid_tickers)]
    tickers = sp500_table["Symbol"].tolist()
    sharpe_series = rank_companies_by_log_sharpe(tickers, start_date, end_date)
    sector_dict = {}
    for idx, row in sp500_table.iterrows():
        ticker = row["Symbol"]
        sector = row["GICS Sector"]
        if ticker in sharpe_series.index and sharpe_series[ticker] >= sharpe_threshold:
            if sector not in sector_dict:
                sector_dict[sector] = []
            sector_dict[sector].append(ticker)
    return sector_dict, sharpe_series

def fetch_data(asset_tickers, start_date, end_date, csv_path=None):
    """Fetch historical asset prices using yfinance and save to CSV if a path is provided."""
    data = yf.download(asset_tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
    data.ffill(inplace=True)
    data.bfill(inplace=True)
    if csv_path:
        data.to_csv(csv_path)
    return data

if __name__ == "__main__":
    sector_tickers_dict, sharpe_series = get_sp500_tickers_by_sector(sp500config["start_date"], sp500config["end_date"], sp500config["sharpe_threshold"])
    total_companies = sum(len(tickers) for tickers in sector_tickers_dict.values())
    print("S&P 500 companies by sector:", total_companies)
    for sector, tickers in sector_tickers_dict.items():
        print(f"{sector}: {tickers}")
    all_tickers = [ticker for tickers in sector_tickers_dict.values() for ticker in tickers]
    data = fetch_data(all_tickers, sp500config["start_date"], sp500config["end_date"], csv_path=sp500config.get("csv_path", None))
    print(data.head(10))
