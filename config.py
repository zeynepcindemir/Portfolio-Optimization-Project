"""Configuration settings for the project."""
CONFIG = {
    "sp500": {
        "start_date": "2023-01-01",
        "end_date": "2025-01-01",
        "sharpe_threshold": 1.5,
        "risk_free_rate": 0.04,
        "csv_path": r"C:\Users\zcindemir\Desktop\YAP471\Proje\Outputs\stock_data.csv"
    },
    "static": {
        "risk_free_rate": 0.04,
        "min_bound": 0.0,
        "max_bound": 0.3,
        "num_portfolios": 10000,
        "target_returns": [0.5, 0.6, 0.7, 0.8, 0.9],
        "csv_path": r"C:\Users\zcindemir\Desktop\YAP471\Proje\Outputs\stock_data.csv",
        "output_dir": r"C:\Users\zcindemir\Desktop\YAP471\Proje\Outputs\MarkoStatic"
    },
    "dynamic": {
        "risk_free_rate": 0.04,
        "window_size": 40,
        "rebalance_frequency": 20,
        "turnover_penalty": 0.05,
        "turnover_penalty_method": "L2",
        "use_garch": True,
        "shrinkage": True,
        "shrinkage_intensity": 0.5,
        "max_weight": 0.3,
        "min_weight": 0.0,
        "epsilon": 1e-6,
        "csv_path": r"C:\Users\zcindemir\Desktop\YAP471\Proje\Outputs\stock_data.csv",
        "output_dir": r"C:\Users\zcindemir\Desktop\YAP471\Proje\Outputs\MarkoDynamic"
    },
    "outlier": {
        "csv_path": r"C:\Users\zcindemir\Desktop\YAP471\Proje\Outputs\stock_data.csv",
        "output_dir": r"C:\Users\zcindemir\Desktop\YAP471\Proje\Outputs\Outliers",
        "winsorized_csv_path": r"C:\Users\zcindemir\Desktop\YAP471\Proje\Outputs\Outliers\all_stocks_winsorized.csv"
    }
}
