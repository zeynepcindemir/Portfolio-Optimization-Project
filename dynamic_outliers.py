import os
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
from arch.univariate.base import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sp500 import get_sp500_tickers_by_sector
from config import CONFIG
from common import (
    annualize_return,
    annualize_volatility,
    compute_sharpe_ratio,
    negative_sharpe_ratio,
    compute_max_drawdown
)
from arch import arch_model

dynamic_config = CONFIG["dynamic"]
OUTPUT_DIR = CONFIG["outlier"]["output_dir"]
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def save_table_as_png(df, filename, output_dir):
    """Save the DataFrame as a PNG image in the specified directory."""
    df_display = df.copy()
    numeric_cols = df_display.select_dtypes(include=[np.number]).columns
    df_display[numeric_cols] = df_display[numeric_cols].round(4)
    
    fig, ax = plt.subplots(
        figsize=(max(6, df_display.shape[1] * 1.8), max(3, df_display.shape[0] * 0.6 + 2))
    )
    ax.axis('off')
    table = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    print(f"Table saved to {filepath}")


def best_garch_model(series, max_p=2, max_q=2):
    """Evaluate different GARCH models and return the best model based on AIC."""
    best_aic = np.inf
    best_res = None
    best_pq = (None, None)
    for p in range(1, max_p+1):
        for q in range(1, max_q+1):
            try:
                am = arch_model(series, vol='Garch', p=p, q=q, dist='normal', rescale=False)
                res = am.fit(disp='off')
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_res = res
                    best_pq = (p, q)
            except Exception as e:
                print(f"Failed to fit GARCH for p={p}, q={q}: {e}")
    return best_res, best_pq, best_aic

def evaluate_garch_models(series, max_p=2, max_q=2):
    """Evaluate GARCH models for different (p, q) combinations and return a DataFrame with AIC and BIC."""
    results = []
    for p in range(1, max_p+1):
        for q in range(1, max_q+1):
            try:
                am = arch_model(series, vol='Garch', p=p, q=q, dist='normal', rescale=False)
                res = am.fit(disp='off')
                results.append({
                    'p': p,
                    'q': q,
                    'AIC': res.aic,
                    'BIC': res.bic
                })
            except:
                results.append({
                    'p': p,
                    'q': q,
                    'AIC': np.nan,
                    'BIC': np.nan
                })
    df = pd.DataFrame(results)
    
    best_aic_idx = None
    best_bic_idx = None
    
    if not df['AIC'].dropna().empty:
        best_aic_idx = df['AIC'].idxmin()
    if not df['BIC'].dropna().empty:
        best_bic_idx = df['BIC'].idxmin()
    
    return df, best_aic_idx, best_bic_idx

def plot_garch_evaluation_table(series, max_p=2, max_q=2, filename="garch_evaluation.png", output_dir=OUTPUT_DIR):
    """Plot a table of GARCH model evaluation metrics (AIC and BIC)."""
    if not dynamic_config["use_garch"]:
        print("GARCH calculations are disabled (use_garch=False).")
        return None, None, None

    df, best_aic_idx, best_bic_idx = evaluate_garch_models(series, max_p, max_q)
    
    df_display = df.copy()
    numeric_cols = df_display.select_dtypes(include=[np.number]).columns
    df_display[numeric_cols] = df_display[numeric_cols].round(4)
    
    fig, ax = plt.subplots(
        figsize=(max(6, df_display.shape[1] * 1.8), max(3, df_display.shape[0] * 0.6 + 2))
    )
    ax.axis('off')
    
    table = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    row_offset = 1
    
    if best_aic_idx is not None:
        row_aic = best_aic_idx + row_offset
        col_aic = 2
        cell = table.get_celld().get((row_aic, col_aic))
        if cell:
            cell.set_text_props(fontweight='bold')
    
    if best_bic_idx is not None:
        row_bic = best_bic_idx + row_offset
        col_bic = 3
        cell = table.get_celld().get((row_bic, col_bic))
        if cell:
            cell.set_text_props(fontweight='bold')
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Table saved to {filepath}")
    
    return df, best_aic_idx, best_bic_idx


def fetch_data(asset_tickers, start_date, end_date, csv_path=CONFIG["outlier"]["winsorized_csv_path"]):
    """Fetch asset price data from CSV if available; otherwise, download from yfinance."""
    if isinstance(asset_tickers, dict):
        all_tickers = [ticker for tickers in asset_tickers.values() for ticker in tickers]
    else:
        all_tickers = asset_tickers
    if csv_path and os.path.exists(csv_path):
        data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        data = yf.download(all_tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
        data = data.dropna()
        if csv_path:
            data.to_csv(csv_path)
    return data

def calculate_cov_matrix(data_window, use_garch=True, shrinkage=False, shrinkage_intensity=0.0, max_p=2, max_q=2):
    """Calculate the covariance matrix for a given data window, optionally using GARCH and shrinkage."""
    if use_garch:
        cond_vols = {}
        for asset in data_window.columns:
            series = data_window[asset]
            try:
                best_res, best_pq, best_aic = best_garch_model(series, max_p=max_p, max_q=max_q)
                forecast = best_res.forecast(horizon=1)
                sigma = np.sqrt(forecast.variance.iloc[-1, 0])
                cond_vols[asset] = sigma
            except Exception:
                cond_vols[asset] = series.std()
        sigma_vec = np.array([cond_vols[asset] for asset in data_window.columns])
        corr_matrix = data_window.corr()
        cov_matrix = np.outer(sigma_vec, sigma_vec) * corr_matrix.values
    else:
        cov_matrix = data_window.cov()
    if shrinkage:
        F = np.diag(np.diag(cov_matrix))
        cov_matrix = shrinkage_intensity * F + (1 - shrinkage_intensity) * cov_matrix
    return cov_matrix

def optimize_once(data_window, 
                  prev_weight=None,
                  turnover_penalty=dynamic_config["turnover_penalty"],
                  turnover_penalty_method=dynamic_config["turnover_penalty_method"],
                  use_garch=dynamic_config["use_garch"],
                  shrinkage=dynamic_config["shrinkage"],
                  shrinkage_intensity=dynamic_config["shrinkage_intensity"],
                  risk_free_rate=dynamic_config["risk_free_rate"],
                  max_weight=dynamic_config["max_weight"],
                  min_weight=dynamic_config["min_weight"],
                  epsilon=dynamic_config["epsilon"]):
    """Optimize portfolio weights for a given data window with optional turnover penalty."""

    mu = data_window.mean() * 252
    cov_matrix = calculate_cov_matrix(data_window, use_garch, shrinkage, shrinkage_intensity) * 252
    cov_matrix += epsilon * np.eye(len(mu))
    num_assets = len(mu)

    def objective(w):
        base_obj = negative_sharpe_ratio(w, mu, cov_matrix, risk_free_rate)
        penalty = 0.0
        if prev_weight is not None and turnover_penalty > 0:
            if turnover_penalty_method == "L1":
                penalty = turnover_penalty * np.sum(np.abs(w - prev_weight))
            else:
                penalty = turnover_penalty * np.sum((w - prev_weight)**2)
        return base_obj + penalty

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(min_weight, max_weight)] * num_assets
    initial_weights = np.ones(num_assets) / num_assets

    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x


def dynamic_portfolio_optimization(returns, window_size, rebalance_frequency,
                                   risk_free_rate=dynamic_config["risk_free_rate"],
                                   use_garch=dynamic_config["use_garch"],
                                   turnover_penalty=dynamic_config["turnover_penalty"],
                                   turnover_penalty_method=dynamic_config["turnover_penalty_method"],
                                   shrinkage=dynamic_config["shrinkage"],
                                   shrinkage_intensity=dynamic_config["shrinkage_intensity"],
                                   max_weight=dynamic_config["max_weight"],
                                   min_weight=dynamic_config["min_weight"],
                                   epsilon=dynamic_config["epsilon"]):
    """Perform dynamic portfolio optimization using a rolling window approach."""
    rebalancing_dates = returns.index[window_size::rebalance_frequency]
    weights_history = pd.DataFrame(index=rebalancing_dates, columns=returns.columns)
    expected_returns_history = pd.Series(index=rebalancing_dates, dtype=float)
    portfolio_risk_history = pd.Series(index=rebalancing_dates, dtype=float)
    sharpe_history = pd.Series(index=rebalancing_dates, dtype=float)
    prev_weight = None
    for date in rebalancing_dates:
        data_window = returns.loc[:date].iloc[-window_size:]
        w_opt = optimize_once(data_window,
                            prev_weight=prev_weight,
                            turnover_penalty=turnover_penalty,
                            turnover_penalty_method=turnover_penalty_method,
                            use_garch=use_garch,
                            shrinkage=shrinkage,
                            shrinkage_intensity=shrinkage_intensity,
                            risk_free_rate=risk_free_rate,
                            max_weight=max_weight,
                            min_weight=min_weight,
                            epsilon=epsilon)
        weights_history.loc[date] = w_opt
        prev_weight = w_opt.copy()
        mu = data_window.mean() * 252
        cov_matrix = calculate_cov_matrix(data_window, use_garch, shrinkage, shrinkage_intensity) * 252
        cov_matrix += epsilon * np.eye(len(mu))
        port_return = np.dot(w_opt, mu)
        port_risk = np.sqrt(w_opt.T @ cov_matrix @ w_opt)
        port_risk = max(port_risk, epsilon)
        sharpe = (port_return - risk_free_rate) / port_risk
        expected_returns_history.loc[date] = port_return
        portfolio_risk_history.loc[date] = port_risk
        sharpe_history.loc[date] = sharpe

    return weights_history, expected_returns_history, portfolio_risk_history, sharpe_history    

def compute_dynamic_portfolio_returns(returns, weights_history):
    """Compute daily portfolio log returns based on the weights history."""
    daily_weights = pd.DataFrame(index=returns.index, columns=weights_history.columns)
    for date in weights_history.index:
        daily_weights.loc[date] = weights_history.loc[date]
    daily_weights.ffill(inplace=True)
    portfolio_log_returns = (daily_weights * returns).sum(axis=1)
    return portfolio_log_returns

def compute_performance_metrics(daily_portfolio_log_returns, risk_free_rate=dynamic_config["risk_free_rate"]):
    """Compute annualized return, volatility, Sharpe ratio, and maximum drawdown."""
    ann_return = annualize_return(daily_portfolio_log_returns)
    ann_vol = annualize_volatility(daily_portfolio_log_returns)
    sharpe = compute_sharpe_ratio(ann_return, ann_vol, risk_free_rate)
    max_dd = compute_max_drawdown(daily_portfolio_log_returns)
    return ann_return, ann_vol, sharpe, max_dd

def plot_summary_metrics_table(ann_return, ann_vol, sharpe, max_dd, filename="summary_metrics.png", show_plot=True):
    """Plot a summary table of dynamic portfolio performance metrics."""
    data = {
        "Annualized Log Return": [ann_return],
        "Annualized Volatility": [ann_vol],
        "Sharpe Ratio": [sharpe],
        "Max Drawdown (Log)": [max_dd]
    }
    df = pd.DataFrame(data, index=["Dynamic Portfolio"])
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('off')
    table = ax.table(cellText=np.round(df.values, 4), colLabels=df.columns, rowLabels=df.index, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    plt.title("Dynamic Portfolio Summary Metrics", fontsize=10)
    if filename:
        plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()

def weights_history_table(weights_history, filename):
    """Save the weights history DataFrame to a CSV file."""
    weights_history.to_csv(os.path.join(OUTPUT_DIR, filename), index=True)
    print(f"Weights history has been saved to '{os.path.join(OUTPUT_DIR, filename)}'.")

def plot_dynamic_metrics_table(expected_returns_history, portfolio_risk_history, sharpe_history, filename=None, max_rows=None, show_plot=True):
    """Plot a table of dynamic portfolio metrics."""
    metrics = pd.DataFrame({
        "Expected Log Return": expected_returns_history, 
        "Volatility": portfolio_risk_history, 
        "Sharpe Ratio": sharpe_history
    })
    if max_rows is not None and len(metrics) > max_rows:
        metrics = metrics.tail(max_rows)
    n_rows = metrics.shape[0]
    fig_height = max(2, n_rows * 0.4)
    fig_width = 8
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    table = ax.table(cellText=np.round(metrics.values, 4), colLabels=metrics.columns, rowLabels=metrics.index.strftime('%Y-%m-%d'), loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('gray')
        cell.set_linewidth(0.5)
    plt.title("Dynamic Portfolio Metrics", fontsize=12)
    try:
        table.auto_set_column_width(col=list(range(len(metrics.columns)+1)))
    except:
        pass
    if filename:
        plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_weights_by_asset_type(weights_history, asset_tickers, title="Portfolio Weights by Asset Type", savefig=None):
    """Plot portfolio weights grouped by asset type."""
    if not isinstance(asset_tickers, dict):
        plot_dynamic_weights_line(weights_history, title=title, savefig=savefig)
        return
    groups = asset_tickers
    num_groups = len(groups)
    fig, axs = plt.subplots(num_groups, 1, figsize=(10, 4*num_groups), sharex=True)
    if num_groups == 1:
        axs = [axs]
    for ax, (group_name, tickers) in zip(axs, groups.items()):
        available = [ticker for ticker in tickers if ticker in weights_history.columns]
        if not available:
            continue
        for ticker in available:
            ax.plot(weights_history.index, weights_history[ticker].astype(float), marker='o', label=ticker)
        ax.set_title(group_name)
        ax.set_ylabel("Weight")
        ax.grid(True)
        ax.legend(loc='upper left')
    axs[-1].set_xlabel("Date")
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.DateFormatter('%Y-%m-%d')
    for ax in axs:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        plt.setp(ax.get_xticklabels(), rotation=45)
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if savefig:
        plt.savefig(os.path.join(OUTPUT_DIR, savefig), bbox_inches='tight')
    plt.show()

def plot_dynamic_weights_line(weights_history, title="Dynamic Portfolio Weights (Line Chart)", savefig=None):
    """Plot dynamic portfolio weights as a line chart."""
    weights_history = weights_history.astype(float).fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    for asset in weights_history.columns:
        ax.plot(weights_history.index, weights_history[asset], marker='o', label=asset)
    ax.legend(loc='upper left')
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.grid(True)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.xticks(rotation=45, fontsize=8)
    if savefig:
        plt.savefig(os.path.join(OUTPUT_DIR, savefig), bbox_inches='tight')
    plt.show()

def plot_portfolio_metrics(expected_returns_history, portfolio_risk_history, sharpe_history, freq_label="Biweekly", savefig=None):
    """Plot portfolio performance metrics as separate line charts."""
    fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    fig.suptitle("Dynamic Portfolio Performance Metrics", fontsize=16)
    axs[0].plot(expected_returns_history.index, expected_returns_history, marker='o', linestyle='-', color='blue')
    axs[0].set_title(f"Expected Log Return ({freq_label})")
    axs[0].set_ylabel("Log Return")
    axs[0].grid(True)
    axs[1].plot(portfolio_risk_history.index, portfolio_risk_history, marker='o', linestyle='-', color='red')
    axs[1].set_title(f"Portfolio Risk (Std. Dev.) ({freq_label})")
    axs[1].set_ylabel("Risk")
    axs[1].grid(True)
    axs[2].plot(sharpe_history.index, sharpe_history, marker='o', linestyle='-', color='green')
    axs[2].set_title(f"Sharpe Ratio ({freq_label})")
    axs[2].set_ylabel("Sharpe")
    axs[2].grid(True)
    axs[2].set_xlabel("Date")
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.DateFormatter('%Y-%m-%d')
    for ax in axs:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        plt.setp(ax.get_xticklabels(), rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if savefig:
        plt.savefig(os.path.join(OUTPUT_DIR, savefig), bbox_inches='tight')
    plt.show()

def plot_average_weights_bar(weights_history, title="Average Portfolio Weights", savefig=None):
    """Plot a bar chart of average portfolio weights."""
    weights_history = weights_history.astype(float).fillna(0)
    avg_weights = weights_history.mean()
    fig, ax = plt.subplots(figsize=(8, 6))
    avg_weights.plot(kind='bar', ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Average Weight")
    ax.set_xlabel("Asset")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if savefig:
        plt.savefig(os.path.join(OUTPUT_DIR, savefig), bbox_inches='tight')
    plt.show()

def print_dynamic_metrics(expected_returns_history, portfolio_risk_history, sharpe_history):
    """Print dynamic portfolio metrics."""
    metrics = pd.DataFrame({
        "Expected Log Return": expected_returns_history, 
        "Volatility": portfolio_risk_history, 
        "Sharpe Ratio": sharpe_history
    })
    print("Dynamic Portfolio Metrics:")
    print(metrics.to_string())

def check_weights_total(weights_history, tolerance=1e-6):
    """Check that the sum of weights for each rebalancing date is within the tolerance."""
    sums = weights_history.sum(axis=1)
    print("Sum of weights for each rebalancing date:")
    print(sums)
    for date, total in sums.items():
        if abs(total - 1) > tolerance:
            print(f"Warning: Sum of weights on {date} is {total}, not within tolerance {tolerance}.")
    print("Weight check completed.")

def run_portfolio_analysis(asset_tickers, start_date, end_date, window_size, rebalance_frequency, 
                           risk_free_rate=dynamic_config["risk_free_rate"],
                           use_garch=dynamic_config["use_garch"],
                           turnover_penalty=dynamic_config["turnover_penalty"],
                           turnover_penalty_method=dynamic_config["turnover_penalty_method"],
                           shrinkage=dynamic_config["shrinkage"],
                           shrinkage_intensity=dynamic_config["shrinkage_intensity"],
                           max_weight=dynamic_config["max_weight"],
                           min_weight=dynamic_config["min_weight"],
                           epsilon=dynamic_config["epsilon"]):
    """Run dynamic portfolio analysis, including optimization, plotting, and performance metrics."""
    if isinstance(asset_tickers, dict):
        all_tickers = [ticker for tickers_list in asset_tickers.values() for ticker in tickers_list]
    else:
        all_tickers = asset_tickers
    data = fetch_data(all_tickers, start_date, end_date, csv_path=CONFIG["outlier"]["winsorized_csv_path"])
    returns = np.log(data / data.shift(1)).dropna()
    weights_history, exp_ret_hist, risk_hist, sharpe_hist = dynamic_portfolio_optimization(
        returns=returns,
        window_size=window_size,
        rebalance_frequency=rebalance_frequency,
        risk_free_rate=risk_free_rate,
        use_garch=use_garch,
        turnover_penalty=turnover_penalty,
        turnover_penalty_method=turnover_penalty_method,
        shrinkage=shrinkage,
        shrinkage_intensity=shrinkage_intensity,
        max_weight=max_weight,
        min_weight=min_weight,
        epsilon=epsilon
    )
    check_weights_total(weights_history, tolerance=1e-6)
    freq_label = f"Every {rebalance_frequency} Days"
    if isinstance(asset_tickers, dict):
        plot_weights_by_asset_type(weights_history, asset_tickers, title="Portfolio Weights by Asset Type", savefig="weights_by_asset_type.png")
    else:
        plot_dynamic_weights_line(weights_history, title=f"Portfolio Weights (Line Chart) - {freq_label}", savefig="portfolio_weights_line.png")
    plot_average_weights_bar(weights_history, title="Average Portfolio Weights Over Period", savefig="average_weights_bar.png")
    plot_portfolio_metrics(exp_ret_hist, risk_hist, sharpe_hist, freq_label=f"{freq_label}", savefig="portfolio_metrics.png")
    plot_dynamic_metrics_table(exp_ret_hist, risk_hist, sharpe_hist, filename="dynamic_metrics_table.png", max_rows=None, show_plot=True)
    print_dynamic_metrics(exp_ret_hist, risk_hist, sharpe_hist)
    daily_portfolio_log_returns = compute_dynamic_portfolio_returns(returns, weights_history)
    ann_return, ann_vol, final_sharpe, max_dd = compute_performance_metrics(daily_portfolio_log_returns, risk_free_rate=risk_free_rate)
    plot_summary_metrics_table(ann_return, ann_vol, final_sharpe, max_dd, filename="summary_metrics.png", show_plot=True)
    print("\n===== Final Performance Metrics (Full Period) =====")
    print(f"Annualized Log Return:    {ann_return:.2%}")
    print(f"Annualized Volatility:    {ann_vol:.2%}")
    print(f"Sharpe Ratio:             {final_sharpe:.2f}")
    print(f"Maximum Drawdown (Log):   {max_dd:.2%}")
    return weights_history, exp_ret_hist, risk_hist, sharpe_hist


def smape(y_true, y_pred):
    """Calculate the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def mase(y_true, y_pred):
    """Calculate the Mean Absolute Scaled Error (MASE)."""
    n = len(y_true)
    naive_mae = np.mean(np.abs(y_true[1:] - y_true[:-1]))
    model_mae = np.mean(np.abs(y_true - y_pred))
    return model_mae / (naive_mae + 1e-8)

def compute_and_plot_error_metrics(forecasted_returns, actual_returns, dates, output_dir=OUTPUT_DIR):
    """Compute and plot error metrics comparing forecasted and actual returns."""
    forecasted = np.array(forecasted_returns)
    actual = np.array(actual_returns)
    errors = forecasted - actual

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    mape = np.mean(np.abs(errors / (actual + 1e-8))) * 100
    smape_score = smape(actual, forecasted)
    mase_score = mase(actual, forecasted)

    print("Error Metrics over rebalancing periods:")
    print(f"  MAE:   {mae:.4f}")
    print(f"  RMSE:  {rmse:.4f}")
    print(f"  MAPE:  {mape:.2f}%")
    print(f"  SMAPE: {smape_score:.2f}%")
    print(f"  MASE:  {mase_score:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(dates, forecasted, marker='o', linestyle='-', color='blue', label='Forecasted Returns')
    plt.plot(dates, actual, marker='s', linestyle='-', color='green', label='Actual Returns')
    plt.plot(dates, errors, marker='d', linestyle='--', color='red', label='Error (Forecast - Actual)')
    plt.title("Backtesting: Forecasted vs Actual Returns and Errors")
    plt.xlabel("Rebalancing Date")
    plt.ylabel("Annualized Return / Error")
    plt.legend()
    plt.grid(True)
    plt.ylim(-4, 6)

    save_path = os.path.join(output_dir, "backtesting_errors.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Error plot saved to {save_path}")
    plt.show()

    return mae, rmse, mape, smape_score, mase_score


def walk_forward_backtest(asset_tickers, start_date, end_date, window_size, rebalance_frequency,
                          risk_free_rate=dynamic_config["risk_free_rate"],
                          use_garch=dynamic_config["use_garch"],
                          turnover_penalty=dynamic_config["turnover_penalty"],
                          turnover_penalty_method=dynamic_config["turnover_penalty_method"],
                          shrinkage=dynamic_config["shrinkage"],
                          shrinkage_intensity=dynamic_config["shrinkage_intensity"],
                          max_weight=dynamic_config["max_weight"],
                          min_weight=dynamic_config["min_weight"],
                          epsilon=dynamic_config["epsilon"]):
    """Perform a walk-forward backtest over the specified period."""
    data = fetch_data(asset_tickers, start_date, end_date, csv_path=CONFIG["outlier"]["winsorized_csv_path"])
    returns = np.log(data / data.shift(1)).dropna()
    dates = returns.index
    portfolio_returns = pd.Series(index=dates, dtype=float)
    prev_weight = None
    t = window_size

    forecasted_list = []
    actual_list = []
    rebal_dates_list = []

    while t < len(dates):
        in_sample_start = t - window_size
        if in_sample_start < 0:
            break
        train_window = returns.iloc[in_sample_start:t]
        w_opt = optimize_once(train_window,
                              prev_weight=prev_weight,
                              turnover_penalty=turnover_penalty,
                              turnover_penalty_method=turnover_penalty_method,
                              use_garch=use_garch,
                              shrinkage=shrinkage,
                              shrinkage_intensity=shrinkage_intensity,
                              risk_free_rate=risk_free_rate,
                              max_weight=max_weight,
                              min_weight=min_weight,
                              epsilon=epsilon)
        forecasted_return = np.dot(w_opt, train_window.mean() * 252)
        out_of_sample_end = min(t + rebalance_frequency, len(dates))
        out_of_sample_window = returns.iloc[t:out_of_sample_end]
        block_returns = [np.dot(w_opt, out_of_sample_window.loc[day]) for day in out_of_sample_window.index]
        actual_return = np.mean(block_returns) * 252

        forecasted_list.append(forecasted_return)
        actual_list.append(actual_return)
        rebal_dates_list.append(returns.index[t])

        for day in out_of_sample_window.index:
            portfolio_returns.loc[day] = np.dot(w_opt, out_of_sample_window.loc[day])
        prev_weight = w_opt
        t += rebalance_frequency

    portfolio_returns = portfolio_returns.dropna()
    ann_factor = 252
    ann_return = portfolio_returns.mean() * ann_factor
    ann_vol = portfolio_returns.std() * np.sqrt(ann_factor)
    sharpe_ratio = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0
    cum_log_returns = portfolio_returns.cumsum()
    running_max = cum_log_returns.cummax()
    drawdown = cum_log_returns - running_max
    max_drawdown = drawdown.min()

    print("Backtest Results:")
    print(f"Annualized Log Return: {ann_return:.2%}")
    print(f"Annualized Volatility: {ann_vol:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown (Log): {max_drawdown:.2%}")

    plt.figure(figsize=(10, 6))
    plt.plot(cum_log_returns.index, cum_log_returns, label='Portfolio')
    plt.title("Walk-Forward Backtest: Cumulative Log Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Log Return")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "walk_forward_backtest.png"), bbox_inches='tight')
    plt.show()

    plot_backtest_summary_table(
        ann_return=ann_return,
        ann_vol=ann_vol,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        filename="walk_forward_backtest_table.png",
        show_plot=True
    )

    compute_and_plot_error_metrics(forecasted_list, actual_list, rebal_dates_list)

    return portfolio_returns, cum_log_returns

def plot_backtest_summary_table(ann_return, ann_vol, sharpe_ratio, max_drawdown, filename="backtest_results_table.png", show_plot=True):
    """Plot a summary table of the walk-forward backtest results."""
    data = {
        "Annualized Log Return": [ann_return],
        "Annualized Volatility": [ann_vol],
        "Sharpe Ratio": [sharpe_ratio],
        "Max Drawdown (Log)": [max_drawdown]
    }
    df = pd.DataFrame(data, index=["Walk-Forward Backtest"])

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis("off")
    
    table = ax.table(
        cellText=np.round(df.values, 4),
        colLabels=df.columns,
        rowLabels=df.index,
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    plt.title("Walk-Forward Backtest Results", fontsize=10)

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"Backtest results are saved: {save_path}")

if __name__ == "__main__":

    sp500_config = CONFIG["sp500"]
    start_date = sp500_config["start_date"]
    end_date = sp500_config["end_date"]
    sharpe_threshold = sp500_config["sharpe_threshold"]
    sector_tickers_dict, sharpe_series = get_sp500_tickers_by_sector(start_date, end_date, sharpe_threshold)
    asset_tickers = sector_tickers_dict
    
    print("Filtered ticker list by sector:")
    for sector, tickers in asset_tickers.items():
        print(f"{sector}: {tickers}")
    
    csv_path = CONFIG["outlier"]["winsorized_csv_path"]
    if os.path.exists(csv_path):
        stock_data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        stock_data.ffill(inplace=True)
        stock_data.bfill(inplace=True)
    else:
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    
    if dynamic_config["use_garch"]:
        for sector, tickers in asset_tickers.items():
            for ticker in tickers:
                if ticker not in stock_data.columns:
                    print(f"{ticker} not found in CSV columns. Skipping.")
                    continue
                print(f"\n--- GARCH Evaluation for: {ticker} (Sector: {sector}) ---")
                series_t = stock_data[ticker]
                series_t = np.log(series_t / series_t.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
                if len(series_t) < 30:
                    print(f"Not enough data for {ticker} (only {len(series_t)} points). Skipping.")
                    continue
                df_garch_eval, best_aic_info, best_bic_info = plot_garch_evaluation_table(
                    series_t, max_p=2, max_q=2,
                    filename=f"garch_evaluation_{ticker}.png",
                    output_dir=OUTPUT_DIR
                )
                print("Best GARCH model based on AIC:", best_aic_info)
                print("Best GARCH model based on BIC:", best_bic_info)
    else:
        print("GARCH calculations have been disabled. (use_garch parameter is False)")

    data = stock_data
    returns = np.log(data / data.shift(1)).dropna()
    
    weights, exp_ret, risk, sharpe = run_portfolio_analysis(
        asset_tickers=asset_tickers,
        start_date=start_date,
        end_date=end_date,
        window_size=dynamic_config["window_size"],
        rebalance_frequency=dynamic_config["rebalance_frequency"],
        risk_free_rate=dynamic_config["risk_free_rate"],
        use_garch=dynamic_config["use_garch"],
        turnover_penalty=dynamic_config["turnover_penalty"],
        turnover_penalty_method=dynamic_config["turnover_penalty_method"],
        shrinkage=dynamic_config["shrinkage"],
        shrinkage_intensity=dynamic_config["shrinkage_intensity"],
        max_weight=dynamic_config["max_weight"],
        min_weight=dynamic_config["min_weight"],
        epsilon=dynamic_config["epsilon"]
    )
    weights_history_table(weights, filename="weights_history.csv")
    backtest_returns, backtest_cum_returns = walk_forward_backtest(
        asset_tickers=asset_tickers,
        start_date=start_date,
        end_date=end_date,
        window_size=dynamic_config["window_size"],
        rebalance_frequency=dynamic_config["rebalance_frequency"],
        risk_free_rate=dynamic_config["risk_free_rate"],
        use_garch=dynamic_config["use_garch"],
        turnover_penalty=dynamic_config["turnover_penalty"],
        turnover_penalty_method=dynamic_config["turnover_penalty_method"],
        shrinkage=dynamic_config["shrinkage"],
        shrinkage_intensity=dynamic_config["shrinkage_intensity"],
        max_weight=dynamic_config["max_weight"],
        min_weight=dynamic_config["min_weight"],
        epsilon=dynamic_config["epsilon"]
    )
