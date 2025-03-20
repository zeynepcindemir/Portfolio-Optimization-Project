import os
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sp500 import get_sp500_tickers_by_sector
from config import CONFIG
from common import (
    portfolio_performance,
    negative_sharpe_ratio,
    compute_max_drawdown
)

static_config = CONFIG["static"]
OUTPUT_DIR = static_config["output_dir"]
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def save_table_as_png(df, filename):
    """Save the given DataFrame as a PNG image with rounded numeric values."""
    df_display = df.copy()
    numeric_cols = df_display.select_dtypes(include=[np.number]).columns
    df_display[numeric_cols] = df_display[numeric_cols].round(4)
    fig, ax = plt.subplots(figsize=(max(6, df_display.shape[1] * 1.8),
                                     max(3, df_display.shape[0] * 0.6 + 2)))
    ax.axis('off')
    table = ax.table(cellText=df_display.values, colLabels=df_display.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close(fig)

def fetch_data(asset_tickers, start_date, end_date, csv_path=static_config["csv_path"]):
    """Retrieve data from CSV if available; otherwise, download using yfinance and save to CSV."""
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

def min_variance_for_target_return(returns, target_return, min_bound, max_bound):
    """
    Optimize for the minimum variance portfolio for a given target return.
    If the target return is lower than the minimum variance portfolio's return,
    the minimum variance portfolio is returned.
    """
    mu_log = returns.mean() * 252
    Sigma = returns.cov() * 252
    num_assets = len(mu_log)
    def variance_obj(w):
        return w @ Sigma @ w
    constraints_mv = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(min_bound, max_bound)] * num_assets
    initial_weights = np.ones(num_assets) / num_assets
    result_mv = minimize(variance_obj, initial_weights, method='SLSQP', constraints=constraints_mv, bounds=bounds)
    if not result_mv.success:
         raise ValueError("Min variance portfolio did not converge: " + result_mv.message)
    w_min = result_mv.x
    min_ret = portfolio_performance(w_min, returns, risk_free_rate=0, ann_factor=252)[0]
    if target_return < min_ret:
         return w_min
    def portfolio_variance(w):
        return w @ Sigma @ w
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                   {'type': 'eq', 'fun': lambda w: np.dot(w, mu_log) - target_return}]
    result = minimize(portfolio_variance, initial_weights, method='SLSQP', constraints=constraints, bounds=bounds)
    if not result.success:
        raise ValueError("Min-variance optimization did not converge: " + result.message)
    return result.x

def get_annualized_metrics(weights, returns, risk_free_rate):
    """Calculate the annualized return, volatility, and Sharpe ratio for the given portfolio weights."""
    return portfolio_performance(weights, returns, risk_free_rate, ann_factor=252)

def markowitz_optimization(returns, risk_free_rate, min_bound, max_bound):
    """Perform Markowitz optimization by minimizing the negative Sharpe ratio."""
    num_assets = len(returns.columns)
    mu_log = returns.mean() * 252
    Sigma = returns.cov() * 252
    def objective(weights):
        return negative_sharpe_ratio(weights, mu_log, Sigma, risk_free_rate)
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = [(min_bound, max_bound)] * num_assets
    initial_weights = np.ones(num_assets) / num_assets
    result = minimize(objective, initial_weights, method='SLSQP', constraints=constraints, bounds=bounds)
    if not result.success:
        raise ValueError("Markowitz optimization did not converge: " + result.message)
    return result.x

def calculate_metrics(weights, returns, risk_free_rate):
    """Calculate portfolio performance metrics based on the provided weights."""
    return portfolio_performance(weights, returns, risk_free_rate, ann_factor=252)

def plot_efficient_frontier_with_cml(returns, risk_free_rate, num_portfolios, filename="efficient_frontier.png"):
    """Plot the efficient frontier and Capital Market Line (CML) based on random portfolios."""
    num_assets = len(returns.columns)
    portfolio_returns = []
    portfolio_risks = []
    portfolio_sharpes = []
    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        pr, pv, ps = portfolio_performance(weights, returns, risk_free_rate, ann_factor=252)
        portfolio_returns.append(pr)
        portfolio_risks.append(pv)
        portfolio_sharpes.append(ps)
    portfolio_returns = np.array(portfolio_returns)
    portfolio_risks = np.array(portfolio_risks)
    portfolio_sharpes = np.array(portfolio_sharpes)
    sorted_indices = np.argsort(portfolio_risks)
    sorted_risks = portfolio_risks[sorted_indices]
    sorted_returns = portfolio_returns[sorted_indices]
    efficient_risks = []
    efficient_returns = []
    max_return_so_far = -np.inf
    for rsk, ret in zip(sorted_risks, sorted_returns):
        if ret > max_return_so_far:
            efficient_risks.append(rsk)
            efficient_returns.append(ret)
            max_return_so_far = ret
    efficient_risks = np.array(efficient_risks)
    efficient_returns = np.array(efficient_returns)
    max_sharpe_idx = np.argmax(portfolio_sharpes)
    max_sharpe_risk = portfolio_risks[max_sharpe_idx]
    max_sharpe_return = portfolio_returns[max_sharpe_idx]
    min_risk_idx = np.argmin(portfolio_risks)
    min_risk_risk = portfolio_risks[min_risk_idx]
    min_risk_return = portfolio_returns[min_risk_idx]
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(portfolio_risks, portfolio_returns, s=20, c=portfolio_sharpes,
                          cmap=plt.get_cmap('tab20', 256), marker='o', edgecolor='k', alpha=0.7)
    plt.colorbar(scatter, label='Sharpe Ratio')
    plt.plot(efficient_risks, efficient_returns, linestyle='--', color='yellow',
             label='Approx. Efficient Frontier', linewidth=2)
    plt.scatter(max_sharpe_risk, max_sharpe_return, color='red', s=60,
                label='Highest Sharpe Ratio', marker='*')
    plt.scatter(min_risk_risk, min_risk_return, color='green', s=60,
                label='Minimum Risk Portfolio', marker='D')
    cml_x = np.linspace(0, max_sharpe_risk * 1.2, 100)
    cml_y = risk_free_rate + (max_sharpe_return - risk_free_rate) / max_sharpe_risk * cml_x
    plt.plot(cml_x, cml_y, linestyle='--', color='blue', label='Capital Market Line (CML)', linewidth=2)
    plt.axhline(y=risk_free_rate, color='gray', linestyle='--', linewidth=1, label='Risk-Free Rate')
    plt.title('Efficient Frontier and CML (Markowitz) - Random Portfolios')
    plt.xlabel('Risk (Annual Std)')
    plt.ylabel('Log Return (Annual)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.show()

def plot_portfolio_allocation_bar(weights, asset_tickers, title="Portfolio Allocation - Bar Chart"):
    """Plot a bar chart showing the portfolio allocation by asset type."""
    tol = 1e-6
    all_tickers = [ticker for tickers in asset_tickers.values() for ticker in tickers]
    unique_tickers = []
    for t in all_tickers:
        if t not in unique_tickers:
            unique_tickers.append(t)
    cmap = plt.get_cmap('tab20', len(unique_tickers))
    color_dict = {ticker: cmap(i) for i, ticker in enumerate(unique_tickers)}
    num_asset_types = len(asset_tickers)
    fig, axes = plt.subplots(1, num_asset_types, figsize=(5.5 * num_asset_types, 6), sharey=True)
    if num_asset_types == 1:
        axes = [axes]
    for i, (asset_type, tickers_in_type) in enumerate(asset_tickers.items()):
        ax = axes[i]
        valid_tickers = []
        valid_weights = []
        for t in tickers_in_type:
            if t in unique_tickers:
                idx = unique_tickers.index(t)
                w = weights[idx]
                if w > tol:
                    valid_tickers.append(t)
                    valid_weights.append(w)
        if len(valid_tickers) == 0:
            ax.bar([], [])
            ax.set_title(asset_type)
            ax.set_ylim(0, 1)
            continue
        x = np.arange(len(valid_tickers))
        bar_colors = [color_dict[t] for t in valid_tickers]
        ax.bar(x, valid_weights, color=bar_colors, edgecolor='black')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(valid_tickers, rotation=45, ha='right')
        ax.set_title(asset_type)
        ax.set_ylabel("Weight")
        ax.set_ylim(0, 1)
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(OUTPUT_DIR, "portfolio_allocation.png"), bbox_inches='tight')
    plt.show()

def plot_weights_vs_target_return(returns, target_returns, asset_names, risk_free_rate, min_bound, max_bound):
    """Plot portfolio weights versus target log return as a line plot and stacked bar chart."""
    import colorsys
    tol = 1e-6
    if asset_names is None or len(asset_names) != len(returns.columns):
        asset_names = list(returns.columns)
    num_assets = len(returns.columns)
    results = []
    for t_ret in target_returns:
        w_opt = min_variance_for_target_return(returns, t_ret, min_bound, max_bound)
        ann_ret, ann_vol, shr = get_annualized_metrics(w_opt, returns, risk_free_rate)
        daily_portfolio_log_returns = (returns * w_opt).sum(axis=1)
        max_dd = compute_max_drawdown(daily_portfolio_log_returns)
        results.append((t_ret, w_opt, ann_ret, ann_vol, shr, max_dd))
    weight_matrix = np.array([res[1] for res in results])
    x_values = np.array([res[0] for res in results])
    nonzero_mask = (np.abs(weight_matrix) > tol).any(axis=0)
    weight_matrix_filtered = weight_matrix[:, nonzero_mask]
    filtered_asset_names = [asset_names[i] for i in range(num_assets) if nonzero_mask[i]]
    n_filtered = weight_matrix_filtered.shape[1]
    def generate_distinct_colors(num_colors):
        color_list = []
        for i in range(num_colors):
            hue = i / num_colors
            saturation = 0.65
            value = 0.8
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            color_list.append((r, g, b))
        return color_list
    colors = generate_distinct_colors(n_filtered)
    fig_line, ax_line = plt.subplots(figsize=(14, 8))
    for idx in range(n_filtered):
        ax_line.plot(x_values, weight_matrix_filtered[:, idx], marker='o', color=colors[idx], label=filtered_asset_names[idx])
    ax_line.set_xlabel("Target Log Return")
    ax_line.set_ylabel("Portfolio Weight")
    ax_line.set_title("Portfolio Weights vs. Target Log Return (Line Plot)")
    ax_line.grid(True)
    ax_line.set_ylim(0, max_bound+0.1)
    legend_handles_line = [Patch(facecolor=colors[i], edgecolor='black', label=filtered_asset_names[i]) for i in range(n_filtered)]
    ax_line.legend(handles=legend_handles_line, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, ncol=1, prop={'size': 8})
    fig_line.subplots_adjust(right=0.75)
    plt.savefig(os.path.join(OUTPUT_DIR, "target_return_weights_line.png"), bbox_inches='tight')
    plt.show()
    fig_bar, ax_bar = plt.subplots(figsize=(14, 8))
    bar_positions = np.arange(len(target_returns))
    bottom = np.zeros(len(target_returns), dtype=float)
    for idx in range(n_filtered):
        bars = ax_bar.bar(bar_positions, weight_matrix_filtered[:, idx], bottom=bottom, color=colors[idx], edgecolor='black')
        for j, rect in enumerate(bars):
            height = rect.get_height()
            if height > 1e-4:
                x_coord = rect.get_x() + rect.get_width() / 2
                y_coord = bottom[j] + height / 2
                ax_bar.text(x_coord, y_coord, f"{height:.2f}", ha='center', va='center', fontsize=9, color='white')
        bottom += weight_matrix_filtered[:, idx]
    ax_bar.set_xticks(bar_positions)
    ax_bar.set_xticklabels([f"{val:.2f}" for val in x_values])
    ax_bar.set_xlabel("Target Log Return")
    ax_bar.set_ylabel("Portfolio Weight")
    ax_bar.set_title("Portfolio Weights vs. Target Log Return (Stacked Bar Chart)")
    ax_bar.grid(True, axis='y')
    ax_bar.set_ylim(0, 1)
    legend_handles_bar = [Patch(facecolor=colors[i], edgecolor='black', label=filtered_asset_names[i]) for i in range(n_filtered)]
    ax_bar.legend(handles=legend_handles_bar, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, ncol=1, prop={'size': 8})
    fig_bar.subplots_adjust(right=0.75)
    plt.savefig(os.path.join(OUTPUT_DIR, "target_return_weights_stacked.png"), bbox_inches='tight')
    plt.show()
    df_metrics = pd.DataFrame({
        "Target Log Return": [res[0] for res in results],
        "Annualized Log Return": [res[2] for res in results],
        "Annualized Volatility": [res[3] for res in results],
        "Sharpe Ratio": [res[4] for res in results],
        "Max Drawdown (Log)": [res[5] for res in results]
    })
    save_table_as_png(df_metrics, "target_return_metrics.png")

if __name__ == "__main__":

    start_date = "2023-01-01"
    end_date   = "2025-01-01"

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    total_days = (end - start).days
    train_end = start + pd.Timedelta(days=int(total_days * 0.8))

    train_end_date = train_end.strftime("%Y-%m-%d")
    test_start_date = (train_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    print("Train set: {} to {}".format(start_date, train_end_date))
    print("Test set: {} to {}".format(test_start_date, end_date))

    train_start_date = "2023-01-01"
    train_end_date   = "2024-08-07"
    test_start_date  = "2024-08-08"
    test_end_date    = "2025-01-01"

    sharpe_threshold = CONFIG["sp500"]["sharpe_threshold"]
    sector_tickers_dict, sharpe_series = get_sp500_tickers_by_sector(start_date=train_start_date,
                                                                      end_date=test_end_date,
                                                                      sharpe_threshold=sharpe_threshold)
    all_tickers = [ticker for tickers in sector_tickers_dict.values() for ticker in tickers]
    print("Filtered tickers by sector:")
    for sector, tickers in sector_tickers_dict.items():
        print(f"{sector}: {tickers}")
    data = fetch_data(all_tickers, train_start_date, test_end_date, csv_path=static_config["csv_path"])
    train_data = data.loc[train_start_date:train_end_date].copy()
    test_data  = data.loc[test_start_date:test_end_date].copy()
    train_returns = np.log(train_data / train_data.shift(1)).dropna()
    test_returns  = np.log(test_data  / test_data.shift(1)).dropna()
    train_weights = markowitz_optimization(train_returns, static_config["risk_free_rate"],
                                           static_config["min_bound"], static_config["max_bound"])
    train_weights /= np.sum(train_weights)
    train_ret, train_vol, train_sharpe = calculate_metrics(train_weights, train_returns, static_config["risk_free_rate"])
    train_daily_log = (train_returns * train_weights).sum(axis=1)
    train_max_dd = compute_max_drawdown(train_daily_log)
    df_train = pd.DataFrame({
        "Metric": ["Annualized Log Return", "Annualized Volatility", "Sharpe Ratio", "Max Drawdown (Log)"],
        "Value": [train_ret, train_vol, train_sharpe, train_max_dd]
    })
    save_table_as_png(df_train, "static_markowitz_results_train.png")
    plot_efficient_frontier_with_cml(train_returns, static_config["risk_free_rate"],
                                     static_config["num_portfolios"], filename="efficient_frontier_train.png")
    plot_portfolio_allocation_bar(train_weights, sector_tickers_dict, title="Static Markowitz Portfolio Allocation (Train)")
    test_daily_log_train = (test_returns * train_weights).sum(axis=1)
    test_ret_train, test_vol_train, test_sharpe_train = calculate_metrics(train_weights, test_returns, static_config["risk_free_rate"])
    test_max_dd_train = compute_max_drawdown(test_daily_log_train)
    df_test_train = pd.DataFrame({
        "Metric": ["Annualized Log Return", "Annualized Volatility", "Sharpe Ratio", "Max Drawdown (Log)"],
        "Value": [test_ret_train, test_vol_train, test_sharpe_train, test_max_dd_train]
    })
    save_table_as_png(df_test_train, "static_markowitz_results_test_train.png")
    plt.figure(figsize=(10,6))
    cumulative_returns = test_daily_log_train.cumsum()
    plt.plot(cumulative_returns.index, cumulative_returns, label="Cumulative Log Return (Test with Train Weights)", color='magenta')
    plt.title("Test Set Performance Using Train Weights")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Log Return")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "test_cumulative_returns_train.png"), bbox_inches='tight')
    plt.show()
    plot_portfolio_allocation_bar(train_weights, sector_tickers_dict, title="Static Markowitz Portfolio Allocation (Test with Train Weights)")
    plot_weights_vs_target_return(test_returns, static_config["target_returns"], asset_names=all_tickers,
                                  risk_free_rate=static_config["risk_free_rate"],
                                  min_bound=static_config["min_bound"],
                                  max_bound=static_config["max_bound"])
