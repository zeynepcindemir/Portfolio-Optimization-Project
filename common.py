import numpy as np

def compute_sharpe_ratio(ann_return, ann_vol, risk_free_rate):
    """Compute the Sharpe ratio given annualized return, volatility, and risk-free rate."""
    if ann_vol == 0:
        return 0.0
    return (ann_return - risk_free_rate) / ann_vol

def annualize_return(daily_returns, ann_factor=252):
    """Annualize the mean of daily returns."""
    return daily_returns.mean() * ann_factor

def annualize_volatility(daily_returns, ann_factor=252):
    """Annualize the volatility of daily returns."""
    return daily_returns.std() * np.sqrt(ann_factor)

def portfolio_performance(weights, returns, risk_free_rate, ann_factor=252):
    """Calculate portfolio performance: annualized log return, volatility, and Sharpe ratio."""
    mu_log = returns.mean() * ann_factor
    port_log_return = np.dot(weights, mu_log)
    Sigma = returns.cov() * ann_factor
    port_vol = np.sqrt(np.dot(weights, np.dot(Sigma, weights)))
    sharpe = compute_sharpe_ratio(port_log_return, port_vol, risk_free_rate)
    return port_log_return, port_vol, sharpe

def compute_max_drawdown(log_returns):
    """Compute the maximum drawdown from log returns."""
    cum_log_returns = log_returns.cumsum()
    running_max = cum_log_returns.cummax()
    drawdown = cum_log_returns - running_max
    return drawdown.min()

def negative_sharpe_ratio(weights, mu_log, Sigma, risk_free_rate):
    """Return the negative Sharpe ratio for optimization purposes."""
    port_return = np.dot(weights, mu_log)
    port_vol = np.sqrt(weights @ Sigma @ weights)
    if port_vol < 1e-12:
        return np.inf
    sharpe = (port_return - risk_free_rate) / port_vol
    return -sharpe
