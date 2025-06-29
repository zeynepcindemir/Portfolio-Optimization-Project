# Computational Finance: Markowitz Portfolio Optimization Project

This project, developed for the Computational Finance course, provides a detailed implementation and comparative analysis of portfolio optimization strategies based on Harry Markowitz's Modern Portfolio Theory (MPT). The study explores static and dynamic approaches, and introduces an outlier-robust model using Winsorization to enhance performance and stability.

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [Core Concept: Modern Portfolio Theory (Markowitz)](#core-concept-modern-portfolio-theory-markowitz)
3.  [Methodology](#methodology)
    - [Data Collection and Preprocessing](#data-collection-and-preprocessing)
    - [1. Static Markowitz Optimization](#1-static-markowitz-optimization)
    - [2. Dynamic Markowitz Optimization (Rolling Window)](#2-dynamic-markowitz-optimization-rolling-window)
    - [3. Outlier Handling: Winsorization](#3-outlier-handling-winsorization)
    - [4. Improved Dynamic Markowitz with Winsorized Data](#4-improved-dynamic-markowitz-with-winsorized-data)
    - [Benchmark Model: 1/N (Naive) Portfolio](#benchmark-model-1n-naive-portfolio)
4.  [Analysis and Findings](#analysis-and-findings)
    - [Performance Comparison on Test Data](#performance-comparison-on-test-data)
    - [Key Insights](#key-insights)
5.  [Key Visualizations](#key-visualizations)
6.  [References](#references)

## Project Overview

The primary goal of this project is to determine optimal investment strategies by applying and comparing different portfolio optimization techniques. The analysis is centered around Markowitz's mean-variance framework.

The project covers the following key stages:
- **Data processing:** Fetching, cleaning, and preparing historical stock price data.
- **Static Optimization:** Implementing a classic Markowitz model on a fixed training period.
- **Dynamic Optimization:** Using a rolling-window approach to allow the portfolio to adapt to changing market conditions.
- **Outlier Analysis:** Identifying and mitigating the effect of extreme market events on the model using Winsorization.
- **Comparative Analysis:** Evaluating the performance of each model using standard financial metrics like Sharpe Ratio, Volatility, and Maximum Drawdown.

## Core Concept: Modern Portfolio Theory (Markowitz)

Introduced by Harry Markowitz in 1952, Modern Portfolio Theory (MPT) revolutionized investment management. MPT is a mathematical framework for assembling a portfolio of assets such that the expected return is maximized for a given level of risk.

The core tenets of MPT used in this project are:
- **Diversification:** The model's fundamental principle is that combining assets with low or negative correlations can reduce overall portfolio risk without sacrificing return.
- **Risk and Return:** Risk is quantified by the variance (or standard deviation/volatility) of portfolio returns, while return is the weighted average of the individual assets' expected returns.
- **The Efficient Frontier:** This is a curve representing the set of optimal portfolios that offer the highest expected return for a defined level of risk. The goal of our optimization is to find a portfolio on this frontier, specifically the one with the highest **Sharpe Ratio** (the portfolio that gives the best return per unit of risk).

## Methodology

### Data Collection and Preprocessing
1.  **Asset Selection:** A list of companies from the S&P 500 index was curated.
2.  **Data Fetching:** Historical daily closing prices were downloaded using the `yfinance` library for the period from January 1, 2023, to January 1, 2025.
3.  **Data Cleaning:** Missing values in the time series were handled using forward-fill (`ffill`) and backward-fill (`bfill`) methods to ensure data integrity.
4.  **Return Calculation:** Daily logarithmic returns were calculated from the closing prices. Log returns are used for their additive properties, which simplify multi-period return calculations.
5.  **Annualization:** Daily log returns and volatility were annualized by multiplying by a factor of 252 (the approximate number of trading days in a year).

### 1. Static Markowitz Optimization
This approach represents a "buy-and-hold" strategy based on a single optimization period.
- **Train/Test Split:** The historical data was divided into a training set (80% of the period, from `01.01.2023` to `07.08.2024`) and a test set (20%, from `08.08.2024` to `01.01.2025`).
- **Optimization:** On the training data, an optimization algorithm was used to find the portfolio weights that **maximize the Sharpe Ratio**.
- **Constraints:**
    - The sum of all weights must equal 1.
    - Individual asset weights are bounded (e.g., between 0% and 30%).
- **Evaluation:** The optimal weights found during training were then applied to the test data to evaluate the strategy's out-of-sample performance.

### 2. Dynamic Markowitz Optimization (Rolling Window)
This method is designed to be more adaptive to changing market dynamics.
- **Rolling Window:** The optimization is performed iteratively over a moving window of historical data (e.g., the last 40 trading days).
- **Rebalancing:** The portfolio weights are re-optimized at regular intervals (e.g., every 20 trading days). At each rebalancing date, the model uses the most recent data window to calculate new expected returns and covariances, then finds the new optimal weights.
- **Backtesting:** This process, known as **walk-forward backtesting**, simulates how the strategy would have performed over the entire period by stitching together the out-of-sample performance of each rebalancing step.

### 3. Outlier Handling: Winsorization
Financial returns are known to have "fat tails," meaning extreme events occur more frequently than a normal distribution would suggest. These outliers can heavily skew the covariance matrix and lead to unstable portfolio weights.
- **Outlier Detection:** Outliers in daily log returns were identified using a robust statistical method based on the **Median Absolute Deviation (MAD)**. Within a rolling window, any return that deviated from the median by more than a set threshold (e.g., 3.5 times the MAD) was flagged as an outlier.
- **Winsorization:** Detected outliers were "capped" at the upper or lower boundary defined by the threshold. This process reduces the impact of extreme values without completely removing them, preserving the data's overall structure.

### 4. Improved Dynamic Markowitz with Winsorized Data
This is the most advanced model in the study. It combines the adaptive nature of the dynamic approach with the stability of outlier-handled data.
- The entire **Dynamic Markowitz Optimization** process was repeated, but this time using the winsorized return series instead of the original raw returns.
- The goal was to test whether a more stable input dataset would lead to better and more consistent out-of-sample performance.

### Benchmark Model: 1/N (Naive) Portfolio
To contextualize the performance of the Markowitz models, a simple **1/N portfolio** was used as a benchmark. In this strategy, capital is divided equally among all `N` assets in the portfolio. It is a common, parameter-free benchmark that is often surprisingly difficult to beat.

## Analysis and Findings

### Performance Comparison on Test Data
The final out-of-sample performance of each model was compared. The following table summarizes the key metrics for the test period.

| Model                                    | Expected Return | Volatility | Sharpe Ratio | Max Drawdown |
| ---------------------------------------- | --------------- | ---------- | ------------ | ------------ |
| Static Markowitz                         | 53.68%          | 17.74%     | 2.80         | -8.8%        |
| Dynamic Markowitz                        | 51.46%          | 17.72%     | 2.68         | -8.4%        |
| **Dynamic Markowitz (Winsorize)**        | **56.32%**      | **16.29%** | **3.21**     | **-7.6%**    |
| 1/N Naive Portfolio                      | 71.51%          | 19.07%     | 3.54         | -6.18%       |

### Key Insights
- **Static vs. Dynamic:** In this specific test period, the classic Static Markowitz model slightly outperformed the standard Dynamic model in terms of Sharpe Ratio. This suggests that the market conditions during the test period were not volatile enough to make frequent rebalancing significantly advantageous.
- **The Power of Winsorization:** The **Dynamic Markowitz (Winsorize)** model was the clear winner among the Markowitz variants. By mitigating the impact of extreme daily returns, it achieved a **higher return**, **lower volatility**, and consequently the **highest Sharpe Ratio (3.21)**. It also had the lowest maximum drawdown, indicating superior risk control.
- **Benchmark Performance:** The 1/N Naive Portfolio performed exceptionally well, achieving the highest Sharpe Ratio overall. This highlights that for certain market regimes, complex optimization does not always guarantee superior performance over simple diversification. However, the 1/N portfolio does not allow for any risk management or strategic tilting.
- **Conclusion:** The study demonstrates that while classic Markowitz models provide strong performance, their stability and risk-adjusted returns can be significantly improved by incorporating robust statistical techniques like **Winsorization**. The Winsorized Dynamic model offers the best combination of adaptability and risk control among the tested optimization strategies.

## References
[1] H. M. Markowitz, "Portfolio Selection," *The Journal of Finance*, vol. 7, no. 1, pp. 77-91, Mar. 1952.
[2] E. J. Elton and M. J. Gruber, "Modern portfolio theory, 1950 to date," *Journal of Banking & Finance*, vol. 21, no. 11-12, pp. 1743-1759, Nov. 1997.
[3] *yfinance - Yahoo Finance API & Python Library*. [Online]. Available: https://pypi.org/project/yfinance/
[4] T. Bollerslev, "Generalized Autoregressive Conditional Heteroskedasticity," *Journal of Econometrics*, vol. 31, no. 3, pp. 307-327, Apr. 1986.
