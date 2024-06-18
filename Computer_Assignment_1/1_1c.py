"""
Task: Calculate the European call and put price using standard MC, determine a 95% confidence interval
"""


import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm  # Normal and lognormal pdf, cdf, and inverse cdf/pdf

# Set display options for pandas
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.width', None)  # Auto-detect width to fit the screen


def generate_stock_paths(S0, r, sigma, T, n_paths, n_years):
    """
    Generate stock price paths using Geometric Brownian Motion.

    Parameters:
    S0 (float): Initial stock price
    r (float): Risk-free interest rate
    sigma (float): Volatility
    T (int): Time to maturity in years
    n_paths (int): Number of simulation paths
    n_years (int): Number of years for simulation

    Returns:
    np.ndarray: Simulated stock price paths
    """
    dt = T / n_years  # Time step
    Z = np.random.standard_normal((n_paths, n_years + 1))
    S = np.full((n_paths, n_years + 1), S0)
    S[:, 1:] = S0 * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt + np.sqrt(dt) * sigma * Z[:, 1:], axis=1))
    return S


def calculate_option_prices(S, K, r, T):
    """
    Calculate call and put option prices and their confidence intervals.

    Parameters:
    S (np.ndarray): Simulated stock price paths
    K (list): List of strike prices
    r (float): Risk-free interest rate
    T (int): Time to maturity in years

    Returns:
    pd.DataFrame: DataFrame containing option prices and confidence intervals
    """
    K = np.array(K)
    call_prices = np.exp(-r * T) * np.maximum(S[:, -1, np.newaxis] - K, 0)
    put_prices = np.exp(-r * T) * np.maximum(K - S[:, -1, np.newaxis], 0)

    call_means = np.mean(call_prices, axis=0)
    put_means = np.mean(put_prices, axis=0)

    call_stds = np.std(call_prices, axis=0)
    put_stds = np.std(put_prices, axis=0)

    call_ses = call_stds / np.sqrt(call_prices.shape[0])
    put_ses = put_stds / np.sqrt(put_prices.shape[0])

    confidence = 0.95
    z_score = stats.norm.ppf((1 + confidence) / 2)

    call_cis = np.column_stack((call_means - z_score * call_ses, call_means + z_score * call_ses))
    put_cis = np.column_stack((put_means - z_score * put_ses, put_means + z_score * put_ses))

    results = []
    for i, strike in enumerate(K):
        results.append({
            'Strike Price': strike,
            'Call Price': call_means[i],
            'Call CI Lower': call_cis[i, 0],
            'Call CI Upper': call_cis[i, 1],
            'Put Price': put_means[i],
            'Put CI Lower': put_cis[i, 0],
            'Put CI Upper': put_cis[i, 1]
        })

    return pd.DataFrame(results)


# Parameters
S0 = 100  # Initial stock price
r = 0.03  # Risk-free interest rate
sigma = 0.2  # Volatility
T = 1  # Time to maturity in years
n_paths = 20000  # Number of simulation paths
n_years = 1  # Number of years for simulation
strike_prices = [70, 100, 130]  # List of strike prices

# Generate stock price paths
stock_paths = generate_stock_paths(S0, r, sigma, T, n_paths, n_years)

# Calculate option prices and confidence intervals
option_prices_df = calculate_option_prices(stock_paths, strike_prices, r, T)

# Display the results
print(option_prices_df)
