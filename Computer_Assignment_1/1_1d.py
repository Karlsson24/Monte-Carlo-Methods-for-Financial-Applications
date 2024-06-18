"""
Task: Calculate the price using antithetic variables and determine a 95% confidence interval. Here you should only
use 10000 simulations to make the comparison fair.
"""


import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm  # Normal and lognormal pdf, cdf, and inverse cdf/pdf
from scipy.stats.mstats import gmean  # Geometric mean

# Set display options for pandas
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.width', None)  # Auto-detect width to fit the screen


def generate_stock_prices(S0, r, sigma, dt, n_paths):
    """
    Generate stock prices for given parameters.

    Parameters:
    S0 (float): Initial stock price
    r (float): Risk-free interest rate
    sigma (float): Volatility
    dt (float): Time step
    n_paths (int): Number of simulation paths

    Returns:
    np.ndarray: Simulated stock prices
    """
    Zi = np.random.standard_normal(n_paths)
    S1 = S0 * np.exp((r - 0.5 * sigma ** 2) * dt + np.sqrt(dt) * sigma * Zi)
    S2 = S0 * np.exp((r - 0.5 * sigma ** 2) * dt + np.sqrt(dt) * sigma * (-Zi))
    return S1, S2


def calculate_option_prices(S1, S2, K, r, n_paths):
    """
    Calculate call and put option prices and their confidence intervals.

    Parameters:
    S1 (np.ndarray): Simulated stock prices for Zi
    S2 (np.ndarray): Simulated stock prices for -Zi
    K (list): List of strike prices
    r (float): Risk-free interest rate
    n_paths (int): Number of simulation paths

    Returns:
    pd.DataFrame: DataFrame containing option prices and confidence intervals
    """
    confidence = 0.95
    z_score = stats.norm.ppf((1 + confidence) / 2)
    results = []

    for strike in K:
        y_call_list = 0.5 * (np.exp(-r) * np.maximum(S1 - strike, 0) + np.exp(-r) * np.maximum(S2 - strike, 0))
        y_put_list = 0.5 * (np.exp(-r) * np.maximum(strike - S1, 0) + np.exp(-r) * np.maximum(strike - S2, 0))

        C_mean = np.mean(y_call_list)
        P_mean = np.mean(y_put_list)
        C_std = np.std(y_call_list)
        P_std = np.std(y_put_list)

        C_se = C_std / np.sqrt(n_paths)
        P_se = P_std / np.sqrt(n_paths)

        C_ci = (C_mean - z_score * C_se, C_mean + z_score * C_se)
        P_ci = (P_mean - z_score * P_se, P_mean + z_score * P_se)

        results.append({
            'Strike Price': strike,
            'Call Price': C_mean,
            'Call CI Lower': C_ci[0],
            'Call CI Upper': C_ci[1],
            'Put Price': P_mean,
            'Put CI Lower': P_ci[0],
            'Put CI Upper': P_ci[1]
        })

    return pd.DataFrame(results)


# Parameters
S0 = 100  # Initial stock price
r = 0.03  # Risk-free interest rate
sigma = 0.2  # Volatility
dt = 1  # Time step
n_paths = 10000  # Number of simulation paths
strike_prices = [70, 100, 130]  # List of strike prices

# Generate stock prices
S1, S2 = generate_stock_prices(S0, r, sigma, dt, n_paths)

# Calculate option prices and confidence intervals
option_prices_df = calculate_option_prices(S1, S2, strike_prices, r, n_paths)

# Display the results
print(option_prices_df)
