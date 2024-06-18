"""
This script simulates the paths of an underlying asset to price Asian options and calculate the Greeks (Delta and Vega) 
using the pathwise method. The script performs the following steps:

1. Simulate the paths of the underlying asset using the Geometric Brownian Motion model.
2. Calculate the price of the Asian option from the simulated paths.
3. Calculate the Greeks (Delta and Vega) using the pathwise method.
4. Print the results in a DataFrame.

Parameters:
- S0: Initial stock price.
- strike: List of strike prices for the options.
- r: Risk-free interest rate.
- sigma: Volatility of the stock.
- T: Time to maturity in years.
- m: Number of periods (time steps).
- n_paths: Number of simulated paths.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

# Parameters for the simulation and option pricing
S0 = 100
strike = [80, 100, 120]
r = 0.03
sigma = 0.2
T = 1
m = 12  # Number of periods (time steps)
n_paths = 20000  # Number of simulated paths

def simulate_paths(S0, r, sigma, T, m, num_simulations):
    dt = T / m
    S = np.zeros((num_simulations, m + 1))
    Z = np.random.standard_normal((num_simulations, m))
    W = np.zeros((num_simulations, m + 1))
    S[:, 0] = S0

    for t in range(1, m + 1):
        W[:, t] = W[:, t - 1] + np.sqrt(dt) * Z[:, t - 1]
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * (W[:, t] - W[:, t - 1]))

    return S, Z

def asian_option_price(S0, K, r, sigma, T, m, n_paths):
    S, _ = simulate_paths(S0, r, sigma, T, m, n_paths)
    average_price = np.mean(S[:, 1:], axis=1)
    payoff = np.maximum(average_price - K, 0)
    price = np.exp(-r * T) * np.mean(payoff)
    return price

def delta_pathwise(S0, K, r, sigma, T, m, num_simulations):
    S, _ = simulate_paths(S0, r, sigma, T, m, num_simulations)
    final_prices = S[:, -1]
    indicator = final_prices > K
    delta = np.exp(-r * T) * np.mean(indicator * final_prices / S0)
    return delta

def vega_pathwise(S0, K, r, sigma, T, m, num_simulations):
    dt = T / m
    S, Z = simulate_paths(S0, r, sigma, T, m, num_simulations)
    dS_dsigma = np.zeros((num_simulations, m + 1))

    for i in range(1, m + 1):
        dS_dsigma[:, i] = dS_dsigma[:, i - 1] * S[:, i] / S[:, i - 1] + \
                          S[:, i] * (-sigma * dt + np.sqrt(dt) * Z[:, i - 1])

    average_price = np.mean(S[:, 1:], axis=1)
    dS_bar_dsigma = np.mean(dS_dsigma[:, 1:], axis=1)
    vega = np.exp(-r * T) * np.mean(dS_bar_dsigma * (average_price > K))

    return vega

results = []
for K in strike:
    option_price = asian_option_price(S0, K, r, sigma, T, m, n_paths)
    delta = delta_pathwise(S0, K, r, sigma, T, m, n_paths)
    vega = vega_pathwise(S0, K, r, sigma, T, m, n_paths)
    results.append({
        'Strike Price (K)': K,
        'Time Steps (m)': m,
        'Option Price': option_price,
        'Delta': delta,
        'Vega': vega
    })

results_df = pd.DataFrame(results)
print(results_df)
