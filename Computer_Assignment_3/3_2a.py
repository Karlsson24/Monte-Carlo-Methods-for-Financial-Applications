"""
This script simulates the paths of an underlying asset to price Asian options and calculate the Greeks (Delta and Vega)
using both forward and central difference methods. The script performs the following steps:

1. Simulate the paths of the underlying asset using the Geometric Brownian Motion model.
2. Calculate the price of the Asian option from the simulated paths.
3. Calculate the Greeks (Delta and Vega) using forward and central difference methods.
4. Print the results in a DataFrame.

Parameters:
- S0: Initial stock price.
- strike: List of strike prices for the options.
- r: Risk-free interest rate.
- sigma: Volatility of the stock.
- T: Time to maturity in years.
- m: Number of periods (time steps).
- n_paths: Number of simulated paths.
- h: Step size for Delta calculation.
- h2: Step size for Vega calculation.
"""

import numpy as np
import pandas as pd

# Parameters for the simulation and option pricing
S0 = 100
strike = [80, 100, 120]
r = 0.03
sigma = 0.2
T = 1
m = 52  # Number of periods (time steps)
n_periods = m
dt = 1 / m  # Time step size
n_paths = 20000  # Number of simulated paths
h = 1  # Step size for Delta calculation
h2 = 0.01  # Step size for Vega calculation

def simulate_paths(S0, r, sigma, T, n_periods, n_paths):
    Z = np.random.normal(size=(n_paths, n_periods + 1))
    dt = np.zeros((n_paths, n_periods + 1))
    dt[:, 1:] = T / n_periods
    dt = np.cumsum(dt, axis=1)
    W = np.zeros((n_paths, n_periods + 1))
    S = np.zeros((n_paths, n_periods + 1))
    W[:, 1:] = np.sqrt(T / n_periods) * Z[:, 1:]
    W = np.cumsum(W, axis=1)
    S[:, 0] = S0
    S[:, 1:] = S0 * np.exp((r - 0.5 * sigma ** 2) * dt[:, 1:] + sigma * W[:, 1:])
    return S

def asian_option_price_from_paths(S, K, r, T):
    average_price = np.mean(S[:, 1:], axis=1)
    payoff = np.maximum(average_price - K, 0)
    price = np.exp(-r * T) * np.mean(payoff)
    return price

def calculate_greeks(S0, K, r, sigma, T, m, n_paths, h_S, h_sigma, S_base):
    # Base price
    base_price = asian_option_price_from_paths(S_base, K, r, T)
    # Delta
    S_up = S_base * np.exp(h_S / S0)
    S_down = S_base * np.exp(-h_S / S0)
    price_up = asian_option_price_from_paths(S_up, K, r, T)
    price_down = asian_option_price_from_paths(S_down, K, r, T)
    delta_fd = (price_up - base_price) / h_S
    delta_cd = (price_up - price_down) / (2 * h_S)
    # Vega
    S_base_up = simulate_paths(S0, r, sigma + h_sigma, T, m, n_paths)
    S_base_down = simulate_paths(S0, r, sigma - h_sigma, T, m, n_paths)
    price_up_sigma = asian_option_price_from_paths(S_base_up, K, r, T)
    price_down_sigma = asian_option_price_from_paths(S_base_down, K, r, T)
    vega_fd = (price_up_sigma - base_price) / h_sigma
    vega_cd = (price_up_sigma - price_down_sigma) / (2 * h_sigma)

    return delta_fd, delta_cd, vega_fd, vega_cd

# Simulate base paths
S_base = simulate_paths(S0, r, sigma, T, m, n_paths)

# Calculate Greeks for each strike price
results = {}
for K in strike:
    delta_fd, delta_cd, vega_fd, vega_cd = calculate_greeks(S0, K, r, sigma, T, m, n_paths, h, h2, S_base)
    results[K] = {
        'Delta Forward Difference': delta_fd,
        'Delta Central Difference': delta_cd,
        'Vega Forward Difference': vega_fd,
        'Vega Central Difference': vega_cd
    }

# Convert results to DataFrame and print
results_df = pd.DataFrame(results)
print(results_df)
