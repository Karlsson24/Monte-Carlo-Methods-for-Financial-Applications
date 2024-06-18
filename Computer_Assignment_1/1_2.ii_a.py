"""
This code performs Monte Carlo simulations to estimate the prices of European call options using
Geometric Brownian Motion. The simulation generates multiple paths for the stock price and
calculates the option prices for different strike prices.

Steps:
1. Initialize the parameters for the simulation, including the number of paths, number of time steps,
   risk-free rate, volatility, and initial stock price.
2. Generate random standard normal numbers for the simulation.
3. Simulate the stock price paths using Geometric Brownian Motion.
4. Calculate the average stock price for each path.
5. Estimate the call option prices for different strike prices by discounting the payoff and averaging over all paths.
6. Print the estimated call option prices.

Parameters:
- n_paths: Number of simulation paths.
- n_month: Number of time steps (weeks).
- r: Risk-free interest rate.
- sigma: Volatility of the stock.
- S0: Initial stock price.
- K: List of strike prices.
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats.mstats import gmean  # geometric mean, used in assignment one

# Parameters
n_paths = 20000  # Number of simulation paths
n_month = 52  # Number of time steps (weeks)
r = 0.03  # Risk-free interest rate
sigma = 0.2  # Volatility
S0 = 100  # Initial stock price

# Initialize stock price matrix
S = np.zeros((n_paths, n_month + 1))
S[:, 0] = S0

# Generate random standard normal numbers for simulation
Z = np.random.standard_normal((n_paths, n_month + 1))

# Time step
dt = 1 / n_month

# Simulate stock price paths
for n in range(n_paths):
    for m in range(1, n_month + 1):
        S[n, m] = S[n, m - 1] * np.exp((r - 0.5 * sigma**2) * dt + np.sqrt(dt) * sigma * Z[n, m])

# List of strike prices
K = [70, 100, 130]

# Estimate call option prices for different strike prices
for strike in K:
    avg_price = np.mean(S[:, 1:], axis=1)
    C = np.exp(-r) * np.maximum(avg_price - strike, 0)
    C_mean = np.mean(C)
    print(f"Call option price estimate with strike: {strike}: {C_mean:.3f}")
